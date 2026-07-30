from __future__ import annotations

import argparse
import json
from dataclasses import fields

import numpy as np
from gymnasium.utils.env_checker import check_env

from env.benchmark import benchmark_vector_env
from env.evaluation import evaluate_oracle, evaluate_policy, evaluate_random
from env.jump_env import JumpEnv, JumpEnvConfig
from td3.agent import BanditTD3
from td3.trainer import TrainConfig, train_distributed


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _env_config_from_metadata(metadata: dict[str, object]) -> JumpEnvConfig:
    values = metadata.get("env_config", {})
    if not isinstance(values, dict):
        return JumpEnvConfig()
    allowed = {field.name for field in fields(JumpEnvConfig)}
    return JumpEnvConfig(**{key: value for key, value in values.items() if key in allowed})


def _command_check(args: argparse.Namespace) -> None:
    env = JumpEnv()
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()
    result = evaluate_oracle(episodes=args.episodes, seed=args.seed)
    payload = {"gymnasium_check": "passed", "oracle": result.as_dict()}
    _print(payload)
    if result.success_rate < 0.99:
        raise SystemExit("Oracle success rate is below the required 99%")


def _command_baseline(args: argparse.Namespace) -> None:
    if args.policy == "oracle":
        result = evaluate_oracle(episodes=args.episodes, seed=args.seed)
    else:
        result = evaluate_random(episodes=args.episodes, seed=args.seed)
    _print(result.as_dict())


def _command_benchmark(args: argparse.Namespace) -> None:
    results = []
    for num_envs in args.env_counts:
        result = benchmark_vector_env(
            num_envs,
            transitions=args.transitions,
            seed=args.seed,
        )
        results.append(result.as_dict())
    _print(results)


def _command_train(args: argparse.Namespace) -> None:
    config = TrainConfig(
        total_transitions=args.transitions,
        num_actors=args.actors,
        envs_per_actor=args.envs_per_actor,
        actor_chunk_size=args.actor_chunk_size,
        transition_queue_size=args.transition_queue_size,
        replay_capacity=args.replay_capacity,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        random_steps=args.random_steps,
        updates_per_transition=args.updates_per_transition,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        final_eval_episodes=args.final_eval_episodes,
        learner_long_wait_seconds=args.learner_long_wait_seconds,
        seed=args.seed,
        device=args.device,
        run_root=args.run_root,
        run_name=args.run_name,
        checkpoint_path=args.checkpoint,
    )
    result = train_distributed(config)
    _print(result.as_dict())


def _command_evaluate(args: argparse.Namespace) -> None:
    agent, metadata = BanditTD3.from_checkpoint(args.checkpoint, device=args.device)
    env_config = _env_config_from_metadata(metadata)

    def policy(observation: np.ndarray, _: dict[str, object]) -> np.ndarray:
        return agent.act(observation[None, :])[0]

    result = evaluate_policy(
        policy,
        episodes=args.episodes,
        config=env_config,
        seed=args.seed,
    )
    _print(result.as_dict())


def build_parser() -> argparse.ArgumentParser:
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(
        prog="td3-cli",
        description="PyBullet 跳一跳环境与单步 TD3 actor-learner",
        formatter_class=formatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser(
        "check", help="运行 Gymnasium 接口检查和 oracle 验收", formatter_class=formatter
    )
    check.add_argument(
        "--episodes", type=int, default=1_000, help="oracle 验收回合数"
    )  # 环境正确性样本量
    check.add_argument(
        "--seed", type=int, default=10_000, help="首个环境随机种子"
    )  # 后续回合逐一递增
    check.set_defaults(func=_command_check)

    baseline = subparsers.add_parser(
        "baseline", help="评测解析或随机基准策略", formatter_class=formatter
    )
    baseline.add_argument(
        "policy", choices=["oracle", "random"], help="需要评测的基准策略"
    )  # oracle 应接近 100% 成功率
    baseline.add_argument(
        "--episodes", type=int, default=1_000, help="基准评测回合数"
    )  # 成功率统计样本量
    baseline.add_argument(
        "--seed", type=int, default=10_000, help="首个评测随机种子"
    )  # 保证不同策略使用同一批场景
    baseline.set_defaults(func=_command_baseline)

    benchmark = subparsers.add_parser(
        "benchmark", help="测试多进程向量环境吞吐量", formatter_class=formatter
    )
    benchmark.add_argument(
        "--env-counts",
        type=lambda value: [int(item) for item in value.split(",")],
        default=[1, 4, 8],
        help="逗号分隔的环境 worker 数量列表",
    )  # 例如 1,4,8 会依次执行三组测试
    benchmark.add_argument(
        "--transitions", type=int, default=2_000, help="每组至少采样的 transition 数"
    )  # 实际数量向上取整到 worker 数的倍数
    benchmark.add_argument(
        "--seed", type=int, default=123, help="吞吐量测试的环境随机种子"
    )  # 便于重复比较配置
    benchmark.set_defaults(func=_command_benchmark)

    train = subparsers.add_parser(
        "train", help="启动分布式单步 TD3 训练", formatter_class=formatter
    )
    train.add_argument(
        "--transitions", type=int, default=100_000, help="learner 接收 transition 的目标总数"
    )  # 训练停止条件；可能按 chunk 略微超出
    train.add_argument(
        "--actors", type=int, default=2, help="并行 CPU actor 进程数"
    )  # 每个 actor 持有独立策略副本
    train.add_argument(
        "--envs-per-actor", type=int, default=4, help="每个 actor 管理的 PyBullet worker 数"
    )  # actor 对这些环境执行批量推理
    train.add_argument(
        "--actor-chunk-size", type=int, default=256, help="actor 每次入队的 transition 数"
    )  # 较大 chunk 降低 IPC 开销但增加延迟
    train.add_argument(
        "--transition-queue-size", type=int, default=8, help="transition chunk 队列容量"
    )  # 队列满时 actor 阻塞且被 TensorBoard 记录
    train.add_argument(
        "--replay-capacity", type=int, default=200_000, help="learner replay buffer 容量"
    )  # 超出后覆盖最旧 transition
    train.add_argument(
        "--batch-size", type=int, default=256, help="每次 TD3 更新采样的 batch 大小"
    )  # critic 和延迟 actor 共用该 batch
    train.add_argument(
        "--learning-starts", type=int, default=2_000, help="开始梯度更新前的 replay 样本数"
    )  # 必须不小于 batch-size
    train.add_argument(
        "--random-steps", type=int, default=2_000, help="训练初期使用均匀随机动作的总步数"
    )  # 总预算会近似平均分配给 actor
    train.add_argument(
        "--updates-per-transition", type=float, default=0.25, help="每条新数据对应的 learner 更新次数"
    )  # 例如 0.25 表示每 4 条数据更新一次
    train.add_argument(
        "--log-interval", type=int, default=5_000, help="TensorBoard/终端日志间隔"
    )  # 单位为 learner 已接收的 transition 数
    train.add_argument(
        "--eval-interval", type=int, default=25_000, help="训练中无噪声策略评测间隔"
    )  # 设为 0 可关闭周期评测
    train.add_argument(
        "--eval-episodes", type=int, default=200, help="每次训练中评测的回合数"
    )  # 影响周期评测耗时和统计方差
    train.add_argument(
        "--final-eval-episodes", type=int, default=1_000, help="训练结束后的最终评测回合数"
    )  # 结果写入 checkpoint metadata 和 TensorBoard
    train.add_argument(
        "--learner-long-wait-seconds", type=float, default=1.0, help="learner 长等待告警阈值（秒）"
    )  # 超过阈值会累加 queue 健康指标
    train.add_argument(
        "--seed", type=int, default=0, help="learner、actor 和环境的基础随机种子"
    )  # 各 actor/worker 会派生互不相同的种子
    train.add_argument(
        "--device", default="cuda", help="learner 使用的 PyTorch device"
    )  # actor 始终在 CPU 上推理
    train.add_argument(
        "--run-root", default="runs", help="TensorBoard run 和默认 checkpoint 的根目录"
    )  # 每次训练会在其下创建时间戳目录
    train.add_argument(
        "--run-name", default="td3", help="追加到时间戳 run 目录后的实验名称"
    )  # 同一秒并发运行仍由 PID 区分
    train.add_argument(
        "--checkpoint",
        default=None,
        help="显式 checkpoint 输出路径；默认保存到本次 run 目录",
    )  # 通常无需设置
    train.set_defaults(func=_command_train)

    evaluate = subparsers.add_parser(
        "evaluate", help="无 GUI 评测 checkpoint", formatter_class=formatter
    )
    evaluate.add_argument(
        "checkpoint", help="需要加载的 checkpoint 文件路径"
    )  # 使用 td3.eval 可进行 GUI 演示
    evaluate.add_argument(
        "--episodes", type=int, default=1_000, help="评测回合数"
    )  # 最终输出平均 reward、误差和成功率
    evaluate.add_argument(
        "--seed", type=int, default=2_000_000, help="首个评测环境随机种子"
    )  # 后续回合逐一递增
    evaluate.add_argument(
        "--device", default="cpu", help="加载和推理模型使用的 PyTorch device"
    )  # CPU 可直接加载 CUDA 训练得到的 checkpoint
    evaluate.set_defaults(func=_command_evaluate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
