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


# RGB 输入经过 CNN，通常需要比一维距离向量更多的样本才能学到稳定策略。
# 这里仅作为 CLI 的“省略 --transitions 时”的默认预算；用户显式传参时，
# --transitions 始终覆盖该值，便于 smoke test、超参数搜索和公平对比。
DEFAULT_TRAINING_TRANSITIONS = {
    "rgb": 500_000,
    "vector": 100_000,
}


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _env_config_from_metadata(metadata: dict[str, object]) -> JumpEnvConfig:
    values = metadata.get("env_config", {})
    if not isinstance(values, dict):
        return JumpEnvConfig()
    # Checkpoints created before observation_mode existed used the distance vector;
    # checkpoints created before charge_exponent existed used linear p=1 physics.
    values = dict(values)
    values.setdefault("observation_mode", "vector")
    values.setdefault("charge_exponent", 1.0)
    allowed = {field.name for field in fields(JumpEnvConfig)}
    return JumpEnvConfig(**{key: value for key, value in values.items() if key in allowed})


def _env_config_from_args(args: argparse.Namespace) -> JumpEnvConfig:
    return JumpEnvConfig(
        observation_mode=args.observation_mode,
        observation_width=args.observation_size,
        observation_height=args.observation_size,
    )


def _add_observation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--observation-mode",
        choices=["rgb", "vector"],
        default="rgb",
        help="观测类型：两通道二值平台像素图或归一化距离向量",
    )  # 默认 rgb 的实际张量形状为 [2, H, W]
    parser.add_argument(
        "--observation-size", type=int, default=64, help="rgb 语义图的正方形边长"
    )  # vector 模式下忽略该参数


def _command_check(args: argparse.Namespace) -> None:
    config = _env_config_from_args(args)
    env = JumpEnv(config=config)
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()
    result = evaluate_oracle(episodes=args.episodes, seed=args.seed, config=config)
    payload = {"gymnasium_check": "passed", "oracle": result.as_dict()}
    _print(payload)
    if result.success_rate < 0.99:
        raise SystemExit("Oracle success rate is below the required 99%")


def _command_baseline(args: argparse.Namespace) -> None:
    config = _env_config_from_args(args)
    if args.policy == "oracle":
        result = evaluate_oracle(
            episodes=args.episodes, seed=args.seed, config=config
        )
    else:
        result = evaluate_random(
            episodes=args.episodes, seed=args.seed, config=config
        )
    _print(result.as_dict())


def _command_benchmark(args: argparse.Namespace) -> None:
    config = _env_config_from_args(args)
    results = []
    for num_envs in args.env_counts:
        result = benchmark_vector_env(
            num_envs,
            transitions=args.transitions,
            seed=args.seed,
            config=config,
        )
        results.append(result.as_dict())
    _print(results)


def _command_train(args: argparse.Namespace) -> None:
    env_config = _env_config_from_args(args)
    # argparse 保留 None，以便在解析完 observation_mode 后再选择预算：
    # RGB 的 CNN 输入默认使用 500k 条 transition，vector 默认使用 100k。
    # 一旦用户显式给出 --transitions，就完全按用户的数值训练。
    total_transitions = args.transitions
    if total_transitions is None:
        total_transitions = DEFAULT_TRAINING_TRANSITIONS[args.observation_mode]
    replay_capacity = args.replay_capacity
    if replay_capacity is None:
        replay_capacity = 50_000 if args.observation_mode == "rgb" else 200_000
    config = TrainConfig(
        total_transitions=total_transitions,
        num_actors=args.actors,
        envs_per_actor=args.envs_per_actor,
        actor_chunk_size=args.actor_chunk_size,
        transition_queue_size=args.transition_queue_size,
        replay_capacity=replay_capacity,
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
    result = train_distributed(config, env_config)
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
    _add_observation_arguments(check)
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
    _add_observation_arguments(baseline)
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
    _add_observation_arguments(benchmark)
    benchmark.set_defaults(func=_command_benchmark)

    train = subparsers.add_parser(
        "train", help="启动分布式单步 TD3 训练", formatter_class=formatter
    )
    train.add_argument(
        "--transitions",
        type=int,
        default=None,
        help="learner 接收 transition 的目标总数；省略时 rgb=500000、vector=100000",
    )  # RGB 比 vector 难，默认预算更大；显式值优先
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
        "--replay-capacity", type=int, default=None, help="learner replay buffer 容量"
    )  # 默认 rgb=50000、vector=200000，超出后覆盖最旧数据
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
        "--run-name",
        default="td3",
        help="实验名称；目录会自动追加观测模式和目标步数",
    )  # 例如 td3 -> rgb-steps500000-td3
    _add_observation_arguments(train)
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
