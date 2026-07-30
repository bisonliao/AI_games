from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from env.evaluation import EvaluationResult, evaluate_policy
from env.jump_env import JumpEnvConfig
from env.vector_env import make_async_vector_env
from td3.agent import Actor, BanditTD3, resolve_device
from td3.replay import ReplayBuffer


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Actor-learner 拓扑、优化频率、监控与输出路径的完整训练配置。"""

    total_transitions: int = 100_000
    num_actors: int = 2
    envs_per_actor: int = 4
    actor_chunk_size: int = 256
    transition_queue_size: int = 8
    replay_capacity: int = 50_000
    batch_size: int = 256
    learning_starts: int = 2_000
    random_steps: int = 2_000
    updates_per_transition: float = 0.25
    hidden_dim: int = 128
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    policy_delay: int = 2
    exploration_noise_start: float = 0.30
    exploration_noise_end: float = 0.03
    parameter_sync_interval: int = 1_000
    log_interval: int = 5_000
    eval_interval: int = 25_000
    eval_episodes: int = 200
    final_eval_episodes: int = 1_000
    learner_long_wait_seconds: float = 1.0
    seed: int = 0
    device: str = "cuda"
    run_root: str = "runs"
    run_name: str | None = None
    checkpoint_path: str | None = None

    def __post_init__(self) -> None:
        positive = {
            "total_transitions": self.total_transitions,
            "num_actors": self.num_actors,
            "envs_per_actor": self.envs_per_actor,
            "actor_chunk_size": self.actor_chunk_size,
            "replay_capacity": self.replay_capacity,
            "batch_size": self.batch_size,
            "parameter_sync_interval": self.parameter_sync_interval,
            "log_interval": self.log_interval,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.learning_starts < self.batch_size:
            raise ValueError("learning_starts must be at least batch_size")
        if self.updates_per_transition <= 0:
            raise ValueError("updates_per_transition must be positive")
        if self.learner_long_wait_seconds <= 0:
            raise ValueError("learner_long_wait_seconds must be positive")


@dataclass(frozen=True, slots=True)
class TrainResult:
    transitions: int
    updates: int
    elapsed_seconds: float
    sample_transitions_per_second: float
    final_evaluation: EvaluationResult
    checkpoint_path: str
    run_dir: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "transitions": self.transitions,
            "updates": self.updates,
            "elapsed_seconds": self.elapsed_seconds,
            "sample_transitions_per_second": self.sample_transitions_per_second,
            "final_evaluation": self.final_evaluation.as_dict(),
            "checkpoint_path": self.checkpoint_path,
            "run_dir": self.run_dir,
        }


@dataclass(slots=True)
class _ActorQueueHealth:
    sent_transitions: int = 0
    blocked_seconds: float = 0.0
    queue_full_events: int = 0
    dropped_transitions: int = 0


def _create_run_directory(root: str, run_name: str | None) -> Path:
    """Reserve a timestamped TensorBoard directory without name collisions."""
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    suffix = ""
    if run_name:
        safe_name = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in run_name
        ).strip("-_")
        if safe_name:
            suffix = f"-{safe_name}"
    base_name = f"{timestamp}-pid{os.getpid()}{suffix}"
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    for index in range(10_000):
        candidate = root_path / (
            base_name if index == 0 else f"{base_name}-{index:02d}"
        )
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not reserve a unique run directory below {root_path}")


def _training_run_name(
    observation_mode: str, total_transitions: int, run_name: str | None
) -> str:
    """为训练 run 生成可读身份，避免只看 TensorBoard 路径时混淆实验。

    ``rgb`` 和 ``vector`` 的网络、样本难度及默认训练预算不同，因此把模式和
    实际目标步数放在用户自定义名称之前。用户仍可用 ``--run-name`` 添加项目、
    seed 或超参数标签；这里不修改 checkpoint 内容，只影响 run 目录名称。
    """
    mode_label = "rgb" if observation_mode == "rgb" else "vec"
    base_name = run_name or "td3"
    return f"{mode_label}-steps{int(total_transitions)}-{base_name}"


def _safe_qsize(target_queue: Any) -> int:
    try:
        return int(target_queue.qsize())
    except (AttributeError, NotImplementedError, OSError):
        return -1


def _replace_latest(weight_queue: Any, item: Any) -> None:
    """发布最新 actor 权重，主动删除尚未被消费的旧版本。"""
    # 参数队列只表达“最新模型”状态，不承担历史消息传递；丢弃旧权重可避免
    # actor 因依次加载过期版本而长期落后 learner。
    try:
        while True:
            weight_queue.get_nowait()
    except queue.Empty:
        pass
    try:
        weight_queue.put_nowait(item)
    except queue.Full:
        # An actor may have raced us and will receive the previous, still-valid
        # version at its next polling point.
        pass


def _put_until_stopped(
    target_queue: Any, item: Any, stop_event: Any
) -> tuple[bool, float, int]:
    """将 transition chunk 无损入队，并统计背压等待时间和 queue-full 次数。"""
    started = time.perf_counter()
    full_events = 0
    try:
        target_queue.put_nowait(item)
        return True, 0.0, 0
    except queue.Full:
        full_events = 1
    while not stop_event.is_set():
        try:
            target_queue.put(item, timeout=0.1)
            return True, time.perf_counter() - started, full_events
        except queue.Full:
            full_events += 1
            continue
    return False, time.perf_counter() - started, full_events


def _emit_actor_health(
    event_queue: Any,
    actor_id: int,
    *,
    sent_transitions: int,
    blocked_seconds: float,
    queue_full_events: int,
    dropped_transitions: int,
) -> None:
    """非阻塞上报 actor 队列健康；监控消息绝不能反向阻塞采样。"""
    try:
        event_queue.put_nowait(
            {
                "kind": "actor_health",
                "actor_id": actor_id,
                "sent_transitions": sent_transitions,
                "blocked_seconds": blocked_seconds,
                "queue_full_events": queue_full_events,
                "dropped_transitions": dropped_transitions,
            }
        )
    except queue.Full:
        # Health telemetry must never block environment collection. Transition
        # data is still lossless while training is active.
        pass


def _actor_main(
    actor_id: int,
    env_config: JumpEnvConfig,
    train_config: TrainConfig,
    transition_queue: Any,
    weight_queue: Any,
    event_queue: Any,
    stop_event: Any,
) -> None:
    """CPU actor 进程：批量推理、驱动向量环境并产生 transition chunk。"""
    try:
        # 每个 actor 下面还会创建 env worker。限制 PyTorch 线程可以避免
        # actor × env × BLAS 线程相乘造成 CPU oversubscription。
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        # actor、learner 和不同 actor 使用相互分离且可复现的随机序列。
        seed = train_config.seed + 100_000 * (actor_id + 1)
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        # actor 只保留 CPU 上的 policy 副本，不会创建 CUDA context。启动后
        # 必须先拿到 learner 的初始参数，才能开始环境交互。
        actor = Actor(
            hidden_dim=train_config.hidden_dim,
            observation_shape=(
                env_config.observation_shape
                if env_config.observation_mode == "rgb"
                else None
            ),
        ).cpu().eval()
        _, weights = weight_queue.get(timeout=60.0)
        actor.load_state_dict(
            {key: torch.as_tensor(value) for key, value in weights.items()}
        )

        # AsyncVectorEnv 对 actor 呈现同步 batch step，但内部 PyBullet 环境由
        # spawn 子进程并行推进，避免单个 Python 进程顺序模拟。
        envs = make_async_vector_env(
            train_config.envs_per_actor, env_config, context="spawn"
        )
        observations, _ = envs.reset(seed=seed)
        local_transitions = 0
        # random_steps 是“所有 actor 合计”的纯随机探索预算，这里用整除把它
        # 近似平均分给每个 actor。例如 random_steps=2000、num_actors=2 时，
        # 每个 actor 的 random_budget=1000；在 local_transitions 达到 1000
        # 之前，下面的动作分支完全忽略 actor 网络，直接从 [-1, 1] 均匀采样。
        # 达到预算后才切换为“actor 输出 + 高斯探索噪声”。由于每次 vector
        # step 会同时产生 envs_per_actor 条 transition，随机阶段只能按整批
        # 切换，实际数量最多会比预算多 envs_per_actor-1 条。max 的第二项
        # 保证即使 random_steps 很小，每个 actor 也至少执行一个完整的随机
        # vector batch，从而让 replay 初始数据覆盖动作空间而非仅来自未训练模型。
        random_budget = max(
            train_config.random_steps // train_config.num_actors,
            train_config.envs_per_actor,
        )
        expected_per_actor = max(
            train_config.total_transitions / train_config.num_actors, 1.0
        )
        # 先在 actor 本地累积多个 vector step，再整块跨进程传输，以摊薄
        # pickle、pipe 和进程唤醒成本。
        chunks: dict[str, list[np.ndarray]] = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "next_observations": [],
            "terminated": [],
            "truncated": [],
            "successes": [],
            "landing_errors": [],
            "simulation_steps": [],
        }
        buffered = 0

        try:
            while not stop_event.is_set():
                # 参数队列容量为 1，但仍采用 drain-to-latest 写法，确保并发
                # 时只加载最后一个可见版本。
                try:
                    while True:
                        _, weights = weight_queue.get_nowait()
                        actor.load_state_dict(
                            {
                                key: torch.as_tensor(value)
                                for key, value in weights.items()
                            }
                        )
                except queue.Empty:
                    pass

                count = train_config.envs_per_actor
                # 训练初期使用均匀随机动作覆盖完整连续动作区间；之后使用
                # actor 输出叠加线性退火的高斯噪声。
                if local_transitions < random_budget:
                    actions = rng.uniform(-1.0, 1.0, size=(count, 1)).astype(
                        np.float32
                    )
                else:
                    with torch.inference_mode():
                        actions = actor(torch.as_tensor(observations)).numpy()
                    progress = min(local_transitions / expected_per_actor, 1.0)
                    noise_std = (
                        train_config.exploration_noise_start
                        + progress
                        * (
                            train_config.exploration_noise_end
                            - train_config.exploration_noise_start
                        )
                    )
                    actions = np.clip(
                        actions + rng.normal(0.0, noise_std, size=actions.shape),
                        -1.0,
                        1.0,
                    ).astype(np.float32)

                # JumpEnv 的一次 step 在内部完成整段跳跃，因此 vector batch
                # 中的每一个 transition 都必须立即 terminated 或 truncated。
                previous_observations = observations.copy()
                observations, rewards, terminated, truncated, infos = envs.step(
                    actions
                )
                if not np.all(np.logical_or(terminated, truncated)):
                    raise RuntimeError("JumpEnv must end after every external step")

                # SAME_STEP autoreset 返回的 observations 已经属于下一回合；
                # 本回合成功率、落点误差等终局信息必须从 final_info 读取。
                final_info = infos["final_info"]
                chunks["observations"].append(previous_observations)
                chunks["actions"].append(actions)
                chunks["rewards"].append(rewards.astype(np.float32))
                # The task is terminal and its observation is constant during a
                # jump. Do not store SAME_STEP's auto-reset observation here.
                chunks["next_observations"].append(previous_observations.copy())
                chunks["terminated"].append(terminated)
                chunks["truncated"].append(truncated)
                chunks["successes"].append(
                    np.asarray(final_info["is_success"], dtype=np.bool_)
                )
                chunks["landing_errors"].append(
                    np.asarray(final_info["landing_error"], dtype=np.float32)
                )
                chunks["simulation_steps"].append(
                    np.asarray(final_info["simulation_steps"], dtype=np.int32)
                )
                buffered += count
                local_transitions += count

                if buffered >= train_config.actor_chunk_size:
                    # 有界 transition queue 满时不丢训练数据，而是让 actor
                    # 等待 learner 形成背压；等待情况通过 health event 上报。
                    batch = {
                        key: np.concatenate(values, axis=0)
                        for key, values in chunks.items()
                    }
                    sent, blocked_seconds, full_events = _put_until_stopped(
                        transition_queue, (actor_id, batch), stop_event
                    )
                    _emit_actor_health(
                        event_queue,
                        actor_id,
                        sent_transitions=buffered if sent else 0,
                        blocked_seconds=blocked_seconds,
                        queue_full_events=full_events,
                        dropped_transitions=0 if sent else buffered,
                    )
                    if not sent:
                        # stop_event 在等待期间触发时，该 chunk 不再有消费者，
                        # 因此计入 dropped_transitions 而不是继续阻塞退出。
                        for values in chunks.values():
                            values.clear()
                        buffered = 0
                        break
                    for values in chunks.values():
                        values.clear()
                    buffered = 0
        finally:
            # 未达到完整 chunk 的尾部 transition 也必须计入停机丢弃指标。
            if buffered:
                _emit_actor_health(
                    event_queue,
                    actor_id,
                    sent_transitions=0,
                    blocked_seconds=0.0,
                    queue_full_events=0,
                    dropped_transitions=buffered,
                )
            envs.close(terminate=True)
    except BaseException:
        error = {
            "kind": "actor_error",
            "actor_id": actor_id,
            "traceback": traceback.format_exc(),
        }
        try:
            event_queue.put(error, timeout=1.0)
        except queue.Full:
            pass
        stop_event.set()


def _drain_actor_events(
    event_queue: Any, health: dict[int, _ActorQueueHealth]
) -> list[dict[str, Any]]:
    """汇总 actor 健康事件，并把需要中止训练的异常单独返回。"""
    errors: list[dict[str, Any]] = []
    while True:
        try:
            event = event_queue.get_nowait()
        except queue.Empty:
            break
        if event.get("kind") == "actor_error":
            errors.append(event)
            continue
        if event.get("kind") != "actor_health":
            continue
        actor = health[int(event["actor_id"])]
        actor.sent_transitions += int(event["sent_transitions"])
        actor.blocked_seconds += float(event["blocked_seconds"])
        actor.queue_full_events += int(event["queue_full_events"])
        actor.dropped_transitions += int(event["dropped_transitions"])
    return errors


def _evaluate_agent(
    agent: BanditTD3,
    env_config: JumpEnvConfig,
    *,
    episodes: int,
    seed: int,
) -> EvaluationResult:
    def policy(observation: np.ndarray, _: dict[str, object]) -> np.ndarray:
        return agent.act(observation[None, :])[0]

    return evaluate_policy(
        policy, episodes=episodes, config=env_config, seed=seed
    )


def train_distributed(
    train_config: TrainConfig | None = None,
    env_config: JumpEnvConfig | None = None,
) -> TrainResult:
    """运行 CPU actor + 单设备 learner 的完整训练、监控、评测和保存流程。"""
    cfg = train_config or TrainConfig()
    env_cfg = env_config or JumpEnvConfig()

    # 1. 在创建任何子进程之前确定设备和输出目录。时间戳目录同时容纳
    # TensorBoard event 与默认 checkpoint，便于并行实验隔离。
    resolve_device(cfg.device)  # Fail before creating any child process.
    effective_run_name = _training_run_name(
        env_cfg.observation_mode, cfg.total_transitions, cfg.run_name
    )
    run_dir = _create_run_directory(cfg.run_root, effective_run_name)
    checkpoint_path = (
        Path(cfg.checkpoint_path)
        if cfg.checkpoint_path is not None
        else run_dir / "checkpoint.pt"
    )
    writer = SummaryWriter(log_dir=str(run_dir), max_queue=100, flush_secs=10)
    # 将完整配置写入 TensorBoard Text，保证仅凭 run 目录即可复现实验。
    writer.add_text(
        "config/train",
        f"```json\n{json.dumps(asdict(cfg), indent=2, default=str)}\n```",
        0,
    )
    writer.add_text(
        "config/environment",
        f"```json\n{json.dumps(asdict(env_cfg), indent=2, default=str)}\n```",
        0,
    )
    writer.add_text("config/run_identity", effective_run_name, 0)
    writer.add_text("config/observation_mode", env_cfg.observation_mode, 0)
    writer.add_scalar("train/target_transitions", cfg.total_transitions, 0)
    writer.add_scalar("queue/transition_capacity", cfg.transition_queue_size, 0)

    # 2. learner 是唯一允许使用 cfg.device 的进程；CPU actor 的模型副本在
    # _actor_main 中创建，因而不会意外争抢 learner GPU。
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.device(cfg.device).type == "cpu":
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    model_config: dict[str, Any] = {
        "obs_dim": 1,
        "action_dim": 1,
        "hidden_dim": cfg.hidden_dim,
        "actor_lr": cfg.actor_lr,
        "critic_lr": cfg.critic_lr,
        "policy_delay": cfg.policy_delay,
    }
    if env_cfg.observation_mode == "rgb":
        # list 可直接写入 checkpoint metadata；BanditTD3 加载时会转为 tuple。
        model_config["observation_shape"] = list(env_cfg.observation_shape)
    learner = BanditTD3(device=cfg.device, **model_config)
    replay = ReplayBuffer(
        cfg.replay_capacity,
        observation_shape=env_cfg.observation_shape,
        observation_dtype=env_cfg.observation_dtype,
        # 本任务每条数据都 terminal，critic target 只等于即时 reward；不保存
        # next observation 可避免像素 replay 内存直接翻倍。
        store_next_observations=False,
        seed=cfg.seed,
    )

    # 3. 所有进程统一使用 spawn：避免 fork 复制 CUDA/PyBullet 状态。
    # transition_queue 传样本，event_queue 传健康/异常，weight_queues 为每个
    # actor 提供互不干扰的 latest-only 参数通道。
    context = mp.get_context("spawn")
    transition_queue = context.Queue(maxsize=cfg.transition_queue_size)
    event_queue = context.Queue(maxsize=max(1_024, cfg.num_actors * 128))
    weight_queues = [context.Queue(maxsize=1) for _ in range(cfg.num_actors)]
    stop_event = context.Event()
    initial_weights = learner.actor_weights_numpy()
    for weight_queue in weight_queues:
        weight_queue.put((0, initial_weights))

    actors: list[mp.Process] = []
    actor_health = {
        actor_id: _ActorQueueHealth() for actor_id in range(cfg.num_actors)
    }
    # 4. 以下计数器分为训练全局累计值和“自上次日志以来”的窗口值；后者
    # 用于展示近期趋势，避免全程平均值掩盖策略退化或队列拥塞。
    received = 0
    updates = 0
    update_credit = 0.0
    successes_since_log = 0
    episodes_since_log = 0
    rewards_since_log = 0.0
    landing_error_since_log = 0.0
    simulation_steps_since_log = 0
    critic_loss_since_log = 0.0
    critic_updates_since_log = 0
    actor_loss_since_log = 0.0
    actor_updates_since_log = 0
    learner_wait_seconds_since_log = 0.0
    learner_wait_count_since_log = 0
    learner_wait_max_since_log = 0.0
    learner_long_wait_count = 0
    learner_wait_timeout_count = 0
    last_log_at = 0
    last_sync_at = 0
    last_eval_at = 0
    started = time.perf_counter()
    discarded_prefetch = 0

    try:
        # 5. actor 必须为 non-daemon，因为它还需要创建 AsyncVectorEnv worker。
        for actor_id in range(cfg.num_actors):
            process = context.Process(
                target=_actor_main,
                args=(
                    actor_id,
                    env_cfg,
                    cfg,
                    transition_queue,
                    weight_queues[actor_id],
                    event_queue,
                    stop_event,
                ),
                name=f"jump-actor-{actor_id}",
                daemon=False,
            )
            process.start()
            actors.append(process)

        try:
            while received < cfg.total_transitions:
                # 6a. 每轮先处理 actor 异常和健康事件，避免 actor 已崩溃时
                # learner 仍在 transition_queue 上无限等待。
                errors = _drain_actor_events(event_queue, actor_health)
                if errors:
                    event = errors[0]
                    raise RuntimeError(
                        f"Actor {event['actor_id']} failed:\n{event['traceback']}"
                    )
                if stop_event.is_set():
                    raise RuntimeError("Actor collection stopped unexpectedly")

                # 6b. learner 对数据队列使用有限超时等待，并记录正常等待、
                # 长等待和完整超时，供 TensorBoard 判断 actor 是否供数不足。
                wait_started = time.perf_counter()
                try:
                    _, batch = transition_queue.get(timeout=2.0)
                    wait_seconds = time.perf_counter() - wait_started
                except queue.Empty:
                    wait_seconds = time.perf_counter() - wait_started
                    learner_wait_seconds_since_log += wait_seconds
                    learner_wait_count_since_log += 1
                    learner_wait_max_since_log = max(
                        learner_wait_max_since_log, wait_seconds
                    )
                    learner_wait_timeout_count += 1
                    if wait_seconds >= cfg.learner_long_wait_seconds:
                        learner_long_wait_count += 1
                    writer.add_scalar(
                        "queue/learner_last_wait_seconds", wait_seconds, received
                    )
                    writer.add_scalar(
                        "queue/learner_wait_timeout_count_total",
                        learner_wait_timeout_count,
                        received,
                    )
                    dead = [
                        process.name for process in actors if not process.is_alive()
                    ]
                    if dead:
                        raise RuntimeError(
                            f"Actor processes exited unexpectedly: {dead}"
                        )
                    continue

                learner_wait_seconds_since_log += wait_seconds
                learner_wait_count_since_log += 1
                learner_wait_max_since_log = max(
                    learner_wait_max_since_log, wait_seconds
                )
                if wait_seconds >= cfg.learner_long_wait_seconds:
                    learner_long_wait_count += 1

                # 6c. learner 是 replay buffer 的唯一写入者，无需跨进程锁。
                # batch 中额外的 successes/landing_errors 仅供监控，不参与 TD3。
                count = len(batch["observations"])
                replay.add_batch(batch)
                received += count
                successes_since_log += int(np.sum(batch["successes"]))
                episodes_since_log += count
                rewards_since_log += float(np.sum(batch["rewards"]))
                landing_error_since_log += float(
                    np.sum(batch["landing_errors"])
                )
                simulation_steps_since_log += int(
                    np.sum(batch["simulation_steps"])
                )
                update_credit += count * cfg.updates_per_transition

                # update_credit 支持非整数 update-to-data ratio。例如 0.25 会
                # 每收到 4 条数据积累出 1 次更新，且不会因 chunk 边界丢精度。
                if len(replay) >= cfg.learning_starts:
                    while update_credit >= 1.0:
                        update = learner.update(replay.sample(cfg.batch_size))
                        critic_loss_since_log += update.critic_loss
                        critic_updates_since_log += 1
                        if update.actor_loss is not None:
                            actor_loss_since_log += update.actor_loss
                            actor_updates_since_log += 1
                        updates += 1
                        update_credit -= 1.0

                # 6d. 定期发布 actor 参数。队列中的旧版本会被替换，而不是让
                # actor 逐个追赶历史 checkpoint。
                if received - last_sync_at >= cfg.parameter_sync_interval:
                    weights = learner.actor_weights_numpy()
                    for weight_queue in weight_queues:
                        _replace_latest(weight_queue, (received, weights))
                    last_sync_at = received

                # 6e. 日志窗口到达或训练即将结束时，统一计算 rollout、loss、
                # throughput 和队列健康指标，并强制 flush event 文件。
                should_log = (
                    received - last_log_at >= cfg.log_interval
                    or received >= cfg.total_transitions
                )
                if should_log:
                    errors = _drain_actor_events(event_queue, actor_health)
                    if errors:
                        event = errors[0]
                        raise RuntimeError(
                            f"Actor {event['actor_id']} failed:\n"
                            f"{event['traceback']}"
                        )
                    elapsed = time.perf_counter() - started
                    success_rate = successes_since_log / max(
                        episodes_since_log, 1
                    )
                    transition_qsize = _safe_qsize(transition_queue)
                    total_blocked = sum(
                        value.blocked_seconds for value in actor_health.values()
                    )
                    total_full = sum(
                        value.queue_full_events for value in actor_health.values()
                    )
                    total_dropped = sum(
                        value.dropped_transitions
                        for value in actor_health.values()
                    )
                    total_sent = sum(
                        value.sent_transitions for value in actor_health.values()
                    )

                    writer.add_scalar("rollout/success_rate", success_rate, received)
                    writer.add_scalar(
                        "rollout/reward_mean",
                        rewards_since_log / max(episodes_since_log, 1),
                        received,
                    )
                    writer.add_scalar(
                        "rollout/landing_error_mean",
                        landing_error_since_log / max(episodes_since_log, 1),
                        received,
                    )
                    writer.add_scalar(
                        "rollout/physics_steps_mean",
                        simulation_steps_since_log / max(episodes_since_log, 1),
                        received,
                    )
                    writer.add_scalar(
                        "throughput/sample_transitions_per_second",
                        received / elapsed,
                        received,
                    )
                    writer.add_scalar("learner/updates", updates, received)
                    writer.add_scalar("learner/replay_size", len(replay), received)
                    writer.add_scalar(
                        "learner/replay_allocated_megabytes",
                        replay.allocated_bytes / (1024**2),
                        received,
                    )
                    if critic_updates_since_log:
                        writer.add_scalar(
                            "learner/critic_loss",
                            critic_loss_since_log / critic_updates_since_log,
                            received,
                        )
                    if actor_updates_since_log:
                        writer.add_scalar(
                            "learner/actor_loss",
                            actor_loss_since_log / actor_updates_since_log,
                            received,
                        )
                    writer.add_scalar(
                        "queue/transition_size", transition_qsize, received
                    )
                    if transition_qsize >= 0:
                        writer.add_scalar(
                            "queue/transition_fill_fraction",
                            transition_qsize / cfg.transition_queue_size,
                            received,
                        )
                    writer.add_scalar(
                        "queue/actor_blocked_seconds_total", total_blocked, received
                    )
                    writer.add_scalar(
                        "queue/actor_queue_full_events_total", total_full, received
                    )
                    writer.add_scalar(
                        "queue/actor_dropped_transitions_total",
                        total_dropped,
                        received,
                    )
                    writer.add_scalar(
                        "queue/actor_sent_transitions_total", total_sent, received
                    )
                    writer.add_scalar(
                        "queue/learner_wait_seconds_mean",
                        learner_wait_seconds_since_log
                        / max(learner_wait_count_since_log, 1),
                        received,
                    )
                    writer.add_scalar(
                        "queue/learner_wait_seconds_max",
                        learner_wait_max_since_log,
                        received,
                    )
                    writer.add_scalar(
                        "queue/learner_long_wait_count_total",
                        learner_long_wait_count,
                        received,
                    )
                    writer.add_scalar(
                        "queue/learner_wait_timeout_count_total",
                        learner_wait_timeout_count,
                        received,
                    )
                    for actor_id, health in actor_health.items():
                        writer.add_scalar(
                            f"actors/actor_{actor_id}/queue_blocked_seconds_total",
                            health.blocked_seconds,
                            received,
                        )
                        writer.add_scalar(
                            f"actors/actor_{actor_id}/queue_full_events_total",
                            health.queue_full_events,
                            received,
                        )
                        writer.add_scalar(
                            f"actors/actor_{actor_id}/dropped_transitions_total",
                            health.dropped_transitions,
                            received,
                        )
                    writer.flush()

                    payload = {
                        "type": "train",
                        "transitions": received,
                        "updates": updates,
                        "sample_tps": received / elapsed,
                        "recent_success_rate": success_rate,
                        "transition_queue_size": transition_qsize,
                        "actor_blocked_seconds_total": total_blocked,
                        "actor_dropped_transitions_total": total_dropped,
                        "learner_long_wait_count_total": learner_long_wait_count,
                        "run_dir": str(run_dir),
                    }
                    print(json.dumps(payload), flush=True)
                    successes_since_log = 0
                    episodes_since_log = 0
                    rewards_since_log = 0.0
                    landing_error_since_log = 0.0
                    simulation_steps_since_log = 0
                    critic_loss_since_log = 0.0
                    critic_updates_since_log = 0
                    actor_loss_since_log = 0.0
                    actor_updates_since_log = 0
                    learner_wait_seconds_since_log = 0.0
                    learner_wait_count_since_log = 0
                    learner_wait_max_since_log = 0.0
                    last_log_at = received

                # 6f. 周期评测关闭探索噪声并使用独立 seed 区间，评测期间 actor
                # 可继续预取数据；有界队列会自然限制其内存占用。
                if (
                    cfg.eval_interval > 0
                    and received - last_eval_at >= cfg.eval_interval
                    and len(replay) >= cfg.learning_starts
                ):
                    evaluation = _evaluate_agent(
                        learner,
                        env_cfg,
                        episodes=cfg.eval_episodes,
                        seed=cfg.seed + 1_000_000 + received,
                    )
                    writer.add_scalar(
                        "evaluation/success_rate",
                        evaluation.success_rate,
                        received,
                    )
                    writer.add_scalar(
                        "evaluation/landing_error_mean",
                        evaluation.mean_landing_error,
                        received,
                    )
                    writer.add_scalar(
                        "evaluation/reward_mean", evaluation.mean_reward, received
                    )
                    writer.flush()
                    print(
                        json.dumps(
                            {"type": "evaluation", **evaluation.as_dict()}
                        ),
                        flush=True,
                    )
                    last_eval_at = received
        finally:
            # 7. 无论正常结束还是异常退出都先通知 actor。停机时持续消费已经
            # 入队的预取 chunk，否则 multiprocessing.Queue feeder 可能因管道
            # 满而阻止 actor 退出；这些不再训练的数据单独计入 discarded_prefetch。
            stop_event.set()
            shutdown_deadline = time.monotonic() + 20.0
            while any(process.is_alive() for process in actors):
                try:
                    _, prefetched = transition_queue.get(timeout=0.1)
                    discarded_prefetch += len(prefetched["observations"])
                except queue.Empty:
                    pass
                _drain_actor_events(event_queue, actor_health)
                for process in actors:
                    process.join(timeout=0.0)
                if time.monotonic() >= shutdown_deadline:
                    break
            # 超过宽限期仍未退出的进程才强制 terminate，正常路径不会走到这里。
            for process in actors:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5.0)
            _drain_actor_events(event_queue, actor_health)
            while True:
                try:
                    _, prefetched = transition_queue.get_nowait()
                    discarded_prefetch += len(prefetched["observations"])
                except queue.Empty:
                    break

            # 停机后的最终队列指标覆盖最后一个 TensorBoard 点，确保 partial
            # chunk、背压等待和预取丢弃不会因为日志窗口已结束而漏报。
            total_dropped = sum(
                value.dropped_transitions for value in actor_health.values()
            )
            total_blocked = sum(
                value.blocked_seconds for value in actor_health.values()
            )
            total_full = sum(
                value.queue_full_events for value in actor_health.values()
            )
            total_sent = sum(
                value.sent_transitions for value in actor_health.values()
            )
            writer.add_scalar(
                "queue/actor_blocked_seconds_total", total_blocked, received
            )
            writer.add_scalar(
                "queue/actor_queue_full_events_total", total_full, received
            )
            writer.add_scalar(
                "queue/actor_dropped_transitions_total", total_dropped, received
            )
            writer.add_scalar(
                "queue/actor_sent_transitions_total", total_sent, received
            )
            writer.add_scalar(
                "queue/learner_discarded_prefetch_transitions_total",
                discarded_prefetch,
                received,
            )
            writer.add_scalar("queue/transition_size", 0, received)
            for actor_id, health in actor_health.items():
                writer.add_scalar(
                    f"actors/actor_{actor_id}/queue_blocked_seconds_total",
                    health.blocked_seconds,
                    received,
                )
                writer.add_scalar(
                    f"actors/actor_{actor_id}/queue_full_events_total",
                    health.queue_full_events,
                    received,
                )
                writer.add_scalar(
                    f"actors/actor_{actor_id}/dropped_transitions_total",
                    health.dropped_transitions,
                    received,
                )
            writer.flush()
            transition_queue.close()
            event_queue.close()
            for weight_queue in weight_queues:
                weight_queue.close()

        # 8. 所有采样进程关闭后，用冻结的确定性 actor 完成最终评测，再保存
        # 网络、配置、评测结果和 run_dir。checkpoint 保存成功后才返回结果。
        elapsed = time.perf_counter() - started
        final_evaluation = _evaluate_agent(
            learner,
            env_cfg,
            episodes=cfg.final_eval_episodes,
            seed=cfg.seed + 2_000_000,
        )
        writer.add_scalar(
            "evaluation/final_success_rate", final_evaluation.success_rate, received
        )
        writer.add_scalar(
            "evaluation/final_landing_error_mean",
            final_evaluation.mean_landing_error,
            received,
        )
        writer.add_scalar(
            "evaluation/final_reward_mean", final_evaluation.mean_reward, received
        )
        metadata = {
            "model_config": model_config,
            "train_config": asdict(cfg),
            "env_config": asdict(env_cfg),
            "run_dir": str(run_dir),
            "transitions": received,
            "updates": updates,
            "final_evaluation": final_evaluation.as_dict(),
        }
        learner.save(checkpoint_path, metadata=metadata)
        writer.add_text("artifacts/checkpoint", str(checkpoint_path), received)
        writer.flush()
        return TrainResult(
            transitions=received,
            updates=updates,
            elapsed_seconds=elapsed,
            sample_transitions_per_second=received / elapsed,
            final_evaluation=final_evaluation,
            checkpoint_path=str(checkpoint_path),
            run_dir=str(run_dir),
        )
    finally:
        # SummaryWriter 在异常路径也必须关闭，否则最后一批 event 可能仍在内存中。
        writer.flush()
        writer.close()
