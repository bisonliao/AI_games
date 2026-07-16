"""Actor–learner训练入口：协调Rollout A/B、GPU learner、策略同步、日志、checkpoint与异步评测。"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import queue
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .agent import DQNAgent, RandomAgent, ReplayBuffer
    from .evaluator import HeuristicEvaluator
    from .run_paths import checkpoint_filename, named_directory, validate_run_name
except ImportError:
    from agent import DQNAgent, RandomAgent, ReplayBuffer
    from evaluator import HeuristicEvaluator
    from run_paths import checkpoint_filename, named_directory, validate_run_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Gomoku black-side DQN agent by self-play.")

    parser.add_argument("--run-name", required=True, help="Required name used to isolate checkpoints and logs.")
    parser.add_argument("--board-size", type=int, default=5)  # 五子棋棋盘边长，支持5～9。
    parser.add_argument("--num-actors", type=int, default=1)  # rollout A的actor进程数，必须至少为1。
    parser.add_argument("--envs-per-actor", type=int, default=16)  # 每个rollout A actor内部的同步环境数。
    parser.add_argument("--actor-batch-size", type=int, default=256)  # rollout A累计多少条transition后发送给learner。
    parser.add_argument("--actor-queue-size", type=int, default=8)  # rollout A数据队列最多容纳的批次数。
    parser.add_argument("--policy-sync-steps", type=int, default=5_000)  # learner向各actor同步黑棋权重的步数间隔。
    parser.add_argument("--updates-per-step", type=float, default=0.25)  # 每条rollout A数据对应的梯度更新次数。
    parser.add_argument("--total-black-steps", type=int, default=50_000_000)  # 本次训练接收的黑棋transition总目标数。
    parser.add_argument("--seed", type=int, default=0)  # 训练、actor和环境的根随机种子。
    parser.add_argument("--device", type=str, default="auto")  # learner训练设备；auto优先使用CUDA。

    parser.add_argument("--black-checkpoint", type=Path, default=None)  # 用指定checkpoint初始化待训练的黑棋网络。
    parser.add_argument("--opponent-checkpoint", type=Path, default=None)  # 用指定checkpoint初始化rollout A白棋对手。
    parser.add_argument("--history-dir", type=Path, default=Path(__file__).resolve().parent / "history")  # checkpoint历史目录根路径。
    parser.add_argument("--save-checkpoint", type=int, default=1_000_000)  # 保存历史checkpoint的黑棋步数间隔。
    parser.add_argument("--load-opponent", type=int, default=2_000_000)  # rollout A重新抽取历史对手的步数间隔。
    parser.add_argument("--checkpoint-rand", type=int, default=10)  # rollout A从最近多少个checkpoint中随机选对手。
    parser.add_argument("--num-sidecar-actors", type=int, default=2)  # rollout B启发式旁路的独立actor进程数。
    parser.add_argument("--sidecar-envs-per-actor", type=int, default=16)  # 每个rollout B actor内部的同步环境数。
    parser.add_argument("--sidecar-batch-size", type=int, default=256)  # rollout B累计多少条双方transition后发送。
    parser.add_argument("--sidecar-queue-size", type=int, default=8)  # rollout B数据队列最多容纳的批次数。
    parser.add_argument("--replay-b-size", type=int, default=50_000)  # learner侧replay buffer B的最大容量。
    parser.add_argument("--replay-b-min-size", type=int, default=5_000)  # replay B达到此大小后才参与采样。
    parser.add_argument("--replay-b-fraction", type=float, default=0.20)  # 每个learner minibatch从replay B采样的目标比例。
    parser.add_argument("--load-optimizer", action="store_true")  # 加载黑棋checkpoint时同时恢复optimizer状态。
    parser.add_argument("--heuristic-eval-games", type=int, default=16)  # 每个checkpoint异步对战启发式机器人的评测局数。
    parser.add_argument("--disable-heuristic-eval", action="store_true")  # 禁用checkpoint的异步启发式评测。

    parser.add_argument("--hidden-channels", type=int, default=96)  # DQN卷积主干的隐藏通道数。
    parser.add_argument("--num-res-blocks", type=int, default=4)  # DQN卷积主干的残差块数量。
    parser.add_argument("--lr", type=float, default=1e-4)  # AdamW optimizer的学习率。
    parser.add_argument("--gamma", type=float, default=0.99)  # TD回报和n-step return的折扣因子。
    parser.add_argument("--batch-size", type=int, default=256)  # 每次梯度更新使用的minibatch大小。
    parser.add_argument("--replay-size", type=int, default=200_000)  # rollout A replay buffer的最大容量。
    parser.add_argument("--min-replay-size", type=int, default=10_000)  # rollout A达到此大小后才开始更新。
    parser.add_argument("--target-update", type=int, default=2_000)  # 同步target network的梯度更新次数间隔。
    parser.add_argument("--grad-clip", type=float, default=10.0)  # 梯度范数裁剪上限；不大于0表示关闭。
    parser.add_argument("--no-double-dqn", action="store_true")  # 关闭Double DQN，改用普通DQN target。
    parser.add_argument(
        "--n-step", type=int, default=1,
        help="N-step return horizon; 1 (the default) uses one-step TD.",
    )
    parser.add_argument(
        "--no-n-step", dest="n_step", action="store_const", const=1,
        help="Disable multi-step returns and use one-step TD targets.",
    )
    parser.add_argument(
        "--reward-shaping-scale", type=float, default=0.02,
        help="Potential-based shaping strength; 0 (the default) disables shaping.",
    )
    parser.add_argument(
        "--no-reward-shaping", dest="reward_shaping_scale",
        action="store_const", const=0.0,
    )  # 强制关闭奖励塑形，将塑形强度设为0。

    parser.add_argument("--epsilon-start", type=float, default=1.0)  # 黑棋epsilon-greedy探索率的初始值。
    parser.add_argument("--epsilon-end", type=float, default=0.05)  # 黑棋epsilon-greedy探索率的最终值。
    parser.add_argument("--epsilon-decay-steps", type=int, default=1_000_000)  # epsilon从初值线性衰减到终值的步数。
    parser.add_argument("--opponent-epsilon", type=float, default=0.0)  # rollout A白棋DQN对手的epsilon探索率。
    parser.add_argument("--log-interval", type=int, default=100_000)  # 控制台和TensorBoard指标的上报步数间隔。
    parser.add_argument("--tb-log-dir", type=Path, default=Path(__file__).resolve().parent / "runs")  # TensorBoard日志目录根路径。
    parser.add_argument("--disable-tensorboard", action="store_true")  # 禁止创建和写入TensorBoard event文件。
    parser.add_argument("--save-final", action="store_true")  # 训练正常结束时额外保存最终checkpoint。

    return parser.parse_args()


def epsilon_by_step(args: argparse.Namespace, total_black_steps: int) -> float:
    if args.epsilon_decay_steps <= 0:
        return args.epsilon_end
    fraction = min(1.0, total_black_steps / args.epsilon_decay_steps)
    return args.epsilon_start + fraction * (args.epsilon_end - args.epsilon_start)


def make_agent(args: argparse.Namespace, seed: int) -> DQNAgent:
    return DQNAgent(
        args.board_size,
        hidden_channels=args.hidden_channels,
        num_res_blocks=args.num_res_blocks,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        min_replay_size=args.min_replay_size,
        target_update=args.target_update,
        grad_clip=args.grad_clip,
        double_dqn=not args.no_double_dqn,
        device=args.device,
        seed=seed,
    )


def history_checkpoints(history_dir: Path) -> List[Path]:
    if not history_dir.exists():
        return []
    return sorted(history_dir.glob("*.pt"), key=checkpoint_sort_key)


def checkpoint_sort_key(path: Path) -> Any:
    match = re.search(r"(\d+)", path.stem)
    return (int(match.group(1)) if match else -1, path.name)


def next_history_index(history_dir: Path) -> int:
    checkpoints = history_checkpoints(history_dir)
    if not checkpoints:
        return 1
    return checkpoint_sort_key(checkpoints[-1])[0] + 1


def save_history_checkpoint(
    agent: DQNAgent,
    history_dir: Path,
    index: int,
    total_black_steps: int,
    args: argparse.Namespace,
) -> Path:
    path = history_dir / checkpoint_filename(args.run_name, index)
    agent.save_checkpoint(
        path,
        total_black_steps=total_black_steps,
        history_index=index,
        extra={"args": vars(args)},
    )
    return path


def build_opponent(args: argparse.Namespace, seed: int) -> Any:
    if args.opponent_checkpoint is None:
        print("Initial opponent: random legal-move agent")
        return RandomAgent(seed=seed)

    opponent = make_agent(args, seed)
    opponent.load_checkpoint(args.opponent_checkpoint, load_optimizer=False)
    print(f"Initial opponent: {args.opponent_checkpoint}")
    return opponent


def maybe_reload_opponent(
    args: argparse.Namespace,
    rng: random.Random,
    seed: int,
) -> Any:
    checkpoints = history_checkpoints(args.history_dir)
    if not checkpoints:
        print("No history checkpoint available; using random legal-move agent")
        return RandomAgent(seed=seed)

    recent = checkpoints[-max(1, args.checkpoint_rand) :]
    checkpoint = rng.choice(recent)
    opponent = make_agent(args, seed)
    opponent.load_checkpoint(checkpoint, load_optimizer=False)
    print(f"Reloaded opponent from history: {checkpoint}")
    return opponent


def print_log(
    *,
    total_black_steps: int,
    interval_steps: int,
    interval_seconds: float,
    epsilon: float,
    replay_size: int,
    updates: int,
    stats: Dict[str, float],
    last_loss: Optional[Dict[str, float]],
    collector_text: str = "",
) -> None:
    steps_per_second = interval_steps / max(1e-6, interval_seconds)
    episodes = max(1, int(stats["episodes"]))
    win_rate = stats["black_wins"] / episodes
    loss_rate = stats["black_losses"] / episodes
    draw_rate = stats["draws"] / episodes
    avg_len = stats["episode_black_steps"] / episodes

    loss_text = ""
    if last_loss is not None:
        loss_text = (
            f" loss={last_loss['loss']:.4f}"
            f" q={last_loss['mean_q']:.3f}"
            f" target={last_loss['mean_target']:.3f}"
        )

    print(
        f"steps={total_black_steps} eps={epsilon:.3f}"
        f" replay={replay_size} updates={updates}"
        f" speed={steps_per_second:.1f}/s"
        f" W/L/D={win_rate:.3f}/{loss_rate:.3f}/{draw_rate:.3f}"
        f" avg_black_len={avg_len:.1f}{loss_text}{collector_text}",
        flush=True,
    )


class TensorBoardLogger:
    def __init__(self, args: argparse.Namespace) -> None:
        self.enabled = not args.disable_tensorboard
        self.writer: Optional[SummaryWriter] = None
        if not self.enabled:
            return

        run_name = time.strftime("dqn_%Y%m%d_%H%M%S")
        self.log_dir = args.tb_log_dir / run_name
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        print(f"TensorBoard log dir: {self.log_dir}")

    def add_scalars(
        self,
        *,
        step: int,
        epsilon: float,
        replay_size: int,
        updates: int,
        stats: Dict[str, float],
        window_stats: Dict[str, float],
        train_metrics: Optional[Dict[str, float]],
    ) -> None:
        if self.writer is None:
            return

        window_episodes = max(1.0, window_stats["episodes"])
        self.writer.add_scalar("Episode/mean_length", window_stats["episode_black_steps"] / window_episodes, step)
        self.writer.add_scalar("Episode/mean_return", window_stats["episode_return"] / window_episodes, step)
        self.writer.add_scalar("Outcome/win_rate", window_stats["black_wins"] / window_episodes, step)
        self.writer.add_scalar("Outcome/loss_rate", window_stats["black_losses"] / window_episodes, step)
        self.writer.add_scalar("Outcome/draw_rate", window_stats["draws"] / window_episodes, step)

        cumulative_episodes = max(1.0, stats["episodes"])
        self.writer.add_scalar("Cumulative/episode_mean_length", stats["episode_black_steps"] / cumulative_episodes, step)
        self.writer.add_scalar("Cumulative/episode_mean_return", stats["episode_return"] / cumulative_episodes, step)
        self.writer.add_scalar("Cumulative/win_rate", stats["black_wins"] / cumulative_episodes, step)
        self.writer.add_scalar("Cumulative/loss_rate", stats["black_losses"] / cumulative_episodes, step)
        self.writer.add_scalar("Cumulative/draw_rate", stats["draws"] / cumulative_episodes, step)

        self.writer.add_scalar("Train/epsilon", epsilon, step)
        self.writer.add_scalar("Train/replay_size", replay_size, step)
        self.writer.add_scalar("Train/update_steps", updates, step)
        if train_metrics is not None:
            for key, value in train_metrics.items():
                self.writer.add_scalar(f"Train/{key}", value, step)
        self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def initial_opponent_payload(opponent: Any) -> Optional[Any]:
    if isinstance(opponent, RandomAgent):
        return ("random",)
    from DQN.async_train import cpu_state_dict
    return ("dqn", cpu_state_dict(opponent.online_net), dict(opponent.model_kwargs))


def opponent_control_command(opponent: Any) -> Any:
    if isinstance(opponent, RandomAgent):
        return ("opponent_random",)
    from DQN.async_train import cpu_state_dict
    return (
        "opponent", cpu_state_dict(opponent.online_net), dict(opponent.model_kwargs)
    )


def report_evaluation_results(
    evaluator: Optional[HeuristicEvaluator],
    tb_logger: TensorBoardLogger,
    failure_count: int,
    messages: Optional[List[Dict[str, Any]]] = None,
) -> int:
    if evaluator is None and messages is None:
        return failure_count
    results = messages if messages is not None else evaluator.poll()
    for result in results:
        step = int(result["step"])
        if result["type"] == "evaluation_error":
            failure_count += 1
            print(
                f"WARNING: heuristic evaluation failed checkpoint={result['checkpoint']} "
                f"step={step}\n{result['traceback']}",
                flush=True,
            )
            if tb_logger.writer is not None:
                tb_logger.writer.add_scalar(
                    "Evaluation/heuristic_failures_total", failure_count, step
                )
                tb_logger.writer.flush()
            continue
        games = max(1, int(result["games"]))
        win_rate = result["wins"] / games
        loss_rate = result["losses"] / games
        draw_rate = result["draws"] / games
        print(
            f"Heuristic evaluation checkpoint={result['checkpoint']} step={step} "
            f"W/L/D={result['wins']}/{result['losses']}/{result['draws']} "
            f"win_rate={win_rate:.3f} duration={result['duration_seconds']:.2f}s",
            flush=True,
        )
        if tb_logger.writer is not None:
            tb_logger.writer.add_scalar("Evaluation/heuristic_win_rate", win_rate, step)
            tb_logger.writer.add_scalar("Evaluation/heuristic_loss_rate", loss_rate, step)
            tb_logger.writer.add_scalar("Evaluation/heuristic_draw_rate", draw_rate, step)
            tb_logger.writer.add_scalar(
                "Evaluation/heuristic_duration_seconds",
                float(result["duration_seconds"]), step,
            )
            tb_logger.writer.flush()
    return failure_count


def update_episode_stats(
    episode: Any,
    stats: Dict[str, float],
) -> None:
    winner, length, episode_return = episode
    stats["episodes"] += 1
    stats["episode_black_steps"] += float(length)
    stats["episode_return"] += float(episode_return)
    if winner == 1:
        stats["black_wins"] += 1
    elif winner == -1:
        stats["black_losses"] += 1
    else:
        stats["draws"] += 1


def stats_since(
    current: Dict[str, float],
    previous: Dict[str, float],
) -> Dict[str, float]:
    """Return per-log-window counters without mutating cumulative statistics."""
    return {key: current[key] - previous.get(key, 0.0) for key in current}


def queue_size(value: Any) -> int:
    try:
        return int(value.qsize())
    except (AttributeError, NotImplementedError):
        return -1


def report_actor_status_events(
    status_queue: Any,
    tb_logger: TensorBoardLogger,
    total_black_steps: int,
    timeout_count: int,
) -> int:
    while True:
        try:
            event = status_queue.get_nowait()
        except queue.Empty:
            break
        if event.get("type") != "queue_put_timeout":
            continue
        timeout_count += 1
        print(
            "WARNING: actor transition queue put timed out "
            f"actor={event['actor_id']} timeout={event['timeout_seconds']:.1f}s "
            f"step={total_black_steps} total_timeouts={timeout_count}",
            flush=True,
        )
        if tb_logger.writer is not None:
            tb_logger.writer.add_scalar(
                "Actors/queue_put_timeouts_total", timeout_count, total_black_steps,
                walltime=float(event["event_time"]),
            )
            tb_logger.writer.flush()
    return timeout_count


def run_actor_learner_training(
    args: argparse.Namespace,
    train_agent: DQNAgent,
    opponent: Any,
    opponent_rng: random.Random,
    tb_logger: TensorBoardLogger,
    evaluator: Optional[HeuristicEvaluator],
) -> None:
    """Run CPU rollout actors and the main-process learner concurrently."""
    from DQN.async_train import actor_worker, cpu_state_dict
    from DQN.heuristic_sidecar import HeuristicSidecar

    for name in ("num_actors", "envs_per_actor", "actor_batch_size", "actor_queue_size"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1")
    if args.policy_sync_steps < 1:
        raise ValueError("--policy-sync-steps must be at least 1")
    if args.updates_per_step < 0:
        raise ValueError("--updates-per-step cannot be negative")
    if args.num_sidecar_actors < 0:
        raise ValueError("--num-sidecar-actors cannot be negative")
    for name in (
        "sidecar_envs_per_actor", "sidecar_batch_size", "sidecar_queue_size",
        "replay_b_size", "replay_b_min_size",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1")
    if args.replay_b_min_size > args.replay_b_size:
        raise ValueError("--replay-b-min-size cannot exceed --replay-b-size")
    if not 0.0 <= args.replay_b_fraction <= 1.0:
        raise ValueError("--replay-b-fraction must be between 0 and 1")

    # Always use the "spawn" start method. Each actor starts a fresh Python
    # interpreter instead of inheriting the learner with fork(). This has more
    # startup overhead, but it is the safe choice when the learner owns CUDA
    # state: forking an initialized CUDA process can deadlock or corrupt state.
    ctx = mp.get_context("spawn")

    # actor -> learner data channel. Actors put complete NumPy transition
    # batches here and the learner consumes them with result_queue.get(). The
    # finite maxsize provides backpressure: when the learner is slower, actors
    # block rather than allowing queued transitions to consume unlimited RAM.
    result_queue = ctx.Queue(maxsize=args.actor_queue_size)

    # actor -> learner event channel for exceptional status such as a timed-out
    # result_queue.put(). It is separate and unbounded so an actor can report
    # that the transition queue is full without sending the report through that
    # same full queue.
    status_queue = ctx.Queue()

    # learner -> actor control channels. Each actor gets its own queue because
    # multiprocessing.Queue delivers each item to only one reader; a single
    # shared queue would not broadcast policy updates to every actor.
    control_queues = [ctx.Queue() for _ in range(args.num_actors)]

    # Shared one-way shutdown flag. Any process can set it; all actors poll it
    # in their collection and queue-put loops. Unlike a Queue, an Event stores
    # state, so every actor observes the same stop request.
    stop_event = ctx.Event()

    # A signed 64-bit integer stored in shared memory ("q" = C long long).
    # Actors increment it under its built-in lock so their epsilon schedules
    # use one global generated-step count instead of separate local counts.
    generated_steps = ctx.Value("q", 0)

    # Process arguments must be pickled under spawn. Convert model tensors to
    # copied CPU NumPy arrays so IPC does not depend on CUDA tensors or PyTorch's
    # file-descriptor-based tensor sharing mechanism.
    initial_policy = cpu_state_dict(train_agent.online_net)

    # Rollout B owns different processes, queues, environments and transition
    # collectors. Nothing from this controller is passed into rollout-A actors.
    sidecar_enabled = args.num_sidecar_actors > 0 and args.replay_b_fraction > 0.0
    replay_b = ReplayBuffer(
        args.replay_b_size if sidecar_enabled else 1,
        args.board_size, args.board_size * args.board_size,
        seed=args.seed + 80_000,
    )
    sidecar: Optional[HeuristicSidecar] = None
    if sidecar_enabled:
        sidecar = HeuristicSidecar(
            config={
                "seed": args.seed,
                "board_size": args.board_size,
                "envs_per_actor": args.sidecar_envs_per_actor,
                "batch_size": args.sidecar_batch_size,
                "n_step": args.n_step,
                "gamma": args.gamma,
                "reward_shaping_scale": args.reward_shaping_scale,
            },
            initial_policy=initial_policy,
            model_kwargs=dict(train_agent.model_kwargs),
            num_actors=args.num_sidecar_actors,
            queue_size=args.sidecar_queue_size,
            context=ctx,
        )

    # Prepare the white-player policy that every actor receives at startup.
    # - RandomAgent -> None: the child constructs its own seeded RandomPolicy.
    # - DQNAgent    -> (CPU NumPy state_dict, model_kwargs): the child rebuilds
    #   a lightweight inference-only white network from this snapshot.
    # This payload is created before process.start() because spawn must pickle
    # all initial arguments and cannot directly share the learner's agent/GPU.
    opponent_payload = initial_opponent_payload(opponent)
    actor_config = {
        # This is a root seed, not the final seed reused by every actor.
        # actor_worker derives an actor-specific seed from (root seed,
        # actor_id), then derives each vector environment's seed from
        # (actor seed, env index). See actor_worker for the exact mapping.
        "seed": args.seed,
        "board_size": args.board_size,
        "envs_per_actor": args.envs_per_actor,
        "actor_batch_size": args.actor_batch_size,
        "actor_device": "cpu",
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "epsilon_decay_steps": args.epsilon_decay_steps,
        "opponent_epsilon": args.opponent_epsilon,
        "n_step": args.n_step,
        "gamma": args.gamma,
        "reward_shaping_scale": args.reward_shaping_scale,
    }
    # Construct Process descriptors only; no child process starts until the
    # later process.start() call. Every actor receives shared queue/event/value
    # handles plus ordinary pickled configuration and initial model snapshots.
    actors = [
        ctx.Process(
            target=actor_worker,
            name=f"gomoku-actor-{actor_id}",
            args=(
                actor_id, actor_config, initial_policy, dict(train_agent.model_kwargs),
                opponent_payload, result_queue, status_queue,
                control_queues[actor_id], stop_event, generated_steps,
            ),
        )
        for actor_id in range(args.num_actors)
    ]

    total_black_steps = 0
    update_credit = 0.0
    next_save_step = args.save_checkpoint
    next_load_step = args.load_opponent
    next_policy_sync = args.policy_sync_steps
    history_index = next_history_index(args.history_dir)
    last_log_step = 0
    last_loss: Optional[Dict[str, float]] = None
    stats = {
        "episodes": 0.0, "black_wins": 0.0, "black_losses": 0.0, "draws": 0.0,
        "episode_black_steps": 0.0, "episode_return": 0.0,
    }
    last_log_stats = {key: 0.0 for key in stats}
    actor_versions = [-1] * args.num_actors
    blocked_seconds = 0.0
    received_batches = 0
    queue_put_timeouts = 0
    last_log_queue_put_timeouts = 0
    evaluation_failures = 0
    start_time = time.time()
    last_log_time = start_time
    updates_at_start = train_agent.update_steps
    last_log_updates = train_agent.update_steps
    last_log_blocked_seconds = 0.0
    last_log_sidecar_transitions = 0
    last_log_sidecar_outcomes = {
        "agent_wins": 0, "agent_losses": 0, "draws": 0,
    }

    print(
        f"Actor-learner training: actors={args.num_actors} envs/actor={args.envs_per_actor} "
        f"batch={args.actor_batch_size} UTD={args.updates_per_step:g}",
        flush=True,
    )
    # start() launches the fresh interpreters and invokes actor_worker(...) in
    # each child. The parent immediately continues as the learner; it does not
    # wait for actor_worker to return here.
    for process in actors:
        process.start()
    if sidecar is not None:
        sidecar.start()
        print(
            f"Rollout B: actors={args.num_sidecar_actors} "
            f"envs/actor={args.sidecar_envs_per_actor} "
            f"replay={args.replay_b_size} warmup={args.replay_b_min_size} "
            f"sample_fraction={args.replay_b_fraction:g}",
            flush=True,
        )

    try:
        while total_black_steps < args.total_black_steps:
            # This is deliberately non-blocking. A slow or failed rollout B
            # cannot delay consumption of the rollout-A transition queue.
            if sidecar is not None:
                sidecar.drain_into(replay_b)
            evaluation_failures = report_evaluation_results(
                evaluator, tb_logger, evaluation_failures
            )
            # Non-blockingly drain the dedicated actor status channel. This is
            # done before consuming transitions so timeout warnings are handled
            # even while the main result queue remains full.
            queue_put_timeouts = report_actor_status_events(
                status_queue, tb_logger, total_black_steps, queue_put_timeouts
            )
            try:
                # Wait at most two seconds for the next actor batch. Queue.get
                # deserializes the message into this learner process; batches
                # are never shared by reference after they cross the queue.
                message = result_queue.get(timeout=2.0)
            except queue.Empty:
                # An empty queue can be normal if actors are still collecting.
                # Use the opportunity to distinguish that from a crashed actor.
                failed = [process for process in actors if not process.is_alive()]
                if failed:
                    codes = ", ".join(f"{p.name}={p.exitcode}" for p in failed)
                    raise RuntimeError(f"Actor process exited unexpectedly: {codes}")
                continue

            # actor_worker catches child exceptions and sends the traceback as
            # a normal queue message because child exceptions cannot otherwise
            # be raised directly in the parent process.
            if message.get("type") == "error":
                raise RuntimeError(
                    f"Actor {message['actor_id']} failed:\n{message['traceback']}"
                )
            received_batches += 1
            actor_id = int(message["actor_id"])
            actor_versions[actor_id] = int(message["policy_version"])
            blocked_seconds += float(message.get("blocked_seconds", 0.0))
            available = len(message["actions"]) # transition numbers in this batch
            accepted = min(available, args.total_black_steps - total_black_steps)
            old_steps = total_black_steps
            # Copy the received actor batch into the learner-owned replay. Only
            # the learner mutates replay, networks, and optimizer; no locks are
            # needed around these training objects.
            for index in range(accepted):
                train_agent.add_transition(
                    message["states"][index], int(message["actions"][index]),
                    float(message["rewards"][index]), message["next_states"][index],
                    message["next_masks"][index], bool(message["dones"][index]),
                    float(message["discounts"][index]),
                )
            total_black_steps += accepted
            for episode in message["episodes"]:
                update_episode_stats(episode, stats)

            warmup_boundary = max(old_steps, args.min_replay_size)  # Earliest step eligible for learning.
            eligible_steps = max(0, total_black_steps - warmup_boundary)  # New post-warmup samples.
            update_credit += eligible_steps * args.updates_per_step  # Accumulate fractional update quota.
            requested_updates = int(update_credit)  # Whole optimizer updates to run now.
            update_credit -= requested_updates  # Preserve the fractional quota for later batches.
            log_due = total_black_steps - last_log_step >= args.log_interval  # Whether metrics are due.
            for update_index in range(requested_updates):
                ready_replay_b = (
                    replay_b if sidecar is not None
                    and len(replay_b) >= args.replay_b_min_size else None
                )
                metrics = train_agent.train_step(
                    force=True,
                    collect_metrics=log_due and update_index == requested_updates - 1,
                    replay_b=ready_replay_b,
                    replay_b_fraction=args.replay_b_fraction,
                )
                if metrics is not None:
                    last_loss = metrics
            # Gradient updates may take long enough for producers to fill and
            # time out on result_queue. Drain their status events again before
            # checkpointing or logging.
            queue_put_timeouts = report_actor_status_events(
                status_queue, tb_logger, total_black_steps, queue_put_timeouts
            )
            if sidecar is not None:
                sidecar.drain_into(replay_b)

            while args.save_checkpoint > 0 and total_black_steps >= next_save_step:
                path = save_history_checkpoint(
                    train_agent, args.history_dir, history_index, total_black_steps, args
                )
                print(f"Saved history checkpoint: {path}")
                if evaluator is not None:
                    evaluator.submit(path, total_black_steps)
                history_index += 1
                next_save_step += args.save_checkpoint

            while args.load_opponent > 0 and total_black_steps >= next_load_step:
                reloaded = maybe_reload_opponent(
                    args, opponent_rng, args.seed + 30_000 + history_index
                )
                if reloaded is not None:
                    opponent = reloaded
                    command = opponent_control_command(opponent)
                    # Queue items are point-to-point, so explicitly put one
                    # opponent command into every actor's private control queue
                    # to implement a broadcast.
                    for control in control_queues:
                        control.put(command)
                next_load_step += args.load_opponent

            while total_black_steps >= next_policy_sync:
                version = next_policy_sync
                state = cpu_state_dict(train_agent.online_net)
                # Send the same learner policy snapshot and monotonically
                # increasing version to every actor. Actors apply pending
                # commands between transition batches, not during a game step.
                for control in control_queues:
                    control.put(("policy", version, state))
                if sidecar is not None:
                    sidecar.sync_policy(version, state)
                next_policy_sync += args.policy_sync_steps

            if total_black_steps - last_log_step >= args.log_interval:
                now = time.time()
                elapsed = max(1e-6, now - start_time)
                interval_seconds = max(1e-6, now - last_log_time)
                interval_steps = total_black_steps - last_log_step
                updates = train_agent.update_steps - updates_at_start
                interval_updates = train_agent.update_steps - last_log_updates
                window_stats = stats_since(stats, last_log_stats)
                valid_versions = [version for version in actor_versions if version >= 0]
                max_lag = total_black_steps - min(valid_versions) if valid_versions else total_black_steps
                interval_actor_wait = (
                    (blocked_seconds - last_log_blocked_seconds)
                    / max(1e-6, args.num_actors * interval_seconds)
                )
                sidecar_transitions = (
                    int(sidecar.stats["transitions"]) if sidecar is not None else 0
                )
                sidecar_rate = (
                    (sidecar_transitions - last_log_sidecar_transitions)
                    / interval_seconds
                )
                active_b_fraction = (
                    args.replay_b_fraction
                    if sidecar is not None and len(replay_b) >= args.replay_b_min_size
                    else 0.0
                )
                sidecar_outcomes = {
                    key: int(sidecar.stats[key]) if sidecar is not None else 0
                    for key in ("agent_wins", "agent_losses", "draws")
                }
                sidecar_window_outcomes = {
                    key: sidecar_outcomes[key] - last_log_sidecar_outcomes[key]
                    for key in sidecar_outcomes
                }
                sidecar_window_episodes = max(
                    1, sum(sidecar_window_outcomes.values())
                )
                sidecar_window_rates = {
                    key: value / sidecar_window_episodes
                    for key, value in sidecar_window_outcomes.items()
                }
                sidecar_outcome_text = (
                    ""
                    if sidecar is None else
                    f" b_W/L/D={sidecar_window_rates['agent_wins']:.3f}/"
                    f"{sidecar_window_rates['agent_losses']:.3f}/"
                    f"{sidecar_window_rates['draws']:.3f}"
                )
                collector_text = (
                    f" sample={interval_steps / interval_seconds:.1f}/s"
                    f" update={interval_updates / interval_seconds:.1f}/s"
                    # qsize() is diagnostic only and may be approximate (or
                    # unsupported) on some multiprocessing platforms.
                    f" queue={queue_size(result_queue)}/{args.actor_queue_size}"
                    f" actor_wait={interval_actor_wait:.3f}"
                    f" policy_lag={max_lag}"
                    f" replay_b={len(replay_b)}"
                    f" b_rate={sidecar_rate:.1f}/s"
                    f" b_sample={active_b_fraction:.3f}"
                    f"{sidecar_outcome_text}"
                )
                print_log(
                    total_black_steps=total_black_steps,
                    interval_steps=interval_steps, interval_seconds=interval_seconds,
                    epsilon=epsilon_by_step(args, total_black_steps), replay_size=len(train_agent.replay),
                    updates=train_agent.update_steps, stats=window_stats, last_loss=last_loss,
                    collector_text=collector_text,
                )
                tb_logger.add_scalars(
                    step=total_black_steps, epsilon=epsilon_by_step(args, total_black_steps),
                    replay_size=len(train_agent.replay), updates=train_agent.update_steps,
                    stats=stats, window_stats=window_stats, train_metrics=last_loss,
                )
                if tb_logger.writer is not None:
                    tb_logger.writer.add_scalar("Throughput/sample_steps_per_second", interval_steps / interval_seconds, total_black_steps)
                    tb_logger.writer.add_scalar("Throughput/updates_per_second", interval_updates / interval_seconds, total_black_steps)
                    tb_logger.writer.add_scalar("Cumulative/sample_steps_per_second", total_black_steps / elapsed, total_black_steps)
                    tb_logger.writer.add_scalar("Cumulative/updates_per_second", updates / elapsed, total_black_steps)
                    tb_logger.writer.add_scalar("Actors/policy_lag_steps", max_lag, total_black_steps)
                    tb_logger.writer.add_scalar("Actors/queue_size", max(0, queue_size(result_queue)), total_black_steps)
                    tb_logger.writer.add_scalar("Actors/queue_put_timeouts_interval", queue_put_timeouts - last_log_queue_put_timeouts, total_black_steps)
                    tb_logger.writer.add_scalar("ReplayB/size", len(replay_b), total_black_steps)
                    tb_logger.writer.add_scalar("ReplayB/transitions_per_second", sidecar_rate, total_black_steps)
                    tb_logger.writer.add_scalar("ReplayB/sample_fraction", active_b_fraction, total_black_steps)
                    if sidecar is not None:
                        cumulative_episodes = max(1, sum(sidecar_outcomes.values()))
                        tb_logger.writer.add_scalar("ReplayB/queue_size", max(0, sidecar.queue_size()), total_black_steps)
                        tb_logger.writer.add_scalar("ReplayB/policy_lag_steps", sidecar.max_policy_lag(total_black_steps), total_black_steps)
                        tb_logger.writer.add_scalar("ReplayB/agent_win_rate", sidecar_window_rates["agent_wins"], total_black_steps)
                        tb_logger.writer.add_scalar("ReplayB/agent_loss_rate", sidecar_window_rates["agent_losses"], total_black_steps)
                        tb_logger.writer.add_scalar("ReplayB/draw_rate", sidecar_window_rates["draws"], total_black_steps)
                        tb_logger.writer.add_scalar("ReplayB/cumulative_agent_win_rate", sidecar_outcomes["agent_wins"] / cumulative_episodes, total_black_steps)
                        tb_logger.writer.add_scalar("ReplayB/cumulative_agent_loss_rate", sidecar_outcomes["agent_losses"] / cumulative_episodes, total_black_steps)
                        tb_logger.writer.add_scalar("ReplayB/cumulative_draw_rate", sidecar_outcomes["draws"] / cumulative_episodes, total_black_steps)
                        tb_logger.writer.add_scalar("ReplayB/queue_put_timeouts_total", sidecar.stats["queue_put_timeouts"], total_black_steps)
                        tb_logger.writer.add_scalar("ReplayB/actor_failures_total", sidecar.stats["failures"], total_black_steps)
                    tb_logger.writer.flush()
                last_log_step = total_black_steps
                last_log_time = now
                last_log_updates = train_agent.update_steps
                last_log_blocked_seconds = blocked_seconds
                last_log_queue_put_timeouts = queue_put_timeouts
                last_log_stats = dict(stats)
                last_log_sidecar_transitions = sidecar_transitions
                last_log_sidecar_outcomes = sidecar_outcomes

        if args.save_final:
            path = save_history_checkpoint(
                train_agent, args.history_dir, history_index, total_black_steps, args
            )
            print(f"Saved final checkpoint: {path}")
            if evaluator is not None:
                evaluator.submit(path, total_black_steps)
    finally:
        # This block runs after normal completion, Ctrl+C, or an exception.
        # Set the shared event first so actors stop collecting and any actor
        # blocked in its retry loop can leave without successfully queueing.
        stop_event.set()

        # Stop and reap the optional sidecar independently. Rollout A has
        # already received its stop event, so waiting for a slow heuristic move
        # cannot make rollout-A actors fill their queue during shutdown.
        if sidecar is not None:
            sidecar.close(replay_b)

        # Also enqueue a point-to-point stop command for each actor. It is
        # redundant with stop_event during normal shutdown, but lets actors
        # notice shutdown while draining their own control queue.
        for control in control_queues:
            try:
                control.put_nowait(("stop",))
            except queue.Full:
                pass
        # join() waits for graceful child exit but does not stop a process. The
        # timeout prevents a broken actor from hanging the learner forever.
        for process in actors:
            process.join(timeout=10.0)

        # terminate() is the last-resort hard stop for children that ignored the
        # cooperative Event/command shutdown. A second join reaps the process.
        for process in actors:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        # Children may have reported a final queue timeout while shutdown was
        # beginning. Consume those events before closing IPC resources.
        queue_put_timeouts = report_actor_status_events(
            status_queue, tb_logger, total_black_steps, queue_put_timeouts
        )
        # close() releases this parent's queue pipe and feeder-thread resources;
        # it does not delete replay data already copied into the learner.
        result_queue.close()
        status_queue.close()
        # Close every learner -> actor control pipe after all children have
        # exited; closing them earlier could make a live actor receive EOF.
        for control in control_queues:
            control.close()


def main() -> None:
    args = parse_args()
    if args.num_actors < 1:
        raise ValueError("--num-actors must be at least 1")
    if args.n_step < 1:
        raise ValueError("--n-step must be at least 1")
    if args.reward_shaping_scale < 0:
        raise ValueError("--reward-shaping-scale cannot be negative")
    if args.heuristic_eval_games < 1:
        raise ValueError("--heuristic-eval-games must be at least 1")

    args.run_name = validate_run_name(args.run_name)
    args.history_dir = named_directory(args.history_dir, args.run_name)
    args.tb_log_dir = named_directory(args.tb_log_dir, args.run_name)
    args.history_dir.mkdir(parents=True, exist_ok=True)
    tb_logger = TensorBoardLogger(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_agent = make_agent(args, args.seed)
    if args.black_checkpoint is not None:
        train_agent.load_checkpoint(args.black_checkpoint, load_optimizer=args.load_optimizer)
        train_agent.sync_target()
        print(f"Loaded black agent checkpoint: {args.black_checkpoint}")

    opponent = build_opponent(args, args.seed + 10_000)
    opponent_rng = random.Random(args.seed + 20_000)

    evaluator: Optional[HeuristicEvaluator] = None
    if not args.disable_heuristic_eval:
        evaluator = HeuristicEvaluator(
            board_size=args.board_size,
            num_games=args.heuristic_eval_games,
            seed=args.seed + 90_000,
        )
    completed = False
    evaluation_failures = 0
    try:
        run_actor_learner_training(
            args, train_agent, opponent, opponent_rng, tb_logger, evaluator
        )
        completed = True
    finally:
        if evaluator is not None:
            final_results = evaluator.close(drain=completed)
            report_evaluation_results(
                None, tb_logger, evaluation_failures, final_results
            )
        tb_logger.close()


if __name__ == "__main__":
    main()
