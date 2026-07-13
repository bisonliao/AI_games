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

from env import make_vector_env

try:
    from .agent import DQNAgent, RandomAgent, encode_boards, slice_obs
except ImportError:
    from agent import DQNAgent, RandomAgent, encode_boards, slice_obs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Gomoku black-side DQN agent by self-play.")

    parser.add_argument("--board-size", type=int, default=5)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--async-envs", action="store_true")
    parser.add_argument("--num-actors", type=int, default=4)
    parser.add_argument("--envs-per-actor", type=int, default=16)
    parser.add_argument("--actor-batch-size", type=int, default=256)
    parser.add_argument("--actor-queue-size", type=int, default=8)
    parser.add_argument("--policy-sync-steps", type=int, default=5_000)
    parser.add_argument("--updates-per-step", type=float, default=0.25)
    parser.add_argument("--actor-device", choices=("cpu",), default="cpu")
    parser.add_argument("--total-black-steps", type=int, default=50_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--black-checkpoint", type=Path, default=None)
    parser.add_argument("--opponent-checkpoint", type=Path, default=None)
    parser.add_argument("--history-dir", type=Path, default=Path(__file__).resolve().parent / "history")
    parser.add_argument("--save-checkpoint", type=int, default=1_000_000)
    parser.add_argument("--load-opponent", type=int, default=2_000_000)
    parser.add_argument("--checkpoint-rand", type=int, default=10)
    parser.add_argument("--load-optimizer", action="store_true")

    parser.add_argument("--hidden-channels", type=int, default=96)
    parser.add_argument("--num-res-blocks", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-size", type=int, default=200_000)
    parser.add_argument("--min-replay-size", type=int, default=10_000)
    parser.add_argument("--target-update", type=int, default=2_000)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--no-double-dqn", action="store_true")

    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=1_000_000)
    parser.add_argument("--opponent-epsilon", type=float, default=0.0)
    parser.add_argument("--log-interval", type=int, default=10_000)
    parser.add_argument("--tb-log-dir", type=Path, default=Path(__file__).resolve().parent / "runs")
    parser.add_argument("--disable-tensorboard", action="store_true")
    parser.add_argument("--save-final", action="store_true")

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
        train_freq=args.train_freq,
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
    path = history_dir / f"{index:06d}.pt"
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
        print("Opponent reload skipped: no history checkpoints yet")
        return None

    recent = checkpoints[-max(1, args.checkpoint_rand) :]
    checkpoint = rng.choice(recent)
    opponent = make_agent(args, seed)
    opponent.load_checkpoint(checkpoint, load_optimizer=False)
    print(f"Reloaded opponent from history: {checkpoint}")
    return opponent


def info_value(
    infos: Dict[str, Any],
    key: str,
    env_index: int,
    done: bool,
    default: Any,
) -> Any:
    """Read per-env info without mixing terminal and reset episodes.

    With SameStep autoreset, top-level ``infos`` belongs to the newly reset
    episode for a done env.  The step that actually terminated is stored under
    ``final_info`` and must be used for its reward and outcome.
    """
    if done:
        final_info_mask = infos.get("_final_info")
        if final_info_mask is None or not final_info_mask[env_index]:
            raise RuntimeError(
                "Done environment is missing final_info; "
                "the vector autoreset contract does not match the training loop."
            )
        final_infos = infos.get("final_info")
        # Gymnasium 1.x vectorizes final_info as a dict of arrays. Older
        # releases exposed an object array containing one dict per env.
        if isinstance(final_infos, dict):
            key_mask = final_infos.get(f"_{key}")
            if key in final_infos and (key_mask is None or key_mask[env_index]):
                return final_infos[key][env_index]
        elif final_infos is not None:
            final_info = final_infos[env_index]
            if final_info is not None:
                return final_info.get(key, default)
        # Never fall through to top-level info for a done SameStep env: those
        # values belong to the reset episode.
        return default

    if key not in infos:
        return default
    value = infos[key]
    try:
        return value[env_index]
    except Exception:
        return default


def print_log(
    *,
    total_black_steps: int,
    start_time: float,
    epsilon: float,
    replay_size: int,
    updates: int,
    stats: Dict[str, float],
    last_loss: Optional[Dict[str, float]],
    collector_text: str = "",
) -> None:
    elapsed = max(1e-6, time.time() - start_time)
    steps_per_second = total_black_steps / elapsed
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

    def add_episode(
        self,
        *,
        step: int,
        length: int,
        episode_return: float,
    ) -> None:
        if self.writer is None:
            return
        self.writer.add_scalar("Episode/length", length, step)
        self.writer.add_scalar("Episode/return", episode_return, step)

    def add_scalars(
        self,
        *,
        step: int,
        epsilon: float,
        replay_size: int,
        updates: int,
        stats: Dict[str, float],
        train_metrics: Optional[Dict[str, float]],
    ) -> None:
        if self.writer is None:
            return

        episodes = max(1.0, stats["episodes"])
        self.writer.add_scalar("Episode/mean_length", stats["episode_black_steps"] / episodes, step)
        self.writer.add_scalar("Episode/mean_return", stats["episode_return"] / episodes, step)

        self.writer.add_scalar("Outcome/win_rate", stats["black_wins"] / episodes, step)
        self.writer.add_scalar("Outcome/loss_rate", stats["black_losses"] / episodes, step)
        self.writer.add_scalar("Outcome/draw_rate", stats["draws"] / episodes, step)

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
        return None
    from DQN.async_train import cpu_state_dict
    return (cpu_state_dict(opponent.online_net), dict(opponent.model_kwargs))


def update_episode_stats(
    episode: Any,
    stats: Dict[str, float],
    tb_logger: TensorBoardLogger,
    total_black_steps: int,
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
    tb_logger.add_episode(step=total_black_steps, length=int(length), episode_return=float(episode_return))


def queue_size(value: Any) -> int:
    try:
        return int(value.qsize())
    except (AttributeError, NotImplementedError):
        return -1


def run_async_training(
    args: argparse.Namespace,
    train_agent: DQNAgent,
    opponent: Any,
    opponent_rng: random.Random,
    tb_logger: TensorBoardLogger,
) -> None:
    from DQN.async_train import actor_worker, cpu_state_dict

    for name in ("num_actors", "envs_per_actor", "actor_batch_size", "actor_queue_size"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1")
    if args.policy_sync_steps < 1:
        raise ValueError("--policy-sync-steps must be at least 1")
    if args.updates_per_step < 0:
        raise ValueError("--updates-per-step cannot be negative")
    if args.async_envs:
        print("Warning: --async-envs is ignored when --num-actors > 0", flush=True)

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=args.actor_queue_size)
    control_queues = [ctx.Queue() for _ in range(args.num_actors)]
    stop_event = ctx.Event()
    generated_steps = ctx.Value("q", 0)
    initial_policy = cpu_state_dict(train_agent.online_net)
    opponent_payload = initial_opponent_payload(opponent)
    actor_config = {
        "seed": args.seed,
        "board_size": args.board_size,
        "envs_per_actor": args.envs_per_actor,
        "actor_batch_size": args.actor_batch_size,
        "actor_device": args.actor_device,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "epsilon_decay_steps": args.epsilon_decay_steps,
        "opponent_epsilon": args.opponent_epsilon,
    }
    actors = [
        ctx.Process(
            target=actor_worker,
            name=f"gomoku-actor-{actor_id}",
            args=(
                actor_id, actor_config, initial_policy, dict(train_agent.model_kwargs),
                opponent_payload, result_queue, control_queues[actor_id], stop_event,
                generated_steps,
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
    actor_versions = [-1] * args.num_actors
    blocked_seconds = 0.0
    received_batches = 0
    start_time = time.time()
    updates_at_start = train_agent.update_steps

    print(
        f"Async training: actors={args.num_actors} envs/actor={args.envs_per_actor} "
        f"batch={args.actor_batch_size} UTD={args.updates_per_step:g}",
        flush=True,
    )
    for process in actors:
        process.start()

    try:
        while total_black_steps < args.total_black_steps:
            try:
                message = result_queue.get(timeout=2.0)
            except queue.Empty:
                failed = [process for process in actors if not process.is_alive()]
                if failed:
                    codes = ", ".join(f"{p.name}={p.exitcode}" for p in failed)
                    raise RuntimeError(f"Actor process exited unexpectedly: {codes}")
                continue

            if message.get("type") == "error":
                raise RuntimeError(
                    f"Actor {message['actor_id']} failed:\n{message['traceback']}"
                )
            received_batches += 1
            actor_id = int(message["actor_id"])
            actor_versions[actor_id] = int(message["policy_version"])
            blocked_seconds += float(message.get("blocked_seconds", 0.0))
            available = len(message["actions"])
            accepted = min(available, args.total_black_steps - total_black_steps)
            old_steps = total_black_steps
            for index in range(accepted):
                train_agent.add_transition(
                    message["states"][index], int(message["actions"][index]),
                    float(message["rewards"][index]), message["next_states"][index],
                    message["next_masks"][index], bool(message["dones"][index]),
                )
            total_black_steps += accepted
            for episode in message["episodes"]:
                update_episode_stats(episode, stats, tb_logger, total_black_steps)

            warmup_boundary = max(old_steps, args.min_replay_size)
            eligible_steps = max(0, total_black_steps - warmup_boundary)
            update_credit += eligible_steps * args.updates_per_step
            requested_updates = int(update_credit)
            update_credit -= requested_updates
            for _ in range(requested_updates):
                metrics = train_agent.train_step(force=True)
                if metrics is not None:
                    last_loss = metrics

            while args.save_checkpoint > 0 and total_black_steps >= next_save_step:
                path = save_history_checkpoint(
                    train_agent, args.history_dir, history_index, total_black_steps, args
                )
                print(f"Saved history checkpoint: {path}")
                history_index += 1
                next_save_step += args.save_checkpoint

            while args.load_opponent > 0 and total_black_steps >= next_load_step:
                reloaded = maybe_reload_opponent(
                    args, opponent_rng, args.seed + 30_000 + history_index
                )
                if reloaded is not None:
                    opponent = reloaded
                    state = cpu_state_dict(opponent.online_net)
                    kwargs = dict(opponent.model_kwargs)
                    for control in control_queues:
                        control.put(("opponent", state, kwargs))
                next_load_step += args.load_opponent

            while total_black_steps >= next_policy_sync:
                version = next_policy_sync
                state = cpu_state_dict(train_agent.online_net)
                for control in control_queues:
                    control.put(("policy", version, state))
                next_policy_sync += args.policy_sync_steps

            if total_black_steps - last_log_step >= args.log_interval:
                elapsed = max(1e-6, time.time() - start_time)
                updates = train_agent.update_steps - updates_at_start
                valid_versions = [version for version in actor_versions if version >= 0]
                max_lag = total_black_steps - min(valid_versions) if valid_versions else total_black_steps
                collector_text = (
                    f" sample={total_black_steps / elapsed:.1f}/s"
                    f" update={updates / elapsed:.1f}/s"
                    f" queue={queue_size(result_queue)}/{args.actor_queue_size}"
                    f" actor_wait={blocked_seconds / max(1, args.num_actors * elapsed):.3f}"
                    f" policy_lag={max_lag}"
                )
                print_log(
                    total_black_steps=total_black_steps, start_time=start_time,
                    epsilon=epsilon_by_step(args, total_black_steps), replay_size=len(train_agent.replay),
                    updates=train_agent.update_steps, stats=stats, last_loss=last_loss,
                    collector_text=collector_text,
                )
                tb_logger.add_scalars(
                    step=total_black_steps, epsilon=epsilon_by_step(args, total_black_steps),
                    replay_size=len(train_agent.replay), updates=train_agent.update_steps,
                    stats=stats, train_metrics=last_loss,
                )
                if tb_logger.writer is not None:
                    tb_logger.writer.add_scalar("Throughput/sample_steps_per_second", total_black_steps / elapsed, total_black_steps)
                    tb_logger.writer.add_scalar("Throughput/updates_per_second", updates / elapsed, total_black_steps)
                    tb_logger.writer.add_scalar("Actors/policy_lag_steps", max_lag, total_black_steps)
                    tb_logger.writer.add_scalar("Actors/queue_size", max(0, queue_size(result_queue)), total_black_steps)
                last_log_step = total_black_steps

        if args.save_final:
            path = save_history_checkpoint(
                train_agent, args.history_dir, history_index, total_black_steps, args
            )
            print(f"Saved final checkpoint: {path}")
    finally:
        stop_event.set()
        for control in control_queues:
            try:
                control.put_nowait(("stop",))
            except queue.Full:
                pass
        for process in actors:
            process.join(timeout=10.0)
        for process in actors:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        result_queue.close()
        for control in control_queues:
            control.close()


def main() -> None:
    args = parse_args()
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

    if args.num_actors < 0:
        raise ValueError("--num-actors cannot be negative")
    if args.num_actors > 0:
        try:
            run_async_training(args, train_agent, opponent, opponent_rng, tb_logger)
        finally:
            tb_logger.close()
        return

    if args.async_envs:
        print(
            "Warning: --async-envs is deprecated; the lightweight Gomoku env "
            "is normally faster with synchronous vectorization.",
            flush=True,
        )

    envs = make_vector_env(
        args.num_envs,
        board_size=args.board_size,
        asynchronous=args.async_envs,
        seed=args.seed,
    )

    pending_states: List[Optional[np.ndarray]] = [None for _ in range(args.num_envs)]
    pending_actions: List[Optional[int]] = [None for _ in range(args.num_envs)]
    episode_black_steps = np.zeros(args.num_envs, dtype=np.int64)

    total_black_steps = 0
    next_save_step = args.save_checkpoint
    next_load_step = args.load_opponent
    history_index = next_history_index(args.history_dir)
    last_log_step = 0
    last_loss: Optional[Dict[str, float]] = None
    stats = {
        "episodes": 0.0,
        "black_wins": 0.0,
        "black_losses": 0.0,
        "draws": 0.0,
        "episode_black_steps": 0.0,
        "episode_return": 0.0,
    }
    start_time = time.time()

    try:
        obs, _ = envs.reset(seed=args.seed)

        while total_black_steps < args.total_black_steps:
            current_players = obs["current_player"].reshape(-1)
            actions = np.zeros(args.num_envs, dtype=np.int64)

            black_indices = np.flatnonzero(current_players == 1)
            white_indices = np.flatnonzero(current_players == -1)

            if len(black_indices) > 0:
                epsilon = epsilon_by_step(args, total_black_steps)
                boards, players, masks = slice_obs(obs, black_indices)
                black_actions = train_agent.select_actions(boards, players, masks, epsilon=epsilon)
                encoded_states = encode_boards(boards, players)
                actions[black_indices] = black_actions

                for row, env_index in enumerate(black_indices):
                    pending_states[env_index] = encoded_states[row]
                    pending_actions[env_index] = int(black_actions[row])
                    episode_black_steps[env_index] += 1

                total_black_steps += len(black_indices)

            if len(white_indices) > 0:
                boards, players, masks = slice_obs(obs, white_indices)
                white_actions = opponent.select_actions(
                    boards,
                    players,
                    masks,
                    epsilon=args.opponent_epsilon,
                )
                actions[white_indices] = white_actions

            # make_vector_env uses SameStep autoreset. For done envs, next_obs
            # is already the initial observation of the next episode; terminal
            # data is in infos["final_obs"] / infos["final_info"].
            next_obs, _, terminated, truncated, infos = envs.step(actions)
            done_flags = np.logical_or(terminated, truncated)

            for env_index in range(args.num_envs):
                acted_player = int(current_players[env_index])
                done = bool(done_flags[env_index])

                # One replay transition is one BLACK decision interval:
                #   black state/action -> optional white reply -> next black state.
                # Therefore a non-terminal black move remains pending until the
                # white move has been stepped. A terminal black move closes it
                # immediately.
                if acted_player == 1:
                    if done:
                        reward_black = float(info_value(infos, "reward_black", env_index, done, 0.0))
                        zero_state = np.zeros(
                            (3, args.board_size, args.board_size),
                            dtype=np.float32,
                        )
                        zero_mask = np.zeros(args.board_size * args.board_size, dtype=np.bool_)
                        train_agent.add_transition(
                            pending_states[env_index],
                            pending_actions[env_index],
                            reward_black,
                            zero_state,
                            zero_mask,
                            True,
                        )
                        pending_states[env_index] = None
                        pending_actions[env_index] = None
                    else:
                        continue

                if pending_states[env_index] is None:
                    if not done:
                        continue
                else:
                    reward_black = float(info_value(infos, "reward_black", env_index, done, 0.0))
                    if done:
                        # next_obs belongs to the next episode under SameStep;
                        # never connect that state to this terminal transition.
                        next_state = np.zeros((3, args.board_size, args.board_size), dtype=np.float32)
                        next_mask = np.zeros(args.board_size * args.board_size, dtype=np.bool_)
                    else:
                        next_state = encode_boards(
                            next_obs["board"][env_index],
                            next_obs["current_player"][env_index],
                        )[0]
                        next_mask = next_obs["action_mask"][env_index].astype(np.bool_)

                    train_agent.add_transition(
                        pending_states[env_index],
                        pending_actions[env_index],
                        reward_black,
                        next_state,
                        next_mask,
                        done,
                    )
                    pending_states[env_index] = None
                    pending_actions[env_index] = None

                if done:
                    winner = int(info_value(infos, "winner", env_index, done, 0))
                    episode_return = float(info_value(infos, "reward_black", env_index, done, 0.0))
                    episode_length = int(episode_black_steps[env_index])
                    stats["episodes"] += 1
                    stats["episode_black_steps"] += float(episode_length)
                    stats["episode_return"] += episode_return
                    if winner == 1:
                        stats["black_wins"] += 1
                    elif winner == -1:
                        stats["black_losses"] += 1
                    else:
                        stats["draws"] += 1
                    tb_logger.add_episode(
                        step=total_black_steps,
                        length=episode_length,
                        episode_return=episode_return,
                    )
                    episode_black_steps[env_index] = 0

            for _ in range(len(black_indices)):
                metrics = train_agent.train_step()
                if metrics is not None:
                    last_loss = metrics

            while args.save_checkpoint > 0 and total_black_steps >= next_save_step:
                checkpoint_path = save_history_checkpoint(
                    train_agent,
                    args.history_dir,
                    history_index,
                    total_black_steps,
                    args,
                )
                print(f"Saved history checkpoint: {checkpoint_path}")
                history_index += 1
                next_save_step += args.save_checkpoint

            while args.load_opponent > 0 and total_black_steps >= next_load_step:
                reloaded = maybe_reload_opponent(args, opponent_rng, args.seed + 30_000 + history_index)
                if reloaded is not None:
                    opponent = reloaded
                next_load_step += args.load_opponent

            if total_black_steps - last_log_step >= args.log_interval:
                print_log(
                    total_black_steps=total_black_steps,
                    start_time=start_time,
                    epsilon=epsilon_by_step(args, total_black_steps),
                    replay_size=len(train_agent.replay),
                    updates=train_agent.update_steps,
                    stats=stats,
                    last_loss=last_loss,
                )
                tb_logger.add_scalars(
                    step=total_black_steps,
                    epsilon=epsilon_by_step(args, total_black_steps),
                    replay_size=len(train_agent.replay),
                    updates=train_agent.update_steps,
                    stats=stats,
                    train_metrics=last_loss,
                )
                last_log_step = total_black_steps

            obs = next_obs

        if args.save_final:
            checkpoint_path = save_history_checkpoint(
                train_agent,
                args.history_dir,
                history_index,
                total_black_steps,
                args,
            )
            print(f"Saved final checkpoint: {checkpoint_path}")

    finally:
        envs.close()
        tb_logger.close()


if __name__ == "__main__":
    main()
