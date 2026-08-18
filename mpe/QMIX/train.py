"""Train QMIX on a fully cooperative archived OpenAI MPE scenario.

Run from the repository root with ``python -m QMIX.train``.  Defaults target
``simple_spread`` through the direct legacy API and the official vector action
semantics; PettingZoo is intentionally not an option in this entry point.
"""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import re
import time

import numpy as np
import torch
from torch.nn import functional as F

from maddpg.common.scenario_metrics import get_scenario_metric_plugin

from .env import LegacyMPEEnv
from .learner import LearnerConfig, QMIXLearner
from .replay_buffer import EpisodeBuilder, EpisodeReplayBuffer
from .tb_logger import TensorBoardLogger, make_run_dir


CHECKPOINT_VERSION = 2
SUPPORTED_CHECKPOINT_VERSIONS = (1, CHECKPOINT_VERSION)
ALGORITHM_NAME = "qmix"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QMIX for fully cooperative legacy OpenAI MPE"
    )
    # Environment defaults mirror the existing MADDPG entry point.
    parser.add_argument("--scenario", default="simple_spread")
    parser.add_argument("--max-episode-len", type=int, default=25)
    parser.add_argument("--num-episodes", type=int, default=2_000_000)

    # Stable QMIX defaults for legacy MPE's comparatively large reward scale.
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="number of complete episodes in one learner batch")
    parser.add_argument("--buffer-size", type=int, default=5_000,
                        help="episode replay capacity")
    parser.add_argument("--updates-per-episode", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--mixing-embed-dim", type=int, default=32)
    parser.add_argument("--hypernet-embed-dim", type=int, default=64)
    parser.add_argument("--optimizer-alpha", type=float, default=0.99)
    parser.add_argument("--optimizer-eps", type=float, default=1e-5)
    parser.add_argument("--grad-norm-clip", type=float, default=10.0)
    parser.add_argument("--td-loss", choices=("huber", "mse"), default="huber")
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument(
        "--max-abs-q",
        type=float,
        default=1_000.0,
        help="abort before an update when |Q| exceeds this threshold; 0 disables",
    )
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=None,
        help="learner reward multiplier; default is 1/number_of_agents",
    )
    parser.add_argument(
        "--bootstrap-time-limit",
        action="store_true",
        help="bootstrap at max-episode-len (off by default for stable finite-horizon training)",
    )
    parser.add_argument("--target-update-interval", type=int, default=200,
                        help="hard target update interval in completed episodes")
    parser.add_argument("--no-double-q", action="store_true",
                        help="disable Double-Q action selection")

    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-finish", type=float, default=0.05)
    parser.add_argument("--epsilon-anneal-steps", type=int, default=50_000)

    parser.add_argument("--tb-log-interval", type=int, default=10_000,
                        help="aggregate and flush TensorBoard every N env steps; 0 disables")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--exp-name", default=None)
    parser.add_argument("--console-log-interval", type=int, default=100,
                        help="print rolling metrics every N completed episodes; 0 disables")

    parser.add_argument("--save-dir", default="QMIX/checkpoints")
    parser.add_argument("--save-rate", type=int, default=10_000,
                        help="checkpoint interval in completed episodes")
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--load-dir", default="",
                        help="checkpoint file/root; defaults to --save-dir")
    parser.add_argument("--checkpoint-eval-episodes", type=int, default=10)
    parser.add_argument("--checkpoint-eval-seed", type=int, default=10_000)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-cuda", action="store_true",
                        help="explicitly use CPU (CUDA is the default)")
    parser.add_argument("--cuda-device", type=int, default=0)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        "max_episode_len",
        "num_episodes",
        "batch_size",
        "buffer_size",
        "updates_per_episode",
        "hidden_dim",
        "mixing_embed_dim",
        "hypernet_embed_dim",
        "target_update_interval",
        "epsilon_anneal_steps",
        "save_rate",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.buffer_size < args.batch_size:
        raise ValueError("--buffer-size must be at least --batch-size")
    for name in ("lr", "grad_norm_clip", "huber_delta"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.reward_scale is not None and args.reward_scale <= 0:
        raise ValueError("--reward-scale must be positive")
    if args.max_abs_q < 0:
        raise ValueError("--max-abs-q cannot be negative")
    for name in ("tb_log_interval", "console_log_interval", "checkpoint_eval_episodes"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative")
    if not 0.0 <= args.gamma <= 1.0:
        raise ValueError("--gamma must be in [0, 1]")
    if not 0.0 <= args.epsilon_finish <= args.epsilon_start <= 1.0:
        raise ValueError(
            "epsilon values must satisfy 0 <= finish <= start <= 1"
        )


def _seed_everything(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _device(args: argparse.Namespace) -> torch.device:
    # Do not silently fall back: training defaults to the promised CUDA device.
    if args.no_cuda:
        return torch.device("cpu")
    return torch.device(f"cuda:{args.cuda_device}")


def _epsilon(args: argparse.Namespace, env_steps: int) -> float:
    fraction = min(1.0, max(0.0, env_steps / args.epsilon_anneal_steps))
    return args.epsilon_start + fraction * (
        args.epsilon_finish - args.epsilon_start
    )


def _state_dir(root: str | Path, scenario: str) -> Path:
    return Path(root) / ALGORITHM_NAME / "legacy" / "official" / scenario


def _legacy_state_dir(root: str | Path, scenario: str) -> Path:
    """Pre-algorithm-directory layout, accepted for restore compatibility."""

    return Path(root) / "legacy" / "official" / scenario


def _metadata(
    args: argparse.Namespace, env: LegacyMPEEnv, config: LearnerConfig
) -> dict:
    return {
        "algorithm": ALGORITHM_NAME,
        "env_backend": "legacy",
        "policy_mode": "official",
        "scenario": args.scenario,
        "max_episode_len": int(args.max_episode_len),
        "n_agents": env.n_agents,
        "observation_dims": list(env.observation_dims),
        "action_branches": [
            list(spec.branch_sizes) for spec in env.action_specs
        ],
        "central_state": "concatenated_local_observations",
        "reward_scale": float(args.reward_scale),
        "bootstrap_time_limit": bool(args.bootstrap_time_limit),
        "learner_config": asdict(config),
    }


def _checkpoint_payload(
    learner: QMIXLearner,
    metadata: dict,
    env_steps: int,
    completed_episodes: int,
    last_target_update_episode: int,
) -> dict:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "metadata": metadata,
        "env_steps": int(env_steps),
        "completed_episodes": int(completed_episodes),
        "last_target_update_episode": int(last_target_update_episode),
        "learner": learner.checkpoint_state(),
    }


def _save_checkpoint(save_dir: Path, payload: dict) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    destination = save_dir / f"state_steps_{payload['env_steps']}.pt"
    temporary = destination.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    print(f"[Checkpoint] saved: {destination.resolve()}")
    return destination


def _checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"state_steps_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else -1


def _find_checkpoint(path: Path, scenario: str) -> Path:
    if path.is_file():
        return path
    candidates = list(path.glob("state_steps_*.pt")) if path.is_dir() else []
    if not candidates:
        scenario_dir = _state_dir(path, scenario)
        candidates = (
            list(scenario_dir.glob("state_steps_*.pt"))
            if scenario_dir.is_dir()
            else []
        )
    if not candidates:
        legacy_scenario_dir = _legacy_state_dir(path, scenario)
        candidates = (
            list(legacy_scenario_dir.glob("state_steps_*.pt"))
            if legacy_scenario_dir.is_dir()
            else []
        )
    candidates = [candidate for candidate in candidates if _checkpoint_step(candidate) >= 0]
    if not candidates:
        raise FileNotFoundError(f"no state_steps_<N>.pt checkpoint found under {path}")
    return max(candidates, key=_checkpoint_step)


def _load_checkpoint(
    path: Path,
    learner: QMIXLearner,
    expected_metadata: dict,
    device: torch.device,
) -> tuple[int, int, int]:
    # Checkpoints are produced locally by this script and contain optimizer state.
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    checkpoint_version = checkpoint.get("checkpoint_version")
    if checkpoint_version not in SUPPORTED_CHECKPOINT_VERSIONS:
        raise ValueError(
            "unsupported checkpoint version {}; supported versions are {}".format(
                checkpoint_version, SUPPORTED_CHECKPOINT_VERSIONS
            )
        )
    metadata_to_compare = expected_metadata
    if checkpoint_version == 1:
        # V1 predates the legacy-reward scaling, finite-horizon target,
        # robust loss, and divergence-guard metadata. Supplying the old
        # lr/gamma explicitly still permits a faithful restore.
        metadata_to_compare = dict(expected_metadata)
        metadata_to_compare.pop("reward_scale", None)
        metadata_to_compare.pop("bootstrap_time_limit", None)
        learner_config = dict(metadata_to_compare["learner_config"])
        for key in ("td_loss", "huber_delta", "max_abs_q"):
            learner_config.pop(key, None)
        metadata_to_compare["learner_config"] = learner_config
    if checkpoint.get("metadata") != metadata_to_compare:
        raise ValueError(
            "checkpoint metadata does not match this environment/configuration"
        )
    learner.load_checkpoint_state(checkpoint["learner"], load_optimizer=True)
    print(f"[Checkpoint] restored: {path.resolve()}")
    print("[Checkpoint] episode replay intentionally starts empty after restore")
    return (
        int(checkpoint["env_steps"]),
        int(checkpoint["completed_episodes"]),
        int(checkpoint.get("last_target_update_episode", 0)),
    )


@contextmanager
def _preserve_rng(device: torch.device):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    )
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _rollout_episode(
    env: LegacyMPEEnv,
    learner: QMIXLearner,
    args: argparse.Namespace,
    env_steps: int,
    seed: int | None,
    deterministic: bool,
) -> tuple[object, dict[str, object], int]:
    observations = env.reset(seed=seed)
    state = env.state(observations)
    builder = EpisodeBuilder(observations, state, args.max_episode_len)
    hidden = learner.initial_hidden()
    last_actions = torch.zeros(
        env.n_agents, learner.config.n_actions, device=learner.device
    )
    team_episode_reward = 0.0
    summed_episode_reward = 0.0
    episode_length = 0
    last_epsilon = 0.0 if deterministic else _epsilon(args, env_steps)

    scaled_team_episode_reward = 0.0
    for timestep in range(args.max_episode_len):
        epsilon = 0.0 if deterministic else _epsilon(args, env_steps)
        last_epsilon = epsilon
        observation_tensor = torch.as_tensor(
            observations, dtype=torch.float32, device=learner.device
        )
        actions, hidden = learner.select_actions(
            observation_tensor,
            last_actions,
            hidden,
            epsilon=epsilon,
            deterministic=deterministic,
        )
        action_array = actions.cpu().numpy()
        next_observations, agent_rewards, dones, _ = env.step(action_array)
        if not np.allclose(agent_rewards, agent_rewards[0], rtol=1e-5, atol=1e-6):
            raise RuntimeError(
                "QMIX entry point only supports a shared-reward cooperative task; "
                f"received per-agent rewards {agent_rewards.tolist()}"
            )
        team_reward = float(np.mean(agent_rewards))
        scaled_team_reward = team_reward * args.reward_scale
        env_terminated = bool(np.all(dones))
        reached_time_limit = timestep + 1 >= args.max_episode_len
        terminated = env_terminated or (
            reached_time_limit and not args.bootstrap_time_limit
        )
        next_state = env.state(next_observations)
        builder.add(
            action_array,
            scaled_team_reward,
            terminated,
            next_observations,
            next_state,
        )
        team_episode_reward += team_reward
        scaled_team_episode_reward += scaled_team_reward
        summed_episode_reward += float(np.sum(agent_rewards))
        episode_length += 1
        env_steps += 1
        observations = next_observations
        last_actions = F.one_hot(
            actions, num_classes=learner.config.n_actions
        ).to(dtype=torch.float32)
        if terminated:
            break

    metrics = {
        "team_episode_reward": team_episode_reward,
        "scaled_team_episode_reward": scaled_team_episode_reward,
        "summed_episode_reward": summed_episode_reward,
        "episode_length": float(episode_length),
        "epsilon": last_epsilon,
        "task_metrics": {},
    }
    scenario_metric_plugin = get_scenario_metric_plugin(args.scenario)
    if scenario_metric_plugin is not None:
        metrics["task_metrics"] = {
            name: float(value)
            for name, value in scenario_metric_plugin(env).items()
        }
    return builder.finish(), metrics, env_steps


def _evaluate(
    learner: QMIXLearner,
    args: argparse.Namespace,
    episodes: int,
    seed: int,
) -> dict[str, object]:
    team_rewards = []
    summed_rewards = []
    lengths = []
    task_metric_values: dict[str, list[float]] = {}
    with _preserve_rng(learner.device):
        env = LegacyMPEEnv(args.scenario)
        try:
            for episode_index in range(episodes):
                _, metrics, _ = _rollout_episode(
                    env,
                    learner,
                    args,
                    env_steps=0,
                    seed=seed + episode_index,
                    deterministic=True,
                )
                team_rewards.append(metrics["team_episode_reward"])
                summed_rewards.append(metrics["summed_episode_reward"])
                lengths.append(metrics["episode_length"])
                for name, value in metrics["task_metrics"].items():
                    task_metric_values.setdefault(name, []).append(float(value))
        finally:
            env.close()
    return {
        "evaluation_episodes": int(episodes),
        "evaluation_seed": int(seed),
        "team_episode_reward_mean": float(np.mean(team_rewards)),
        "team_episode_reward_std": float(np.std(team_rewards)),
        "episode_reward_mean": float(np.mean(summed_rewards)),
        "episode_reward_std": float(np.std(summed_rewards)),
        "episode_length_mean": float(np.mean(lengths)),
        "team_episode_rewards": [float(value) for value in team_rewards],
        "episode_rewards": [float(value) for value in summed_rewards],
        "task_metrics": {
            name: float(np.mean(values))
            for name, values in task_metric_values.items()
        },
    }


def _save_evaluation(
    save_dir: Path, evaluation: dict, env_steps: int, completed_episodes: int
) -> None:
    report = dict(evaluation)
    report["env_steps"] = int(env_steps)
    report["completed_episodes"] = int(completed_episodes)
    for path in (
        save_dir / f"evaluation_steps_{env_steps}.json",
        save_dir / "evaluation.json",
    ):
        with path.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")


def _evaluation_tensorboard_metrics(evaluation: dict) -> dict[str, float]:
    return {
        "eval/episode_reward_mean": evaluation["episode_reward_mean"],
        "eval/episode_reward_std": evaluation["episode_reward_std"],
        "eval/team_episode_reward_mean": evaluation[
            "team_episode_reward_mean"
        ],
        "eval/team_episode_reward_std": evaluation[
            "team_episode_reward_std"
        ],
        "eval/episode_length_mean": evaluation["episode_length_mean"],
        **{
            f"eval/task_{name}": value
            for name, value in evaluation["task_metrics"].items()
        },
    }


def _evaluate_checkpoint(
    learner: QMIXLearner,
    args: argparse.Namespace,
    save_dir: Path,
    logger: TensorBoardLogger,
    env_steps: int,
    completed_episodes: int,
) -> None:
    evaluation = _evaluate(
        learner,
        args,
        args.checkpoint_eval_episodes,
        args.checkpoint_eval_seed,
    )
    _save_evaluation(save_dir, evaluation, env_steps, completed_episodes)
    print(
        "[Checkpoint Eval] "
        f"episode_reward={evaluation['episode_reward_mean']:.6f} "
        f"team_reward={evaluation['team_episode_reward_mean']:.6f}"
    )
    logger.immediate(_evaluation_tensorboard_metrics(evaluation), env_steps)


def train(args: argparse.Namespace) -> None:
    _validate_args(args)
    device = _device(args)
    _seed_everything(args.seed, device)
    env = LegacyMPEEnv(args.scenario)
    if not env.shared_reward:
        env.close()
        raise ValueError(
            f"scenario {args.scenario!r} is not marked collaborative/shared-reward"
        )
    if len(set(env.observation_dims)) != 1:
        env.close()
        raise ValueError("all agents must have the same observation dimension")
    n_actions_set = {spec.n_actions for spec in env.action_specs}
    if len(n_actions_set) != 1:
        env.close()
        raise ValueError("all agents must have the same discrete action count")
    if args.reward_scale is None:
        # Archived MultiAgentEnv sums one reward callback result per agent and
        # then broadcasts it in collaborative scenarios. Averaging restores a
        # scale comparable to the scenario-level cooperative objective.
        args.reward_scale = 1.0 / env.n_agents

    config = LearnerConfig(
        n_agents=env.n_agents,
        obs_dim=env.observation_dims[0],
        state_dim=env.state_dim,
        n_actions=n_actions_set.pop(),
        hidden_dim=args.hidden_dim,
        mixing_embed_dim=args.mixing_embed_dim,
        hypernet_embed_dim=args.hypernet_embed_dim,
        gamma=args.gamma,
        lr=args.lr,
        optimizer_alpha=args.optimizer_alpha,
        optimizer_eps=args.optimizer_eps,
        grad_norm_clip=args.grad_norm_clip,
        double_q=not args.no_double_q,
        td_loss=args.td_loss,
        huber_delta=args.huber_delta,
        max_abs_q=args.max_abs_q,
    )
    learner = QMIXLearner(config, device)
    replay = EpisodeReplayBuffer(args.buffer_size)
    metadata = _metadata(args, env, config)
    save_dir = _state_dir(args.save_dir, args.scenario)

    env_steps = 0
    completed_episodes = 0
    last_target_update_episode = 0
    if args.restore:
        load_root = Path(args.load_dir if args.load_dir else args.save_dir)
        checkpoint_path = _find_checkpoint(load_root, args.scenario)
        env_steps, completed_episodes, last_target_update_episode = _load_checkpoint(
            checkpoint_path, learner, metadata, device
        )

    log_dir = None
    if args.tb_log_interval > 0:
        log_dir = make_run_dir(args.runs_dir, args.scenario, args.exp_name)
        print(f"TensorBoard log_dir: {log_dir.resolve()}")
    else:
        print("TensorBoard logging disabled (--tb-log-interval=0)")
    logger = TensorBoardLogger(
        args.tb_log_interval,
        env_steps,
        log_dir,
        config={**vars(args), "device": str(device), "metadata": metadata},
    )

    print(
        f"QMIX legacy+official | scenario={args.scenario} | "
        f"agents={env.n_agents} | obs={config.obs_dim} | "
        f"actions={config.n_actions} | state={config.state_dim} | device={device} | "
        f"reward_scale={args.reward_scale:.6g} | td_loss={args.td_loss} | "
        f"time_limit_bootstrap={args.bootstrap_time_limit}"
    )
    recent_summed_rewards: deque[float] = deque(maxlen=100)
    recent_team_rewards: deque[float] = deque(maxlen=100)
    recent_losses: deque[float] = deque(maxlen=100)
    last_checkpoint_episode: int | None = None
    session_start_steps = env_steps
    started = time.perf_counter()
    interrupted = False
    completed_normally = False

    try:
        while completed_episodes < args.num_episodes:
            episode, episode_metrics, env_steps = _rollout_episode(
                env,
                learner,
                args,
                env_steps,
                seed=args.seed if completed_episodes == 0 and env_steps == 0 else None,
                deterministic=False,
            )
            replay.add(episode)
            completed_episodes += 1
            recent_summed_rewards.append(
                episode_metrics["summed_episode_reward"]
            )
            recent_team_rewards.append(episode_metrics["team_episode_reward"])

            logger.mean(
                "reward/episode_reward",
                episode_metrics["summed_episode_reward"],
            )
            logger.mean(
                "reward/team_episode_reward",
                episode_metrics["team_episode_reward"],
            )
            logger.mean(
                "reward/scaled_team_episode_reward",
                episode_metrics["scaled_team_episode_reward"],
            )
            logger.mean("env/episode_length", episode_metrics["episode_length"])
            logger.record_task_metrics(episode_metrics["task_metrics"])
            logger.latest("policy/epsilon", episode_metrics["epsilon"])
            logger.latest("replay/episodes", len(replay))

            if len(replay) >= args.batch_size:
                for _ in range(args.updates_per_episode):
                    batch = replay.sample(args.batch_size, device)
                    update = learner.train(batch)
                    recent_losses.append(update["loss"])
                    logger.mean("loss/td_loss", update["loss"])
                    logger.mean("loss/td_error_abs", update["td_error_abs"])
                    logger.mean(
                        "grad/pre_clip_global_norm",
                        update["pre_clip_grad_norm"],
                    )
                    logger.mean(
                        "grad/post_clip_global_norm",
                        update["post_clip_grad_norm"],
                    )
                    logger.mean("q/chosen_total", update["chosen_q"])
                    logger.mean("q/target_total", update["target_q"])
                    logger.mean(
                        "q/chosen_total_abs_max",
                        update["chosen_q_abs_max"],
                    )
                    logger.mean(
                        "q/target_total_abs_max",
                        update["target_q_abs_max"],
                    )
                    logger.latest("learner/updates", learner.train_updates)

            if (
                completed_episodes - last_target_update_episode
                >= args.target_update_interval
            ):
                learner.update_targets()
                last_target_update_episode = completed_episodes
                logger.latest(
                    "learner/last_target_update_episode",
                    last_target_update_episode,
                )

            elapsed = max(time.perf_counter() - started, 1e-9)
            logger.latest(
                "system/env_steps_per_second",
                (env_steps - session_start_steps) / elapsed,
            )
            logger.maybe_flush(env_steps)

            if (
                args.console_log_interval > 0
                and completed_episodes % args.console_log_interval == 0
            ):
                loss_text = (
                    f"{np.mean(recent_losses):.6f}"
                    if recent_losses
                    else "warming-up"
                )
                print(
                    f"steps={env_steps} episodes={completed_episodes} "
                    f"reward_roll100={np.mean(recent_summed_rewards):.3f} "
                    f"team_reward_roll100={np.mean(recent_team_rewards):.3f} "
                    f"loss_roll100={loss_text} "
                    f"epsilon={episode_metrics['epsilon']:.4f}"
                )

            if completed_episodes % args.save_rate == 0:
                payload = _checkpoint_payload(
                    learner,
                    metadata,
                    env_steps,
                    completed_episodes,
                    last_target_update_episode,
                )
                _save_checkpoint(save_dir, payload)
                last_checkpoint_episode = completed_episodes
                if args.checkpoint_eval_episodes > 0:
                    _evaluate_checkpoint(
                        learner,
                        args,
                        save_dir,
                        logger,
                        env_steps,
                        completed_episodes,
                    )
        completed_normally = True
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted; saving the latest completed-episode state...")
    finally:
        try:
            if (
                completed_episodes > 0
                and last_checkpoint_episode != completed_episodes
            ):
                payload = _checkpoint_payload(
                    learner,
                    metadata,
                    env_steps,
                    completed_episodes,
                    last_target_update_episode,
                )
                _save_checkpoint(save_dir, payload)
                if completed_normally and args.checkpoint_eval_episodes > 0:
                    _evaluate_checkpoint(
                        learner,
                        args,
                        save_dir,
                        logger,
                        env_steps,
                        completed_episodes,
                    )
        finally:
            logger.close(env_steps)
            env.close()

    status = "interrupted" if interrupted else "finished"
    print(
        f"QMIX {status}: steps={env_steps}, episodes={completed_episodes}, "
        f"updates={learner.train_updates}"
    )


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
