"""Evaluate or render a checkpoint produced by :mod:`QMIX.train`.

Examples::

    # Deterministic, headless evaluation.
    python -m QMIX.play \
      --checkpoint QMIX/checkpoints/qmix/legacy/official/simple_spread/state_steps_4000000.pt \
      --episodes 100

    # Open the archived MPE GUI at roughly 10 environment steps per second.
    python -m QMIX.play --checkpoint /path/to/state_steps_4000000.pt --render

The environment and network configuration are reconstructed from checkpoint
metadata.  Current command-line training defaults therefore cannot silently
change the policy being evaluated.
"""

from __future__ import annotations

import argparse
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
from .train import ALGORITHM_NAME, SUPPORTED_CHECKPOINT_VERSIONS


_CHECKPOINT_PATTERN = re.compile(r"state_steps_(\d+)\.pt")
_LANDMARK_CENTER_SUCCESS_METRIC = "landmark_center_success"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play or evaluate a legacy OpenAI MPE QMIX checkpoint",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "checkpoint .pt file or a directory containing state_steps_<N>.pt; "
            "a directory selects the greatest saved step"
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="number of deterministic evaluation episodes (default: 10)",
    )
    parser.add_argument(
        "--env-seed",
        type=int,
        default=10_000,
        help=(
            "base environment seed; episode i uses env-seed+i "
            "(default: 10000)"
        ),
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="open the archived MPE GUI (headless evaluation is the default)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="GUI speed in environment steps/second; 0 disables throttling",
    )
    parser.add_argument(
        "--report-json",
        default="",
        help="optional path for a JSON evaluation report",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="explicitly run inference on CPU (CUDA is the default)",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=0,
        help="CUDA device index (default: 0)",
    )
    return parser.parse_args(argv)


def _checkpoint_step(path: Path) -> int:
    match = _CHECKPOINT_PATTERN.fullmatch(path.name)
    return int(match.group(1)) if match else -1


def resolve_checkpoint(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    if candidate.is_dir():
        checkpoints = [
            item
            for item in candidate.glob("state_steps_*.pt")
            if _checkpoint_step(item) >= 0
        ]
        if checkpoints:
            return max(checkpoints, key=_checkpoint_step).resolve()
        raise FileNotFoundError(
            f"no state_steps_<N>.pt checkpoint found in {candidate}"
        )
    raise FileNotFoundError(f"checkpoint path does not exist: {candidate}")


def _device(options: argparse.Namespace) -> torch.device:
    if options.no_cuda:
        return torch.device("cpu")
    return torch.device(f"cuda:{options.cuda_device}")


def _seed_everything(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _load_checkpoint(
    path: Path, device: torch.device
) -> tuple[dict, dict, LearnerConfig, QMIXLearner]:
    # QMIX checkpoints are local training artifacts containing optimizer state.
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("unsupported checkpoint: expected a dictionary")
    version = checkpoint.get("checkpoint_version")
    if version not in SUPPORTED_CHECKPOINT_VERSIONS:
        raise ValueError(
            "unsupported checkpoint version {}; supported versions are {}".format(
                version, SUPPORTED_CHECKPOINT_VERSIONS
            )
        )
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("unsupported checkpoint: metadata must be a dictionary")
    if metadata.get("algorithm") != ALGORITHM_NAME:
        raise ValueError(
            "unsupported checkpoint algorithm {!r}; expected {!r}".format(
                metadata.get("algorithm"), ALGORITHM_NAME
            )
        )
    if metadata.get("env_backend") != "legacy":
        raise ValueError("QMIX play only supports the legacy MPE backend")
    if metadata.get("policy_mode") != "official":
        raise ValueError("QMIX play only supports official vector actions")
    scenario = metadata.get("scenario")
    if not isinstance(scenario, str) or not scenario:
        raise ValueError("unsupported checkpoint: missing scenario metadata")
    config_values = metadata.get("learner_config")
    if not isinstance(config_values, dict):
        raise ValueError("unsupported checkpoint: missing learner_config metadata")
    try:
        config = LearnerConfig(**config_values)
    except TypeError as exc:
        raise ValueError(
            "unsupported checkpoint learner_config: {}".format(exc)
        ) from exc
    learner_state = checkpoint.get("learner")
    if not isinstance(learner_state, dict):
        raise ValueError("unsupported checkpoint: missing learner state")

    learner = QMIXLearner(config, device)
    learner.load_checkpoint_state(learner_state, load_optimizer=False)
    learner.agent.eval()
    learner.mixer.eval()
    learner.target_agent.eval()
    learner.target_mixer.eval()
    return checkpoint, metadata, config, learner


def _validate_environment(
    env: LegacyMPEEnv, metadata: dict, config: LearnerConfig
) -> None:
    expected_branches = [list(spec.branch_sizes) for spec in env.action_specs]
    checks = {
        "n_agents": (env.n_agents, config.n_agents),
        "observation dimension": (env.observation_dims[0], config.obs_dim),
        "state dimension": (env.state_dim, config.state_dim),
        "number of actions": (env.action_specs[0].n_actions, config.n_actions),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(
                f"checkpoint/environment {name} mismatch: "
                f"checkpoint={expected}, environment={actual}"
            )
    if list(metadata.get("observation_dims", [])) != list(env.observation_dims):
        raise ValueError("checkpoint/environment observation_dims mismatch")
    if list(metadata.get("action_branches", [])) != expected_branches:
        raise ValueError("checkpoint/environment action branches mismatch")


def _play_episode(
    env: LegacyMPEEnv,
    learner: QMIXLearner,
    max_episode_len: int,
    seed: int,
    render: bool,
    render_delay: float,
) -> dict[str, object]:
    observations = env.reset(seed=seed)
    hidden = learner.initial_hidden()
    last_actions = torch.zeros(
        env.n_agents,
        learner.config.n_actions,
        dtype=torch.float32,
        device=learner.device,
    )
    team_episode_reward = 0.0
    episode_reward = 0.0
    episode_length = 0

    if render:
        env.render()
    for _ in range(max_episode_len):
        actions, hidden = learner.select_actions(
            torch.as_tensor(
                observations, dtype=torch.float32, device=learner.device
            ),
            last_actions,
            hidden,
            epsilon=0.0,
            deterministic=True,
        )
        next_observations, agent_rewards, dones, _ = env.step(
            actions.cpu().numpy()
        )
        if not np.allclose(agent_rewards, agent_rewards[0], rtol=1e-5, atol=1e-6):
            raise RuntimeError(
                "QMIX play requires a shared-reward cooperative scenario; "
                f"received rewards {agent_rewards.tolist()}"
            )
        team_episode_reward += float(np.mean(agent_rewards))
        episode_reward += float(np.sum(agent_rewards))
        episode_length += 1
        observations = next_observations
        last_actions = F.one_hot(
            actions, num_classes=learner.config.n_actions
        ).to(dtype=torch.float32)
        if render:
            env.render()
            if render_delay > 0:
                time.sleep(render_delay)
        if bool(np.all(dones)):
            break

    task_metrics: dict[str, float] = {}
    metric_plugin = get_scenario_metric_plugin(env.scenario_name)
    if metric_plugin is not None:
        task_metrics = {
            name: float(value) for name, value in metric_plugin(env).items()
        }
    return {
        "seed": int(seed),
        "team_episode_reward": team_episode_reward,
        "episode_reward": episode_reward,
        "episode_length": episode_length,
        "task_metrics": task_metrics,
    }


def _mean_task_metrics(episodes: list[dict[str, object]]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for episode in episodes:
        for name, value in episode["task_metrics"].items():
            values.setdefault(name, []).append(float(value))
    return {name: float(np.mean(items)) for name, items in values.items()}


def _print_episode(index: int, total: int, result: dict[str, object]) -> None:
    task_text = ""
    if result["task_metrics"]:
        task_text = ", task: " + ", ".join(
            f"{name}={value:.6f}"
            for name, value in sorted(result["task_metrics"].items())
        )
    print(
        "[Play {}/{}] seed={}, team_reward={:.6f}, reward={:.6f}, "
        "length={}{}".format(
            index,
            total,
            result["seed"],
            result["team_episode_reward"],
            result["episode_reward"],
            result["episode_length"],
            task_text,
        )
    )


def play(options: argparse.Namespace) -> dict[str, object]:
    if options.episodes <= 0:
        raise ValueError("--episodes must be greater than 0")
    if options.fps < 0:
        raise ValueError("--fps must be greater than or equal to 0")
    if options.cuda_device < 0:
        raise ValueError("--cuda-device must be greater than or equal to 0")

    device = _device(options)
    _seed_everything(options.env_seed, device)
    checkpoint_path = resolve_checkpoint(options.checkpoint)
    checkpoint, metadata, config, learner = _load_checkpoint(
        checkpoint_path, device
    )
    max_episode_len = int(metadata.get("max_episode_len", 25))
    if max_episode_len <= 0:
        raise ValueError("checkpoint max_episode_len must be greater than 0")

    env = LegacyMPEEnv(metadata["scenario"])
    try:
        _validate_environment(env, metadata, config)
        print(f"Checkpoint: {checkpoint_path}")
        print(
            "Saved progress: env_steps={}, episodes={}, learner_updates={}".format(
                checkpoint.get("env_steps", "unknown"),
                checkpoint.get("completed_episodes", "unknown"),
                checkpoint.get("learner", {}).get("train_updates", "unknown"),
            )
        )
        print(
            "Environment: legacy / {}, policy: official, device: {}, GUI: {}".format(
                metadata["scenario"],
                device,
                "on" if options.render else "off",
            )
        )
        print(
            "Learner: lr={:.6g}, gamma={:.6g}, hidden_dim={}, "
            "mixing_embed_dim={}, double_q={}".format(
                config.lr,
                config.gamma,
                config.hidden_dim,
                config.mixing_embed_dim,
                config.double_q,
            )
        )
        render_delay = (
            1.0 / options.fps
            if options.render and options.fps > 0
            else 0.0
        )
        episode_results = []
        for episode_index in range(options.episodes):
            result = _play_episode(
                env,
                learner,
                max_episode_len,
                options.env_seed + episode_index,
                options.render,
                render_delay,
            )
            episode_results.append(result)
            _print_episode(episode_index + 1, options.episodes, result)
    finally:
        env.close()

    team_rewards = np.asarray(
        [item["team_episode_reward"] for item in episode_results],
        dtype=np.float64,
    )
    episode_rewards = np.asarray(
        [item["episode_reward"] for item in episode_results],
        dtype=np.float64,
    )
    episode_lengths = np.asarray(
        [item["episode_length"] for item in episode_results],
        dtype=np.float64,
    )
    task_metrics = _mean_task_metrics(episode_results)
    success_count = None
    if "episode_success" in task_metrics:
        success_count = sum(
            float(item["task_metrics"].get("episode_success", 0.0)) >= 0.5
            for item in episode_results
        )
    landmark_center_success_count = None
    if _LANDMARK_CENTER_SUCCESS_METRIC in task_metrics:
        landmark_center_success_count = sum(
            float(
                item["task_metrics"].get(
                    _LANDMARK_CENTER_SUCCESS_METRIC, 0.0
                )
            )
            >= 0.5
            for item in episode_results
        )
    report: dict[str, object] = {
        "checkpoint": os.fspath(checkpoint_path),
        "checkpoint_version": checkpoint.get("checkpoint_version"),
        "saved_env_steps": checkpoint.get("env_steps"),
        "saved_completed_episodes": checkpoint.get("completed_episodes"),
        "evaluation_episodes": int(options.episodes),
        "evaluation_env_seed": int(options.env_seed),
        "render": bool(options.render),
        "device": str(device),
        "team_episode_reward_mean": float(team_rewards.mean()),
        "team_episode_reward_std": float(team_rewards.std()),
        "episode_reward_mean": float(episode_rewards.mean()),
        "episode_reward_std": float(episode_rewards.std()),
        "episode_length_mean": float(episode_lengths.mean()),
        "task_metrics": task_metrics,
        "episodes": episode_results,
        "metadata": metadata,
    }
    if success_count is not None:
        report["task_success_count"] = int(success_count)
        report["task_success_rate"] = float(success_count / options.episodes)
    if landmark_center_success_count is not None:
        report["task_landmark_center_success_count"] = int(
            landmark_center_success_count
        )
        report["task_landmark_center_success_rate"] = float(
            landmark_center_success_count / options.episodes
        )
    print(
        "[Summary] episodes={}, team_reward={:.6f} +/- {:.6f}, "
        "reward={:.6f} +/- {:.6f}, mean_length={:.2f}".format(
            options.episodes,
            report["team_episode_reward_mean"],
            report["team_episode_reward_std"],
            report["episode_reward_mean"],
            report["episode_reward_std"],
            report["episode_length_mean"],
        )
    )
    if task_metrics:
        print(
            "[Summary] task metrics: "
            + ", ".join(
                f"{name}={value:.6f}"
                for name, value in sorted(task_metrics.items())
            )
        )
    if success_count is not None:
        print(
            "[Summary] strict full-coverage success (d < 0.10): "
            "{}/{} ({:.6f})".format(
                success_count,
                options.episodes,
                success_count / options.episodes,
            )
        )
    if landmark_center_success_count is not None:
        print(
            "[Summary] landmark-center success (d < 0.15): "
            "{}/{} ({:.6f})".format(
                landmark_center_success_count,
                options.episodes,
                landmark_center_success_count / options.episodes,
            )
        )

    if options.report_json:
        report_path = Path(options.report_json).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(f"Report: {report_path}")
    return report


def main(argv: list[str] | None = None) -> None:
    options = parse_args(argv)
    try:
        play(options)
    except KeyboardInterrupt:
        print("\nPlay interrupted; environment closed.")


if __name__ == "__main__":
    main()
