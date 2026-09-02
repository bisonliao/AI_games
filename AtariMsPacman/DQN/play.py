"""Load a DQN checkpoint and run greedy Ms. Pac-Man evaluation episodes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, replace
from pathlib import Path
import time

import numpy as np
import torch

from DQN.config import DQNConfig
from DQN.network import DuelingQNetwork
from DQN.rewards import shape_reward
from PacManEnv import MsPacmanEnvConfig, make_env


@dataclass(frozen=True, slots=True)
class PlayResult:
    episode_lengths: list[int]
    episode_returns: list[float]
    episode_raw_scores: list[float]
    capped_episodes: int


def config_from_checkpoint(checkpoint: dict) -> DQNConfig:
    """Rebuild available config fields while tolerating older checkpoints."""
    saved = checkpoint.get("config")
    if not isinstance(saved, dict):
        return DQNConfig()

    base = DQNConfig()
    saved_actor_env = saved.get("actor_env", {})
    if isinstance(saved_actor_env, MsPacmanEnvConfig):
        actor_env = saved_actor_env
    elif isinstance(saved_actor_env, dict):
        env_field_names = {field.name for field in fields(MsPacmanEnvConfig)}
        actor_env = MsPacmanEnvConfig(
            **{
                key: value
                for key, value in saved_actor_env.items()
                if key in env_field_names
            }
        )
    else:
        actor_env = base.actor_env

    ignored_fields = {
        "actor_env",
        "runs_dir",
        "checkpoints_dir",
        "resume_checkpoint",
    }
    config_field_names = {
        field.name for field in fields(DQNConfig) if field.name not in ignored_fields
    }
    overrides = {
        key: value
        for key, value in saved.items()
        if key in config_field_names
    }
    return replace(base, actor_env=actor_env, **overrides)


def load_model_and_config(
    checkpoint_path: Path,
) -> tuple[DuelingQNetwork, DQNConfig]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "online_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain online_state_dict")
    config = config_from_checkpoint(checkpoint)
    model = DuelingQNetwork(config.observation_shape, config.action_count).cpu()
    model.load_state_dict(checkpoint["online_state_dict"])
    model.eval()
    return model, config


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    episodes: int,
    gui: bool,
    fps: float = 30.0,
) -> PlayResult:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if fps <= 0 or not np.isfinite(fps):
        raise ValueError("fps must be finite and positive")
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    torch.set_num_threads(1)
    model, config = load_model_and_config(checkpoint_path)
    env_config = replace(config.actor_env, num_envs=1)
    env = make_env(env_config, render_mode="human" if gui else None)
    lengths: list[int] = []
    returns: list[float] = []
    raw_scores: list[float] = []
    capped_episodes = 0

    try:
        for episode_index in range(episodes):
            observation, _ = env.reset(seed=config.evaluation_seed + episode_index)
            episode_length = 0
            episode_return = 0.0
            capped = False
            while True:
                step_started = time.monotonic()
                with torch.inference_mode():
                    q_values = model(torch.from_numpy(observation[None]))
                    action = int(q_values.argmax(dim=1).item())
                observation, reward, terminated, truncated, info = env.step(action)
                if truncated:
                    raise RuntimeError("Play environment unexpectedly truncated")
                if not np.isclose(reward, float(info["raw_reward"])):
                    raise RuntimeError("Play environment must return raw rewards")
                episode_return += shape_reward(
                    float(info["raw_reward"]),
                    bool(info["life_lost"]),
                    bool(info["game_over"]),
                    config,
                )
                episode_length += 1
                if terminated:
                    break
                if episode_length >= config.evaluation_max_episode_steps:
                    capped = True
                    capped_episodes += 1
                    break
                if gui:
                    remaining = 1.0 / fps - (time.monotonic() - step_started)
                    if remaining > 0:
                        time.sleep(remaining)

            raw_score = float(info["raw_score"])
            lengths.append(episode_length)
            returns.append(episode_return)
            raw_scores.append(raw_score)
            print(
                f"episode={episode_index + 1}/{episodes} "
                f"length={episode_length} return={episode_return:.3f} "
                f"raw_score={raw_score:.0f} capped={capped}",
                flush=True,
            )
    finally:
        env.close()

    return PlayResult(lengths, returns, raw_scores, capped_episodes)


def print_summary(result: PlayResult) -> None:
    lengths = np.asarray(result.episode_lengths, dtype=np.float64)
    returns = np.asarray(result.episode_returns, dtype=np.float64)
    raw_scores = np.asarray(result.episode_raw_scores, dtype=np.float64)
    print("\nsummary")
    print(f"episodes={len(lengths)} capped={result.capped_episodes}")
    print(f"episode_length_mean={lengths.mean():.2f}")
    print(f"episode_return_mean={returns.mean():.3f}")
    print(
        f"raw_score_mean={raw_scores.mean():.2f} "
        f"median={np.median(raw_scores):.2f} "
        f"p25={np.percentile(raw_scores, 25):.2f} "
        f"p75={np.percentile(raw_scores, 75):.2f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint_step_*.pt",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of greedy evaluation episodes (default: 10)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Render the ALE game window in real time",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Target GUI decision-frame rate (default: 30)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_checkpoint(
        args.checkpoint,
        episodes=args.episodes,
        gui=args.gui,
        fps=args.fps,
    )
    print_summary(result)


if __name__ == "__main__":
    main()
