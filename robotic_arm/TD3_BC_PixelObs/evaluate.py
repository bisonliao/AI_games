"""Evaluate one TD3+BC checkpoint in the online PyBullet environment."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from .algorithm import OfflineTD3BC
from .config import EnvConfig, setup
from .pixel_env import make_env


def evaluate_checkpoint(checkpoint: Path, config: EnvConfig, episodes: int, seed_start: int,
                        device: str = "cpu", randomize: bool = False, gui: bool = False) -> dict:
    def factory():
        return make_env(config, seed_start, randomize=randomize, gui=gui)

    env = DummyVecEnv([factory])
    model = OfflineTD3BC.load(str(checkpoint), env=env, device=device)
    success = grasp = lift = place = release = premature = 0
    returns = []
    episode_lengths = []
    failures: dict[str, int] = {}
    started = time.perf_counter()
    for episode_id in range(episodes):
        observation = env.reset()
        episode_return = 0.0
        episode_length = 0
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, done, infos = env.step(action)
            episode_return += float(reward[0])
            episode_length += 1
            if done[0]:
                info = infos[0]
                success += int(info.get("success", False))
                grasp += int(info.get("ever_grasped", False))
                lift += int(info.get("ever_lifted", False))
                stage_index = int(info.get("stage_index", -1))
                place += int(stage_index >= 3)
                release += int(stage_index >= 4)
                premature += int(info.get("premature_release", False))
                reason = str(info.get("failure_reason", ""))
                failures[reason] = failures.get(reason, 0) + 1
                returns.append(episode_return)
                episode_lengths.append(episode_length)
                break
    env.close()
    return {
        "checkpoint": str(checkpoint),
        "episodes": episodes,
        "success_rate": success / episodes,
        "grasp_rate": grasp / episodes,
        "lift_rate": lift / episodes,
        "place_rate": place / episodes,
        "release_rate": release / episodes,
        "premature_release_rate": premature / episodes,
        "mean_return": float(np.mean(returns)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "failure_reasons": failures,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    setup(args.seed_start, args.device)
    result = evaluate_checkpoint(
        args.checkpoint,
        EnvConfig(),
        args.episodes,
        args.seed_start,
        args.device,
        args.randomize,
        args.gui,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
