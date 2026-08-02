"""Greedy checkpoint evaluation CLI and asynchronous evaluator process."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import os
from pathlib import Path
import queue
import time
from typing import Any

import numpy as np
import torch

from env import BicycleBalanceEnv

from .checkpoint import load_checkpoint, online_state_from_checkpoint
from .config import DQNConfig
from .network import DuelingQNetwork, load_numpy_state_dict


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Aggregate metrics from a fixed-seed greedy evaluation batch."""

    env_steps: int
    success_rate: float
    mean_return: float
    mean_distance_m: float
    mean_length: float
    fall_rate: float
    timeout_rate: float
    roll_rms: float


def evaluate_network(
    network: DuelingQNetwork,
    episodes: int = 100,
    seed_start: int = 100_000,
    display: bool = False,
) -> EvaluationResult:
    """Evaluate a greedy CPU policy on a deterministic sequence of seeds.

    Evaluation never applies epsilon exploration. ``display=False`` runs
    headless as fast as possible; GUI mode follows the bicycle and sleeps at the
    20 Hz control rate so a human can inspect behavior.
    """
    network.cpu().eval()
    env = BicycleBalanceEnv(render_mode="human" if display else None)
    outcomes: list[str] = []
    returns: list[float] = []
    distances: list[float] = []
    lengths: list[int] = []
    roll_squares = 0.0
    roll_count = 0
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed_start + episode)
            total_reward = 0.0
            length = 0
            while True:
                with torch.no_grad():
                    action = int(
                        network(torch.as_tensor(observation).unsqueeze(0)).argmax(1).item()
                    )
                observation, reward, terminated, truncated, info = env.step(action)
                if display:
                    time.sleep(1.0 / env.config.control_hz)
                total_reward += reward
                length += 1
                roll_squares += float(info["roll_rad"]) ** 2
                roll_count += 1
                if terminated or truncated:
                    break
            outcomes.append(str(info["outcome"]))
            returns.append(total_reward)
            distances.append(float(info["progress_m"]))
            lengths.append(length)
            if display:
                print(
                    f"episode={episode} outcome={info['outcome']} "
                    f"distance={info['progress_m']:.2f} return={total_reward:.3f}"
                )
    finally:
        env.close()
    return EvaluationResult(
        env_steps=0,
        success_rate=outcomes.count("success") / episodes,
        mean_return=float(np.mean(returns)),
        mean_distance_m=float(np.mean(distances)),
        mean_length=float(np.mean(lengths)),
        fall_rate=outcomes.count("fall") / episodes,
        timeout_rate=outcomes.count("timeout") / episodes,
        roll_rms=float(np.sqrt(roll_squares / max(1, roll_count))),
    )


def evaluator_process(
    config: DQNConfig,
    request_queue: Any,
    result_queue: Any,
    stop_event: Any,
    episodes: int,
) -> None:
    """Serve evaluation requests without blocking learner optimization.

    The queue holds NumPy state dicts, keeping this process CPU-only and avoiding
    CUDA multiprocessing storage. Every request uses the same seed range, making
    success-rate curves comparable across learner checkpoints.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)
    network = DuelingQNetwork(
        config.observation_dim, config.action_dim, config.hidden_dim
    ).cpu()
    while not stop_event.is_set():
        try:
            env_steps, state = request_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if env_steps is None:
            return
        load_numpy_state_dict(network, state)
        result = replace(evaluate_network(network, episodes=episodes), env_steps=int(env_steps))
        result_queue.put(result)


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone evaluation command-line interface."""
    parser = argparse.ArgumentParser(
        description="Evaluate a bicycle DQN checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="checkpoint file containing learner.online weights",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="number of greedy evaluation episodes",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=100_000,
        help="first seed in the deterministic consecutive evaluation set",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="render evaluation in the PyBullet GUI at real-time speed",
    )
    return parser


def main() -> None:
    """Load one checkpoint, run evaluation, and print aggregate metrics."""
    args = build_parser().parse_args()
    state = load_checkpoint(args.checkpoint)
    saved_config = state.get("config", {})
    config = DQNConfig.from_dict(saved_config) if saved_config else DQNConfig()
    network = DuelingQNetwork(
        config.observation_dim, config.action_dim, config.hidden_dim
    )
    network.load_state_dict(online_state_from_checkpoint(state))
    result = evaluate_network(network, args.episodes, args.seed_start, args.display)
    print(
        f"success_rate={result.success_rate:.3f} fall_rate={result.fall_rate:.3f} "
        f"timeout_rate={result.timeout_rate:.3f} mean_return={result.mean_return:.3f} "
        f"mean_distance_m={result.mean_distance_m:.2f}"
    )


if __name__ == "__main__":
    main()
