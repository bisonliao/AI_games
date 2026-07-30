from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np

from env.jump_env import JumpEnv, JumpEnvConfig


Policy = Callable[[np.ndarray, dict[str, object]], np.ndarray]


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    episodes: int
    success_rate: float
    mean_reward: float
    mean_landing_error: float
    transitions_per_second: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "episodes": self.episodes,
            "success_rate": self.success_rate,
            "mean_reward": self.mean_reward,
            "mean_landing_error": self.mean_landing_error,
            "transitions_per_second": self.transitions_per_second,
        }


def evaluate_policy(
    policy: Policy,
    *,
    episodes: int = 1_000,
    config: JumpEnvConfig | None = None,
    seed: int = 10_000,
) -> EvaluationResult:
    env = JumpEnv(config=config)
    successes = 0
    rewards = 0.0
    errors = 0.0
    started = perf_counter()
    try:
        for episode in range(episodes):
            observation, info = env.reset(seed=seed + episode)
            action = policy(observation, info)
            _, reward, _, _, final_info = env.step(action)
            successes += int(final_info["is_success"])
            rewards += reward
            errors += float(final_info["landing_error"])
    finally:
        env.close()
    elapsed = perf_counter() - started
    return EvaluationResult(
        episodes=episodes,
        success_rate=successes / episodes,
        mean_reward=rewards / episodes,
        mean_landing_error=errors / episodes,
        transitions_per_second=episodes / elapsed,
    )


def evaluate_oracle(
    *,
    episodes: int = 1_000,
    config: JumpEnvConfig | None = None,
    seed: int = 10_000,
) -> EvaluationResult:
    cfg = config or JumpEnvConfig()

    def oracle(_: np.ndarray, info: dict[str, object]) -> np.ndarray:
        return cfg.oracle_action(float(info["target_distance"]))

    return evaluate_policy(oracle, episodes=episodes, config=cfg, seed=seed)


def evaluate_random(
    *,
    episodes: int = 1_000,
    config: JumpEnvConfig | None = None,
    seed: int = 10_000,
) -> EvaluationResult:
    rng = np.random.default_rng(seed)

    def random_policy(_: np.ndarray, __: dict[str, object]) -> np.ndarray:
        return rng.uniform(-1.0, 1.0, size=1).astype(np.float32)

    return evaluate_policy(random_policy, episodes=episodes, config=config, seed=seed)
