from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from env.jump_env import JumpEnvConfig
from env.vector_env import make_async_vector_env


@dataclass(frozen=True, slots=True)
class ThroughputResult:
    num_envs: int
    transitions: int
    elapsed_seconds: float
    transitions_per_second: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "num_envs": self.num_envs,
            "transitions": self.transitions,
            "elapsed_seconds": self.elapsed_seconds,
            "transitions_per_second": self.transitions_per_second,
        }


def benchmark_vector_env(
    num_envs: int,
    *,
    transitions: int = 2_000,
    config: JumpEnvConfig | None = None,
    seed: int = 123,
) -> ThroughputResult:
    cfg = config or JumpEnvConfig()
    envs = make_async_vector_env(num_envs, cfg, context="spawn")
    completed = 0
    try:
        observations, _ = envs.reset(seed=seed)
        started = perf_counter()
        while completed < transitions:
            distances = observations[:, 0] * cfg.max_distance
            actions = np.stack(
                [cfg.oracle_action(float(distance)) for distance in distances]
            )
            observations, _, _, _, _ = envs.step(actions)
            completed += num_envs
        elapsed = perf_counter() - started
    finally:
        envs.close(terminate=True)
    return ThroughputResult(
        num_envs=num_envs,
        transitions=completed,
        elapsed_seconds=elapsed,
        transitions_per_second=completed / elapsed,
    )
