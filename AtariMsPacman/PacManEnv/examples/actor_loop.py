"""Minimal transition collection loop with explicit masked resets."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from PacManEnv import MsPacmanEnvConfig, make_vector_env


def choose_actions(observations: np.ndarray, action_count: int) -> np.ndarray:
    """Replace this random policy with batched DDDQN inference."""
    return np.random.randint(
        action_count, size=observations.shape[0], dtype=np.int64
    )


CONFIG = replace(MsPacmanEnvConfig(), num_envs=4)


def main() -> None:
    env = make_vector_env(CONFIG)
    try:
        observations, _ = env.reset(seed=2026)
        for _ in range(1_000):
            actions = choose_actions(
                observations, env.single_action_space.n  # type: ignore[attr-defined]
            )
            next_observations, rewards, terminated, truncated, infos = env.step(
                actions
            )
            assert not np.any(truncated)

            # Store the transition before resetting. next_observations contains the
            # terminal frame stack for rows where terminated is true.
            transition = (
                observations,
                actions,
                rewards,
                next_observations,
                terminated,
                infos,
            )
            del transition  # Send it to replay in a real Actor.

            observations = next_observations
            if np.any(terminated):
                observations, _ = env.reset(
                    options={"reset_mask": terminated.astype(np.bool_)}
                )
    finally:
        env.close()


if __name__ == "__main__":
    main()
