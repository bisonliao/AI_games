"""Helpers for constructing synchronous vector environments."""
from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence

from gymnasium.vector import AutoresetMode, SyncVectorEnv

from .placement_env import PlacementTetrisEnv


def make_sync_vector_env(
    num_envs: int,
    *,
    seed: int | None = 0,
    seeds: Sequence[int] | None = None,
    **env_kwargs,
) -> SyncVectorEnv:
    """Build placement-level training environments with distinct seeds."""
    if seeds is None:
        if seed is None:
            raise ValueError("provide seed or seeds")
        seeds = tuple(int(seed) + index for index in range(num_envs))
    if len(seeds) != num_envs:
        raise ValueError("seeds must contain one distinct seed per environment")
    if len(set(int(value) for value in seeds)) != num_envs:
        raise ValueError("each vector environment must have a distinct seed")
    def factory(index: int) -> Callable[[], PlacementTetrisEnv]:
        def thunk() -> PlacementTetrisEnv:
            return PlacementTetrisEnv(**env_kwargs)

        return thunk

    # SAME_STEP prevents Gymnasium's NEXT_STEP mode from inserting a reset-only
    # pseudo-transition with an all-zero terminal action mask. The returned
    # terminated flag still blocks bootstrap, while next_obs is immediately a
    # valid reset observation for the actor's following action.
    return SyncVectorEnv(
        [factory(i) for i in range(num_envs)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
