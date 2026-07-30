from __future__ import annotations

from functools import partial
from typing import Any

from gymnasium.vector import AsyncVectorEnv, AutoresetMode

from env.jump_env import JumpEnv, JumpEnvConfig


def _make_env(config: JumpEnvConfig) -> JumpEnv:
    return JumpEnv(config=config)


def make_async_vector_env(
    num_envs: int,
    config: JumpEnvConfig | None = None,
    *,
    context: str = "spawn",
) -> AsyncVectorEnv:
    """Create a synchronous-batch, process-parallel vector environment."""
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    cfg = config or JumpEnvConfig()
    env_fns = [partial(_make_env, cfg) for _ in range(num_envs)]
    return AsyncVectorEnv(
        env_fns,
        shared_memory=True,
        context=context,
        autoreset_mode=AutoresetMode.SAME_STEP,
    )


def final_info_at(infos: dict[str, Any], index: int) -> dict[str, Any]:
    """Read terminal info from a SAME_STEP AsyncVectorEnv result."""
    final_infos = infos.get("final_info")
    if final_infos is None:
        raise KeyError("Expected SAME_STEP vector info containing 'final_info'")
    if not isinstance(final_infos, dict):
        info = final_infos[index]
        if info is None:
            raise ValueError(f"Environment {index} did not return terminal info")
        return info

    def unbatch(mapping: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in mapping.items():
            if key.startswith("_"):
                continue
            mask = mapping.get(f"_{key}")
            if mask is not None and not bool(mask[index]):
                continue
            result[key] = unbatch(value) if isinstance(value, dict) else value[index]
        return result

    return unbatch(final_infos)
