"""Configuration for the Ms. Pac-Man environment."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class MsPacmanEnvConfig:
    """Configuration shared by single and vector Ms. Pac-Man environments."""

    num_envs: int = 8
    frame_skip: int = 4
    frame_stack: int = 4
    screen_size: int = 84
    repeat_action_probability: float = 0.25
    noop_max: int = 30
    mode: int = 0
    difficulty: int = 0
    step_cost: float = 0.0
    clip_training_reward: bool = False
    include_ram_metrics: bool = False
    multiprocessing_context: str = "spawn"

    def __post_init__(self) -> None:
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if self.frame_skip not in (1, 2, 4):
            raise ValueError("frame_skip must be one of 1, 2, or 4")
        if self.frame_stack <= 0:
            raise ValueError("frame_stack must be positive")
        if self.screen_size <= 0:
            raise ValueError("screen_size must be positive")
        if not 0.0 <= self.repeat_action_probability <= 1.0:
            raise ValueError("repeat_action_probability must be in [0, 1]")
        if self.noop_max < 0:
            raise ValueError("noop_max must be non-negative")
        if self.mode not in (0, 1, 2, 3):
            raise ValueError("Ms. Pac-Man mode must be one of 0, 1, 2, or 3")
        if self.difficulty != 0:
            raise ValueError("The packaged Ms. Pac-Man ROM only supports difficulty 0")
        if not math.isfinite(self.step_cost) or self.step_cost < 0.0:
            raise ValueError("step_cost must be a finite, non-negative number")
        if self.multiprocessing_context not in ("spawn", "forkserver"):
            raise ValueError(
                "multiprocessing_context must be 'spawn' or 'forkserver'"
            )
