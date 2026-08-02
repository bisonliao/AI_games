"""Validated physics, task, and seeded wind configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class WindConfig:
    """Cross-wind force, timing, duration, and application-point defaults."""
    enabled: bool = True
    min_force_n: float = 5.0
    max_force_n: float = 15.0
    min_interval_s: float = 1.0
    max_interval_s: float = 4.0
    min_duration_s: float = 0.5
    max_duration_s: float = 2.0
    application_height_m: float = 1.0

    def __post_init__(self) -> None:
        if not 0 <= self.min_force_n <= self.max_force_n:
            raise ValueError("wind force range must be non-negative and ordered")
        if not 0 <= self.min_interval_s <= self.max_interval_s:
            raise ValueError("wind interval range must be non-negative and ordered")
        if not 0 < self.min_duration_s <= self.max_duration_s:
            raise ValueError("wind duration range must be positive and ordered")


@dataclass(frozen=True, slots=True)
class BicycleEnvConfig:
    """All deterministic physics and episode parameters for the environment."""
    physics_hz: int = 240
    control_hz: int = 20
    target_speed_mps: float = 2.0
    wheel_radius_m: float = 0.33
    drive_force_nm: float = 40.0
    reaction_torque_nm: float = 15.0
    reaction_max_speed_rad_s: float = 120.0
    steering_force_nm: float = 80.0
    goal_distance_m: float = 60.0
    max_episode_seconds: float = 40.0
    fall_roll_rad: float = 0.7853981633974483
    initial_roll_rad: float = 0.08726646259971647
    initial_roll_rate_rad_s: float = 0.25
    progress_reward_per_m: float = 0.01
    success_reward: float = 1.0
    fall_penalty: float = -1.0
    solver_iterations: int = 50
    wheel_friction: float = 1.2
    wind: WindConfig = field(default_factory=WindConfig)

    def __post_init__(self) -> None:
        if self.physics_hz <= 0 or self.control_hz <= 0:
            raise ValueError("simulation frequencies must be positive")
        if self.physics_hz % self.control_hz:
            raise ValueError("physics_hz must be divisible by control_hz")
        if self.goal_distance_m <= 0 or self.max_episode_seconds <= 0:
            raise ValueError("episode distance and duration must be positive")

    @property
    def substeps(self) -> int:
        return self.physics_hz // self.control_hz

    @property
    def max_episode_steps(self) -> int:
        return int(self.max_episode_seconds * self.control_hz)
