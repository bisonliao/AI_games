"""Validated physics and task configuration for steering-only balance."""

from __future__ import annotations

from dataclasses import dataclass, field

from env.config import WindConfig


@dataclass(frozen=True, slots=True)
class BicycleSteeringEnvConfig:
    """All deterministic physics and episode parameters for environment 2."""

    physics_hz: int = 240
    control_hz: int = 20
    target_speed_mps: float = 2.0
    wheel_radius_m: float = 0.33
    drive_force_nm: float = 40.0
    steering_force_nm: float = 35.0
    steering_max_angle_rad: float = 0.35
    goal_distance_m: float = 40.0
    max_episode_seconds: float = 30.0
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
        if not 0 < self.steering_max_angle_rad <= 0.6:
            raise ValueError("steering_max_angle_rad must be in (0, 0.6]")

    @property
    def substeps(self) -> int:
        """Number of PyBullet steps executed for one agent action."""
        return self.physics_hz // self.control_hz

    @property
    def max_episode_steps(self) -> int:
        """Gymnasium control-step time limit."""
        return int(self.max_episode_seconds * self.control_hz)
