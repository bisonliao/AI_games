from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any

import gymnasium as gym
import numpy as np
import pybullet as pybullet
from gymnasium import spaces
from pybullet_utils.bullet_client import BulletClient


@dataclass(frozen=True, slots=True)
class JumpEnvConfig:
    """Configuration for the one-action jump task, in SI units."""

    observation_mode: str = "rgb"
    observation_width: int = 64
    observation_height: int = 64
    region_size: float = 6.0
    platform_size: float = 0.70
    platform_height: float = 0.25
    min_distance: float = 1.0
    max_distance: float = 4.0
    player_radius: float = 0.12
    player_cylinder_height: float = 0.28
    gravity: float = 9.81
    vertical_speed: float = 4.0
    charge_scale: float = 5.0
    charge_exponent: float = 3.0
    max_hold_seconds: float = 1.0
    physics_hz: int = 240
    max_flight_seconds: float = 2.0
    settle_steps: int = 12
    reward_mode: str = "dense"
    rgb_width: int = 640
    rgb_height: int = 480

    def __post_init__(self) -> None:
        if self.observation_mode not in {"rgb", "vector"}:
            raise ValueError("observation_mode must be 'rgb' or 'vector'")
        if self.observation_width < 16 or self.observation_height < 16:
            raise ValueError("observation_width and observation_height must be >= 16")
        if self.region_size <= 0 or self.platform_size <= 0:
            raise ValueError("region_size and platform_size must be positive")
        if (
            self.gravity <= 0
            or self.vertical_speed <= 0
            or self.charge_scale <= 0
            or self.charge_exponent <= 0
            or self.max_hold_seconds <= 0
        ):
            raise ValueError(
                "gravity, vertical_speed, charge_scale, charge_exponent and "
                "max_hold_seconds must be positive"
            )
        if not 0 < self.min_distance < self.max_distance:
            raise ValueError("Require 0 < min_distance < max_distance")
        if self.max_distance >= self.region_size * np.sqrt(2.0):
            raise ValueError("max_distance cannot exceed the region diagonal")
        if self.reward_mode not in {"dense", "sparse"}:
            raise ValueError("reward_mode must be 'dense' or 'sparse'")
        if self.physics_hz <= 0 or self.settle_steps <= 0:
            raise ValueError("physics_hz and settle_steps must be positive")

    @property
    def flight_time(self) -> float:
        """Ideal time for a ballistic return to the launch height."""
        return 2.0 * self.vertical_speed / self.gravity

    @property
    def maximum_jump_distance(self) -> float:
        """Ideal distance at full charge under the equal-height assumption."""
        maximum_horizontal_speed = self.charge_scale * self.max_hold_seconds
        return maximum_horizontal_speed * self.flight_time

    @property
    def observation_shape(self) -> tuple[int, ...]:
        if self.observation_mode == "rgb":
            return (2, self.observation_height, self.observation_width)
        return (1,)

    @property
    def observation_dtype(self) -> np.dtype[Any]:
        return np.dtype(np.uint8 if self.observation_mode == "rgb" else np.float32)

    def oracle_action(self, distance: float) -> np.ndarray:
        """Analytic inverse of the power-law charge curve."""
        # Ideal flight distance is
        #   d(t) = d_max * (t / T) ** p,
        # so the exact inverse is
        #   t(d) = T * (d / d_max) ** (1 / p).
        # Clipping also makes the oracle well-defined for diagnostic distances
        # outside the configured reachable interval.
        normalized_distance = float(
            np.clip(distance / self.maximum_jump_distance, 0.0, 1.0)
        )
        normalized_hold = normalized_distance ** (1.0 / self.charge_exponent)
        hold = self.max_hold_seconds * normalized_hold
        normalized = 2.0 * hold / self.max_hold_seconds - 1.0
        return np.asarray([normalized], dtype=np.float32)

    def horizontal_speed_from_hold_time(self, hold_seconds: float) -> float:
        """Map hold duration to release speed using the configured power law.

        ``p=1`` reproduces the original linear rule. The default ``p=3`` keeps
        the same full-charge speed while requiring increasingly precise timing
        for long jumps.
        """
        normalized_hold = float(
            np.clip(hold_seconds / self.max_hold_seconds, 0.0, 1.0)
        )
        maximum_horizontal_speed = self.charge_scale * self.max_hold_seconds
        return maximum_horizontal_speed * normalized_hold**self.charge_exponent

    def action_from_hold_time(self, hold_seconds: float) -> np.ndarray:
        """Convert a real keyboard hold duration to a normalized action."""
        hold = float(np.clip(hold_seconds, 0.0, self.max_hold_seconds))
        normalized = 2.0 * hold / self.max_hold_seconds - 1.0
        return np.asarray([normalized], dtype=np.float32)


class JumpEnv(gym.Env[np.ndarray, np.ndarray]):
    """A one-step PyBullet environment inspired by WeChat Jump-and-Jump.

    A call to :meth:`step` launches the player and internally simulates the
    entire trajectory. The returned transition is therefore always terminal,
    except that an internal safety timeout is reported as truncation.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        config: JumpEnvConfig | None = None,
        render_mode: str | None = None,
        playback_speed: float | None = None,
    ) -> None:
        self.config = config or JumpEnvConfig()
        if render_mode not in {None, *self.metadata["render_modes"]}:
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        self.render_mode = render_mode
        if playback_speed is not None and playback_speed <= 0:
            raise ValueError("playback_speed must be positive or None")
        self.playback_speed = playback_speed

        if self.config.observation_mode == "rgb":
            # “rgb” 在本项目中表示平台语义像素观测，而不是相机的三通道颜色图：
            # channel 0=A 平台，channel 1=B 平台，所有背景像素均为 0。
            self.observation_space = spaces.Box(
                low=0,
                high=1,
                shape=self.config.observation_shape,
                dtype=np.uint8,
            )
        else:
            self.observation_space = spaces.Box(
                low=np.asarray([0.0], dtype=np.float32),
                high=np.asarray([1.0], dtype=np.float32),
                dtype=np.float32,
            )
        self.action_space = spaces.Box(
            low=np.asarray([-1.0], dtype=np.float32),
            high=np.asarray([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        connection_mode = pybullet.GUI if render_mode == "human" else pybullet.DIRECT
        self._p = BulletClient(connection_mode=connection_mode)
        self._closed = False
        self._episode_done = True
        self._platform_a_xy = np.zeros(2, dtype=np.float64)
        self._platform_b_xy = np.zeros(2, dtype=np.float64)
        self._target_distance = 0.0
        self._build_world()
        if self.render_mode == "human":
            self._p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)

    @property
    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)

    def _build_world(self) -> None:
        p = self._p
        cfg = self.config
        p.resetSimulation()
        p.setGravity(0.0, 0.0, -cfg.gravity)
        p.setTimeStep(1.0 / cfg.physics_hz)
        p.setPhysicsEngineParameter(
            deterministicOverlappingPairs=1,
            numSolverIterations=80,
        )

        ground_collision = p.createCollisionShape(
            pybullet.GEOM_BOX,
            halfExtents=[cfg.region_size / 2.0, cfg.region_size / 2.0, 0.05],
        )
        ground_visual = p.createVisualShape(
            pybullet.GEOM_BOX,
            halfExtents=[cfg.region_size / 2.0, cfg.region_size / 2.0, 0.05],
            rgbaColor=[0.18, 0.20, 0.24, 1.0],
        )
        self._ground_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=ground_collision,
            baseVisualShapeIndex=ground_visual,
            basePosition=[0.0, 0.0, -0.05],
        )

        platform_half = [cfg.platform_size / 2.0] * 2 + [cfg.platform_height / 2.0]
        platform_collision = p.createCollisionShape(
            pybullet.GEOM_BOX, halfExtents=platform_half
        )
        visual_a = p.createVisualShape(
            pybullet.GEOM_BOX,
            halfExtents=platform_half,
            rgbaColor=[0.25, 0.55, 0.95, 1.0],
        )
        visual_b = p.createVisualShape(
            pybullet.GEOM_BOX,
            halfExtents=platform_half,
            rgbaColor=[0.20, 0.85, 0.35, 1.0],
        )
        hidden = [0.0, 0.0, -10.0]
        self._platform_a_id = p.createMultiBody(
            0.0, platform_collision, visual_a, hidden
        )
        self._platform_b_id = p.createMultiBody(
            0.0, platform_collision, visual_b, hidden
        )

        player_collision = p.createCollisionShape(
            pybullet.GEOM_CAPSULE,
            radius=cfg.player_radius,
            height=cfg.player_cylinder_height,
        )
        player_visual = p.createVisualShape(
            pybullet.GEOM_CAPSULE,
            radius=cfg.player_radius,
            length=cfg.player_cylinder_height,
            rgbaColor=[1.0, 0.72, 0.16, 1.0],
        )
        self._player_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=player_collision,
            baseVisualShapeIndex=player_visual,
            basePosition=hidden,
        )
        p.changeDynamics(
            self._player_id,
            -1,
            lateralFriction=8.0,
            spinningFriction=0.2,
            rollingFriction=0.2,
            restitution=0.0,
            linearDamping=0.0,
            angularDamping=0.95,
        )
        for body in (self._platform_a_id, self._platform_b_id):
            p.changeDynamics(body, -1, lateralFriction=8.0, restitution=0.0)

    @property
    def _player_half_height(self) -> float:
        return self.config.player_radius + self.config.player_cylinder_height / 2.0

    @property
    def _platform_top(self) -> float:
        return self.config.platform_height

    def _sample_platforms(self) -> tuple[np.ndarray, np.ndarray, float]:
        cfg = self.config
        margin = cfg.platform_size / 2.0 + 0.05
        low = -cfg.region_size / 2.0 + margin
        high = cfg.region_size / 2.0 - margin

        for _ in range(10_000):
            a_xy = self.np_random.uniform(low, high, size=2)
            distance = float(
                self.np_random.uniform(cfg.min_distance, cfg.max_distance)
            )
            angle = float(self.np_random.uniform(-np.pi, np.pi))
            b_xy = a_xy + distance * np.asarray(
                [np.cos(angle), np.sin(angle)], dtype=np.float64
            )
            if np.all(b_xy >= low) and np.all(b_xy <= high):
                return a_xy, b_xy, distance
        raise RuntimeError("Could not sample a valid platform pair")

    def _vector_observation(self) -> np.ndarray:
        normalized = self._target_distance / self.config.max_distance
        return np.asarray([normalized], dtype=np.float32)

    def _rgb_observation(self) -> np.ndarray:
        """Rasterize A/B top faces into separate binary, channel-first masks."""
        cfg = self.config
        observation = np.zeros(cfg.observation_shape, dtype=np.uint8)
        world_half = cfg.region_size / 2.0
        platform_half = cfg.platform_size / 2.0

        def draw_platform(channel: int, center: np.ndarray) -> None:
            # x grows from left to right. Image row 0 represents +y so that the
            # semantic image has the conventional top-down map orientation.
            col_start = int(
                np.floor(
                    (center[0] - platform_half + world_half)
                    / cfg.region_size
                    * cfg.observation_width
                )
            )
            col_stop = int(
                np.ceil(
                    (center[0] + platform_half + world_half)
                    / cfg.region_size
                    * cfg.observation_width
                )
            )
            row_start = int(
                np.floor(
                    (world_half - (center[1] + platform_half))
                    / cfg.region_size
                    * cfg.observation_height
                )
            )
            row_stop = int(
                np.ceil(
                    (world_half - (center[1] - platform_half))
                    / cfg.region_size
                    * cfg.observation_height
                )
            )
            col_start = int(np.clip(col_start, 0, cfg.observation_width - 1))
            col_stop = int(np.clip(col_stop, col_start + 1, cfg.observation_width))
            row_start = int(np.clip(row_start, 0, cfg.observation_height - 1))
            row_stop = int(np.clip(row_stop, row_start + 1, cfg.observation_height))
            observation[channel, row_start:row_stop, col_start:col_stop] = 1

        draw_platform(0, self._platform_a_xy)
        draw_platform(1, self._platform_b_xy)
        return observation

    def _observation(self) -> np.ndarray:
        if self.config.observation_mode == "rgb":
            return self._rgb_observation()
        return self._vector_observation()

    def _base_info(self) -> dict[str, Any]:
        return {
            "platform_a_xy": self._platform_a_xy.astype(np.float32).copy(),
            "platform_b_xy": self._platform_b_xy.astype(np.float32).copy(),
            "target_distance": float(self._target_distance),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        self._platform_a_xy, self._platform_b_xy, self._target_distance = (
            self._sample_platforms()
        )

        platform_z = self.config.platform_height / 2.0
        self._p.resetBasePositionAndOrientation(
            self._platform_a_id,
            [*self._platform_a_xy, platform_z],
            [0.0, 0.0, 0.0, 1.0],
        )
        self._p.resetBasePositionAndOrientation(
            self._platform_b_id,
            [*self._platform_b_xy, platform_z],
            [0.0, 0.0, 0.0, 1.0],
        )
        player_z = self._platform_top + self._player_half_height + 1e-3
        self._p.resetBasePositionAndOrientation(
            self._player_id,
            [*self._platform_a_xy, player_z],
            [0.0, 0.0, 0.0, 1.0],
        )
        self._p.resetBaseVelocity(
            self._player_id,
            linearVelocity=[0.0, 0.0, 0.0],
            angularVelocity=[0.0, 0.0, 0.0],
        )
        self._p.performCollisionDetection()
        self._episode_done = False
        if self.render_mode == "human":
            midpoint = (self._platform_a_xy + self._platform_b_xy) / 2.0
            self._p.resetDebugVisualizerCamera(
                cameraDistance=max(4.5, self.config.region_size * 0.8),
                cameraYaw=45.0,
                cameraPitch=-42.0,
                cameraTargetPosition=[midpoint[0], midpoint[1], 0.25],
            )
        return self._observation(), self._base_info()

    def _inside_target(self, xy: np.ndarray) -> bool:
        usable_half = self.config.platform_size / 2.0 - self.config.player_radius
        return bool(np.all(np.abs(xy - self._platform_b_xy) <= usable_half))

    def _contacting(self, body_id: int) -> bool:
        return bool(self._p.getContactPoints(self._player_id, body_id))

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._episode_done:
            raise RuntimeError("step() called outside an active episode; call reset()")

        scalar_action = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        scalar_action = float(np.clip(scalar_action, -1.0, 1.0))
        cfg = self.config
        hold_time = (scalar_action + 1.0) * 0.5 * cfg.max_hold_seconds
        direction = self._platform_b_xy - self._platform_a_xy
        direction /= np.linalg.norm(direction)
        # Release impulse follows the configured power law. With equal platform
        # heights the flight time remains fixed, hence landing distance is
        # d(t) = maximum_jump_distance * (t / max_hold_seconds) ** p.
        horizontal_speed = cfg.horizontal_speed_from_hold_time(hold_time)
        self._p.resetBaseVelocity(
            self._player_id,
            linearVelocity=[
                horizontal_speed * direction[0],
                horizontal_speed * direction[1],
                cfg.vertical_speed,
            ],
            angularVelocity=[0.0, 0.0, 0.0],
        )

        max_steps = int(round(cfg.max_flight_seconds * cfg.physics_hz))
        stable_steps = 0
        airborne = False
        latched = False
        success = False
        truncated = False
        landing_platform = "none"
        touchdown_error: float | None = None
        touchdown_xy: np.ndarray | None = None
        simulation_steps = 0

        for simulation_steps in range(1, max_steps + 1):
            self._p.stepSimulation()
            if self.render_mode == "human" and self.playback_speed is not None:
                time.sleep(1.0 / (cfg.physics_hz * self.playback_speed))
            position, _ = self._p.getBasePositionAndOrientation(self._player_id)
            velocity, _ = self._p.getBaseVelocity(self._player_id)
            xy = np.asarray(position[:2], dtype=np.float64)
            error = float(np.linalg.norm(xy - self._platform_b_xy))
            bottom_z = position[2] - self._player_half_height

            on_a = self._contacting(self._platform_a_id)
            on_b = self._contacting(self._platform_b_id)
            on_ground = self._contacting(self._ground_id)
            if not on_a and bottom_z > self._platform_top + 0.02:
                airborne = True

            if (
                touchdown_error is None
                and velocity[2] <= 0.0
                and bottom_z <= self._platform_top + 0.025
            ):
                touchdown_error = error
                touchdown_xy = xy.copy()

            # Bullet's contact solver can leave a tiny positive separating
            # velocity at rest, so use a small tolerance instead of <= 0.
            if on_b and velocity[2] <= 0.1 and position[2] > self._platform_top:
                if not self._inside_target(xy):
                    landing_platform = "B_edge"
                    break
                if not latched:
                    # WeChat-style landing: a valid top contact arrests the jump,
                    # then physics verifies that the avatar remains supported.
                    self._p.resetBaseVelocity(
                        self._player_id,
                        linearVelocity=[0.0, 0.0, 0.0],
                        angularVelocity=[0.0, 0.0, 0.0],
                    )
                    latched = True
                stable_steps += 1
                if stable_steps >= cfg.settle_steps:
                    success = True
                    landing_platform = "B"
                    break
            elif latched:
                landing_platform = "B_slip"
                break

            if airborne and on_a:
                landing_platform = "A"
                break
            if on_ground:
                landing_platform = "ground"
                break
            half_region = cfg.region_size / 2.0
            if np.any(np.abs(xy) > half_region + 0.5) or position[2] < -0.5:
                landing_platform = "out_of_bounds"
                break
        else:
            truncated = True
            landing_platform = "timeout"

        final_position, _ = self._p.getBasePositionAndOrientation(self._player_id)
        final_xy = np.asarray(final_position[:2], dtype=np.float64)
        landing_error = (
            touchdown_error
            if touchdown_error is not None
            else float(np.linalg.norm(final_xy - self._platform_b_xy))
        )
        landing_xy = touchdown_xy if touchdown_xy is not None else final_xy
        if cfg.reward_mode == "sparse":
            reward = float(success)
        else:
            dense = 1.0 - 2.0 * min(landing_error / cfg.max_distance, 1.0)
            reward = float(dense + float(success))

        self._episode_done = True
        info = self._base_info()
        info.update(
            {
                "is_success": success,
                "hold_time_s": hold_time,
                "landing_error": landing_error,
                "landing_xy": landing_xy.astype(np.float32),
                "landing_platform": landing_platform,
                "simulation_steps": simulation_steps,
            }
        )
        return self._observation(), reward, not truncated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "human":
            return None
        if self.render_mode != "rgb_array":
            raise RuntimeError("render() requires render_mode='human' or 'rgb_array'")

        midpoint = (self._platform_a_xy + self._platform_b_xy) / 2.0
        view = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[midpoint[0], midpoint[1], 0.3],
            distance=max(5.0, self.config.region_size * 0.9),
            yaw=45.0,
            pitch=-48.0,
            roll=0.0,
            upAxisIndex=2,
        )
        projection = self._p.computeProjectionMatrixFOV(
            fov=55.0,
            aspect=self.config.rgb_width / self.config.rgb_height,
            nearVal=0.05,
            farVal=30.0,
        )
        _, _, rgba, _, _ = self._p.getCameraImage(
            width=self.config.rgb_width,
            height=self.config.rgb_height,
            viewMatrix=view,
            projectionMatrix=projection,
            renderer=pybullet.ER_TINY_RENDERER,
        )
        image = np.asarray(rgba, dtype=np.uint8).reshape(
            self.config.rgb_height, self.config.rgb_width, 4
        )
        return image[..., :3].copy()

    def get_keyboard_events(self) -> dict[int, int]:
        """Return PyBullet GUI keyboard events for interactive tools."""
        if self.render_mode != "human":
            raise RuntimeError("Keyboard events require render_mode='human'")
        return self._p.getKeyboardEvents()

    def show_message(
        self,
        text: str,
        *,
        color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        duration: float = 1.0,
    ) -> int:
        """Display a short world-space message above the target platform."""
        if self.render_mode != "human":
            raise RuntimeError("Debug messages require render_mode='human'")
        return int(
            self._p.addUserDebugText(
                text,
                [self._platform_b_xy[0], self._platform_b_xy[1], 1.2],
                textColorRGB=color,
                textSize=1.6,
                lifeTime=duration,
            )
        )

    @property
    def is_connected(self) -> bool:
        return bool(self._p.getConnectionInfo().get("isConnected", 0))

    def close(self) -> None:
        if not self._closed:
            self._p.disconnect()
            self._closed = True
