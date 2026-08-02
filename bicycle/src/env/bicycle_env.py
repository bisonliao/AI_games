"""PyBullet/Gymnasium bicycle task with reaction-wheel balance and gusts."""

from __future__ import annotations

from importlib.resources import files
import math
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p

from .config import BicycleEnvConfig
from .wind import SmoothGustGenerator, WindState


class BicycleBalanceEnv(gym.Env[np.ndarray, int]):
    """Balance a driven bicycle for 60 m using three reaction-wheel torques.

    The simulation runs at 240 Hz while the agent acts at 20 Hz. Observations
    contain roll sin/cos, local roll rate, reaction-wheel speed, and forward
    speed. All physics clients are instance-local, so separate processes can run
    environments safely in headless DIRECT mode.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        config: BicycleEnvConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError(f"unsupported render_mode: {render_mode}")
        self.config = config or BicycleEnvConfig()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        mode = p.GUI if render_mode == "human" else p.DIRECT
        self._client = p.connect(mode)
        if self._client < 0:
            raise RuntimeError("failed to connect to PyBullet")
        self._bike_id = -1
        self._plane_id = -1
        self._joints: dict[str, int] = {}
        self._wheel_links: set[int] = set()
        self._elapsed_substeps = 0
        self._episode_steps = 0
        self._start_x = 0.0
        self._max_progress_m = 0.0
        self._wind = SmoothGustGenerator(self.config.wind)
        self._wind_state = WindState()
        self._last_action = 1
        self._reaction_saturated = False
        self._closed = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Rebuild deterministic physics and sample seeded initial disturbances."""
        super().reset(seed=seed)
        options = options or {}
        p.resetSimulation(physicsClientId=self._client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        p.setTimeStep(1.0 / self.config.physics_hz, physicsClientId=self._client)
        p.setPhysicsEngineParameter(
            numSolverIterations=self.config.solver_iterations,
            enableFileCaching=0,
            deterministicOverlappingPairs=1,
            physicsClientId=self._client,
        )
        plane_shape = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=self._client)
        self._plane_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=plane_shape,
            physicsClientId=self._client,
        )

        roll = float(
            options.get(
                "initial_roll_rad",
                self.np_random.uniform(
                    -self.config.initial_roll_rad, self.config.initial_roll_rad
                ),
            )
        )
        roll_rate = float(
            options.get(
                "initial_roll_rate_rad_s",
                self.np_random.uniform(
                    -self.config.initial_roll_rate_rad_s,
                    self.config.initial_roll_rate_rad_s,
                ),
            )
        )
        urdf_path = files("env").joinpath("assets/bicycle.urdf")
        self._bike_id = p.loadURDF(
            str(urdf_path),
            basePosition=(0.0, 0.0, 0.77),
            baseOrientation=p.getQuaternionFromEuler((roll, 0.0, 0.0)),
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=self._client,
        )
        p.resetBaseVelocity(
            self._bike_id,
            linearVelocity=(self.config.target_speed_mps, 0.0, 0.0),
            angularVelocity=(roll_rate, 0.0, 0.0),
            physicsClientId=self._client,
        )
        self._index_joints()
        self._configure_dynamics()
        self._elapsed_substeps = 0
        self._episode_steps = 0
        self._start_x = p.getBasePositionAndOrientation(
            self._bike_id, physicsClientId=self._client
        )[0][0]
        self._max_progress_m = 0.0
        self._wind_state = self._wind.reset(self.np_random)
        self._last_action = 1
        self._reaction_saturated = False
        self._apply_motors(0.0)
        observation = self._observation()
        return observation, self._info("running", False)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply one discrete torque for 12 physics substeps and score progress."""
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action}")
        self._last_action = int(action)
        requested_torque = (int(action) - 1) * self.config.reaction_torque_nm
        for _ in range(self.config.substeps):
            torque = self._limited_reaction_torque(requested_torque)
            self._apply_motors(torque)
            time_s = self._elapsed_substeps / self.config.physics_hz
            self._wind_state = self._wind.value(time_s)
            self._apply_wind(self._wind_state.force_y_n)
            p.stepSimulation(physicsClientId=self._client)
            self._elapsed_substeps += 1

        self._episode_steps += 1
        observation = self._observation()
        position, _ = p.getBasePositionAndOrientation(
            self._bike_id, physicsClientId=self._client
        )
        progress = max(0.0, float(position[0] - self._start_x))
        capped_progress = min(progress, self.config.goal_distance_m)
        progress_delta = max(0.0, capped_progress - self._max_progress_m)
        self._max_progress_m = max(self._max_progress_m, capped_progress)
        reward = progress_delta * self.config.progress_reward_per_m

        fell = self._has_fallen()
        success = progress >= self.config.goal_distance_m and not fell
        terminated = fell or success
        truncated = not terminated and self._episode_steps >= self.config.max_episode_steps
        outcome = "running"
        if success:
            reward += self.config.success_reward
            outcome = "success"
        elif fell:
            reward += self.config.fall_penalty
            outcome = "fall"
        elif truncated:
            outcome = "timeout"
        info = self._info(outcome, success)
        if self.render_mode == "human":
            self.render()
        return observation, float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Follow the bicycle in GUI mode or return a TinyRenderer RGB frame."""
        if self.render_mode == "human":
            if self._bike_id >= 0:
                position = p.getBasePositionAndOrientation(
                    self._bike_id, physicsClientId=self._client
                )[0]
                p.resetDebugVisualizerCamera(
                    cameraDistance=4.0,
                    cameraYaw=120.0,
                    cameraPitch=-20.0,
                    cameraTargetPosition=position,
                    physicsClientId=self._client,
                )
            return None
        if self.render_mode != "rgb_array":
            return None
        position = p.getBasePositionAndOrientation(
            self._bike_id, physicsClientId=self._client
        )[0]
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=position,
            distance=4.0,
            yaw=120,
            pitch=-20,
            roll=0,
            upAxisIndex=2,
        )
        projection = p.computeProjectionMatrixFOV(60, 16 / 9, 0.1, 100)
        _, _, rgba, _, _ = p.getCameraImage(
            640,
            360,
            viewMatrix=view,
            projectionMatrix=projection,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self._client,
        )
        return np.asarray(rgba, dtype=np.uint8)[..., :3]

    def close(self) -> None:
        """Release this environment's independent PyBullet client."""
        if not self._closed and p.isConnected(self._client):
            p.disconnect(self._client)
        self._closed = True

    def _index_joints(self) -> None:
        self._joints.clear()
        for index in range(p.getNumJoints(self._bike_id, physicsClientId=self._client)):
            info = p.getJointInfo(self._bike_id, index, physicsClientId=self._client)
            self._joints[info[1].decode("utf-8")] = index
        required = {
            "rear_wheel_joint",
            "steering_joint",
            "front_wheel_joint",
            "reaction_wheel_joint",
        }
        if missing := required.difference(self._joints):
            raise RuntimeError(f"bicycle URDF is missing joints: {sorted(missing)}")
        self._wheel_links = {
            self._joints["rear_wheel_joint"],
            self._joints["front_wheel_joint"],
        }

    def _configure_dynamics(self) -> None:
        for joint in self._joints.values():
            p.setJointMotorControl2(
                self._bike_id,
                joint,
                p.VELOCITY_CONTROL,
                force=0,
                physicsClientId=self._client,
            )
        for link in self._wheel_links:
            p.changeDynamics(
                self._bike_id,
                link,
                lateralFriction=self.config.wheel_friction,
                rollingFriction=0.002,
                spinningFriction=0.002,
                restitution=0.0,
                physicsClientId=self._client,
            )
        p.changeDynamics(
            self._plane_id,
            -1,
            lateralFriction=self.config.wheel_friction,
            restitution=0.0,
            physicsClientId=self._client,
        )

    def _apply_motors(self, reaction_torque: float) -> None:
        p.setJointMotorControl2(
            self._bike_id,
            self._joints["rear_wheel_joint"],
            p.VELOCITY_CONTROL,
            targetVelocity=self.config.target_speed_mps / self.config.wheel_radius_m,
            force=self.config.drive_force_nm,
            physicsClientId=self._client,
        )
        p.setJointMotorControl2(
            self._bike_id,
            self._joints["front_wheel_joint"],
            p.VELOCITY_CONTROL,
            force=0,
            physicsClientId=self._client,
        )
        p.setJointMotorControl2(
            self._bike_id,
            self._joints["steering_joint"],
            p.POSITION_CONTROL,
            targetPosition=0.0,
            force=self.config.steering_force_nm,
            physicsClientId=self._client,
        )
        p.setJointMotorControl2(
            self._bike_id,
            self._joints["reaction_wheel_joint"],
            p.TORQUE_CONTROL,
            force=reaction_torque,
            physicsClientId=self._client,
        )

    def _limited_reaction_torque(self, requested: float) -> float:
        speed = p.getJointState(
            self._bike_id,
            self._joints["reaction_wheel_joint"],
            physicsClientId=self._client,
        )[1]
        further_into_saturation = (speed >= self.config.reaction_max_speed_rad_s and requested > 0) or (
            speed <= -self.config.reaction_max_speed_rad_s and requested < 0
        )
        self._reaction_saturated = abs(speed) >= self.config.reaction_max_speed_rad_s
        return 0.0 if further_into_saturation else requested

    def _apply_wind(self, force_y_n: float) -> None:
        if force_y_n == 0.0:
            return
        position, orientation = p.getBasePositionAndOrientation(
            self._bike_id, physicsClientId=self._client
        )
        local_height = self.config.wind.application_height_m - position[2]
        point, _ = p.multiplyTransforms(
            position,
            orientation,
            (0.0, 0.0, local_height),
            (0.0, 0.0, 0.0, 1.0),
        )
        p.applyExternalForce(
            self._bike_id,
            -1,
            forceObj=(0.0, force_y_n, 0.0),
            posObj=point,
            flags=p.WORLD_FRAME,
            physicsClientId=self._client,
        )

    def _observation(self) -> np.ndarray:
        _, orientation = p.getBasePositionAndOrientation(
            self._bike_id, physicsClientId=self._client
        )
        roll = p.getEulerFromQuaternion(orientation)[0]
        linear_world, angular_world = p.getBaseVelocity(
            self._bike_id, physicsClientId=self._client
        )
        rotation = np.asarray(p.getMatrixFromQuaternion(orientation), dtype=np.float64).reshape(3, 3)
        linear_local = rotation.T @ np.asarray(linear_world)
        angular_local = rotation.T @ np.asarray(angular_world)
        reaction_speed = p.getJointState(
            self._bike_id,
            self._joints["reaction_wheel_joint"],
            physicsClientId=self._client,
        )[1]
        return np.asarray(
            [
                math.sin(roll),
                math.cos(roll),
                np.clip(angular_local[0] / 10.0, -1.0, 1.0),
                np.clip(reaction_speed / self.config.reaction_max_speed_rad_s, -1.0, 1.0),
                np.clip(linear_local[0] / 4.0, -1.0, 1.0),
            ],
            dtype=np.float32,
        )

    def _roll(self) -> float:
        orientation = p.getBasePositionAndOrientation(
            self._bike_id, physicsClientId=self._client
        )[1]
        return float(p.getEulerFromQuaternion(orientation)[0])

    def _has_fallen(self) -> bool:
        if abs(self._roll()) >= self.config.fall_roll_rad:
            return True
        contacts = p.getContactPoints(
            bodyA=self._bike_id,
            bodyB=self._plane_id,
            physicsClientId=self._client,
        )
        return any(contact[3] not in self._wheel_links for contact in contacts)

    def _info(self, outcome: str, success: bool) -> dict[str, Any]:
        position, orientation = p.getBasePositionAndOrientation(
            self._bike_id, physicsClientId=self._client
        )
        linear_world, _ = p.getBaseVelocity(self._bike_id, physicsClientId=self._client)
        rotation = np.asarray(p.getMatrixFromQuaternion(orientation), dtype=np.float64).reshape(3, 3)
        forward_speed = float((rotation.T @ np.asarray(linear_world))[0])
        reaction_speed = float(
            p.getJointState(
                self._bike_id,
                self._joints["reaction_wheel_joint"],
                physicsClientId=self._client,
            )[1]
        )
        return {
            "outcome": outcome,
            "success": bool(success),
            "progress_m": float(self._max_progress_m),
            "roll_rad": self._roll(),
            "forward_speed_mps": forward_speed,
            "lateral_drift_m": float(position[1]),
            "reaction_wheel_speed_rad_s": reaction_speed,
            "reaction_wheel_saturated": self._reaction_saturated,
            "wind_force_y_n": self._wind_state.force_y_n,
            "wind_peak_force_n": self._wind_state.peak_force_n,
            "wind_direction": self._wind_state.direction,
            "gust_count": self._wind_state.gust_count,
            "episode_step": self._episode_steps,
        }
