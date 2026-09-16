"""Local randomized wrapper for the existing three-view PixelTaskEnv."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pybullet as p

from SAC_PixelObs.env import (
    CAMERA_VIEW_NAMES,
    DEFAULT_CAMERA_SCALE,
    DEFAULT_FRAME_STACK,
    DEFAULT_IMAGE_SIZE,
    PixelTaskEnv,
)


class RandomizedPixelTaskEnv(PixelTaskEnv):
    """PixelTaskEnv with seeded scene, pose and camera perturbations."""

    def __init__(
        self,
        *args: Any,
        camera_jitter: float = 0.03,
        object_position_jitter: float = 0.04,
        goal_position_jitter: float = 0.04,
        initial_joint_jitter: float = 0.05,
        randomize: bool = True,
        **kwargs: Any,
    ) -> None:
        self.camera_jitter = float(camera_jitter if randomize else 0.0)
        self.object_position_jitter = float(object_position_jitter if randomize else 0.0)
        self.goal_position_jitter = float(goal_position_jitter if randomize else 0.0)
        self.initial_joint_jitter = float(initial_joint_jitter if randomize else 0.0)
        self.randomize = bool(randomize)
        self.randomization_metadata: dict[str, Any] = {}
        self._camera_params: dict[str, Any] = {}
        super().__init__(*args, **kwargs)

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        rng = self.np_random
        base = self.task_env.base_env

        object_position = base.object_position.copy()
        goal_position = base.goal_position.copy()
        if self.randomize:
            object_position[:2] += rng.uniform(
                -self.object_position_jitter,
                self.object_position_jitter,
                size=2,
            )
            goal_position[:2] += rng.uniform(
                -self.goal_position_jitter,
                self.goal_position_jitter,
                size=2,
            )
            object_position[:2] = np.clip(object_position[:2], [0.38, -0.34], [0.76, 0.34])
            goal_position[:2] = np.clip(goal_position[:2], [0.38, -0.34], [0.76, 0.34])
            while np.linalg.norm(object_position[:2] - goal_position[:2]) < 0.12:
                goal_position[:2] = np.clip(
                    base._sample_position(base.object_half_extent)[:2]
                    + rng.uniform(-self.goal_position_jitter, self.goal_position_jitter, size=2),
                    [0.38, -0.34],
                    [0.76, 0.34],
                )
        base.object_position = np.asarray(object_position, dtype=np.float32)
        p.resetBasePositionAndOrientation(
            base.object_id,
            base.object_position.tolist(),
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=base._client,
        )
        base._set_goal_position(goal_position)

        joint_offsets = np.zeros(len(base.ARM_JOINTS), dtype=np.float32)
        if self.randomize and self.initial_joint_jitter > 0:
            joint_offsets = rng.uniform(
                -self.initial_joint_jitter,
                self.initial_joint_jitter,
                size=len(base.ARM_JOINTS),
            ).astype(np.float32)
            joint_positions = base._rest_pose + joint_offsets
            for joint_index, joint_position in zip(base.ARM_JOINTS, joint_positions):
                p.resetJointState(
                    base.robot_id,
                    joint_index,
                    float(joint_position),
                    targetVelocity=0.0,
                    physicsClientId=base._client,
                )
            p.setJointMotorControlArray(
                base.robot_id,
                list(base.ARM_JOINTS),
                p.POSITION_CONTROL,
                targetPositions=joint_positions.tolist(),
                forces=[200.0] * 7,
                positionGains=[0.12] * 7,
                velocityGains=[1.0] * 7,
                physicsClientId=base._client,
            )
            for _ in range(5):
                p.stepSimulation(physicsClientId=base._client)

        camera_offset = np.zeros(3, dtype=np.float64)
        camera_scale_factor = 1.0
        yaw = pitch = roll = 0.0
        if self.randomize and self.camera_jitter > 0:
            camera_offset = rng.uniform(-self.camera_jitter, self.camera_jitter, size=3)
            camera_scale_factor = float(rng.uniform(0.97, 1.03))
            yaw = float(rng.uniform(-3.0, 3.0))
            pitch = float(rng.uniform(-3.0, 3.0))
            roll = float(rng.uniform(-2.0, 2.0))
        self._camera_params = {
            "center_offset": camera_offset.tolist(),
            "camera_scale_factor": camera_scale_factor,
            "yaw_deg": yaw,
            "pitch_deg": pitch,
            "roll_deg": roll,
        }
        self.randomization_metadata = {
            "seed": seed,
            "object_position": base.object_position.tolist(),
            "goal_position": base.goal_position.tolist(),
            "initial_joint_offsets": joint_offsets.tolist(),
            "camera": dict(self._camera_params),
        }
        info = dict(info)
        info["randomization"] = dict(self.randomization_metadata)
        # Re-render after randomization so the returned frame matches the state.
        frame = self._render_views()
        self._frames.clear()
        for _ in range(self.frame_stack):
            self._frames.append(frame.copy())
        return self._make_observation(), info

    @staticmethod
    def _rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
        yaw, pitch, roll = [math.radians(value) for value in (yaw, pitch, roll)]
        cz, sz = math.cos(yaw), math.sin(yaw)
        cy, sy = math.cos(pitch), math.sin(pitch)
        cx, sx = math.cos(roll), math.sin(roll)
        rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        return rz @ ry @ rx

    def _render_views(self) -> np.ndarray:
        client = self.task_env.base_env._client
        size = self.image_size
        center = np.array([0.52, 0.0, 0.30], dtype=np.float64)
        center += np.asarray(self._camera_params.get("center_offset", [0, 0, 0]))
        distance = 5.0
        near, far = 0.05, 7.0
        rotation = self._rotation_matrix(
            self._camera_params.get("yaw_deg", 0.0),
            self._camera_params.get("pitch_deg", 0.0),
            self._camera_params.get("roll_deg", 0.0),
        )
        views = (
            (np.array([0.0, 0.0, distance]), np.array([0.0, 1.0, 0.0])),
            (np.array([0.0, distance, 0.0]), np.array([0.0, 0.0, 1.0])),
            (np.array([distance, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
        )
        frames = []
        scale = self.camera_scale * self._camera_params.get("camera_scale_factor", 1.0)
        for index, (eye_vector, up_vector) in enumerate(views):
            eye = center + rotation @ eye_vector
            up = rotation @ up_vector
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=eye.tolist(),
                cameraTargetPosition=center.tolist(),
                cameraUpVector=up.tolist(),
                physicsClientId=client,
            )
            half = scale * 0.5 * near / distance
            projection_matrix = p.computeProjectionMatrix(
                left=-half, right=half, bottom=-half, top=half,
                nearVal=near, farVal=far,
            )
            _, _, rgba, _, _ = p.getCameraImage(
                width=size, height=size, viewMatrix=view_matrix,
                projectionMatrix=projection_matrix,
                renderer=p.ER_TINY_RENDERER, physicsClientId=client,
            )
            rgb = np.asarray(rgba, dtype=np.uint8).reshape(size, size, 4)[..., :3]
            if self.task == "pick_place":
                rgb = self._enhance_goal_marker(rgb, view_matrix, projection_matrix)
            if index == 2:
                rgb = np.fliplr(rgb)
            frames.append(np.transpose(rgb, (2, 0, 1)))
        return np.concatenate(frames, axis=0)


def make_randomized_env_factory(
    *,
    task: str,
    rank: int,
    seed: int,
    config,
    randomize: bool = True,
    render_mode: str | None = None,
):
    """Return a picklable factory for learner evaluation environments."""
    def _init():
        return RandomizedPixelTaskEnv(
            task=task,
            image_size=config.image_size,
            frame_stack=config.frame_stack,
            camera_scale=config.camera_scale,
            max_episode_steps=config.max_episode_steps,
            action_repeat=config.action_repeat,
            seed=seed + rank,
            randomize=randomize,
            render_mode=render_mode,
        )
    return _init


__all__ = ["RandomizedPixelTaskEnv", "make_randomized_env_factory"]
