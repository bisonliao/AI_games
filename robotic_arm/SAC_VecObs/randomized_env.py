"""Vector-task evaluation environment with reset-time physical jitter."""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
import pybullet as p

from .env import SACVectorTaskEnv


class RandomizedVectorTaskEnv(SACVectorTaskEnv):
    """SACVectorTaskEnv with optional object, goal and initial-pose jitter.

    The reward and phase machine are inherited unchanged. Jitter is applied
    only after the normal reset has sampled the original task distribution.
    """

    def __init__(
        self,
        *args: Any,
        object_position_jitter: float = 0.04,
        goal_position_jitter: float = 0.04,
        initial_joint_jitter: float = 0.05,
        **kwargs: Any,
    ) -> None:
        self.object_position_jitter = float(object_position_jitter)
        self.goal_position_jitter = float(goal_position_jitter)
        self.initial_joint_jitter = float(initial_joint_jitter)
        self.randomization_metadata: dict[str, Any] = {}
        super().__init__(*args, **kwargs)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ):
        observation, info = super().reset(seed=seed, options=options)
        rng = self.np_random
        base = self.base_env

        object_position = base.object_position.copy()
        goal_position = base.goal_position.copy()
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

        self.randomization_metadata = {
            "seed": seed,
            "object_position": base.object_position.tolist(),
            "goal_position": base.goal_position.tolist(),
            "initial_joint_offsets": joint_offsets.tolist(),
            "object_position_jitter": self.object_position_jitter,
            "goal_position_jitter": self.goal_position_jitter,
            "initial_joint_jitter": self.initial_joint_jitter,
        }
        info = dict(info)
        info["randomization"] = dict(self.randomization_metadata)
        return self._get_observation(), info


def make_randomized_env_factory(
    *,
    task: str,
    rank: int,
    seed: int,
    max_episode_steps: int,
    action_repeat: int,
    object_position_jitter: float,
    goal_position_jitter: float,
    initial_joint_jitter: float,
    render_mode: Optional[str] = None,
):
    """Return a picklable factory for a jitter evaluation environment."""

    def _init() -> gym.Env:
        return RandomizedVectorTaskEnv(
            task=task,
            max_episode_steps=max_episode_steps,
            action_repeat=action_repeat,
            seed=seed + rank,
            object_position_jitter=object_position_jitter,
            goal_position_jitter=goal_position_jitter,
            initial_joint_jitter=initial_joint_jitter,
            render_mode=render_mode,
        )

    return _init


__all__ = ["RandomizedVectorTaskEnv", "make_randomized_env_factory"]
