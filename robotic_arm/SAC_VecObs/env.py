"""Task adapter used by the vector-observation SAC experiments.

``RobotEnv`` owns the PyBullet scene and low-level Panda control.  This module
adds the RL-facing observation, phase machine, and rewards.  Keeping this layer
separate means a future pixel-observation experiment can reuse the same
physics without silently changing the task definition.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium import spaces

from RobotEnv import PandaTabletopEnv


class PickPlaceStage(IntEnum):
    APPROACH = 0
    GRASP = 1
    TRANSPORT = 2
    PLACE = 3
    RELEASE = 4


STAGE_NAMES = tuple(stage.name.lower() for stage in PickPlaceStage)
VECTOR_OBS_SIZE = 52

# The task is intentionally a one-way state machine.  These limits are
# measured in RL environment steps (after action_repeat physics steps).  They
# are generous relative to the scripted controller, but stop a policy from
# camping forever in a phase instead of attempting the required transition.
STAGE_STEP_LIMITS = {
    PickPlaceStage.APPROACH: 50,
    PickPlaceStage.GRASP: 30,
    PickPlaceStage.TRANSPORT: 75,
    PickPlaceStage.PLACE: 100,
    PickPlaceStage.RELEASE: 20,
}
GRASP_CONFIRM_STEPS = 2
VIOLATION_GRACE_STEPS = 4
# Hysteresis is intentional: entering PLACE uses the tighter 7.5 cm target
# band, while leaving it requires a visibly larger excursion.  This absorbs
# the cube's short settling motion without permitting a phase-loop exploit.
PLACE_EXIT_DISTANCE = 0.18
RELEASE_EXIT_DISTANCE = 0.10


class SACVectorTaskEnv(gym.Env):
    """Privileged vector-observation environment for reach and pick-place.

    The action is a normalized Cartesian delta.  Reach uses three values
    ``[dx, dy, dz]`` and keeps the gripper open; pick-place uses four values
    ``[dx, dy, dz, gripper]``.

    The phase machine is part of the environment's task semantics, not the SAC
    trainer.  It provides dense, phase-appropriate rewards while one policy is
    trained over the complete episode.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        task: str = "reach",
        render_mode: Optional[str] = None,
        max_episode_steps: int = 150,
        action_repeat: int = 8,
        seed: Optional[int] = None,
    ) -> None:
        if task not in {"reach", "pick_place"}:
            raise ValueError("task must be 'reach' or 'pick_place'")
        self.task = task
        self.render_mode = render_mode
        self.max_episode_steps = int(max_episode_steps)

        self.base_env = PandaTabletopEnv(
            task=task,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            action_repeat=action_repeat,
            seed=seed,
        )
        # Reach in this experiment means reaching the red object's exact
        # position.  Freeze it so accidental contact cannot move the target.
        if task == "reach":
            p.changeDynamics(
                self.base_env.object_id,
                -1,
                mass=0.0,
                physicsClientId=self.base_env._client,
            )
            p.changeVisualShape(
                self.base_env.goal_id,
                -1,
                rgbaColor=[0.15, 0.75, 0.20, 0.0],
                physicsClientId=self.base_env._client,
            )

        action_dim = 3 if task == "reach" else 4
        self.action_space = spaces.Box(
            low=-np.ones(action_dim, dtype=np.float32),
            high=np.ones(action_dim, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(VECTOR_OBS_SIZE,),
            dtype=np.float32,
        )

        self.stage = PickPlaceStage.APPROACH
        self.ever_grasped = False
        self.ever_lifted = False
        self.grasp_bonus_given = False
        self.lift_bonus_given = False
        self.place_bonus_given = False
        self.release_bonus_given = False
        self.stable_steps = 0
        self._stage_steps = 0
        self._contact_steps = 0
        self._violation_steps = 0
        self._failure_reason = ""
        self._last_reward_terms: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        del options
        super().reset(seed=seed)
        raw_obs, _ = self.base_env.reset(seed=seed)
        del raw_obs
        if self.task == "reach":
            # Reach has no independent placement goal.  Keep the otherwise
            # shared goal fields equal to the object position instead of
            # exposing an irrelevant hidden random target to the policy.
            self.base_env._set_goal_position(self.base_env.object_position)
        self.stage = PickPlaceStage.APPROACH
        self.ever_grasped = False
        self.ever_lifted = False
        self.grasp_bonus_given = False
        self.lift_bonus_given = False
        self.place_bonus_given = False
        self.release_bonus_given = False
        self.stable_steps = 0
        self._stage_steps = 0
        self._contact_steps = 0
        self._violation_steps = 0
        self._failure_reason = ""
        self._last_reward_terms = {}
        observation = self._get_observation()
        info = self._make_info(success=False, failure=False)
        return observation, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(f"expected action shape {self.action_space.shape}, got {action.shape}")
        action = np.clip(action, self.action_space.low, self.action_space.high)

        previous = self._metrics()
        if self.task == "reach":
            base_action = np.concatenate([action, np.array([1.0], dtype=np.float32)])
        else:
            base_action = action
        # The base environment has a looser pick-place success condition used
        # by its scripted demo.  Do not inherit its terminated/truncated flags:
        # this adapter owns both the stricter RL success semantics and horizon.
        self.base_env.step(base_action)

        current = self._metrics()
        if self.task == "reach":
            reward, success, failure = self._reach_reward(previous, current, action)
        else:
            reward, success, failure = self._pick_place_reward(previous, current, action)
        terminated = bool(success or failure)
        time_limit_reached = self.base_env.step_count >= self.max_episode_steps
        truncated = bool(time_limit_reached and not terminated)
        observation = self._get_observation()
        info = self._make_info(success=success, failure=failure)
        info["time_limit_reached"] = bool(time_limit_reached)
        return observation, float(reward), terminated, truncated, info

    def close(self) -> None:
        self.base_env.close()

    def render(self):
        return self.base_env.render()

    # ------------------------------------------------------------------
    # State and phase detection
    # ------------------------------------------------------------------
    def _metrics(self) -> Dict[str, Any]:
        ee_position = self.base_env._get_ee_position()
        object_position = self.base_env.object_position.copy()
        goal_position = self.base_env.goal_position.copy()
        linear_velocity, angular_velocity = p.getBaseVelocity(
            self.base_env.object_id,
            physicsClientId=self.base_env._client,
        )
        ee_object_distance = float(np.linalg.norm(ee_position - object_position))
        object_goal_distance = float(np.linalg.norm(object_position - goal_position))
        object_goal_xy_distance = float(np.linalg.norm(object_position[:2] - goal_position[:2]))
        return {
            "ee_position": ee_position,
            "object_position": object_position,
            "goal_position": goal_position,
            "linear_velocity": np.asarray(linear_velocity, dtype=np.float32),
            "angular_velocity": np.asarray(angular_velocity, dtype=np.float32),
            "ee_object_distance": ee_object_distance,
            "object_goal_distance": object_goal_distance,
            "object_goal_xy_distance": object_goal_xy_distance,
            "gripper_width": self.base_env._get_gripper_width(),
            "finger_contact": self._finger_contact(),
            "lifted": bool(
                object_position[2] > self.base_env.object_half_extent + 0.035
            ),
        }

    def _finger_contact(self) -> bool:
        client = self.base_env._client
        left_contacts = p.getContactPoints(
            bodyA=self.base_env.robot_id,
            bodyB=self.base_env.object_id,
            linkIndexA=9,
            physicsClientId=client,
        )
        right_contacts = p.getContactPoints(
            bodyA=self.base_env.robot_id,
            bodyB=self.base_env.object_id,
            linkIndexA=10,
            physicsClientId=client,
        )
        # Requiring both fingers prevents a single accidental bump from being
        # reported as a successful grasp.  A small width margin accommodates
        # the cube stopping the fingers before their zero-width target.
        return bool(left_contacts and right_contacts and self.base_env._get_gripper_width() < 0.078)

    def _enter_stage(self, stage: PickPlaceStage) -> None:
        """Advance to a later phase and reset phase-local debounce state."""

        if stage <= self.stage:
            raise RuntimeError(f"pick-place stages must advance monotonically: {self.stage} -> {stage}")
        self.stage = stage
        self._stage_steps = 0
        self._contact_steps = 0
        self._violation_steps = 0

    def _update_stage(self, metrics: Dict[str, Any]) -> str:
        """Advance the one-way phase machine and return a failure reason.

        Short contact/position glitches are debounced.  Once a grasp has been
        confirmed, however, a sustained regression is terminal rather than a
        transition back to an earlier phase.  This prevents policies from
        repeatedly collecting the approach reward.
        """

        if self.task == "reach":
            self.stage = PickPlaceStage.APPROACH
            return ""

        contact_grasp = bool(metrics["finger_contact"])
        lifted = bool(metrics["lifted"])
        near_goal = bool(metrics["object_goal_xy_distance"] < 0.075)
        released = bool(metrics["gripper_width"] > 0.065)

        if self.stage == PickPlaceStage.APPROACH:
            if contact_grasp:
                self._contact_steps += 1
            else:
                self._contact_steps = 0
            if self._contact_steps >= GRASP_CONFIRM_STEPS:
                self._enter_stage(PickPlaceStage.GRASP)
                self.ever_grasped = True
            return ""

        if self.stage == PickPlaceStage.GRASP:
            if lifted and contact_grasp:
                self._enter_stage(PickPlaceStage.TRANSPORT)
                self.ever_lifted = True
            elif contact_grasp:
                self._violation_steps = 0
            else:
                self._violation_steps += 1
                if self._violation_steps >= VIOLATION_GRACE_STEPS:
                    return "grasp_lost"
            return ""

        if self.stage == PickPlaceStage.TRANSPORT:
            if near_goal and lifted:
                self._enter_stage(PickPlaceStage.PLACE)
                return ""
            if lifted and contact_grasp:
                self._violation_steps = 0
            else:
                self._violation_steps += 1
                if self._violation_steps >= VIOLATION_GRACE_STEPS:
                    return "object_dropped"
            return ""

        if self.stage == PickPlaceStage.PLACE:
            # Small excursions around the near-goal threshold are tolerated,
            # but PLACE never transitions back to TRANSPORT.
            if metrics["object_goal_xy_distance"] > PLACE_EXIT_DISTANCE:
                self._violation_steps += 1
                if self._violation_steps >= VIOLATION_GRACE_STEPS:
                    return "object_left_goal"
            else:
                self._violation_steps = 0
            if near_goal and released and not lifted:
                self._enter_stage(PickPlaceStage.RELEASE)
            return ""

        # RELEASE is also monotonic.  Re-grasping or letting the object roll
        # clearly outside the target is a failed release, not another attempt.
        release_invalid = bool(
            not released
            or metrics["object_goal_xy_distance"] > RELEASE_EXIT_DISTANCE
        )
        if release_invalid:
            self._violation_steps += 1
            if self._violation_steps >= VIOLATION_GRACE_STEPS:
                return "release_regressed"
        else:
            self._violation_steps = 0
        return ""

    def _settled_at_goal(self, metrics: Dict[str, Any]) -> bool:
        return bool(
            metrics["object_goal_xy_distance"] < 0.06
            and abs(float(metrics["object_position"][2]) - self.base_env.object_half_extent) < 0.025
            and float(np.linalg.norm(metrics["linear_velocity"])) < 0.08
            and float(np.linalg.norm(metrics["angular_velocity"])) < 0.8
            and metrics["gripper_width"] > 0.065
        )

    # ------------------------------------------------------------------
    # Reward functions
    # ------------------------------------------------------------------
    def _reach_reward(
        self,
        previous: Dict[str, Any],
        current: Dict[str, Any],
        action: np.ndarray,
    ) -> Tuple[float, bool, bool]:
        distance_before = previous["ee_object_distance"]
        distance_after = current["ee_object_distance"]
        success = bool(distance_after < 0.045)
        reward_terms = {
            "progress": 2.0 * (distance_before - distance_after),
            "distance": -0.1 * distance_after,
            "time": -0.01,
            "action": -0.001 * float(np.sum(action * action)),
            "success": 2.0 if success else 0.0,
        }
        self._last_reward_terms = reward_terms
        return float(sum(reward_terms.values())), success, False

    def _pick_place_reward(
        self,
        previous: Dict[str, Any],
        current: Dict[str, Any],
        action: np.ndarray,
    ) -> Tuple[float, bool, bool]:
        self._stage_steps += 1
        failure_reason = self._update_stage(current)
        if not failure_reason and self._stage_steps >= STAGE_STEP_LIMITS[self.stage]:
            failure_reason = f"{STAGE_NAMES[int(self.stage)]}_timeout"
        self._failure_reason = failure_reason

        drop_failure = failure_reason in {"object_dropped", "object_left_goal"}
        other_failure = bool(failure_reason and not drop_failure)

        reward_terms = {
            "time": -0.01,
            "action": -0.001 * float(np.sum(action * action)),
            "progress": 0.0,
            "event": 0.0,
            "drop": -5.0 if drop_failure else 0.0,
            "failure": -5.0 if other_failure else 0.0,
        }
        if self.stage == PickPlaceStage.APPROACH:
            reward_terms["progress"] = 2.0 * (
                previous["ee_object_distance"] - current["ee_object_distance"]
            )
        elif self.stage == PickPlaceStage.GRASP:
            # Signed height progress makes an up/down oscillation sum to zero;
            # clipping negative deltas was an exploitable reward pump.
            reward_terms["progress"] = 4.0 * float(
                current["object_position"][2] - previous["object_position"][2]
            )
        elif self.stage == PickPlaceStage.TRANSPORT:
            reward_terms["progress"] = 2.0 * (
                previous["object_goal_xy_distance"] - current["object_goal_xy_distance"]
            )
        elif self.stage == PickPlaceStage.PLACE:
            previous_height_error = abs(
                float(previous["object_position"][2]) - self.base_env.object_half_extent
            )
            current_height_error = abs(
                float(current["object_position"][2]) - self.base_env.object_half_extent
            )
            reward_terms["progress"] = 2.0 * (previous_height_error - current_height_error)

        if self.ever_grasped and not self.grasp_bonus_given:
            reward_terms["event"] += 1.0
            self.grasp_bonus_given = True
        if self.ever_lifted and not self.lift_bonus_given:
            reward_terms["event"] += 2.0
            self.lift_bonus_given = True
        if self.stage in {PickPlaceStage.PLACE, PickPlaceStage.RELEASE} and not self.place_bonus_given:
            reward_terms["event"] += 1.0
            self.place_bonus_given = True
        if self.stage == PickPlaceStage.RELEASE and not self.release_bonus_given:
            reward_terms["event"] += 1.0
            self.release_bonus_given = True

        if self.stage == PickPlaceStage.RELEASE and self._settled_at_goal(current):
            self.stable_steps += 1
        else:
            self.stable_steps = 0
        success = self.stable_steps >= 4
        if success:
            reward_terms["event"] += 10.0
        self._last_reward_terms = reward_terms
        return float(sum(reward_terms.values())), success, bool(failure_reason)

    # ------------------------------------------------------------------
    # Vector observation and diagnostics
    # ------------------------------------------------------------------
    def _get_observation(self) -> np.ndarray:
        base_observation = self.base_env._get_observation()
        metrics = self._metrics()
        stage_one_hot = np.zeros(len(PickPlaceStage), dtype=np.float32)
        stage_one_hot[int(self.stage)] = 1.0
        extra = np.concatenate(
            [
                np.asarray(metrics["ee_position"] - metrics["object_position"], dtype=np.float32),
                np.asarray(metrics["object_position"] - metrics["goal_position"], dtype=np.float32),
                metrics["linear_velocity"],
                metrics["angular_velocity"],
                stage_one_hot,
                np.array(
                    [
                        float(metrics["finger_contact"]),
                        float(self.ever_grasped),
                        float(self.ever_lifted),
                        min(1.0, self.stable_steps / 4.0),
                        min(
                            1.0,
                            self._stage_steps / STAGE_STEP_LIMITS[self.stage],
                        ),
                        min(1.0, self._contact_steps / GRASP_CONFIRM_STEPS),
                        min(1.0, self._violation_steps / VIOLATION_GRACE_STEPS),
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        observation = np.concatenate([base_observation, extra]).astype(np.float32)
        if observation.shape != self.observation_space.shape:
            raise RuntimeError(f"internal observation has shape {observation.shape}")
        return observation

    def _make_info(self, success: bool, failure: bool) -> Dict[str, Any]:
        return {
            "task": self.task,
            "success": bool(success),
            # Stable-Baselines3's evaluation helper looks for this conventional
            # key when reporting success rate.
            "is_success": bool(success),
            "failure": bool(failure),
            "stage": STAGE_NAMES[int(self.stage)],
            "stage_index": int(self.stage),
            "is_grasped": bool(self._metrics()["finger_contact"]),
            "ever_grasped": bool(self.ever_grasped),
            "ever_lifted": bool(self.ever_lifted),
            "stable_steps": int(self.stable_steps),
            "stage_steps": int(self._stage_steps),
            "failure_reason": self._failure_reason,
            "reward_terms": dict(self._last_reward_terms),
        }


__all__ = [
    "GRASP_CONFIRM_STEPS",
    "PickPlaceStage",
    "SACVectorTaskEnv",
    "STAGE_NAMES",
    "STAGE_STEP_LIMITS",
    "VIOLATION_GRACE_STEPS",
]
