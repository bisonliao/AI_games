"""A minimal Panda tabletop manipulation environment.

The environment deliberately starts with privileged state observations.  It is
small enough to debug with a scripted controller before adding an RL algorithm
or camera observations.

Actions are normalized Cartesian end-effector deltas and a gripper command::

    action = [dx, dy, dz, gripper]

``gripper=-1`` closes the gripper and ``gripper=1`` opens it.  The Cartesian
action is converted to joint position targets through PyBullet IK.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pybullet as p
import pybullet_data

try:  # Gymnasium is optional at import time, but is present in the dev env.
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - useful for a simulator-only install
    import gym
    from gym import spaces


class PandaTabletopEnv(gym.Env):
    """Panda reach or single-object tabletop pick-and-place task.

    Parameters
    ----------
    task:
        ``"reach"`` trains the end effector to reach a target marker.
        ``"pick_place"`` trains a Panda to move a small cube to the marker.
    render_mode:
        ``None`` uses headless DIRECT mode, ``"human"`` opens a GUI, and
        ``"rgb_array"`` makes :meth:`render` return an RGB camera image.
    max_episode_steps:
        Truncation horizon for one episode.
    action_repeat:
        Number of fixed PyBullet simulation steps for each environment action.
    seed:
        Optional initial random seed.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    ARM_JOINTS = tuple(range(7))
    FINGER_JOINTS = (9, 10)
    END_EFFECTOR_LINK = 11

    def __init__(
        self,
        task: str = "pick_place",
        render_mode: Optional[str] = None,
        max_episode_steps: int = 150,
        action_repeat: int = 8,
        seed: Optional[int] = None,
    ) -> None:
        if task not in {"reach", "pick_place"}:
            raise ValueError("task must be 'reach' or 'pick_place'")
        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError("render_mode must be None, 'human', or 'rgb_array'")
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if action_repeat <= 0:
            raise ValueError("action_repeat must be positive")

        self.task = task
        self.render_mode = render_mode
        self.max_episode_steps = int(max_episode_steps)
        self.action_repeat = int(action_repeat)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

        self.action_space = spaces.Box(
            low=np.full(4, -1.0, dtype=np.float32),
            high=np.full(4, 1.0, dtype=np.float32),
            dtype=np.float32,
        )
        # q(7), dq(7), ee position(3), object position(3), object
        # quaternion(4), goal position(3), gripper width(1).
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(28,),
            dtype=np.float32,
        )

        self._client = p.connect(p.GUI if render_mode == "human" else p.DIRECT)
        if self._client < 0:
            raise RuntimeError("Could not connect to PyBullet")

        self.object_id: Optional[int] = None
        self.goal_id: Optional[int] = None
        self._configure_simulation()
        self._load_scene()

        self.step_count = 0
        self.object_position = np.zeros(3, dtype=np.float32)
        self.goal_position = np.zeros(3, dtype=np.float32)
        self._heuristic_phase = 0
        self._heuristic_phase_steps = 0

        # A reset is needed before the first observation is meaningful.
        self.reset(seed=seed)

    # ------------------------------------------------------------------
    # PyBullet setup and scene management
    # ------------------------------------------------------------------
    def _configure_simulation(self) -> None:
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client)
        p.resetSimulation(physicsClientId=self._client)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self._client)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self._client)
        p.setPhysicsEngineParameter(
            fixedTimeStep=1.0 / 240.0,
            numSolverIterations=80,
            numSubSteps=1,
            physicsClientId=self._client,
        )

    def _load_scene(self) -> None:
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self._client)
        # The standard Bullet table has its top surface close to z=0.
        self.table_id = p.loadURDF(
            "table/table.urdf",
            basePosition=[0.5, 0.0, -0.65],
            useFixedBase=True,
            physicsClientId=self._client,
        )
        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0.0, 0.0, 0.0],
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=self._client,
        )

        self._joint_lower = np.array(
            [p.getJointInfo(self.robot_id, i, physicsClientId=self._client)[8] for i in self.ARM_JOINTS],
            dtype=np.float64,
        )
        self._joint_upper = np.array(
            [p.getJointInfo(self.robot_id, i, physicsClientId=self._client)[9] for i in self.ARM_JOINTS],
            dtype=np.float64,
        )
        # Some URDFs expose an unlimited joint with upper <= lower.  The Panda
        # model does not, but using safe ranges makes this wrapper less brittle.
        invalid = self._joint_upper <= self._joint_lower
        self._joint_lower[invalid] = -math.pi
        self._joint_upper[invalid] = math.pi
        self._joint_ranges = self._joint_upper - self._joint_lower
        self._rest_pose = np.array(
            [0.0, -0.40, 0.0, -2.0, 0.0, 1.60, 0.80], dtype=np.float64
        )

        # Cube dimensions are intentionally modest so that the default Panda
        # gripper can envelop the object without requiring custom meshes.
        self.object_half_extent = 0.025
        object_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[self.object_half_extent] * 3,
            rgbaColor=[0.85, 0.15, 0.10, 1.0],
            physicsClientId=self._client,
        )
        object_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[self.object_half_extent] * 3,
            physicsClientId=self._client,
        )
        self.object_id = p.createMultiBody(
            baseMass=0.08,
            baseCollisionShapeIndex=object_collision,
            baseVisualShapeIndex=object_visual,
            basePosition=[0.55, 0.0, self.object_half_extent],
            physicsClientId=self._client,
        )
        p.changeDynamics(
            self.object_id,
            -1,
            lateralFriction=0.8,
            spinningFriction=0.2,
            rollingFriction=0.1,
            restitution=0.0,
            physicsClientId=self._client,
        )

        goal_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.045, 0.045, 0.002],
            rgbaColor=[0.15, 0.75, 0.20, 0.75],
            physicsClientId=self._client,
        )
        self.goal_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=goal_visual,
            basePosition=[0.70, 0.20, 0.002],
            physicsClientId=self._client,
        )

        self._reset_robot()

    def _reset_robot(self) -> None:
        for joint_index, joint_position in zip(self.ARM_JOINTS, self._rest_pose):
            p.resetJointState(
                self.robot_id,
                joint_index,
                float(joint_position),
                targetVelocity=0.0,
                physicsClientId=self._client,
            )
        # Reset motor targets as well as joint states.  Without this, a second
        # reset after a previous action would continue driving toward stale IK
        # targets while the settling steps run, breaking seeded determinism.
        p.setJointMotorControlArray(
            self.robot_id,
            list(self.ARM_JOINTS),
            p.POSITION_CONTROL,
            targetPositions=self._rest_pose.tolist(),
            forces=[200.0] * 7,
            positionGains=[0.12] * 7,
            velocityGains=[1.0] * 7,
            physicsClientId=self._client,
        )
        for joint_index in self.FINGER_JOINTS:
            p.resetJointState(self.robot_id, joint_index, 0.04, physicsClientId=self._client)
        self._command_gripper(1.0)

    def _sample_position(self, z: float) -> np.ndarray:
        return np.array(
            [
                self.np_random.uniform(0.42, 0.72),
                self.np_random.uniform(-0.28, 0.28),
                z,
            ],
            dtype=np.float32,
        )

    def _set_goal_position(self, position: np.ndarray) -> None:
        self.goal_position = np.asarray(position, dtype=np.float32).copy()
        p.resetBasePositionAndOrientation(
            self.goal_id,
            self.goal_position.tolist(),
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=self._client,
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        del options  # Reserved for future task configuration.
        super().reset(seed=seed)
        self.step_count = 0
        self._heuristic_phase = 0
        self._heuristic_phase_steps = 0
        self._reset_robot()

        self.object_position = self._sample_position(self.object_half_extent)
        self._set_goal_position(self._sample_position(self.object_half_extent))
        while np.linalg.norm(self.object_position[:2] - self.goal_position[:2]) < 0.12:
            self._set_goal_position(self._sample_position(self.object_half_extent))

        p.resetBasePositionAndOrientation(
            self.object_id,
            self.object_position.tolist(),
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=self._client,
        )
        p.resetBaseVelocity(self.object_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], physicsClientId=self._client)

        # Let the arm settle at its initial configuration before observations.
        for _ in range(10):
            p.stepSimulation(physicsClientId=self._client)
        return self._get_observation(), {"success": False, "task": self.task}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (4,):
            raise ValueError(f"expected action shape (4,), got {action.shape}")
        action = np.clip(action, self.action_space.low, self.action_space.high)
        for _ in range(self.action_repeat):
            self._apply_action(action)
            p.stepSimulation(physicsClientId=self._client)

        self.step_count += 1
        observation = self._get_observation()
        reward, success, failure = self._compute_reward()
        terminated = bool(success or failure)
        truncated = bool(self.step_count >= self.max_episode_steps and not terminated)
        info = {
            "success": bool(success),
            "failure": bool(failure),
            "is_grasped": bool(self._is_grasped()),
            "object_position": self.object_position.copy(),
            "goal_position": self.goal_position.copy(),
            "step": self.step_count,
        }
        return observation, float(reward), terminated, truncated, info

    def close(self) -> None:
        if getattr(self, "_client", -1) >= 0:
            p.disconnect(self._client)
            self._client = -1

    # ------------------------------------------------------------------
    # Control and observations
    # ------------------------------------------------------------------
    def _apply_action(self, action: np.ndarray) -> None:
        link_state = p.getLinkState(
            self.robot_id,
            self.END_EFFECTOR_LINK,
            computeForwardKinematics=True,
            physicsClientId=self._client,
        )
        current_position = np.asarray(link_state[4], dtype=np.float64)
        target_position = current_position + np.asarray(action[:3], dtype=np.float64) * 0.035
        target_position = np.clip(
            target_position,
            # Link 11 is below the Panda finger joints in this URDF.  Allowing
            # its target slightly below the tabletop places the actual finger
            # tips at the cube height instead of stopping them too high.
            np.array([0.28, -0.48, -0.08]),
            np.array([0.82, 0.48, 0.72]),
        )
        target_orientation = p.getQuaternionFromEuler([0.0, math.pi, 0.0])
        ik_solution = p.calculateInverseKinematics(
            self.robot_id,
            self.END_EFFECTOR_LINK,
            target_position.tolist(),
            targetOrientation=target_orientation,
            lowerLimits=self._joint_lower.tolist(),
            upperLimits=self._joint_upper.tolist(),
            jointRanges=self._joint_ranges.tolist(),
            restPoses=self._rest_pose.tolist(),
            maxNumIterations=50,
            residualThreshold=1e-4,
            physicsClientId=self._client,
        )
        arm_targets = np.asarray(ik_solution[:7], dtype=np.float64)
        arm_targets = np.clip(arm_targets, self._joint_lower, self._joint_upper)
        p.setJointMotorControlArray(
            self.robot_id,
            list(self.ARM_JOINTS),
            p.POSITION_CONTROL,
            targetPositions=arm_targets.tolist(),
            forces=[200.0] * 7,
            positionGains=[0.12] * 7,
            velocityGains=[1.0] * 7,
            physicsClientId=self._client,
        )
        self._command_gripper(float(action[3]))

    def _command_gripper(self, command: float) -> None:
        # Each Panda finger joint has a roughly [0, 0.04] range.  The exposed
        # command uses [-1, 1] so policies have a consistent normalized space.
        finger_target = 0.04 * (float(np.clip(command, -1.0, 1.0)) + 1.0) / 2.0
        p.setJointMotorControlArray(
            self.robot_id,
            list(self.FINGER_JOINTS),
            p.POSITION_CONTROL,
            targetPositions=[finger_target, finger_target],
            forces=[40.0, 40.0],
            positionGains=[0.3, 0.3],
            velocityGains=[1.0, 1.0],
            physicsClientId=self._client,
        )

    def _get_joint_state(self) -> Tuple[np.ndarray, np.ndarray]:
        states = p.getJointStates(self.robot_id, list(self.ARM_JOINTS), physicsClientId=self._client)
        positions = np.array([state[0] for state in states], dtype=np.float32)
        velocities = np.array([state[1] for state in states], dtype=np.float32)
        return positions, velocities

    def _get_ee_position(self) -> np.ndarray:
        state = p.getLinkState(
            self.robot_id,
            self.END_EFFECTOR_LINK,
            computeForwardKinematics=True,
            physicsClientId=self._client,
        )
        return np.asarray(state[4], dtype=np.float32)

    def _get_gripper_width(self) -> float:
        states = p.getJointStates(self.robot_id, list(self.FINGER_JOINTS), physicsClientId=self._client)
        return float(states[0][0] + states[1][0])

    def _get_observation(self) -> np.ndarray:
        joint_positions, joint_velocities = self._get_joint_state()
        ee_position = self._get_ee_position()
        object_position, object_orientation = p.getBasePositionAndOrientation(
            self.object_id, physicsClientId=self._client
        )
        self.object_position = np.asarray(object_position, dtype=np.float32)
        object_orientation = np.asarray(object_orientation, dtype=np.float32)
        values = np.concatenate(
            [
                joint_positions,
                joint_velocities,
                ee_position,
                self.object_position,
                object_orientation,
                self.goal_position,
                np.array([self._get_gripper_width()], dtype=np.float32),
            ]
        ).astype(np.float32)
        if values.shape != self.observation_space.shape:
            raise RuntimeError(f"internal observation has shape {values.shape}")
        return values

    # ------------------------------------------------------------------
    # Reward, success and a simple scripted baseline
    # ------------------------------------------------------------------
    def _is_grasped(self) -> bool:
        if self.task != "pick_place":
            return False
        ee_position = self._get_ee_position()
        object_position = self.object_position
        close_enough = np.linalg.norm(ee_position - object_position) < 0.075
        lifted = object_position[2] > self.object_half_extent + 0.035
        contacts = p.getContactPoints(
            bodyA=self.robot_id,
            bodyB=self.object_id,
            physicsClientId=self._client,
        )
        return bool(close_enough and lifted and len(contacts) > 0)

    def _compute_reward(self) -> Tuple[float, bool, bool]:
        ee_position = self._get_ee_position()
        ee_goal_distance = float(np.linalg.norm(ee_position - self.goal_position))
        if self.task == "reach":
            success = ee_goal_distance < 0.045
            reward = -ee_goal_distance + (2.0 if success else 0.0)
            return reward, success, False

        object_goal_distance = float(np.linalg.norm(self.object_position - self.goal_position))
        ee_object_distance = float(np.linalg.norm(ee_position - self.object_position))
        lifted = self.object_position[2] > self.object_half_extent + 0.035
        success = bool(object_goal_distance < 0.065 and not lifted)
        # Dense shaping makes the early approach phase learnable, while the
        # lift bonus distinguishes a genuine grasp from merely touching a cube.
        reward = -0.35 * ee_object_distance - 0.65 * object_goal_distance
        if lifted:
            reward += 0.35
        if self._is_grasped():
            reward += 0.15
        if success:
            reward += 5.0
        failure = bool(self.object_position[2] < -0.10)
        if failure:
            reward -= 2.0
        return reward, success, failure

    def _point_action(self, target: np.ndarray, gripper: float) -> np.ndarray:
        current = self._get_ee_position()
        delta = (np.asarray(target, dtype=np.float32) - current) / 0.035
        return np.concatenate([np.clip(delta, -1.0, 1.0), [np.clip(gripper, -1.0, 1.0)]]).astype(np.float32)

    def heuristic_action(self) -> np.ndarray:
        """Return one action from a simple scripted reach/pick-place policy.

        This is intended for smoke tests and debugging, not as a replacement
        for a learned policy.  It is useful for checking that contact and
        reward logic work before starting RL training.
        """
        if self.task == "reach":
            return self._point_action(self.goal_position, gripper=1.0)

        object_position = self.object_position.copy()
        goal_position = self.goal_position.copy()
        phase_targets = [
            (object_position + np.array([0.0, 0.0, 0.12]), 1.0),
            (object_position + np.array([0.0, 0.0, -0.045]), 1.0),
            (object_position + np.array([0.0, 0.0, -0.045]), -1.0),
            (object_position + np.array([0.0, 0.0, 0.20]), -1.0),
            (goal_position + np.array([0.0, 0.0, 0.20]), -1.0),
            (goal_position + np.array([0.0, 0.0, 0.075]), -1.0),
            (goal_position + np.array([0.0, 0.0, 0.075]), 1.0),
        ]
        target, gripper = phase_targets[min(self._heuristic_phase, len(phase_targets) - 1)]
        action = self._point_action(target, gripper)
        distance = float(np.linalg.norm(self._get_ee_position() - target))
        self._heuristic_phase_steps += 1
        if distance < 0.035 or self._heuristic_phase_steps > 35:
            self._heuristic_phase = min(self._heuristic_phase + 1, len(phase_targets) - 1)
            self._heuristic_phase_steps = 0
        return action

    # ------------------------------------------------------------------
    # Optional camera rendering
    # ------------------------------------------------------------------
    def render(self) -> Optional[np.ndarray]:
        if self.render_mode != "rgb_array":
            return None
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.52, 0.0, 0.12],
            distance=1.05,
            yaw=50.0,
            pitch=-45.0,
            roll=0.0,
            upAxisIndex=2,
            physicsClientId=self._client,
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60.0,
            aspect=1.0,
            nearVal=0.01,
            farVal=2.0,
        )
        width, height, rgba, _, _ = p.getCameraImage(
            width=256,
            height=256,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self._client,
        )
        del width, height
        return np.asarray(rgba, dtype=np.uint8)[:, :, :3]


__all__ = ["PandaTabletopEnv"]
