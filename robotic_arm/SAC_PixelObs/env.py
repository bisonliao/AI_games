"""Three-view RGB observation adapter for the tabletop SAC tasks.

The validated ``SAC_VecObs`` task adapter remains the source of truth for
actions, rewards, phase transitions, and success/failure semantics. This
module exposes only rendered camera images and robot proprioception to the
policy; object/goal coordinates and phase flags stay inside the environment.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium import spaces

from SAC_VecObs.env import SACVectorTaskEnv


CAMERA_VIEW_NAMES = ("xy", "xz", "yz")
N_VIEWS = len(CAMERA_VIEW_NAMES)
RGB_CHANNELS = 3
# These defaults favor visual localization accuracy. They can be overridden
# from the training and evaluation CLIs for resource-constrained runs.
DEFAULT_IMAGE_SIZE = 96
DEFAULT_FRAME_STACK = 1
DEFAULT_CAMERA_SCALE = 1.0
PROPRIO_SIZE = 26

# Normalize the Cartesian end-effector position using the same controller
# workspace limits as PandaTabletopEnv._apply_action().  Values normally lie
# in [-1, 1], with a small amount of headroom retained for physics transients.
EE_WORKSPACE_LOW = np.array([0.28, -0.48, -0.08], dtype=np.float32)
EE_WORKSPACE_HIGH = np.array([0.82, 0.48, 0.72], dtype=np.float32)
EE_WORKSPACE_CENTER = (EE_WORKSPACE_LOW + EE_WORKSPACE_HIGH) * 0.5
EE_WORKSPACE_HALF_RANGE = (EE_WORKSPACE_HIGH - EE_WORKSPACE_LOW) * 0.5
EE_VELOCITY_SCALE = np.float32(1.0)  # metres per second


class PixelTaskEnv(gym.Env):
    """Pixel-observation reach or pick-place environment.

    A single observation contains three synchronized RGB projections. With
    the default single-frame stack, ``image`` has shape ``(9, H, W)``:
    1 frame x 3 views x RGB. ``frame_stack`` can be increased explicitly.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        task: str = "pick_place",
        image_size: int = DEFAULT_IMAGE_SIZE,
        frame_stack: int = DEFAULT_FRAME_STACK,
        max_episode_steps: int = 150,
        action_repeat: int = 8,
        camera_scale: float = DEFAULT_CAMERA_SCALE,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        if task not in {"reach", "pick_place"}:
            raise ValueError("task must be 'reach' or 'pick_place'")
        if image_size <= 0 or frame_stack <= 0:
            raise ValueError("image_size and frame_stack must be positive")
        if camera_scale <= 0.0:
            raise ValueError("camera_scale must be positive")
        if render_mode not in {None, "human"}:
            raise ValueError("render_mode must be None or 'human'")

        self.task = task
        self.image_size = int(image_size)
        self.frame_stack = int(frame_stack)
        self.camera_scale = float(camera_scale)
        self.render_mode = render_mode
        self.task_env = SACVectorTaskEnv(
            task=task,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            action_repeat=action_repeat,
            seed=seed,
        )
        self.action_space = self.task_env.action_space
        image_channels = self.frame_stack * N_VIEWS * RGB_CHANNELS
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(image_channels, self.image_size, self.image_size),
                    dtype=np.uint8,
                ),
                "proprio": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(PROPRIO_SIZE,),
                    dtype=np.float32,
                ),
            }
        )
        self._frames: deque[np.ndarray] = deque(maxlen=self.frame_stack)
        self._last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self._last_gripper_width = np.float32(0.08)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        del options
        super().reset(seed=seed)
        _, info = self.task_env.reset(seed=seed)
        self._last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self._last_gripper_width = self._gripper_width()
        frame = self._render_views()
        self._frames.clear()
        for _ in range(self.frame_stack):
            self._frames.append(frame.copy())
        return self._make_observation(), self._pixel_info(info)

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        _, reward, terminated, truncated, info = self.task_env.step(action)
        self._last_action = np.clip(action, -1.0, 1.0).astype(np.float32)
        frame = self._render_views()
        self._frames.append(frame)
        return (
            self._make_observation(),
            float(reward),
            bool(terminated),
            bool(truncated),
            self._pixel_info(info),
        )

    def close(self) -> None:
        self.task_env.close()

    def render(self):
        if self.render_mode == "human":
            return self.task_env.render()
        return None

    def _render_views(self) -> np.ndarray:
        """Return synchronized RGB views as ``(9, H, W)`` uint8."""

        client = self.task_env.base_env._client
        size = self.image_size
        # At the target plane, camera_scale is the visible width/height in
        # metres. The default 1.0 m crop covers the full controller workspace:
        # x=[0.02, 1.02], y=[-0.50, 0.50], z=[-0.20, 0.80].
        center = np.array([0.52, 0.0, 0.30], dtype=np.float64)
        distance = 5.0
        near = 0.05
        far = 7.0
        views = (
            (center + np.array([0.0, 0.0, distance]), center, np.array([0.0, 1.0, 0.0])),
            (center + np.array([0.0, distance, 0.0]), center, np.array([0.0, 0.0, 1.0])),
            (center + np.array([distance, 0.0, 0.0]), center, np.array([0.0, 0.0, 1.0])),
        )
        frames = []
        for index, (eye, target, up) in enumerate(views):
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=eye.tolist(),
                cameraTargetPosition=target.tolist(),
                cameraUpVector=up.tolist(),
                physicsClientId=client,
            )
            # PyBullet's computeProjectionMatrix expects frustum bounds on
            # the near plane; it is perspective, despite the generic name.
            # Passing workspace-scale bounds directly (the previous code)
            # produced an approximately 177-degree FOV and shrank the cube to
            # zero pixels. Scale the bounds by near/distance so the target
            # plane spans camera_scale metres. A distant camera makes the
            # resulting narrow-FOV projection effectively orthographic while
            # retaining TinyRenderer compatibility.
            half = self.camera_scale * 0.5 * near / distance
            projection_matrix = p.computeProjectionMatrix(
                left=-half,
                right=half,
                bottom=-half,
                top=half,
                nearVal=near,
                farVal=far,
            )
            _, _, rgba, _, _ = p.getCameraImage(
                width=size,
                height=size,
                viewMatrix=view_matrix,
                projectionMatrix=projection_matrix,
                renderer=p.ER_TINY_RENDERER,
                physicsClientId=client,
            )
            rgb = np.asarray(rgba, dtype=np.uint8).reshape(size, size, 4)[..., :3]
            # Canonicalize the +X view's horizontal direction.
            if index == 2:
                rgb = np.fliplr(rgb)
            frames.append(np.transpose(rgb, (2, 0, 1)))
        return np.concatenate(frames, axis=0)

    def _make_observation(self) -> Dict[str, np.ndarray]:
        image = np.concatenate(tuple(self._frames), axis=0).astype(np.uint8)
        return {"image": image, "proprio": self._get_proprio()}

    def _get_proprio(self) -> np.ndarray:
        raw = self.task_env.base_env._get_observation()
        q = np.asarray(raw[0:7], dtype=np.float32) / np.float32(np.pi)
        dq = np.clip(np.asarray(raw[7:14], dtype=np.float32) / 10.0, -5.0, 5.0)
        ee_state = p.getLinkState(
            self.task_env.base_env.robot_id,
            self.task_env.base_env.END_EFFECTOR_LINK,
            computeLinkVelocity=True,
            computeForwardKinematics=True,
            physicsClientId=self.task_env.base_env._client,
        )
        ee_position = np.asarray(ee_state[4], dtype=np.float32)
        ee_position_norm = np.clip(
            (ee_position - EE_WORKSPACE_CENTER) / EE_WORKSPACE_HALF_RANGE,
            -1.5,
            1.5,
        )
        ee_linear_velocity = np.clip(
            np.asarray(ee_state[6], dtype=np.float32) / EE_VELOCITY_SCALE,
            -5.0,
            5.0,
        )
        width = self._gripper_width()
        width_norm = np.float32(np.clip(width / 0.08, 0.0, 1.5))
        width_velocity = np.float32(
            np.clip((width - self._last_gripper_width) / 0.08, -2.0, 2.0)
        )
        self._last_gripper_width = width
        # Keep reach and pick-place observation shapes identical: reach has a
        # three-dimensional action, so its fourth action slot is zero.
        action = np.zeros(4, dtype=np.float32)
        action[: self._last_action.shape[0]] = self._last_action
        return np.concatenate(
            [
                q,
                dq,
                ee_position_norm,
                ee_linear_velocity,
                np.array([width_norm, width_velocity], dtype=np.float32),
                action,
            ]
        ).astype(np.float32)

    def _gripper_width(self) -> np.float32:
        return np.float32(self.task_env.base_env._get_gripper_width())

    @staticmethod
    def _pixel_info(info: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(info)
        result["camera_views"] = CAMERA_VIEW_NAMES
        return result


__all__ = [
    "CAMERA_VIEW_NAMES",
    "DEFAULT_FRAME_STACK",
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_CAMERA_SCALE",
    "N_VIEWS",
    "PROPRIO_SIZE",
    "PixelTaskEnv",
]
