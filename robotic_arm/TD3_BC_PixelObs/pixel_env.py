"""Evaluation environment adapter.

The validated project PixelTaskEnv already owns the camera projection and
phase/reward semantics. This thin local adapter fixes the configuration and
keeps the TD3 package independent from SAC training code.
"""
from __future__ import annotations

from .config import EnvConfig


def make_env(config: EnvConfig, seed: int, randomize: bool = False, gui: bool = False):
    from SAC_BC_PixelObs.randomized_env import RandomizedPixelTaskEnv

    return RandomizedPixelTaskEnv(
        task=config.task,
        image_size=config.image_size,
        frame_stack=config.frame_stack,
        camera_scale=config.camera_scale,
        max_episode_steps=config.max_episode_steps,
        action_repeat=config.action_repeat,
        seed=seed,
        randomize=randomize,
        render_mode="human" if gui else None,
    )
