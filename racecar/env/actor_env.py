"""Synchronous actor-local vector wrapper for :class:`RaceCarEnv`.

Each actor owns several independent PyBullet clients and steps them directly,
which avoids a multiprocessing queue round trip for every environment step.
The actor--learner queues are managed by ``ppo/actor.py`` and ``ppo/train.py``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .RaceCarEnv import DEFAULT_MAX_STEPS, RaceCarEnv


class DirectRaceCarVectorEnv:
    """多个独立 PyBullet client，由当前 actor 进程直接同步驱动。"""

    def __init__(self, num_envs: int, *, render: bool = False, fps: int = 20,
                 max_steps: int = DEFAULT_MAX_STEPS, start_position_noise=0.0,
                 start_heading_noise=0.0):
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.num_envs = int(num_envs)
        self._envs = [
            RaceCarEnv(writer=None, render=render, fps=fps, max_steps=max_steps,
                       start_position_noise=start_position_noise,
                       start_heading_noise=start_heading_noise)
            for _ in range(self.num_envs)
        ]
        self.observation_space = self._envs[0].observation_space
        self.action_space = self._envs[0].action_space

    def reset(self, seeds: Optional[Sequence[Optional[int]]] = None):
        if seeds is None:
            seeds = [None] * self.num_envs
        if len(seeds) != self.num_envs:
            raise ValueError("seeds must have one entry per environment")
        results = [env.reset(seed=seed) for env, seed in zip(self._envs, seeds)]
        observations, infos = zip(*results)
        return np.stack(observations), list(infos)

    def reset_at(self, indices: Sequence[int], seeds=None):
        unique_indices = list(dict.fromkeys(int(index) for index in indices))
        if seeds is None:
            seeds = [None] * len(unique_indices)
        if len(seeds) != len(unique_indices):
            raise ValueError("seeds must have one entry per reset index")
        results = {}
        for index, seed in zip(unique_indices, seeds):
            if not 0 <= index < self.num_envs:
                raise IndexError(f"environment index out of range: {index}")
            results[index] = self._envs[index].reset(seed=seed)
        return results

    def step(self, actions):
        actions = np.asarray(actions)
        if len(actions) != self.num_envs:
            raise ValueError("actions must have one entry per environment")
        results = [env.step(action) for env, action in zip(self._envs, actions)]
        observations, rewards, terminated, truncated, infos = zip(*results)
        return (np.stack(observations), np.asarray(rewards, dtype=np.float32),
                np.asarray(terminated, dtype=bool), np.asarray(truncated, dtype=bool),
                list(infos))

    def close(self):
        for env in self._envs:
            env.close()
        self._envs.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
