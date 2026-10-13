import multiprocessing as mp
from dataclasses import replace

import numpy as np

from MahjongEnv.config import EnvConfig
from MahjongEnv.env import MahjongEnv


def _worker(remote, parent_remote, config, seed, rank):
    parent_remote.close()
    env = MahjongEnv(replace(config, observation_mode='vector', render_mode=None), seed=seed)
    episode_index = 0

    def reset_env():
        nonlocal episode_index
        episode_seed = None if seed is None else int(seed) + episode_index * 1009
        episode_index += 1
        return env.reset(seed=episode_seed)

    try:
        obs, info = reset_env()
        while info.get('terminal_reason') is not None:
            obs, info = reset_env()
        while True:
            command, payload = remote.recv()
            if command == 'reset':
                if payload is not None:
                    episode_index = int(payload)
                obs, info = reset_env()
                while info.get('terminal_reason') is not None:
                    obs, info = reset_env()
                remote.send((obs, info))
            elif command == 'step':
                next_obs, reward, terminated, truncated, step_info = env.step(int(payload))
                if terminated or truncated:
                    final_obs = next_obs
                    final_info = step_info
                    next_obs, reset_info = reset_env()
                    # A newly dealt episode can itself be immediately
                    # terminal (for example, an opponent's heaven win).
                    # Never expose that zero-action-mask state to the agent;
                    # keep advancing seeds until a playable episode exists.
                    while reset_info.get('terminal_reason') is not None:
                        next_obs, reset_info = reset_env()
                    step_info = dict(step_info)
                    step_info['final_observation'] = final_obs
                    step_info['final_info'] = final_info
                    step_info['reset_info'] = reset_info
                obs = next_obs
                remote.send((next_obs, float(reward), bool(terminated), bool(truncated), step_info))
            elif command == 'close':
                break
            else:
                raise RuntimeError(f'unknown worker command: {command}')
    finally:
        env.close()


class SubprocessVectorEnv:
    """Small spawn-based vector environment with auto-reset workers."""

    def __init__(self, num_envs: int, config: EnvConfig | None = None, seed: int = 1):
        self.num_envs = int(num_envs)
        self.config = config or EnvConfig(observation_mode='vector', render_mode=None)
        if not self.config.include_action_mask_in_obs:
            raise ValueError(
                'SubprocessVectorEnv requires include_action_mask_in_obs=True'
            )
        self.ctx = mp.get_context('spawn')
        self.parent_remotes = []
        self.processes = []
        for rank in range(self.num_envs):
            parent_remote, child_remote = self.ctx.Pipe()
            process = self.ctx.Process(
                target=_worker,
                args=(child_remote, parent_remote, self.config, seed + rank, rank),
                daemon=True,
            )
            process.start()
            child_remote.close()
            self.parent_remotes.append(parent_remote)
            self.processes.append(process)
        self.closed = False

    def reset(self):
        for remote in self.parent_remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.parent_remotes]
        observations = [result[0] for result in results]
        infos = [result[1] for result in results]
        return self._stack_observations(observations), infos

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.num_envs,):
            raise ValueError(f'expected actions shape {(self.num_envs,)}, got {actions.shape}')
        for remote, action in zip(self.parent_remotes, actions):
            remote.send(('step', int(action)))
        results = [remote.recv() for remote in self.parent_remotes]
        observations = [result[0] for result in results]
        rewards = np.asarray([result[1] for result in results], dtype=np.float32)
        terminated = np.asarray([result[2] for result in results], dtype=np.bool_)
        truncated = np.asarray([result[3] for result in results], dtype=np.bool_)
        infos = [result[4] for result in results]
        return self._stack_observations(observations), rewards, terminated, truncated, infos

    @staticmethod
    def _stack_observations(observations):
        return {
            'vector': np.stack([obs['vector'] for obs in observations]).astype(np.float32, copy=False),
            'action_mask': np.stack([obs['action_mask'] for obs in observations]).astype(np.uint8, copy=False),
        }

    def close(self):
        if self.closed:
            return
        self.closed = True
        for remote in self.parent_remotes:
            try:
                remote.send(('close', None))
            except (BrokenPipeError, EOFError):
                pass
        for remote in self.parent_remotes:
            remote.close()
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
