from dataclasses import dataclass

import numpy as np


@dataclass
class ReplayBatch:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    action_mask: np.ndarray
    next_action_mask: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, seed: int = 1):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.rng = np.random.default_rng(seed)
        self.obs = np.empty((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.empty((capacity, obs_dim), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int16)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=np.bool_)
        self.truncated = np.empty(capacity, dtype=np.bool_)
        self.action_mask = np.empty((capacity, action_dim), dtype=np.uint8)
        self.next_action_mask = np.empty((capacity, action_dim), dtype=np.uint8)
        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add_batch(self, obs, actions, rewards, next_obs, terminated, truncated, action_mask, next_action_mask):
        obs = np.asarray(obs, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int16)
        rewards = np.asarray(rewards, dtype=np.float32)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        terminated = np.asarray(terminated, dtype=np.bool_)
        truncated = np.asarray(truncated, dtype=np.bool_)
        action_mask = np.asarray(action_mask, dtype=np.uint8)
        next_action_mask = np.asarray(next_action_mask, dtype=np.uint8)
        n = len(actions)
        if obs.shape != (n, self.obs_dim) or next_obs.shape != (n, self.obs_dim):
            raise ValueError('invalid observation batch shape')
        if action_mask.shape != (n, self.action_dim) or next_action_mask.shape != (n, self.action_dim):
            raise ValueError('invalid action mask batch shape')
        if n == 0:
            return
        indices = (self.position + np.arange(n)) % self.capacity
        self.obs[indices] = obs
        self.next_obs[indices] = next_obs
        self.actions[indices] = actions
        self.rewards[indices] = rewards
        self.terminated[indices] = terminated
        self.truncated[indices] = truncated
        self.action_mask[indices] = action_mask
        self.next_action_mask[indices] = next_action_mask
        self.position = int((self.position + n) % self.capacity)
        self.size = min(self.capacity, self.size + n)

    def sample(self, batch_size: int) -> ReplayBatch:
        if self.size < batch_size:
            raise ValueError('not enough replay samples')
        indices = self.rng.integers(0, self.size, size=batch_size)
        return ReplayBatch(
            self.obs[indices], self.actions[indices], self.rewards[indices],
            self.next_obs[indices], self.terminated[indices], self.truncated[indices],
            self.action_mask[indices], self.next_action_mask[indices],
        )
