"""Policy/value network and the discrete training adapter."""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces
from torch import nn
from torch.distributions import Categorical


# PPO action id -> RaceCarEnv continuous steering command.
# Keeping this mapping here makes saved policy outputs stable and auditable.
DISCRETE_TO_STEER = np.asarray([0.0, -1.0, 1.0], dtype=np.float32)
ACTION_NAMES = ("straight", "left", "right")
POLICY_ACTION_SPACE = spaces.Discrete(len(ACTION_NAMES))


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    """Shared MLP trunk with categorical policy and scalar critic heads."""

    def __init__(self, observation_size: int = 9, action_size: int = 3):
        super().__init__()
        self.trunk = nn.Sequential(
            layer_init(nn.Linear(observation_size, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
        )
        self.policy = layer_init(nn.Linear(256, action_size), std=0.01)
        self.value = layer_init(nn.Linear(256, 1), std=1.0)

    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        return self.value(self.trunk(observation)).squeeze(-1)

    def get_action_and_value(self, observation: torch.Tensor,
                             action: torch.Tensor | None = None):
        hidden = self.trunk(observation)
        distribution = Categorical(logits=self.policy(hidden))
        if action is None:
            action = distribution.sample()
        return (action, distribution.log_prob(action), distribution.entropy(),
                self.value(hidden).squeeze(-1))


def actions_to_steering(actions: np.ndarray) -> np.ndarray:
    """Map a categorical action batch to ``RaceCarEnv`` Box actions."""
    actions = np.asarray(actions, dtype=np.int64)
    if np.any((actions < 0) | (actions >= len(DISCRETE_TO_STEER))):
        raise ValueError("discrete action must be 0 (straight), 1 (left), or 2 (right)")
    return DISCRETE_TO_STEER[actions, None]


def shape_rewards(observations: np.ndarray, next_observations: np.ndarray,
                  terminated: np.ndarray, truncated: np.ndarray, infos: list[dict],
                  finish_xy=(14.5, 7.5)) -> np.ndarray:
    """Reward efficiency rather than accumulated time near the finish.

    Dense reward is five times the reduction in Euclidean goal distance minus
    a small per-step cost. Terminal outcomes replace that dense reward. The
    original environment reward is intentionally not used because it is
    positive every step and can reward deliberate delay.
    """
    finish = np.asarray(finish_xy, dtype=np.float32)
    old_distance = np.linalg.norm(observations[:, :2] - finish, axis=1)
    new_distance = np.linalg.norm(next_observations[:, :2] - finish, axis=1)
    rewards = 5.0 * (old_distance - new_distance) - 0.01
    for index, info in enumerate(infos):
        reason = info.get("termination_reason")
        if reason == "goal":
            rewards[index] = 20.0
        elif reason == "collision":
            rewards[index] = -5.0
        elif truncated[index] or reason == "time_limit":
            rewards[index] = -2.0
    return rewards.astype(np.float32)
