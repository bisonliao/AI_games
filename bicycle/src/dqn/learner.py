"""Single-device Double/Dueling DQN learner and optimization step."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .config import DQNConfig
from .network import DuelingQNetwork, numpy_state_dict
from .replay import ReplayBuffer


@dataclass(frozen=True, slots=True)
class LearnerMetrics:
    """Scalar diagnostics emitted by one learner optimization update."""

    loss: float
    td_error_mean: float
    td_error_max: float
    q_mean: float
    target_mean: float
    grad_norm: float


class DQNLearner:
    """Own online/target networks and all optimizer state on one device."""

    def __init__(self, config: DQNConfig, device: torch.device):
        self.config = config
        self.device = device
        self.online = DuelingQNetwork(
            config.observation_dim, config.action_dim, config.hidden_dim
        ).to(device)
        self.target = DuelingQNetwork(
            config.observation_dim, config.action_dim, config.hidden_dim
        ).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.updates = 0

    def update(
        self,
        replay: ReplayBuffer,
        rng: np.random.Generator,
    ) -> LearnerMetrics:
        """Run one Double-DQN update from a uniform replay minibatch.

        The online network selects the bootstrap action and the frozen target
        network evaluates it. A mean Huber loss updates only the online network;
        the target network is copied at a fixed learner-update interval.
        """
        # Replay arrays stay in CPU memory until the selected minibatch is moved
        # to the learner device. Actors never initialize CUDA.
        batch = replay.sample(self.config.batch_size, rng)
        observations = torch.as_tensor(batch.observations, device=self.device)
        actions = torch.as_tensor(batch.actions, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(batch.rewards, device=self.device)
        next_observations = torch.as_tensor(batch.next_observations, device=self.device)
        discounts = torch.as_tensor(batch.discounts, device=self.device)

        # Q(s, a) for the actions that the behavior actors actually executed.
        q_values = self.online(observations).gather(1, actions).squeeze(1)
        # Double DQN separates action selection (online) from evaluation (target).
        targets = double_dqn_targets(
            self.online,
            self.target,
            next_observations,
            rewards,
            discounts,
        )
        td_errors = targets - q_values
        loss = nn.functional.smooth_l1_loss(q_values, targets)
        # Huber loss is less sensitive than MSE to occasional large TD errors.
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.online.parameters(), self.config.gradient_clip_norm
        )
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.config.target_update_interval == 0:
            self.sync_target()
        return LearnerMetrics(
            loss=float(loss.detach().cpu()),
            td_error_mean=float(td_errors.detach().abs().mean().cpu()),
            td_error_max=float(td_errors.detach().abs().max().cpu()),
            q_mean=float(q_values.detach().mean().cpu()),
            target_mean=float(targets.detach().mean().cpu()),
            grad_norm=float(grad_norm.detach().cpu()),
        )

    def sync_target(self) -> None:
        """Hard-copy online parameters into the target network."""
        self.target.load_state_dict(self.online.state_dict())

    def actor_state_dict(self) -> dict[str, np.ndarray]:
        return numpy_state_dict(self.online)

    def state_dict(self) -> dict[str, object]:
        return {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "updates": self.updates,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = int(state["updates"])


def double_dqn_targets(
    online: nn.Module,
    target: nn.Module,
    next_observations: torch.Tensor,
    rewards: torch.Tensor,
    discounts: torch.Tensor,
) -> torch.Tensor:
    """Compute r + discount * Q_target(s', argmax Q_online(s', ·))."""
    with torch.no_grad():
        next_actions = online(next_observations).argmax(dim=1, keepdim=True)
        next_q = target(next_observations).gather(1, next_actions).squeeze(1)
        return rewards + discounts * next_q
