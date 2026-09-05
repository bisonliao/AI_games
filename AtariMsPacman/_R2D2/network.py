"""Recurrent dueling Q-network used by R2D2."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


HiddenState = tuple[torch.Tensor, torch.Tensor]


class RecurrentDuelingQNetwork(nn.Module):
    """Atari CNN + LSTM + dueling heads.

    The recurrent input at time ``t`` contains the encoded observation at ``t``
    together with the action and reward observed at ``t-1``.
    """

    def __init__(
        self,
        observation_shape: tuple[int, int, int] = (4, 84, 84),
        action_count: int = 9,
        hidden_size: int = 512,
    ) -> None:
        super().__init__()
        if len(observation_shape) != 3:
            raise ValueError("observation_shape must be (channels, height, width)")
        if action_count <= 0 or hidden_size <= 0:
            raise ValueError("action_count and hidden_size must be positive")
        channels, height, width = observation_shape
        self.observation_shape = tuple(observation_shape)
        self.action_count = int(action_count)
        self.hidden_size = int(hidden_size)
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            feature_size = int(
                self.encoder(torch.zeros(1, channels, height, width)).shape[1]
            )
        self.projection = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(inplace=True),
        )
        self.recurrent = nn.LSTM(
            hidden_size + action_count + 1,
            hidden_size,
            batch_first=True,
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, action_count),
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.zeros_(module.bias)

    def initial_hidden(self, batch_size: int, *, device=None) -> HiddenState:
        device = device if device is not None else next(self.parameters()).device
        zeros = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return zeros.clone(), zeros.clone()

    def forward(
        self,
        observation: torch.Tensor,
        previous_action: torch.Tensor | None = None,
        previous_reward: torch.Tensor | None = None,
        hidden_state: Optional[HiddenState] = None,
    ):
        """Convenience dispatch for one-step or sequence inference."""
        if observation.ndim == 4:
            batch = observation.shape[0]
            if previous_action is None:
                previous_action = torch.zeros(
                    batch, self.action_count, device=observation.device
                )
                previous_action[:, 0] = 1.0
            if previous_reward is None:
                previous_reward = torch.zeros(batch, device=observation.device)
            return self.step(observation, previous_action, previous_reward, hidden_state)
        if observation.ndim == 5:
            batch, steps = observation.shape[:2]
            if previous_action is None:
                previous_action = torch.zeros(
                    batch, steps, self.action_count, device=observation.device
                )
                previous_action[..., 0] = 1.0
            if previous_reward is None:
                previous_reward = torch.zeros(batch, steps, device=observation.device)
            return self.unroll(
                observation, previous_action, previous_reward, hidden_state
            )
        raise ValueError("observation must have shape (B,C,H,W) or (B,T,C,H,W)")

    def _prepare_previous_action(self, action: torch.Tensor, batch: int, steps: int) -> torch.Tensor:
        if action.ndim == 2 and action.shape[-1] == self.action_count:
            result = action.float()
        elif action.ndim in (1, 2):
            indices = action.long().reshape(batch, steps)
            result = torch.zeros(
                batch, steps, self.action_count, device=action.device, dtype=torch.float32
            )
            result.scatter_(2, indices.unsqueeze(-1), 1.0)
        else:
            raise ValueError("previous_action must be one-hot or integer indices")
        return result

    def _q_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        advantage = self.advantage_stream(hidden)
        value = self.value_stream(hidden)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)

    def step(
        self,
        observation: torch.Tensor,
        previous_action: torch.Tensor,
        previous_reward: torch.Tensor,
        hidden_state: Optional[HiddenState] = None,
    ) -> tuple[torch.Tensor, HiddenState]:
        """Run one recurrent step for a batch of observations."""
        if observation.ndim != 4:
            raise ValueError("observation must have shape (B, C, H, W)")
        batch = observation.shape[0]
        if hidden_state is None:
            hidden_state = self.initial_hidden(batch, device=observation.device)
        obs_features = self.projection(self.encoder(observation.float() / 255.0))
        if previous_action.ndim == 1:
            previous_action = torch.nn.functional.one_hot(
                previous_action.long(), self.action_count
            ).float()
        previous_action = previous_action.reshape(batch, self.action_count).float()
        previous_reward = previous_reward.reshape(batch, 1).float()
        recurrent_input = torch.cat((obs_features, previous_action, previous_reward), dim=-1)
        output, new_hidden = self.recurrent(recurrent_input[:, None, :], hidden_state)
        return self._q_from_hidden(output[:, 0]), new_hidden

    def unroll(
        self,
        observations: torch.Tensor,
        previous_actions: torch.Tensor,
        previous_rewards: torch.Tensor,
        hidden_state: Optional[HiddenState] = None,
        burn_in_steps: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, HiddenState]:
        """Unroll padded sequences and return all Q-values.

        If ``burn_in_steps`` is supplied, each row's recurrent state is
        detached immediately after its burn-in prefix.  Later losses therefore
        train through the learning unroll but never through burn-in.
        """
        if observations.ndim != 5:
            raise ValueError("observations must have shape (B, T, C, H, W)")
        batch, steps = observations.shape[:2]
        if lengths is not None:
            lengths = lengths.to(observations.device).long().reshape(batch)
            if bool(torch.any(lengths < 0)) or bool(torch.any(lengths > steps)):
                raise ValueError("sequence lengths must be between zero and T")
        if valid_mask is not None:
            if valid_mask.shape != (batch, steps):
                raise ValueError("valid_mask must have shape (B, T)")
            valid_mask = valid_mask.to(observations.device, dtype=torch.bool)
            if lengths is None:
                lengths = valid_mask.long().sum(dim=1)
        if hidden_state is None:
            hidden_state = self.initial_hidden(batch, device=observations.device)
        flattened = observations.reshape(batch * steps, *observations.shape[2:]).float() / 255.0
        features = self.projection(self.encoder(flattened)).reshape(batch, steps, -1)
        if previous_actions.ndim == 3:
            actions = previous_actions.float()
        else:
            actions = self._prepare_previous_action(previous_actions, batch, steps)
        rewards = previous_rewards.float().reshape(batch, steps, 1)
        recurrent_input = torch.cat((features, actions, rewards), dim=-1)
        if burn_in_steps is None and lengths is None:
            output, final_hidden = self.recurrent(recurrent_input, hidden_state)
        else:
            if burn_in_steps is None:
                burn_in_steps = torch.zeros(batch, device=observations.device, dtype=torch.long)
            else:
                burn_in_steps = burn_in_steps.to(observations.device).reshape(batch)
            outputs: list[torch.Tensor] = []
            recurrent_hidden = hidden_state
            for step_index in range(steps):
                detach_rows = burn_in_steps == step_index
                if bool(detach_rows.any()):
                    mask = detach_rows.view(1, batch, 1)
                    recurrent_hidden = (
                        torch.where(mask, recurrent_hidden[0].detach(), recurrent_hidden[0]),
                        torch.where(mask, recurrent_hidden[1].detach(), recurrent_hidden[1]),
                    )
                old_hidden = recurrent_hidden
                item, candidate_hidden = self.recurrent(
                    recurrent_input[:, step_index : step_index + 1], recurrent_hidden
                )
                if lengths is not None:
                    active = (step_index < lengths).view(1, batch, 1)
                    recurrent_hidden = (
                        torch.where(active, candidate_hidden[0], old_hidden[0]),
                        torch.where(active, candidate_hidden[1], old_hidden[1]),
                    )
                    item = torch.where(active.transpose(0, 1), item, torch.zeros_like(item))
                else:
                    recurrent_hidden = candidate_hidden
                outputs.append(item)
            output = torch.cat(outputs, dim=1)
            final_hidden = recurrent_hidden
        q_values = self._q_from_hidden(output)
        if valid_mask is not None:
            q_values = q_values.masked_fill(~valid_mask[..., None], 0.0)
        return q_values, final_hidden

    @staticmethod
    def value_rescale(value: torch.Tensor, epsilon: float = 1.0e-3) -> torch.Tensor:
        return value.sign() * (torch.sqrt(value.abs() + 1.0) - 1.0) + epsilon * value

    @staticmethod
    def inverse_value_rescale(value: torch.Tensor, epsilon: float = 1.0e-3) -> torch.Tensor:
        inside = torch.sqrt(1.0 + 4.0 * epsilon * (value.abs() + 1.0 + epsilon))
        transformed = (inside - 1.0) / (2.0 * epsilon)
        return value.sign() * (transformed.square() - 1.0)


# Familiar short name for callers migrating from the reference implementation.
Network = RecurrentDuelingQNetwork
