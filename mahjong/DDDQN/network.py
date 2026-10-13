import torch
from torch import nn


class DuelingQNetwork(nn.Module):
    def __init__(self, obs_dim: int = 306, action_dim: int = 141, hidden_dim: int = 512):
        super().__init__()
        self.action_dim = action_dim
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value = nn.Linear(hidden_dim, 1)
        self.advantage = nn.Linear(hidden_dim, action_dim)

    def raw(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return finite Q values with a legal-action dueling baseline."""
        hidden = self.trunk(obs)
        value = self.value(hidden)
        advantage = self.advantage(hidden)
        if action_mask is None:
            mean_advantage = advantage.mean(dim=1, keepdim=True)
        else:
            valid = action_mask.to(dtype=advantage.dtype)
            mean_advantage = (advantage * valid).sum(
                dim=1,
                keepdim=True,
            ) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        return value + advantage - mean_advantage

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_values = self.raw(obs, action_mask)
        if action_mask is not None:
            q_values = q_values.masked_fill(~action_mask.bool(), -torch.inf)
        return q_values


def masked_argmax(q_values: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    masked = q_values.masked_fill(~action_mask.bool(), -torch.inf)
    return masked.argmax(dim=1)
