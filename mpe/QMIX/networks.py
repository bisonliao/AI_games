"""Recurrent per-agent Q network and monotonic QMIX mixing network."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RecurrentAgent(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_actions: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(
        self, inputs: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(inputs))
        hidden = self.rnn(x, hidden)
        return self.fc2(hidden), hidden


class QMixer(nn.Module):
    """QMIX mixer with non-negative state-conditioned hypernetwork weights."""

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        mixing_embed_dim: int = 32,
        hypernet_embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.n_agents = int(n_agents)
        self.state_dim = int(state_dim)
        self.embed_dim = int(mixing_embed_dim)

        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed_dim),
            nn.ReLU(),
            nn.Linear(hypernet_embed_dim, n_agents * mixing_embed_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, mixing_embed_dim)
        self.hyper_w_final = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed_dim),
            nn.ReLU(),
            nn.Linear(hypernet_embed_dim, mixing_embed_dim),
        )
        self.value = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = agent_qs.shape
        flat_states = states.reshape(-1, self.state_dim)
        flat_agent_qs = agent_qs.reshape(-1, 1, self.n_agents)

        w1 = torch.abs(self.hyper_w1(flat_states)).view(
            -1, self.n_agents, self.embed_dim
        )
        b1 = self.hyper_b1(flat_states).view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(flat_agent_qs, w1) + b1)

        w_final = torch.abs(self.hyper_w_final(flat_states)).view(
            -1, self.embed_dim, 1
        )
        value = self.value(flat_states).view(-1, 1, 1)
        total_q = torch.bmm(hidden, w_final) + value
        return total_q.view(batch_size, sequence_length, 1)
