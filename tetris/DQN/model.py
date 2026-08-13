"""Dueling Q network for placement-level Tetris observations."""
from __future__ import annotations

import torch
from torch import nn


class DuelingDQN(nn.Module):
    NUM_ACTIONS = 40

    def __init__(self) -> None:
        super().__init__()
        self.board_encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            board_size = self.board_encoder(torch.zeros(1, 2, 20, 10)).shape[1]
        self.scalar_encoder = nn.Sequential(nn.Linear(7 + 7 + 4 + 2, 64), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(board_size + 64, 256), nn.ReLU())
        self.value = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        # Keep rotation x target-column as one joint 40-action head. Two
        # independent 4/10 heads would assume their values can be separated,
        # although the best/legal column depends on the chosen rotation. The
        # joint head also uses one straightforward 40-element action mask.
        self.advantage = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, self.NUM_ACTIONS)
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        board = obs["board"].float()
        active = obs["active"].float()
        if board.ndim == 2:
            board = board.unsqueeze(0)
            active = active.unsqueeze(0)
        if board.ndim == 3:
            board = board.unsqueeze(1)
            active = active.unsqueeze(1)
        scalar_inputs = []
        for key in ("current_piece", "next_piece", "rotation", "position"):
            value = obs[key].float()
            scalar_inputs.append(value.unsqueeze(0) if value.ndim == 1 else value)
        image = torch.cat((board, active), dim=1)
        image_features = self.board_encoder(image)
        scalars = torch.cat(
            tuple(scalar_inputs), dim=-1
        )
        fused = self.fusion(torch.cat((image_features, self.scalar_encoder(scalars)), dim=-1))
        value = self.value(fused)
        advantage = self.advantage(fused)
        if "action_mask" in obs:
            mask = obs["action_mask"].to(device=advantage.device, dtype=advantage.dtype)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            valid_count = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            advantage_mean = (advantage * mask).sum(dim=-1, keepdim=True) / valid_count
        else:
            advantage_mean = advantage.mean(dim=-1, keepdim=True)
        return value + advantage - advantage_mean


def observations_to_torch(obs: dict, device: torch.device | str) -> dict[str, torch.Tensor]:
    return {key: torch.as_tensor(value, device=device) for key, value in obs.items()}


def masked_q_values(q_values: torch.Tensor, action_mask: torch.Tensor | None) -> torch.Tensor:
    """Set invalid actions to the smallest finite value before max/argmax.

    A terminal observation legitimately has an all-zero mask. Action zero is a
    harmless sentinel for such rows because its Q value is later multiplied by
    the termination mask.
    """
    if action_mask is None:
        return q_values
    mask = action_mask.to(device=q_values.device, dtype=torch.bool)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if mask.shape != q_values.shape:
        raise ValueError(f"action mask shape {tuple(mask.shape)} != Q shape {tuple(q_values.shape)}")
    # A terminal row can have no legal actions. Select action zero as its
    # harmless sentinel without synchronizing a CUDA boolean back to Python.
    mask = mask.clone()
    mask[:, 0] |= ~mask.any(dim=1)
    return q_values.masked_fill(~mask, torch.finfo(q_values.dtype).min)
