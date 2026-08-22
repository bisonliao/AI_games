"""BC-compatible policy and dueling Q networks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from .common import encode_boards, load_numpy_state_dict, random_legal_actions


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.relu(value + self.body(value))


def make_trunk(in_channels: int, hidden_channels: int, num_res_blocks: int) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(hidden_channels),
        nn.ReLU(inplace=True),
    ]
    layers.extend(ResidualBlock(hidden_channels) for _ in range(num_res_blocks))
    return nn.Sequential(*layers)


class GomokuPolicyNet(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_channels: int = 96,
                 num_res_blocks: int = 4) -> None:
        super().__init__()
        self.trunk = make_trunk(in_channels, hidden_channels, num_res_blocks)
        self.policy = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.policy(self.trunk(value)).flatten(1)


class DuelingGomokuQNet(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_channels: int = 96,
                 num_res_blocks: int = 4) -> None:
        super().__init__()
        self.trunk = make_trunk(in_channels, hidden_channels, num_res_blocks)
        self.advantage = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        self.value_pool = nn.AdaptiveAvgPool2d(1)
        self.value = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        features = self.trunk(value)
        advantage = self.advantage(features).flatten(1)
        state_value = self.value(self.value_pool(features))
        return state_value + advantage - advantage.mean(dim=1, keepdim=True)


def read_bc_checkpoint(path: Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    if int(checkpoint.get("board_size", -1)) != 9:
        raise ValueError("afterBC requires the 9x9 BC_BEST checkpoint")
    kwargs = checkpoint.get("model_kwargs")
    if kwargs != {"hidden_channels": 128, "num_res_blocks": 8}:
        raise ValueError(f"unexpected BC_BEST model configuration: {kwargs}")
    if "model_state_dict" not in checkpoint:
        raise ValueError("BC checkpoint does not contain model_state_dict")
    return checkpoint


def make_bc_policy(path: Path, *, device: str | torch.device = "cpu") -> tuple[GomokuPolicyNet, dict[str, Any]]:
    checkpoint = read_bc_checkpoint(path, device)
    model = GomokuPolicyNet(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def make_dueling_from_bc(
    path: Path, *, device: str | torch.device = "cpu"
) -> tuple[DuelingGomokuQNet, dict[str, Any]]:
    checkpoint = read_bc_checkpoint(path, "cpu")
    model = DuelingGomokuQNet(**checkpoint["model_kwargs"])
    bc_state = checkpoint["model_state_dict"]
    target_state = model.state_dict()
    transferred: set[str] = set()
    for name in list(target_state):
        source_name = name if name.startswith("trunk.") else None
        if name.startswith("advantage."):
            source_name = "policy." + name[len("advantage."):]
        if source_name is not None:
            if source_name not in bc_state or bc_state[source_name].shape != target_state[name].shape:
                raise ValueError(f"BC parameter cannot initialize DQN parameter: {source_name} -> {name}")
            target_state[name] = bc_state[source_name].detach().cpu().clone()
            transferred.add(source_name)
    expected = {name for name in bc_state if name.startswith(("trunk.", "policy."))}
    if transferred != expected:
        raise ValueError(f"BC transfer missed parameters: {sorted(expected - transferred)}")
    target_state["value.3.weight"].zero_()
    target_state["value.3.bias"].zero_()
    model.load_state_dict(target_state)
    model.to(device)
    return model, checkpoint


class NetworkPolicy:
    """CPU or GPU inference wrapper with legal-action masking."""

    def __init__(self, model: nn.Module, board_size: int, *, device: str | torch.device,
                 seed: int = 0) -> None:
        self.model = model
        self.board_size = int(board_size)
        self.action_dim = self.board_size * self.board_size
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.model.to(self.device).eval()

    def load_numpy_weights(self, state: Mapping[str, np.ndarray]) -> None:
        load_numpy_state_dict(self.model, state)
        self.model.eval()

    def values(self, boards: np.ndarray, players: np.ndarray | int) -> np.ndarray:
        states = torch.from_numpy(encode_boards(boards, players)).to(self.device)
        with torch.inference_mode():
            return self.model(states).cpu().numpy()

    def select_actions(self, boards: np.ndarray, players: np.ndarray | int,
                       action_masks: np.ndarray, *, epsilon: float = 0.0) -> np.ndarray:
        boards = np.asarray(boards, dtype=np.int8)
        if boards.ndim == 2:
            boards = boards[None]
        masks = np.asarray(action_masks, dtype=np.bool_).reshape(len(boards), self.action_dim)
        actions = random_legal_actions(masks, self.rng)
        greedy = np.flatnonzero(self.rng.random(len(boards)) >= float(epsilon))
        if len(greedy):
            values = self.values(boards[greedy], np.asarray(players).reshape(-1)[greedy]
                                 if np.asarray(players).size > 1 else players)
            values[~masks[greedy]] = -np.inf
            actions[greedy] = values.argmax(1)
        return actions
