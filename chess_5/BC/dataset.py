"""Compact sharded BC data loading and online symmetry augmentation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .symmetry import transform_action, transform_board


def encode_boards(boards: np.ndarray, players: np.ndarray) -> np.ndarray:
    boards = np.asarray(boards, dtype=np.int8)
    if boards.ndim == 2:
        boards = boards[None]
    players = np.asarray(players, dtype=np.int8).reshape(-1, 1, 1)
    return np.stack((boards == players, boards == -players, boards == 0), axis=1).astype(np.float32)


def discover_shards(roots: Sequence[Path]) -> list[Path]:
    shards: list[Path] = []
    for root in roots:
        metadata = json.loads((Path(root) / "metadata.json").read_text())
        if metadata.get("status") != "complete":
            raise ValueError(f"dataset is not complete: {root}")
        shards.extend(Path(root) / name for name in metadata["shards"])
    return shards


class GomokuDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, shards: Sequence[Path], *, split: str, val_fraction: float = 0.1,
                 augment: bool = False, seed: int = 0, max_samples: int | None = None) -> None:
        boards, players, actions = [], [], []
        for path in shards:
            with np.load(path) as data:
                games = data["games"]
                val = np.asarray([((int(g) * 2654435761 + seed) & 0xffffffff) / 2**32 < val_fraction
                                  for g in games])
                take = val if split == "val" else ~val
                selected_boards = data["boards"][take]
                selected_actions = data["actions"][take]
                if len(selected_actions) and not np.all(
                    selected_boards.reshape(len(selected_boards), -1)[np.arange(len(selected_boards)), selected_actions] == 0
                ):
                    raise ValueError(f"dataset contains an illegal expert label: {path}")
                boards.append(selected_boards); players.append(data["players"][take])
                actions.append(selected_actions)
        self.boards = np.concatenate(boards) if boards else np.empty((0, 0, 0), np.int8)
        self.players = np.concatenate(players) if players else np.empty(0, np.int8)
        self.actions = np.concatenate(actions) if actions else np.empty(0, np.int64)
        if max_samples is not None and len(self.actions) > max_samples:
            rng = np.random.default_rng(seed); idx = rng.choice(len(self.actions), max_samples, replace=False)
            self.boards, self.players, self.actions = self.boards[idx], self.players[idx], self.actions[idx]
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        board = self.boards[index]; action = int(self.actions[index])
        if self.augment:
            transform = int(self.rng.integers(8)); action = transform_action(action, board.shape[0], transform)
            board = transform_board(board, transform)
        state = encode_boards(board, [self.players[index]])[0]
        return torch.from_numpy(state), torch.tensor(action), torch.from_numpy((board.reshape(-1) == 0).copy())
