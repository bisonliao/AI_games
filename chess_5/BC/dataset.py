"""Compact sharded BC data loading and online symmetry augmentation."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .symmetry import transform_action, transform_board
from .oracle import DEFAULT_ORACLE_TOP_K, REASON_TO_ID, SOURCE_TO_ID
from .symmetry import canonicalize


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


def is_validation_group(group: bytes, seed: int, val_fraction: float) -> bool:
    digest = hashlib.sha256(int(seed).to_bytes(8, "little", signed=True) + bytes(group)).digest()
    return int.from_bytes(digest[:8], "little") / 2**64 < val_fraction


class GomokuDataset(Dataset[Any]):
    def __init__(self, shards: Sequence[Path], *, split: str, val_fraction: float = 0.1,
                 augment: bool = False, seed: int = 0, max_samples: int | None = None,
                 rich: bool = False, excluded_keys: set[bytes] | None = None) -> None:
        arrays: dict[str, list[np.ndarray]] = {name: [] for name in (
            "boards", "players", "actions", "candidate_actions", "candidate_counts",
            "tactical", "reasons", "plies", "sources", "rounds", "behavior_actions",
            "canonical_keys",
        )}
        excluded_keys = excluded_keys or set()
        for path in shards:
            with np.load(path) as data:
                games = data["games"]
                if "trajectory_groups" in data:
                    groups = data["trajectory_groups"]
                    val = np.asarray([is_validation_group(bytes(group), seed, val_fraction)
                                      for group in groups])
                else:
                    val = np.asarray([((int(g) * 2654435761 + seed) & 0xffffffff) / 2**32
                                      < val_fraction for g in games])
                take = val if split == "val" else ~val
                if "canonical_keys" in data and excluded_keys:
                    permitted = np.asarray([bytes(key) not in excluded_keys
                                            for key in data["canonical_keys"]])
                    take &= permitted
                selected_boards = data["boards"][take]
                selected_actions = data["actions"][take]
                if len(selected_actions) and not np.all(
                    selected_boards.reshape(len(selected_boards), -1)[np.arange(len(selected_boards)), selected_actions] == 0
                ):
                    raise ValueError(f"dataset contains an illegal expert label: {path}")
                selected_players = np.asarray(data["players"][take], dtype=np.int8)
                count = len(selected_actions)
                if "candidate_actions" in data:
                    raw_candidates = np.asarray(data["candidate_actions"][take], dtype=np.int16)
                    candidates = np.full((count, DEFAULT_ORACLE_TOP_K), -1, dtype=np.int16)
                    width = min(raw_candidates.shape[1], DEFAULT_ORACLE_TOP_K)
                    candidates[:, :width] = raw_candidates[:, :width]
                else:
                    candidates = np.full((count, DEFAULT_ORACLE_TOP_K), -1, dtype=np.int16)
                    candidates[:, 0] = np.asarray(selected_actions, dtype=np.int16)
                candidate_counts = (np.asarray(data["candidate_counts"][take], dtype=np.int8)
                                    if "candidate_counts" in data else
                                    np.ones(count, dtype=np.int8))
                keys = (np.asarray(data["canonical_keys"][take], dtype="S32")
                        if "canonical_keys" in data else
                        np.asarray([canonicalize(board, int(player))[0]
                                    for board, player in zip(selected_boards, selected_players)],
                                   dtype="S32"))
                defaults = {
                    "tactical": np.zeros(count, dtype=np.bool_),
                    "reasons": np.full(count, REASON_TO_ID["normal"], dtype=np.int8),
                    "plies": np.count_nonzero(selected_boards, axis=(1, 2)).astype(np.int16),
                    "sources": np.full(count, SOURCE_TO_ID["expert_selfplay"], dtype=np.int8),
                    "rounds": np.zeros(count, dtype=np.int8),
                    "behavior_actions": np.asarray(selected_actions, dtype=np.int16),
                }
                values = {
                    "boards": selected_boards, "players": selected_players,
                    "actions": selected_actions, "candidate_actions": candidates,
                    "candidate_counts": candidate_counts, "canonical_keys": keys,
                }
                for name, default in defaults.items():
                    values[name] = np.asarray(data[name][take]) if name in data else default
                for name, value in values.items():
                    arrays[name].append(np.asarray(value))
        self.boards = np.concatenate(arrays["boards"]) if arrays["boards"] else np.empty((0, 0, 0), np.int8)
        self.players = np.concatenate(arrays["players"]) if arrays["players"] else np.empty(0, np.int8)
        self.actions = np.concatenate(arrays["actions"]) if arrays["actions"] else np.empty(0, np.int64)
        self.candidate_actions = (np.concatenate(arrays["candidate_actions"])
                                  if arrays["candidate_actions"] else np.empty((0, 1), np.int16))
        for name in ("candidate_counts", "tactical", "reasons", "plies", "sources",
                     "rounds", "behavior_actions", "canonical_keys"):
            setattr(self, name, np.concatenate(arrays[name]) if arrays[name] else np.empty(0))
        if max_samples is not None and len(self.actions) > max_samples:
            rng = np.random.default_rng(seed); idx = rng.choice(len(self.actions), max_samples, replace=False)
            for name in ("boards", "players", "actions", "candidate_actions", "candidate_counts",
                         "tactical", "reasons", "plies", "sources", "rounds",
                         "behavior_actions", "canonical_keys"):
                setattr(self, name, getattr(self, name)[idx])
        if len(self.canonical_keys):
            _, inverse, counts = np.unique(self.canonical_keys, return_inverse=True,
                                           return_counts=True)
            self.frequency_weights = 1.0 / np.sqrt(np.minimum(counts[inverse], 64))
        else:
            self.frequency_weights = np.empty(0, dtype=np.float32)
        self.augment = augment
        self.rich = bool(rich)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        board = self.boards[index]; action = int(self.actions[index])
        candidates = np.asarray(self.candidate_actions[index], dtype=np.int64).copy()
        if self.augment:
            transform = int(self.rng.integers(8)); action = transform_action(action, board.shape[0], transform)
            for candidate_index in np.flatnonzero(candidates >= 0):
                candidates[candidate_index] = transform_action(
                    int(candidates[candidate_index]), board.shape[0], transform
                )
            board = transform_board(board, transform)
        state = encode_boards(board, [self.players[index]])[0]
        mask = torch.from_numpy((board.reshape(-1) == 0).copy())
        if not self.rich:
            return torch.from_numpy(state), torch.tensor(action), mask
        return {
            "states": torch.from_numpy(state), "targets": torch.tensor(action),
            "masks": mask, "candidates": torch.from_numpy(candidates),
            "candidate_counts": torch.tensor(int(self.candidate_counts[index])),
            "tactical": torch.tensor(bool(self.tactical[index])),
            "reasons": torch.tensor(int(self.reasons[index])),
            "plies": torch.tensor(int(self.plies[index])),
            "sources": torch.tensor(int(self.sources[index])),
            "rounds": torch.tensor(int(self.rounds[index])),
            "frequency_weights": torch.tensor(float(self.frequency_weights[index]),
                                                dtype=torch.float32),
        }
