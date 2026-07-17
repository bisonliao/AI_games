"""Inference and checkpoint wrapper for BC policies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .dataset import encode_boards
from .network import GomokuPolicyNet


class BCAgent:
    def __init__(self, board_size: int, *, hidden_channels: int = 96,
                 num_res_blocks: int = 4, device: str = "cpu") -> None:
        self.board_size = int(board_size)
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                                   else "cpu" if device == "auto" else device)
        self.model_kwargs = {"hidden_channels": hidden_channels, "num_res_blocks": num_res_blocks}
        self.net = GomokuPolicyNet(**self.model_kwargs).to(self.device)

    def select_actions(self, boards: np.ndarray, current_players: np.ndarray,
                       action_masks: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
        del epsilon
        logits = self.action_logits(boards, current_players)
        masks = np.asarray(action_masks).reshape(len(logits), -1).astype(bool)
        logits[~masks] = -np.inf
        return logits.argmax(1)

    def action_logits(self, boards: np.ndarray, current_players: np.ndarray) -> np.ndarray:
        """Return raw policy logits for controlled play or diagnostics."""
        boards = np.asarray(boards, dtype=np.int8)
        if boards.ndim == 2: boards = boards[None]
        players = np.asarray(current_players).reshape(-1)
        was_training = self.net.training; self.net.eval()
        with torch.no_grad():
            states = torch.from_numpy(encode_boards(boards, players)).to(self.device)
            logits = self.net(states).cpu().numpy()
        self.net.train(was_training)
        return logits

    def load_checkpoint(self, path: Path) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if int(checkpoint["board_size"]) != self.board_size:
            raise ValueError("checkpoint board size does not match agent")
        kwargs = checkpoint.get("model_kwargs", self.model_kwargs)
        if kwargs != self.model_kwargs:
            self.model_kwargs = kwargs; self.net = GomokuPolicyNet(**kwargs).to(self.device)
        self.net.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint
