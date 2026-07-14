from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

try:
    from .network import DuelingGomokuQNet
except ImportError:
    from network import DuelingGomokuQNet


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        board_size: int,
        action_dim: int,
        seed: int = 0,
    ) -> None:
        self.capacity = int(capacity)
        self.board_size = int(board_size)
        self.action_dim = int(action_dim)
        self.rng = np.random.default_rng(seed)

        self.states = np.zeros((capacity, 3, board_size, board_size), dtype=np.float32)
        self.next_states = np.zeros_like(self.states)
        self.next_masks = np.zeros((capacity, action_dim), dtype=np.bool_)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.discounts = np.zeros(capacity, dtype=np.float32)

        self.pos = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_mask: np.ndarray,
        done: bool,
        discount: float,
    ) -> None:
        self.states[self.pos] = state
        self.actions[self.pos] = int(action)
        self.rewards[self.pos] = float(reward)
        self.next_states[self.pos] = next_state
        self.next_masks[self.pos] = next_mask.astype(np.bool_)
        self.dones[self.pos] = bool(done)
        self.discounts[self.pos] = float(discount)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self) -> int:
        return self.size

    def sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, ...]:
        indices = self.rng.integers(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.states[indices], device=device),
            torch.as_tensor(self.actions[indices], device=device).long(),
            torch.as_tensor(self.rewards[indices], device=device),
            torch.as_tensor(self.next_states[indices], device=device),
            torch.as_tensor(self.next_masks[indices], device=device),
            torch.as_tensor(self.dones[indices], device=device).float(),
            torch.as_tensor(self.discounts[indices], device=device),
        )


class RandomAgent:
    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def select_actions(
        self,
        boards: np.ndarray,
        current_players: np.ndarray,
        action_masks: np.ndarray,
        epsilon: float = 0.0,
    ) -> np.ndarray:
        del boards, current_players, epsilon
        return random_legal_actions(action_masks, self.rng)


class DQNAgent:
    def __init__(
        self,
        board_size: int,
        *,
        hidden_channels: int = 96,
        num_res_blocks: int = 4,
        lr: float = 1e-4,
        gamma: float = 0.99,
        batch_size: int = 256,
        replay_size: int = 200_000,
        min_replay_size: int = 10_000,
        target_update: int = 2_000,
        train_freq: int = 1,
        grad_clip: float = 10.0,
        double_dqn: bool = True,
        device: Optional[str] = None,
        seed: int = 0,
    ) -> None:
        self.board_size = int(board_size)
        self.action_dim = self.board_size * self.board_size
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.min_replay_size = int(min_replay_size)
        self.target_update = int(target_update)
        self.train_freq = int(train_freq)
        self.grad_clip = float(grad_clip)
        self.double_dqn = bool(double_dqn)
        self.device = choose_device(device)
        self.rng = np.random.default_rng(seed)
        self.learn_calls = 0
        self.update_steps = 0
        self.model_kwargs = {
            "hidden_channels": int(hidden_channels),
            "num_res_blocks": int(num_res_blocks),
        }

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        self.online_net = self._make_network().to(self.device)
        self.target_net = self._make_network().to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.AdamW(self.online_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(replay_size, self.board_size, self.action_dim, seed=seed)

    def _make_network(self) -> DuelingGomokuQNet:
        return DuelingGomokuQNet(**self.model_kwargs)

    def select_actions(
        self,
        boards: np.ndarray,
        current_players: np.ndarray,
        action_masks: np.ndarray,
        epsilon: float = 0.0,
    ) -> np.ndarray:
        boards = ensure_batch_board(boards)
        current_players = np.asarray(current_players).reshape(-1).astype(np.int8)
        masks = np.asarray(action_masks).reshape((boards.shape[0], self.action_dim)).astype(bool)

        actions = random_legal_actions(masks, self.rng)
        greedy_indices = np.flatnonzero(self.rng.random(boards.shape[0]) >= float(epsilon))
        if len(greedy_indices) == 0:
            return actions

        states = encode_boards(boards[greedy_indices], current_players[greedy_indices])
        was_training = self.online_net.training
        self.online_net.eval()
        with torch.no_grad():
            q_values = self.online_net(torch.as_tensor(states, device=self.device))
            mask_tensor = torch.as_tensor(masks[greedy_indices], device=self.device)
            q_values = q_values.masked_fill(~mask_tensor, -1e9)
            greedy_actions = q_values.argmax(dim=1).cpu().numpy()
        if was_training:
            self.online_net.train()
        actions[greedy_indices] = greedy_actions
        return actions

    def add_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_mask: np.ndarray,
        done: bool,
        discount: Optional[float] = None,
    ) -> None:
        self.replay.add(
            state, action, reward, next_state, next_mask, done,
            self.gamma if discount is None else discount,
        )

    def train_step(
        self,
        *,
        force: bool = False,
        collect_metrics: bool = True,
    ) -> Optional[Dict[str, float]]:
        self.learn_calls += 1
        if len(self.replay) < self.min_replay_size:
            return None
        if not force and self.learn_calls % self.train_freq != 0:
            return None

        states, actions, rewards, next_states, next_masks, dones, discounts = self.replay.sample(
            self.batch_size,
            self.device,
        )

        q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.double_dqn:
                next_online_q = self.online_net(next_states).masked_fill(~next_masks, -1e9)
                next_actions = next_online_q.argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_states).masked_fill(~next_masks, -1e9)
                next_values = next_q.gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_net(next_states).masked_fill(~next_masks, -1e9)
                next_values = next_q.max(dim=1).values
            targets = rewards + discounts * next_values * (1.0 - dones)

        loss = nn.functional.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_max = 0.0
        grad_norm_sq = 0.0
        if collect_metrics:
            for parameter in self.online_net.parameters():
                if parameter.grad is None:
                    continue
                grad = parameter.grad.detach()
                grad_max = max(grad_max, float(grad.abs().max().item()))
                grad_norm_sq += float(grad.pow(2).sum().item())
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self.update_steps += 1
        if self.update_steps % self.target_update == 0:
            self.sync_target()

        if not collect_metrics:
            return None
        return {
            "loss": float(loss.item()),
            "mean_q": float(q_values.detach().mean().item()),
            "mean_target": float(targets.detach().mean().item()),
            "td_error_abs": float((targets.detach() - q_values.detach()).abs().mean().item()),
            "grad_max": grad_max,
            "grad_norm": float(grad_norm_sq ** 0.5),
        }

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save_checkpoint(
        self,
        path: Path,
        *,
        total_black_steps: int,
        history_index: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "online_state_dict": self.online_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "board_size": self.board_size,
            "action_dim": self.action_dim,
            "model_kwargs": self.model_kwargs,
            "gamma": self.gamma,
            "total_black_steps": int(total_black_steps),
            "history_index": history_index,
            "update_steps": self.update_steps,
            "extra": extra or {},
        }
        torch.save(checkpoint, str(path))

    def load_checkpoint(
        self,
        path: Path,
        *,
        load_optimizer: bool = False,
        strict: bool = True,
    ) -> Dict[str, Any]:
        # Checkpoints contain optimizer state and training metadata (including
        # pathlib.Path values), so they are intentionally full trusted training
        # checkpoints rather than weights-only files.
        checkpoint = torch.load(str(path), map_location=self.device, weights_only=False)
        model_kwargs = checkpoint.get("model_kwargs")
        if model_kwargs and model_kwargs != self.model_kwargs:
            self.model_kwargs = dict(model_kwargs)
            self.online_net = self._make_network().to(self.device)
            self.target_net = self._make_network().to(self.device)
            self.optimizer = torch.optim.AdamW(
                self.online_net.parameters(),
                lr=self.optimizer.param_groups[0]["lr"],
            )

        self.online_net.load_state_dict(checkpoint["online_state_dict"], strict=strict)
        target_state = checkpoint.get("target_state_dict", checkpoint["online_state_dict"])
        self.target_net.load_state_dict(target_state, strict=strict)
        self.target_net.eval()

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.update_steps = int(checkpoint.get("update_steps", self.update_steps))
        return checkpoint


def choose_device(device: Optional[str]) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def ensure_batch_board(boards: np.ndarray) -> np.ndarray:
    boards = np.asarray(boards)
    if boards.ndim == 2:
        boards = boards[None, ...]
    return boards.astype(np.int8, copy=False)


def encode_boards(boards: np.ndarray, current_players: np.ndarray) -> np.ndarray:
    boards = ensure_batch_board(boards)
    players = np.asarray(current_players).reshape(-1, 1, 1).astype(np.int8)
    own = boards == players
    opponent = boards == -players
    empty = boards == 0
    return np.stack([own, opponent, empty], axis=1).astype(np.float32)


def random_legal_actions(action_masks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    masks = np.asarray(action_masks).astype(bool)
    if masks.ndim == 1:
        masks = masks[None, :]

    actions = np.zeros(masks.shape[0], dtype=np.int64)
    for idx, mask in enumerate(masks):
        legal_actions = np.flatnonzero(mask)
        if len(legal_actions) == 0:
            raise RuntimeError("No legal actions available.")
        actions[idx] = int(rng.choice(legal_actions))
    return actions


def slice_obs(obs: Dict[str, np.ndarray], indices: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    index_array = np.asarray(indices, dtype=np.int64)
    return (
        obs["board"][index_array],
        obs["current_player"][index_array].reshape(-1),
        obs["action_mask"][index_array],
    )
