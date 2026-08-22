"""GPU learner, Double-DQN update, and checkpoint persistence."""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .common import (
    EXPECTED_BC_SHA256,
    TransitionPacket,
    atomic_torch_save,
    cpu_state_dict,
    encode_boards,
)
from .network import DuelingGomokuQNet, make_dueling_from_bc
from .replay import PrioritizedReplayBuffer


class DQNLearner:
    def __init__(
        self,
        bc_checkpoint: Path,
        *,
        device: str = "cuda",
        board_size: int = 9,
        lr: float = 1e-4,
        gamma: float = 0.99,
        batch_size: int = 256,
        replay_size: int = 1_000_000,
        min_replay_size: int = 50_000,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        target_update: int = 2_500,
        grad_clip: float = 10.0,
        seed: int = 0,
    ) -> None:
        self.device = torch.device(device)
        self.board_size = int(board_size)
        self.action_dim = board_size * board_size
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.min_replay_size = int(min_replay_size)
        self.per_beta_start = float(per_beta_start)
        self.target_update = int(target_update)
        self.grad_clip = float(grad_clip)
        self.rng = np.random.default_rng(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        self.online_net, bc_metadata = make_dueling_from_bc(bc_checkpoint, device=self.device)
        self.model_kwargs = dict(bc_metadata["model_kwargs"])
        self.target_net = DuelingGomokuQNet(**self.model_kwargs).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.AdamW(self.online_net.parameters(), lr=lr)
        self.replay = PrioritizedReplayBuffer(
            replay_size, board_size, alpha=per_alpha, seed=seed
        )
        self.global_step = 0
        self.update_steps = 0
        self.update_credit = 0.0

    def add_packet(self, packet: TransitionPacket, global_step: int) -> None:
        self.replay.add_packet(packet)
        self.global_step = int(global_step)

    def beta(self, total_training_steps: int) -> float:
        if total_training_steps <= 0:
            return 1.0
        fraction = min(1.0, max(0.0, self.global_step / float(total_training_steps)))
        return self.per_beta_start + fraction * (1.0 - self.per_beta_start)

    def train_step(self, *, total_training_steps: int) -> dict[str, float]:
        if len(self.replay) < max(self.min_replay_size, self.batch_size):
            raise RuntimeError("replay has not reached min_replay_size")
        beta = self.beta(total_training_steps)
        sample = self.replay.sample(self.batch_size, beta, augment=True)
        states = torch.from_numpy(encode_boards(sample.states, -1)).to(self.device)
        actions = torch.from_numpy(sample.actions).long().to(self.device)
        rewards = torch.from_numpy(sample.rewards).to(self.device)
        next_states = torch.from_numpy(encode_boards(sample.next_states, -1)).to(self.device)
        next_masks = torch.from_numpy(sample.next_masks).to(self.device)
        dones = torch.from_numpy(sample.dones).to(self.device)
        discounts = torch.from_numpy(sample.discounts).to(self.device)
        weights = torch.from_numpy(sample.weights).to(self.device)

        self.online_net.train()
        q_values = self.online_net(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            self.online_net.eval()
            next_online = self.online_net(next_states).masked_fill(~next_masks, -1e9)
            next_actions = next_online.argmax(1, keepdim=True)
            next_target = self.target_net(next_states).masked_fill(~next_masks, -1e9)
            next_values = next_target.gather(1, next_actions).squeeze(1)
            next_values = torch.where(dones, torch.zeros_like(next_values), next_values)
            targets = rewards + discounts * next_values
        self.online_net.train()
        td_errors = targets - q_values
        losses = nn.functional.smooth_l1_loss(q_values, targets, reduction="none")
        loss = (losses * weights).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(nn.utils.clip_grad_norm_(
            self.online_net.parameters(), self.grad_clip
        ).item()) if self.grad_clip > 0 else 0.0
        self.optimizer.step()
        self.update_steps += 1
        if self.update_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
            self.target_net.eval()
        td_numpy = td_errors.detach().abs().cpu().numpy()
        self.replay.update_priorities(sample.indices, td_numpy)
        return {
            "loss": float(loss.item()),
            "mean_q": float(q_values.detach().mean().item()),
            "mean_target": float(targets.mean().item()),
            "mean_abs_td_error": float(td_numpy.mean()),
            "max_abs_td_error": float(td_numpy.max()),
            "mean_importance_weight": float(sample.weights.mean()),
            "grad_norm": grad_norm,
            "beta": beta,
        }

    def snapshot(self) -> dict[str, np.ndarray]:
        return cpu_state_dict(self.online_net)

    def checkpoint_payload(self, config: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format_version": 1,
            "algorithm": "Ape-X Dueling Double DQN",
            "board_size": self.board_size,
            "model_kwargs": self.model_kwargs,
            "online_state_dict": self.online_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "update_steps": self.update_steps,
            "update_credit": self.update_credit,
            "bc_sha256": EXPECTED_BC_SHA256,
            "config": config,
            "numpy_rng_state": self.rng.bit_generator.state,
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
        }
        if self.device.type == "cuda":
            payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        return payload

    def save_checkpoint(self, path: Path, config: dict[str, Any], *, latest: Path | None = None) -> None:
        atomic_torch_save(self.checkpoint_payload(config), path)
        if latest is not None:
            temporary = latest.with_name(f".{latest.name}.tmp")
            latest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, temporary)
            temporary.replace(latest)

    def load_checkpoint(self, path: Path) -> dict[str, Any]:
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=False)
        if int(checkpoint.get("board_size", -1)) != self.board_size:
            raise ValueError("resume checkpoint board size does not match")
        if checkpoint.get("model_kwargs") != self.model_kwargs:
            raise ValueError("resume checkpoint model configuration does not match BC_BEST")
        if checkpoint.get("bc_sha256") != EXPECTED_BC_SHA256:
            raise ValueError("resume checkpoint was initialized from a different BC model")
        self.online_net.load_state_dict(checkpoint["online_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.target_net.eval()
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = int(checkpoint.get("global_step", 0))
        self.update_steps = int(checkpoint.get("update_steps", 0))
        self.update_credit = float(checkpoint.get("update_credit", 0.0))
        if "numpy_rng_state" in checkpoint:
            self.rng.bit_generator.state = checkpoint["numpy_rng_state"]
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if self.device.type == "cuda" and "cuda_rng_state_all" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        return checkpoint
