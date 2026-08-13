"""GPU-resident Dueling Double DQN learner."""
from __future__ import annotations

import time
import random

import numpy as np

import torch
from torch import nn

from .model import DuelingDQN, masked_q_values, observations_to_torch
from .replay import ReplayBuffer, TransitionBatch
from .utils import cpu_state_dict


class Learner:
    def __init__(self, config, device: str = "cuda", seed: int = 0) -> None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
        self.device = torch.device(device)
        self.config = config
        self.num_actions = 40
        self.online = DuelingDQN().to(self.device)
        self.target = DuelingDQN().to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.replay = ReplayBuffer(config.replay_capacity, seed=seed)
        self.gradient_updates = 0
        self.transitions = 0
        # 更新时钟独立于 IPC 消息边界。收到一个大 batch 后，trainer 会反复调用
        # update()，直到 transitions 追上这个时钟，保持每 update_every 条
        # transition 做一次梯度更新。
        self._next_update_transition = max(1, int(config.learning_starts))
        self.started_at = time.perf_counter()

    def add(self, batch: TransitionBatch) -> int:
        self.replay.add_batch(batch)
        count = len(batch.actions)
        self.transitions += count
        return count

    def update(self) -> dict[str, float] | None:
        if len(self.replay) < max(self.config.learning_starts, self.config.batch_size):
            return None
        if self.transitions < self._next_update_transition:
            return None
        batch = self.replay.sample(self.config.batch_size)
        obs = observations_to_torch(batch.obs, self.device)
        next_obs = observations_to_torch(batch.next_obs, self.device)
        actions = torch.as_tensor(batch.actions, device=self.device, dtype=torch.long)
        rewards = torch.as_tensor(batch.rewards, device=self.device, dtype=torch.float32)
        terminated = torch.as_tensor(batch.terminated, device=self.device, dtype=torch.float32)
        q = self.online(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_mask = next_obs.get("action_mask")
            # Auto-reset vector environments return the reset observation after a
            # terminal transition.  Its mask is harmless because bootstrap is
            # multiplied by zero for terminated samples.
            next_actions = masked_q_values(self.online(next_obs), next_mask).argmax(dim=1)
            next_q = self.target(next_obs).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rewards + self.config.gamma * (1.0 - terminated) * next_q
        loss = nn.functional.smooth_l1_loss(q, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.config.gradient_clip_norm))
        self.optimizer.step()
        self.gradient_updates += 1
        # 关键点：按固定 cadence 前进，而不是把游标直接设为当前 transitions。
        # 后者会让“大 IPC batch”只触发一次更新，降低有效训练频率。
        self._next_update_transition += self.config.update_every
        if self.gradient_updates % self.config.target_update_every == 0:
            self.target.load_state_dict(self.online.state_dict())
        elapsed = max(time.perf_counter() - self.started_at, 1e-6)
        return {
            "loss": float(loss.item()),
            "q_mean": float(q.mean().item()),
            "target_mean": float(target.mean().item()),
            "gradient_norm": grad_norm,
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "replay_size": float(len(self.replay)),
            "throughput": float(self.transitions / elapsed),
        }

    def state_dict_cpu(self) -> dict[str, object]:
        return cpu_state_dict(self.online)

    def checkpoint(self, path, extra: dict | None = None) -> None:
        payload = {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "transitions": self.transitions,
            "gradient_updates": self.gradient_updates,
            "next_update_transition": self._next_update_transition,
            "config": self.config.to_dict(),
            "random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            payload["cuda_random_state"] = torch.cuda.get_rng_state_all()
        if extra:
            payload.update(extra)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
