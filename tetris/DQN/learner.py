"""GPU-resident Dueling Double DQN learner."""
from __future__ import annotations

import math
import time
import random
from pathlib import Path
from typing import Any

import numpy as np

import torch
from torch import nn

from .model import DuelingDQN, masked_q_values, observations_to_torch
from .replay import ReplayBuffer, TransitionBatch
from .schedule import decay_progress, learning_rate_for_schedule
from .utils import cpu_state_dict


def double_dqn_next_values(
    online_q: torch.Tensor,
    target_q: torch.Tensor,
    action_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Select next actions online, then evaluate those actions with the target net."""
    next_actions = masked_q_values(online_q, action_mask).argmax(dim=1)
    return target_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)


class Learner:
    def __init__(
        self,
        config,
        device: str = "cuda",
        seed: int = 0,
    ) -> None:
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
        self._train_metric_sums = torch.zeros(4, device=self.device)
        self._train_metric_count = 0
        # 更新时钟独立于 IPC 消息边界。收到一个大 batch 后，trainer 会反复调用
        # update()，直到 transitions 追上这个时钟，保持每 update_every 条
        # transition 做一次梯度更新。
        self._next_update_transition = float(max(1, int(config.learning_starts)))
        self._resume_replay_warmup = False
        self._resume_transition = 0
        self._resume_warmup_target = 0
        self._throughput_start_transition = 0
        self.started_at = time.perf_counter()

    def _validate_checkpoint_config(self, checkpoint_config: dict[str, Any]) -> None:
        for key in (
            "gamma",
            "piece_placed_reward",
            "line_clear_reward",
            "terminal_penalty",
        ):
            if key not in checkpoint_config:
                continue
            current = float(getattr(self.config, key))
            saved = float(checkpoint_config[key])
            if not math.isclose(current, saved, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(
                    f"checkpoint {key}={saved} is incompatible with current {key}={current}"
                )
        saved_final_epsilon = checkpoint_config.get(
            "final_epsilon",
            checkpoint_config.get("post_trigger_epsilon"),
        )
        if saved_final_epsilon is not None and not math.isclose(
            float(self.config.final_epsilon),
            float(saved_final_epsilon),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"checkpoint final_epsilon={float(saved_final_epsilon)} is "
                f"incompatible with current final_epsilon="
                f"{float(self.config.final_epsilon)}"
            )

    def load_checkpoint(
        self,
        checkpoint: str | Path,
        *,
        total_transitions: int,
    ) -> None:
        """Warm-start from a checkpoint whose replay contents were not serialized.

        The model, target, optimizer, counters, and RNG states are restored.  The
        network is then frozen until a fresh replay buffer is
        completely populated.  Resetting the update cursor when that happens
        prevents catching up millions of updates against the new small buffer.
        """
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self._validate_checkpoint_config(payload.get("config", {}))
        self.online.load_state_dict(payload["online"], strict=True)
        self.target.load_state_dict(payload["target"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])
        # Optimizer state tensors need to follow parameters when resuming on CUDA.
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(self.device)
        self.transitions = int(payload.get("transitions", 0))
        self._throughput_start_transition = self.transitions
        self.gradient_updates = int(payload.get("gradient_updates", 0))
        if total_transitions <= self.transitions:
            raise ValueError(
                f"total_transitions ({total_transitions}) must exceed checkpoint "
                f"transitions ({self.transitions})"
            )

        self._resume_transition = self.transitions
        self._resume_warmup_target = int(self.config.replay_capacity)
        self._resume_replay_warmup = self._resume_warmup_target > 0
        warmup_end = self.transitions + self._resume_warmup_target
        self._next_update_transition = float(warmup_end)
        self._apply_learning_rate()

        if "random_state" in payload:
            random.setstate(payload["random_state"])
        if "numpy_random_state" in payload:
            np.random.set_state(payload["numpy_random_state"])
        if "torch_random_state" in payload:
            torch.set_rng_state(payload["torch_random_state"].cpu())
        if torch.cuda.is_available() and "cuda_random_state" in payload:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in payload["cuda_random_state"]]
            )
        if "replay_rng_state" in payload:
            self.replay.rng.bit_generator.state = payload["replay_rng_state"]

    def add(self, batch: TransitionBatch) -> int:
        self.replay.add_batch(batch)
        count = len(batch.actions)
        self.transitions += count
        # Linear decay advances with accepted transitions even while replay is
        # warming up and no optimizer step is performed.
        self._apply_learning_rate()
        if self._resume_replay_warmup and len(self.replay) >= self._resume_warmup_target:
            self._resume_replay_warmup = False
            # Begin from the actual post-batch step.  Do not make up updates that
            # would have occurred while the checkpoint's missing replay was rebuilt.
            self._next_update_transition = float(self.transitions)
        return count

    @property
    def replay_warming_up(self) -> bool:
        return self._resume_replay_warmup

    def decay_progress_at(self, transition: int | float | None = None) -> float:
        if transition is None:
            transition = self.transitions
        return decay_progress(transition)

    def learning_rate_at(self, transition: int | float | None = None) -> float:
        return learning_rate_for_schedule(
            self.config.learning_rate,
            self.decay_progress_at(transition),
        )

    @property
    def gamma(self) -> float:
        return float(self.config.gamma)

    def updates_per_transition_at(self, transition: float | None = None) -> float:
        if self._resume_replay_warmup:
            return 0.0
        update_every = float(self.config.update_every)
        return 1.0 / update_every

    def _update_interval_at(self, transition: float) -> float:
        updates_per_transition = self.updates_per_transition_at(transition)
        if updates_per_transition <= 0:
            return math.inf
        return 1.0 / updates_per_transition

    def _apply_learning_rate(self) -> None:
        learning_rate = self.learning_rate_at()
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

    def update(self) -> bool:
        if self._resume_replay_warmup:
            return False
        if len(self.replay) < max(self.config.learning_starts, self.config.batch_size):
            return False
        if self.transitions < self._next_update_transition:
            return False
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
            next_q = double_dqn_next_values(
                self.online(next_obs), self.target(next_obs), next_mask
            )
            target = rewards + self.gamma * (1.0 - terminated) * next_q
        loss = nn.functional.smooth_l1_loss(q, target)
        self._apply_learning_rate()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.config.gradient_clip_norm)
        self.optimizer.step()
        self.gradient_updates += 1
        with torch.no_grad():
            self._train_metric_sums += torch.stack(
                (loss.detach(), q.detach().mean(), target.detach().mean(), grad_norm.detach())
            )
        self._train_metric_count += 1
        # 关键点：按固定 cadence 前进，而不是把游标直接设为当前 transitions。
        # 后者会让“大 IPC batch”只触发一次更新，降低有效训练频率。
        self._next_update_transition += self._update_interval_at(
            self._next_update_transition
        )
        if self.gradient_updates % self.config.target_update_every == 0:
            self.target.load_state_dict(self.online.state_dict())
        return True

    def pop_training_stats(self) -> dict[str, float]:
        """Return interval metrics with a single device-to-host synchronization."""
        if not self._train_metric_count:
            return {}
        means = (self._train_metric_sums / self._train_metric_count).cpu().tolist()
        self._train_metric_sums.zero_()
        self._train_metric_count = 0
        elapsed = max(time.perf_counter() - self.started_at, 1e-6)
        return {
            "loss": means[0],
            "q_mean": means[1],
            "target_mean": means[2],
            "gradient_norm": means[3],
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "replay_size": float(len(self.replay)),
            "throughput": float(
                (self.transitions - self._throughput_start_transition) / elapsed
            ),
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
            "replay_rng_state": self.replay.rng.bit_generator.state,
        }
        if torch.cuda.is_available():
            payload["cuda_random_state"] = torch.cuda.get_rng_state_all()
        if extra:
            payload.update(extra)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
