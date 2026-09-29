"""SB3 TD3 subclass implementing the offline TD3+BC update."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import TD3
from stable_baselines3.common.utils import polyak_update

from .augmentation import ConsistentMultiViewAugmentation
from .dataset import OfflineDataset


class OfflineTD3BC(TD3):
    def __init__(
        self,
        *args: Any,
        alpha: float = 2.5,
        bc_coef: float = 1.0,
        augmentation_shift: int = 4,
        **kwargs: Any,
    ):
        self.alpha = float(alpha)
        self.bc_coef = float(bc_coef)
        self.augmentation_shift = int(augmentation_shift)
        self.image_augmentation = ConsistentMultiViewAugmentation(
            self.augmentation_shift
        )
        self.last_metrics: dict[str, float] = {}
        # SB3 constructs a replay buffer during setup. Keep it tiny: all real
        # transitions come from OfflineDataset.sample().
        kwargs.setdefault("buffer_size", 1)
        kwargs.setdefault("learning_starts", 0)
        super().__init__(*args, **kwargs)

    @staticmethod
    def _gradient_norm(module) -> float:
        norms = [
            parameter.grad.detach().norm(2)
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        if not norms:
            return 0.0
        return float(th.stack(norms).norm(2).detach().cpu())

    def offline_update(self, dataset: OfflineDataset, batch_size: int) -> dict[str, float]:
        self.policy.set_training_mode(True)
        if hasattr(self, "_logger"):
            self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])
        else:
            learning_rate = self.lr_schedule(self._current_progress_remaining)
            for optimizer in (self.actor.optimizer, self.critic.optimizer):
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
        batch = dataset.sample(batch_size, self.device)
        self._n_updates += 1
        # K=1/M=1 DrQ: one independent, geometry-consistent shift for each
        # target, critic-current, and actor-current observation path.
        next_observations = self._augment_observation(batch.next_observations)
        with th.no_grad():
            noise = th.randn_like(batch.actions) * self.target_policy_noise
            noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
            next_actions = (self.actor_target(next_observations) + noise).clamp(-1.0, 1.0)
            target_q = th.cat(self.critic_target(next_observations, next_actions), dim=1).min(dim=1, keepdim=True).values
            target_q = batch.rewards + (1.0 - batch.terminals) * self.gamma * target_q
        observations = self._augment_observation(batch.observations)
        current_qs = self.critic(observations, batch.actions)
        critic_loss = sum(F.mse_loss(current, target_q) for current in current_qs)
        self.critic.optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = self._gradient_norm(self.critic)
        self.critic.optimizer.step()
        current_q_tensor = th.cat(current_qs, dim=1)
        td_error = current_q_tensor - target_q
        metrics = {
            "critic_loss": float(critic_loss.detach().cpu()),
            "target_q_mean": float(target_q.mean().detach().cpu()),
            "target_q_std": float(target_q.std(unbiased=False).detach().cpu()),
            "current_q_mean": float(current_q_tensor.mean().detach().cpu()),
            "current_q_std": float(current_q_tensor.std(unbiased=False).detach().cpu()),
            "td_error_abs_mean": float(td_error.abs().mean().detach().cpu()),
            "q1_q2_gap": float((current_qs[0] - current_qs[1]).abs().mean().detach().cpu()),
            "critic_grad_norm": critic_grad_norm,
            "dataset_action_abs_mean": float(batch.actions.abs().mean().detach().cpu()),
            "stage_mean": float(batch.stages.float().mean().detach().cpu()),
            "actor_loss": float("nan"),
            "bc_loss": float("nan"),
            "lambda": float("nan"),
            "actor_grad_norm": float("nan"),
            "actor_action_abs_mean": float("nan"),
            "actor_action_saturation_rate": float("nan"),
            "actor_updated": 0.0,
        }
        if self._n_updates % self.policy_delay == 0:
            actor_observations = self._augment_observation(batch.observations)
            predicted = self.actor(actor_observations)
            q_value = self.critic.q1_forward(actor_observations, predicted)
            scale = self.alpha / (q_value.abs().mean().detach() + 1e-6)
            bc_loss = F.mse_loss(predicted, batch.actions)
            actor_loss = -scale * q_value.mean() + self.bc_coef * bc_loss
            self.actor.optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad_norm = self._gradient_norm(self.actor)
            self.actor.optimizer.step()
            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
            polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
            polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
            polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)
            metrics.update(
                {
                    "actor_loss": float(actor_loss.detach().cpu()),
                    "bc_loss": float(bc_loss.detach().cpu()),
                    "lambda": float(scale.detach().cpu()),
                    "actor_grad_norm": actor_grad_norm,
                    "actor_action_abs_mean": float(predicted.abs().mean().detach().cpu()),
                    "actor_action_saturation_rate": float(
                        (predicted.abs() >= 0.99).float().mean().detach().cpu()
                    ),
                    "actor_updated": 1.0,
                }
            )
        for stage_id in range(5):
            metrics[f"stage_{stage_id}_fraction"] = float(
                (batch.stages == stage_id).float().mean().detach().cpu()
            )
        metrics["augmentation_shift"] = float(self.augmentation_shift)
        metrics["augmentation_enabled"] = float(self.augmentation_shift > 0)
        self.last_metrics = metrics
        return metrics

    def _augment_observation(self, observation: dict[str, th.Tensor]) -> dict[str, th.Tensor]:
        result = dict(observation)
        result["image"] = self.image_augmentation(result["image"])
        return result

    def _excluded_save_params(self) -> list[str]:
        return [*super()._excluded_save_params(), "image_augmentation"]

    def fit(self, dataset: OfflineDataset, total_updates: int, batch_size: int) -> list[dict[str, float]]:
        if total_updates < 1:
            raise ValueError("total_updates must be positive")
        return [self.offline_update(dataset, batch_size) for _ in range(total_updates)]
