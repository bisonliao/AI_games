from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def resolve_device(device: str) -> torch.device:
    """Resolve a learner device, failing clearly when CUDA was requested."""
    resolved = torch.device(device)
    # bison updated manually
    '''if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested for the learner, but PyTorch cannot access a CUDA "
            "device. Fix the NVIDIA/WSL device passthrough or run with --device cpu."
        )'''
    return resolved


class Actor(nn.Module):
    def __init__(self, obs_dim: int = 1, action_dim: int = 1, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


class Critic(nn.Module):
    def __init__(self, obs_dim: int = 1, action_dim: int = 1, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((observation, action), dim=-1))


@dataclass(slots=True)
class TD3Update:
    critic_loss: float
    actor_loss: float | None


class BanditTD3:
    """TD3 adapted to a terminal, one-action contextual bandit.

    There is deliberately no target network or bootstrapped next-state value:
    both critics regress directly onto the observed immediate reward.
    """

    def __init__(
        self,
        *,
        obs_dim: int = 1,
        action_dim: int = 1,
        hidden_dim: int = 128,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        policy_delay: int = 2,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = resolve_device(str(device))
        self.actor = Actor(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic1 = Critic(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic2 = Critic(obs_dim, action_dim, hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=critic_lr,
        )
        self.policy_delay = policy_delay
        self.update_count = 0

    @torch.inference_mode()
    def act(self, observations: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        )
        return self.actor(tensor).cpu().numpy()

    def update(self, batch: Mapping[str, np.ndarray]) -> TD3Update:
        """用 terminal transition 更新双 critic，并按延迟频率更新 actor。"""
        # 这个环境每回合只有一个动作，所有样本都在动作后终止。因此 critic
        # target 就是即时 reward，不读取 next_observations，也不存在 gamma、
        # target actor、target critic 或 target-policy smoothing。
        observations = torch.as_tensor(
            batch["observations"], dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            batch["actions"], dtype=torch.float32, device=self.device
        )
        rewards = torch.as_tensor(
            batch["rewards"], dtype=torch.float32, device=self.device
        ).reshape(-1, 1)

        # 两个 critic 独立估计同一个 Q(s, a)=reward 回归目标。双 critic 虽然
        # 不再用于抑制 bootstrap 误差，仍可降低 actor 利用单一函数近似误差的风险。
        q1 = self.critic1(observations, actions)
        q2 = self.critic2(observations, actions)
        critic_loss = F.mse_loss(q1, rewards) + F.mse_loss(q2, rewards)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        self.update_count += 1
        actor_loss_value: float | None = None
        # 延迟 actor 更新让 critic 先跟上不断变化的 replay 数据分布。
        if self.update_count % self.policy_delay == 0:
            # actor 反向传播只需要 Q 对 action 的梯度；冻结 critic 参数可避免
            # 生成无用的 critic parameter gradients，同时保留对 action 的梯度链。
            for parameter in list(self.critic1.parameters()) + list(
                self.critic2.parameters()
            ):
                parameter.requires_grad_(False)
            predicted_actions = self.actor(observations)
            # 最大化两个 critic 中较保守的估计，减少 actor 钻某一个 critic
            # 近似误差的空子；负号将最大化目标转换成优化器的最小化 loss。
            conservative_q = torch.minimum(
                self.critic1(observations, predicted_actions),
                self.critic2(observations, predicted_actions),
            )
            actor_loss = -conservative_q.mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            for parameter in list(self.critic1.parameters()) + list(
                self.critic2.parameters()
            ):
                parameter.requires_grad_(True)
            actor_loss_value = float(actor_loss.detach().cpu())

        return TD3Update(
            critic_loss=float(critic_loss.detach().cpu()),
            actor_loss=actor_loss_value,
        )

    def actor_weights_numpy(self) -> dict[str, np.ndarray]:
        return {
            key: value.detach().cpu().numpy().copy()
            for key, value in self.actor.state_dict().items()
        }

    def load_actor_weights_numpy(self, weights: Mapping[str, np.ndarray]) -> None:
        state = {key: torch.as_tensor(value) for key, value in weights.items()}
        self.actor.load_state_dict(state)

    def save(
        self,
        path: str | Path,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "update_count": self.update_count,
                "metadata": dict(metadata or {}),
            },
            path,
        )

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, *, device: str = "cpu"
    ) -> tuple[BanditTD3, dict[str, Any]]:
        resolve_device(device)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        metadata = checkpoint.get("metadata", {})
        model_config = metadata.get("model_config", {})
        agent = cls(device=device, **model_config)
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.critic1.load_state_dict(checkpoint["critic1"])
        agent.critic2.load_state_dict(checkpoint["critic2"])
        agent.update_count = int(checkpoint.get("update_count", 0))
        return agent, metadata
