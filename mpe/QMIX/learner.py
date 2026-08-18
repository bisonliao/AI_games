"""QMIX learner and shared recurrent multi-agent controller."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .networks import QMixer, RecurrentAgent


@dataclass(frozen=True)
class LearnerConfig:
    n_agents: int
    obs_dim: int
    state_dim: int
    n_actions: int
    hidden_dim: int = 64
    mixing_embed_dim: int = 32
    hypernet_embed_dim: int = 64
    gamma: float = 0.95
    lr: float = 1e-4
    optimizer_alpha: float = 0.99
    optimizer_eps: float = 1e-5
    grad_norm_clip: float = 10.0
    double_q: bool = True
    td_loss: str = "huber"
    huber_delta: float = 1.0
    max_abs_q: float = 1_000.0


class QMIXLearner:
    def __init__(self, config: LearnerConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        if config.td_loss not in ("huber", "mse"):
            raise ValueError(f"unsupported TD loss: {config.td_loss}")
        if config.huber_delta <= 0:
            raise ValueError("huber_delta must be positive")
        # Local observation + previous one-hot action + one-hot agent id.
        input_dim = config.obs_dim + config.n_actions + config.n_agents
        self.agent = RecurrentAgent(
            input_dim, config.hidden_dim, config.n_actions
        ).to(device)
        self.mixer = QMixer(
            config.n_agents,
            config.state_dim,
            config.mixing_embed_dim,
            config.hypernet_embed_dim,
        ).to(device)
        self.target_agent = deepcopy(self.agent).to(device)
        self.target_mixer = deepcopy(self.mixer).to(device)
        parameters = list(self.agent.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.RMSprop(
            parameters,
            lr=config.lr,
            alpha=config.optimizer_alpha,
            eps=config.optimizer_eps,
        )
        self._parameters = parameters
        self.train_updates = 0

    def initial_hidden(self) -> torch.Tensor:
        return self.agent.initial_hidden(self.config.n_agents, self.device)

    def select_actions(
        self,
        observations: torch.Tensor,
        last_actions: torch.Tensor,
        hidden: torch.Tensor,
        epsilon: float,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One recurrent controller step for all agents."""

        agent_ids = torch.eye(self.config.n_agents, device=self.device)
        inputs = torch.cat((observations, last_actions, agent_ids), dim=-1)
        with torch.no_grad():
            q_values, next_hidden = self.agent(inputs, hidden)
            greedy_actions = q_values.argmax(dim=-1)
            if deterministic or epsilon <= 0.0:
                actions = greedy_actions
            else:
                random_actions = torch.randint(
                    self.config.n_actions,
                    (self.config.n_agents,),
                    device=self.device,
                )
                choose_random = torch.rand(
                    self.config.n_agents, device=self.device
                ) < float(epsilon)
                actions = torch.where(
                    choose_random, random_actions, greedy_actions
                )
        return actions, next_hidden

    def _sequence_q(
        self,
        network: RecurrentAgent,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, n_agents, _ = observations.shape
        last_actions = torch.zeros(
            batch_size,
            sequence_length,
            n_agents,
            self.config.n_actions,
            device=observations.device,
        )
        if sequence_length > 1:
            last_actions[:, 1:] = F.one_hot(
                actions, num_classes=self.config.n_actions
            ).to(dtype=observations.dtype)
        agent_ids = torch.eye(
            n_agents, device=observations.device, dtype=observations.dtype
        ).view(1, 1, n_agents, n_agents)
        agent_ids = agent_ids.expand(batch_size, sequence_length, -1, -1)
        inputs = torch.cat((observations, last_actions, agent_ids), dim=-1)

        hidden = network.initial_hidden(batch_size * n_agents, observations.device)
        outputs = []
        for timestep in range(sequence_length):
            q_values, hidden = network(
                inputs[:, timestep].reshape(batch_size * n_agents, -1),
                hidden,
            )
            outputs.append(
                q_values.view(batch_size, n_agents, self.config.n_actions)
            )
        return torch.stack(outputs, dim=1)

    def train(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        observations = batch["observations"]
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        terminated = batch["terminated"]
        mask = batch["filled"]

        online_q = self._sequence_q(self.agent, observations, actions)
        chosen_q = torch.gather(
            online_q[:, :-1], dim=3, index=actions.unsqueeze(-1)
        ).squeeze(-1)

        with torch.no_grad():
            target_q_all = self._sequence_q(
                self.target_agent, observations, actions
            )[:, 1:]
            if self.config.double_q:
                online_next_actions = online_q[:, 1:].argmax(
                    dim=3, keepdim=True
                )
                target_agent_q = torch.gather(
                    target_q_all, dim=3, index=online_next_actions
                ).squeeze(-1)
            else:
                target_agent_q = target_q_all.max(dim=3).values

        chosen_total_q = self.mixer(chosen_q, states[:, :-1])
        with torch.no_grad():
            target_total_q = self.target_mixer(target_agent_q, states[:, 1:])
            targets = rewards + self.config.gamma * (1.0 - terminated) * target_total_q

        td_error = chosen_total_q - targets
        valid = mask.bool().expand_as(chosen_total_q)
        valid_chosen_q = chosen_total_q.detach()[valid]
        valid_target_q = targets.detach()[valid]
        valid_td_error = td_error.detach()[valid]
        finite = (
            torch.isfinite(valid_chosen_q).all()
            & torch.isfinite(valid_target_q).all()
            & torch.isfinite(valid_td_error).all()
        )
        chosen_q_abs_max = valid_chosen_q.abs().max()
        target_q_abs_max = valid_target_q.abs().max()
        q_abs_max = torch.maximum(chosen_q_abs_max, target_q_abs_max)
        if not bool(finite):
            raise FloatingPointError(
                "QMIX divergence guard: Q values or TD errors became non-finite"
            )
        if self.config.max_abs_q > 0 and q_abs_max.item() > self.config.max_abs_q:
            raise FloatingPointError(
                "QMIX divergence guard: |Q| reached {:.6g}, exceeding "
                "max_abs_q={:.6g}".format(q_abs_max.item(), self.config.max_abs_q)
            )

        if self.config.td_loss == "huber":
            elementwise_loss = F.huber_loss(
                chosen_total_q,
                targets,
                reduction="none",
                delta=self.config.huber_delta,
            )
        else:
            elementwise_loss = td_error.square()
        loss = (elementwise_loss * mask).sum() / mask.sum().clamp(min=1.0)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                "QMIX divergence guard: TD loss became non-finite"
            )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        pre_clip_grad_norm = nn.utils.clip_grad_norm_(
            self._parameters,
            self.config.grad_norm_clip,
            error_if_nonfinite=True,
        )
        squared_post_clip_norms = [
            parameter.grad.detach().norm(2).square()
            for parameter in self._parameters
            if parameter.grad is not None
        ]
        post_clip_grad_norm = (
            torch.stack(squared_post_clip_norms).sum().sqrt()
            if squared_post_clip_norms
            else torch.zeros((), device=self.device)
        )
        self.optimizer.step()
        self.train_updates += 1

        return {
            "loss": float(loss.detach().item()),
            # Keep the old key for consumers while exposing unambiguous tags.
            "grad_norm": float(pre_clip_grad_norm.detach().item()),
            "pre_clip_grad_norm": float(pre_clip_grad_norm.detach().item()),
            "post_clip_grad_norm": float(post_clip_grad_norm.detach().item()),
            "chosen_q": float(valid_chosen_q.mean().item()),
            "target_q": float(valid_target_q.mean().item()),
            "chosen_q_abs_max": float(chosen_q_abs_max.item()),
            "target_q_abs_max": float(target_q_abs_max.item()),
            "td_error_abs": float(valid_td_error.abs().mean().item()),
        }

    def update_targets(self) -> None:
        self.target_agent.load_state_dict(self.agent.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def checkpoint_state(self) -> dict:
        return {
            "agent": self.agent.state_dict(),
            "mixer": self.mixer.state_dict(),
            "target_agent": self.target_agent.state_dict(),
            "target_mixer": self.target_mixer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_updates": int(self.train_updates),
        }

    def load_checkpoint_state(self, state: dict, load_optimizer: bool = True) -> None:
        self.agent.load_state_dict(state["agent"])
        self.mixer.load_state_dict(state["mixer"])
        self.target_agent.load_state_dict(state["target_agent"])
        self.target_mixer.load_state_dict(state["target_mixer"])
        if load_optimizer:
            self.optimizer.load_state_dict(state["optimizer"])
        self.train_updates = int(state.get("train_updates", 0))
