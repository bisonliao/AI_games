from dataclasses import asdict

import numpy as np
import torch
from torch import nn

from .config import TrainConfig
from .network import DuelingQNetwork, masked_argmax
from .replay import ReplayBatch


class DDDQNAgent:
    def __init__(self, config: TrainConfig, device: torch.device):
        self.config = config
        self.device = device
        self.online = DuelingQNetwork(config.obs_dim, config.action_dim, config.hidden_dim).to(device)
        self.target = DuelingQNetwork(config.obs_dim, config.action_dim, config.hidden_dim).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.AdamW(self.online.parameters(), lr=config.learning_rate)
        self.learner_updates = 0

    def epsilon(self, env_steps: int) -> float:
        fraction = min(1.0, max(0.0, env_steps / self.config.epsilon_decay_steps))
        return self.config.epsilon_start + fraction * (self.config.epsilon_final - self.config.epsilon_start)

    @torch.no_grad()
    def select_actions(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        epsilon: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        valid_counts = np.asarray(action_mask, dtype=np.uint8).sum(axis=1)
        if np.any(valid_counts == 0):
            raise ValueError('action_mask contains a row with no legal action')
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
        greedy = masked_argmax(self.online(obs_tensor, mask_tensor), mask_tensor).cpu().numpy()
        if epsilon <= 0:
            return greedy.astype(np.int64)
        actions = greedy.astype(np.int64)
        for row, valid in enumerate(action_mask):
            if rng.random() < epsilon:
                choices = np.flatnonzero(valid)
                if len(choices):
                    actions[row] = int(rng.choice(choices))
        return actions

    def train_batch(self, batch: ReplayBatch) -> dict[str, float]:
        obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch.next_obs, dtype=torch.float32, device=self.device)
        terminated = torch.as_tensor(batch.terminated, dtype=torch.float32, device=self.device)
        action_mask = torch.as_tensor(batch.action_mask, dtype=torch.bool, device=self.device)
        next_mask = torch.as_tensor(batch.next_action_mask, dtype=torch.bool, device=self.device)

        # Keep the legal-action dueling baseline while gathering from the
        # finite raw Q values. Masking is applied only when selecting the next
        # action; gathering a masked -inf value would create an infinite loss.
        current_q = self.online.raw(obs, action_mask).gather(
            1,
            actions.unsqueeze(1),
        ).squeeze(1)
        with torch.no_grad():
            online_next_q = self.online.raw(next_obs, next_mask)
            online_next_q = online_next_q.masked_fill(~next_mask, -torch.inf)
            next_actions = masked_argmax(online_next_q, next_mask)
            target_next_q = self.target.raw(next_obs, next_mask).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            # A terminal observation has no legal action.  Treat it as having
            # zero continuation value even if the environment wrapper hands
            # us an all-zero mask (and even if its done flag is delayed).
            target_next_q = target_next_q.masked_fill(~next_mask.any(dim=1), 0.0)
            target_q = rewards + self.config.gamma * (1.0 - terminated) * target_next_q

        td_error = target_q - current_q
        loss = nn.functional.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(nn.utils.clip_grad_norm_(self.online.parameters(), self.config.max_grad_norm))
        self.optimizer.step()
        self.learner_updates += 1
        if self.learner_updates % self.config.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return {
            'loss': float(loss.item()),
            'td_error': float(td_error.abs().mean().item()),
            'q_mean': float(current_q.mean().item()),
            'q_max': float(current_q.max().item()),
            'grad_norm': grad_norm,
        }

    def checkpoint_state(
        self,
        env_steps: int,
        best_metric: tuple[float, float, float] | None,
        config: TrainConfig,
        evaluation_metrics: dict,
    ):
        return {
            'online': self.online.state_dict(),
            'target': self.target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'env_steps': int(env_steps),
            'learner_updates': int(self.learner_updates),
            'best_metric': best_metric,
            'config': asdict(config),
            'evaluation_metrics': evaluation_metrics,
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    def load_checkpoint(self, checkpoint: dict) -> int:
        self.online.load_state_dict(checkpoint['online'])
        self.target.load_state_dict(checkpoint['target'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.learner_updates = int(checkpoint.get('learner_updates', 0))
        torch_rng_state = checkpoint.get('torch_rng_state')
        if torch_rng_state is not None:
            # torch.load(map_location='cuda') also moves RNG state tensors.
            # The CPU generator requires its byte state on CPU regardless
            # of the device used by the model and optimizer.
            torch_rng_state = torch.as_tensor(
                torch_rng_state,
                dtype=torch.uint8,
                device='cpu',
            ).contiguous()
            torch.set_rng_state(torch_rng_state)
        cuda_rng_state = checkpoint.get('cuda_rng_state')
        if torch.cuda.is_available() and cuda_rng_state is not None:
            normalized_cuda_states = [
                torch.as_tensor(state, dtype=torch.uint8, device='cpu').contiguous()
                for state in cuda_rng_state
            ]
            torch.cuda.set_rng_state_all(normalized_cuda_states)
        return int(checkpoint.get('env_steps', 0))
