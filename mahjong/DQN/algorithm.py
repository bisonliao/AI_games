import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import DQN
from .policy import MaskedDuelingPolicy

class MaskedDoubleDQN(DQN):
    policy_aliases = {"MlpPolicy": MaskedDuelingPolicy, "MultiInputPolicy": MaskedDuelingPolicy}
    def __init__(self, policy, env, **kwargs):
        kwargs.setdefault('policy_kwargs', {})
        kwargs['policy_kwargs'] = dict(kwargs['policy_kwargs'])
        kwargs['policy_kwargs'].setdefault('hidden_dim', 512)
        super().__init__(policy, env, **kwargs)

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        if self._last_obs is None:
            raise RuntimeError('missing current observation')
        masks = np.asarray(self._last_obs['action_mask'], dtype=bool)
        if masks.ndim == 1: masks = masks[None, :]
        if not masks.any(axis=1).all():
            raise RuntimeError('empty action mask in rollout observation')
        if self.num_timesteps < learning_starts:
            actions = np.asarray(
                [np.random.choice(np.flatnonzero(mask)) for mask in masks],
                dtype=np.int64,
            )
        else:
            actions, _ = self.predict(self._last_obs, deterministic=False)
            actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        return actions, actions.copy()

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        if not deterministic and np.random.random() < self.exploration_rate:
            masks = np.asarray(observation['action_mask'], dtype=bool)
            if masks.ndim == 1: masks = masks[None, :]
            if not masks.any(axis=1).all():
                raise RuntimeError('empty action mask during exploration')
            actions = np.asarray(
                [np.random.choice(np.flatnonzero(mask)) for mask in masks],
                dtype=np.int64,
            )
            if observation['action_mask'].ndim == 1: actions = actions[0]
            return actions, state
        return super().predict(observation, state, episode_start, deterministic=True)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses = []
        td_values = []
        grad_values = []
        q_values = []
        for _ in range(gradient_steps):
            data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            obs, next_obs = data.observations, data.next_observations
            with th.no_grad():
                next_mask = next_obs['action_mask'].bool()
                online_next = self.policy.q_net.raw(next_obs).masked_fill(~next_mask, -th.inf)
                next_actions = online_next.argmax(1, keepdim=True)
                target_next = self.policy.q_net_target.raw(next_obs).gather(1, next_actions).squeeze(1)
                target_next = target_next.masked_fill(~next_mask.any(1), 0.0)
                targets = data.rewards.flatten() + (1 - data.dones.flatten()) * self.gamma * target_next
            current_all = self.policy.q_net.raw(obs)
            current = current_all.gather(1, data.actions.long()).squeeze(1)
            td = targets - current
            loss = F.smooth_l1_loss(current, targets)
            self.policy.optimizer.zero_grad()
            loss.backward()
            grad = float(th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm))
            self.policy.optimizer.step()
            losses.append(float(loss.item()))
            td_values.append(float(td.abs().mean().detach()))
            grad_values.append(grad)
            q_values.append(float(current.mean().detach()))
        self._n_updates += gradient_steps
        self.logger.record('train/loss', np.mean(losses))
        self.logger.record('train/td_error', np.mean(td_values))
        self.logger.record('train/q_mean', np.mean(q_values))
        self.logger.record('train/q_max', float(current.max().detach().item()))
        self.logger.record('train/grad_norm', np.mean(grad_values))
        self.logger.record('train/n_updates', self._n_updates)
