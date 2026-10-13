import torch as th
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.dqn.policies import DQNPolicy

class VectorExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=int(observation_space['vector'].shape[0]))
        self._features_dim = int(observation_space['vector'].shape[0])
    def forward(self, observations: PyTorchObs) -> th.Tensor:
        return observations['vector'].float()

class MaskedDuelingQNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__(); self.action_dim = action_dim
        self.trunk = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.value = nn.Linear(hidden_dim, 1); self.advantage = nn.Linear(hidden_dim, action_dim)
    def raw(self, obs: PyTorchObs) -> th.Tensor:
        """Return finite Q values using the legal-action dueling baseline."""
        hidden = self.trunk(obs['vector'].float())
        value = self.value(hidden)
        advantage = self.advantage(hidden)
        mask = obs['action_mask'].bool()
        mean_advantage = (advantage * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        return value + advantage - mean_advantage
    def forward(self, obs: PyTorchObs) -> th.Tensor:
        mask = obs['action_mask'].bool()
        if not bool(mask.any(dim=1).all()):
            raise RuntimeError('received an observation with an empty action mask')
        q = self.raw(obs)
        return q.masked_fill(~mask, -th.inf)
    def _predict(self, obs: PyTorchObs, deterministic: bool = True) -> th.Tensor:
        mask = obs['action_mask'].bool()
        if not bool(mask.any(dim=1).all()):
            raise RuntimeError('received an observation with an empty action mask')
        q = self.forward(obs)
        if deterministic: return q.argmax(1).reshape(-1, 1)
        choices = [th.where(row)[0][th.randint(int(row.sum()), (1,), device=row.device)] for row in mask]
        return th.stack(choices).reshape(-1, 1)
    def set_training_mode(self, mode: bool): self.train(mode)

class MaskedDuelingPolicy(DQNPolicy):
    def __init__(self, *args, hidden_dim: int = 512, **kwargs):
        self.hidden_dim = hidden_dim
        kwargs['features_extractor_class'] = VectorExtractor
        super().__init__(*args, **kwargs)

    def make_q_net(self):
        return MaskedDuelingQNetwork(
            int(self.observation_space['vector'].shape[0]),
            int(self.action_space.n),
            self.hidden_dim,
        ).to(self.device)

    def _predict(self, observation, deterministic=True):
        return self.q_net._predict(observation, deterministic)

    def forward(self, observation, deterministic=True): return self._predict(observation, deterministic)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters(); data['hidden_dim'] = self.hidden_dim; return data
