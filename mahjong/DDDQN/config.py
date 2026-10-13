from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrainConfig:
    total_env_steps: int = 10_000_000
    num_envs: int = 8
    eval_num_envs: int = 4
    seed: int = 1
    obs_dim: int = 306
    action_dim: int = 141
    gamma: float = 0.99
    learning_rate: float = 1e-4
    batch_size: int = 256
    replay_capacity: int = 500_000
    learning_starts: int = 20_000
    train_frequency: int = 4 #每train_frequency个时间步进行训练，训练gradient_steps个batch
    gradient_steps: int = 1
    target_update_interval: int = 2_000
    hidden_dim: int = 512
    epsilon_start: float = 1.0
    epsilon_final: float = 0.2
    epsilon_decay_steps: int = 2_000_000
    max_grad_norm: float = 10.0
    log_interval_steps: int = 10_000
    checkpoint_interval_steps: int = 1_000_000
    eval_episodes: int = 100
    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'tb_logs'
    device: str = 'auto'
    opponent_action_delay: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> 'TrainConfig':
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in values.items() if k in fields})

    def checkpoint_path(self, name: str) -> Path:
        return Path(self.checkpoint_dir) / name
