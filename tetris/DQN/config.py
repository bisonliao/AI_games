from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import tomllib


@dataclass
class TrainConfig:
    piece_placed_reward: float = 0.01
    line_clear_reward: float = 0.75
    terminal_penalty: float = 1.0
    replay_capacity: int = 500_000
    batch_size: int = 256
    learning_starts: int = 20_000
    update_every: int = 4
    target_update_every: int = 10_000
    broadcast_every: int = 1_000
    learning_rate: float = 1e-4
    gamma: float = 0.99
    gradient_clip_norm: float = 10.0
    num_actors: int = 4
    envs_per_actor: int = 8
    queue_size: int = 32
    transition_put_poll_timeout: float = 1.0
    transition_batch_size: int = 256
    transition_batch_max_wait: float = 0.1
    transition_get_poll_timeout: float = 0.0
    learner_idle_sleep: float = 0.001
    actor_stats_every: int = 10_000
    tb_log_every: int = 10_000
    total_transitions: int = 5_000_000
    checkpoint_every: int = 250_000
    eval_every: int = 250_000
    eval_episodes: int = 5
    eval_max_steps: int = 5_000
    eval_device: str = "cpu"
    max_pending_evals: int = 2
    eval_shutdown_timeout: float = 30.0
    log_root: str = "runs/dddqn"
    checkpoint_root: str = "checkpoints"
    seed: int = 0

    @classmethod
    def from_toml(cls, path: str | Path) -> "TrainConfig":
        with open(path, "rb") as f:
            data = tomllib.load(f)
        section = data.get("train", data)
        return cls(**{k: v for k, v in section.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        if self.piece_placed_reward < 0 or self.line_clear_reward < 0 or self.terminal_penalty < 0:
            raise ValueError("reward magnitudes must be non-negative")
