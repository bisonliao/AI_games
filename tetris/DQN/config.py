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
    epsilon_start_min: float = 0.05
    epsilon_start_max: float = 0.40
    epsilon_final_min: float = 0.01
    epsilon_final_max: float = 0.15
    epsilon_decay_transitions: int = 1_500_000
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
        epsilon_values = (
            self.epsilon_start_min,
            self.epsilon_start_max,
            self.epsilon_final_min,
            self.epsilon_final_max,
        )
        if not all(0.0 <= value <= 1.0 for value in epsilon_values):
            raise ValueError("epsilon values must be in [0, 1]")
        if self.epsilon_start_min > self.epsilon_start_max:
            raise ValueError("epsilon_start_min must not exceed epsilon_start_max")
        if self.epsilon_final_min > self.epsilon_final_max:
            raise ValueError("epsilon_final_min must not exceed epsilon_final_max")
        if self.epsilon_final_min > self.epsilon_start_min or self.epsilon_final_max > self.epsilon_start_max:
            raise ValueError("final epsilon range must not exceed the start range")
        if self.epsilon_decay_transitions <= 0:
            raise ValueError("epsilon_decay_transitions must be positive")
