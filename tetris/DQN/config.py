from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import tomllib


@dataclass
class TrainConfig:
    piece_placed_reward: float = 0.01
    line_clear_reward: float = 0.75
    terminal_penalty: float = 1.0
    replay_capacity: int = 1_000_000
    batch_size: int = 256
    learning_starts: int = 20_000
    update_every: int = 4
    target_update_every: int = 10_000
    broadcast_every: int = 1_000
    learning_rate: float = 1e-4
    final_epsilon: float = 0.01
    gamma: float = 0.99
    gradient_clip_norm: float = 10.0
    num_actors: int = 4
    envs_per_actor: int = 8
    queue_size: int = 32
    transition_put_poll_timeout: float = 1.0
    transition_batch_size: int = 256
    learner_idle_sleep: float = 0.001
    actor_stats_every: int = 10_000
    tb_log_every: int = 100_000
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
        if self.replay_capacity < 1 or self.batch_size < 1:
            raise ValueError("replay_capacity and batch_size must be positive")
        if self.update_every <= 0 or self.learning_rate <= 0:
            raise ValueError("update_every and learning_rate must be positive")
        if self.broadcast_every <= 0:
            raise ValueError("broadcast_every must be positive")
        if not 0 <= self.final_epsilon <= 1:
            raise ValueError("final_epsilon must be in [0, 1]")
        if not 0 <= self.gamma < 1:
            raise ValueError("gamma must be in [0, 1)")
        if self.transition_batch_size < 1:
            raise ValueError("transition_batch_size must be positive")
        if self.transition_put_poll_timeout <= 0:
            raise ValueError("transition_put_poll_timeout must be positive")
        if self.learner_idle_sleep < 0:
            raise ValueError("learner_idle_sleep must be non-negative")
