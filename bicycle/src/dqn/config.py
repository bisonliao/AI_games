"""Configuration defaults for networks, replay, actors, and the learner."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class DQNConfig:
    """Immutable algorithm defaults saved into every checkpoint and run config."""

    observation_dim: int = 5
    action_dim: int = 3
    hidden_dim: int = 256
    gamma: float = 0.99
    n_step: int = 3
    replay_capacity: int = 1_000_000
    replay_warmup: int = 50_000
    batch_size: int = 512
    learning_rate: float = 1e-4
    gradient_clip_norm: float = 10.0
    replay_ratio: float = 4.0
    target_update_interval: int = 2_000
    actor_update_interval: int = 200
    actor_chunk_size: int = 256
    checkpoint_interval: int = 100_000
    evaluation_interval_steps: int = 500_000
    evaluation_episodes: int = 100

    def __post_init__(self) -> None:
        if self.observation_dim <= 0 or self.action_dim <= 1:
            raise ValueError("invalid observation or action dimensions")
        if not 0 < self.gamma <= 1 or self.n_step <= 0:
            raise ValueError("gamma and n_step must be positive")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay capacity must fit at least one batch")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "DQNConfig":
        """Load current or legacy checkpoint configuration.

        Unknown fields are intentionally ignored so checkpoints written by the
        former prioritized-replay implementation remain readable.
        """
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in allowed})


DEFAULT_EPSILONS = (0.4, 0.047, 0.006, 0.001)


def actor_epsilon(actor_id: int, actor_count: int) -> float:
    """Return an Ape-X-style exploration rate for one distributed actor.

    Diverse epsilon values let some actors explore aggressively while others
    collect trajectories close to the learner's greedy policy. This exploration
    schedule is independent of the now-uniform replay sampling policy.
    """
    if actor_count <= 1:
        return DEFAULT_EPSILONS[0]
    if actor_count == len(DEFAULT_EPSILONS):
        return DEFAULT_EPSILONS[actor_id]
    exponent = 1.0 + actor_id / (actor_count - 1) * 7.0
    return 0.4**exponent
