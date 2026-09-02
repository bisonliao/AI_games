"""Central configuration for training, rollout, evaluation, and storage."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path

from PacManEnv import MsPacmanEnvConfig


DQN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DQN_DIR.parent


def default_actor_env_config() -> MsPacmanEnvConfig:
    return MsPacmanEnvConfig(
        num_envs=8,
        frame_skip=4,
        frame_stack=4,
        screen_size=84,
        repeat_action_probability=0.25,
        noop_max=30,
        mode=0,
        difficulty=0,
        step_cost=0.0,
        clip_training_reward=False,
        include_ram_metrics=False,
        multiprocessing_context="spawn",
    )


@dataclass(frozen=True, slots=True)
class DQNConfig:
    """All mutable experiment choices live in this single configuration."""

    seed: int = 2026
    num_actors: int = 2
    actor_env: MsPacmanEnvConfig = field(default_factory=default_actor_env_config)
    actor_transition_batch_size: int = 64
    actor_torch_threads: int = 1
    actor_seed_stride: int = 100_000
    rollout_queue_capacity: int = 8
    metrics_queue_capacity: int = 128
    queue_retry_timeout_seconds: float = 0.5
    initial_parameters_timeout_seconds: float = 120.0

    epsilon_start: float = 0.9
    epsilon_end: float = 0.05
    epsilon_decay_transitions: int = 1_000_000

    reward_log_scale: float = 10.0
    reward_clip_min: float = 0.0
    reward_clip_max: float = 5.0
    decision_step_cost: float = 0.01
    lost_life_penalty: float = -5.0
    game_over_penalty: float = -10.0

    total_transitions: int = 10_000_000
    replay_capacity: int = 500_000
    learning_starts: int = 20_000
    learner_batch_size: int = 64
    updates_per_transition: float = 0.25
    gamma: float = 0.99
    learning_rate: float = 1.0e-4
    adam_epsilon: float = 1.5e-4
    gradient_clip_norm: float = 10.0
    target_sync_interval_updates: int = 10_000
    learner_device: str = "cuda:0"

    prioritized_replay_alpha: float = 0.6
    prioritized_replay_beta_start: float = 0.4
    prioritized_replay_beta_end: float = 1.0
    prioritized_replay_beta_transitions: int = 10_000_000
    priority_epsilon: float = 1.0e-6

    tensorboard_interval_transitions: int = 200_000
    checkpoint_interval_transitions: int = 1_000_000
    evaluation_episodes: int = 50
    evaluation_enabled: bool = True
    evaluation_max_episode_steps: int = 30_000
    evaluation_seed: int = 20_260_000
    evaluator_torch_threads: int = 1
    runs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "runs")
    checkpoints_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "chkpt")
    resume_checkpoint: Path | None = None
    shutdown_timeout_seconds: float = 30.0
    evaluation_shutdown_timeout_seconds: float = 3_600.0

    def __post_init__(self) -> None:
        positive_ints = {
            "num_actors": self.num_actors,
            "actor_transition_batch_size": self.actor_transition_batch_size,
            "actor_torch_threads": self.actor_torch_threads,
            "actor_seed_stride": self.actor_seed_stride,
            "rollout_queue_capacity": self.rollout_queue_capacity,
            "metrics_queue_capacity": self.metrics_queue_capacity,
            "epsilon_decay_transitions": self.epsilon_decay_transitions,
            "total_transitions": self.total_transitions,
            "replay_capacity": self.replay_capacity,
            "learning_starts": self.learning_starts,
            "learner_batch_size": self.learner_batch_size,
            "target_sync_interval_updates": self.target_sync_interval_updates,
            "prioritized_replay_beta_transitions": self.prioritized_replay_beta_transitions,
            "tensorboard_interval_transitions": self.tensorboard_interval_transitions,
            "checkpoint_interval_transitions": self.checkpoint_interval_transitions,
            "evaluation_episodes": self.evaluation_episodes,
            "evaluation_max_episode_steps": self.evaluation_max_episode_steps,
            "evaluator_torch_threads": self.evaluator_torch_threads,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        if self.learning_starts > self.replay_capacity:
            raise ValueError("learning_starts cannot exceed replay_capacity")
        if self.actor_transition_batch_size < self.actor_env.num_envs:
            raise ValueError(
                "actor_transition_batch_size must fit at least one vector step"
            )
        if self.actor_transition_batch_size % self.actor_env.num_envs != 0:
            raise ValueError(
                "actor_transition_batch_size must be divisible by actor_env.num_envs"
            )
        if self.actor_env.step_cost != 0.0 or self.actor_env.clip_training_reward:
            raise ValueError(
                "DQN actor_env must return raw rewards; configure shaping in DQNConfig"
            )
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError("epsilon must satisfy 0 <= end <= start <= 1")
        reward_values = (
            self.reward_log_scale,
            self.reward_clip_min,
            self.reward_clip_max,
            self.decision_step_cost,
            self.lost_life_penalty,
            self.game_over_penalty,
        )
        if not all(math.isfinite(value) for value in reward_values):
            raise ValueError("reward shaping values must be finite")
        if self.reward_clip_min >= self.reward_clip_max:
            raise ValueError("reward_clip_min must be less than reward_clip_max")
        if self.reward_log_scale <= 0:
            raise ValueError("reward_log_scale must be positive")
        if self.decision_step_cost < 0:
            raise ValueError("decision_step_cost must be non-negative")
        if not math.isfinite(self.updates_per_transition) or self.updates_per_transition <= 0:
            raise ValueError("updates_per_transition must be finite and positive")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if self.learning_rate <= 0 or self.adam_epsilon <= 0:
            raise ValueError("optimizer parameters must be positive")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if not 0.0 <= self.prioritized_replay_alpha <= 1.0:
            raise ValueError("prioritized_replay_alpha must be in [0, 1]")
        if not (
            0.0
            < self.prioritized_replay_beta_start
            <= self.prioritized_replay_beta_end
            <= 1.0
        ):
            raise ValueError("prioritized replay beta must satisfy 0 < start <= end <= 1")
        if self.priority_epsilon <= 0:
            raise ValueError("priority_epsilon must be positive")
        if self.queue_retry_timeout_seconds <= 0:
            raise ValueError("queue_retry_timeout_seconds must be positive")
        if self.initial_parameters_timeout_seconds <= 0:
            raise ValueError("initial_parameters_timeout_seconds must be positive")
        if self.shutdown_timeout_seconds <= 0:
            raise ValueError("shutdown_timeout_seconds must be positive")
        if self.evaluation_shutdown_timeout_seconds <= 0:
            raise ValueError("evaluation_shutdown_timeout_seconds must be positive")

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return (
            self.actor_env.frame_stack,
            self.actor_env.screen_size,
            self.actor_env.screen_size,
        )

    @property
    def action_count(self) -> int:
        return 9
