"""Configuration for the self-contained R2D2 trainer."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path

from PacManEnv import MsPacmanEnvConfig


R2D2_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = R2D2_DIR.parent


def default_actor_env_config() -> MsPacmanEnvConfig:
    """Return the Atari preprocessing used by the R2D2 actors/evaluator."""
    return MsPacmanEnvConfig(
        num_envs=2,
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
class R2D2Config:
    """All experiment choices used by actors, learner, and evaluator."""

    seed: int = 2026
    # Eight actors give a denser fixed R2D2 epsilon ladder while remaining
    # practical on this host.
    num_actors: int = 8
    actor_env: MsPacmanEnvConfig = field(default_factory=default_actor_env_config)
    actor_torch_threads: int = 1
    evaluator_torch_threads: int = 1
    actor_parameter_update_interval: int = 400
    actor_sequence_chunk_size: int = 8
    rollout_queue_capacity: int = 16
    metrics_queue_capacity: int = 128
    queue_timeout_seconds: float = 0.5
    initial_parameters_timeout_seconds: float = 120.0

    # Ape-X/R2D2 exploration ladder.  The learner uses these only for metrics;
    # each actor owns its own RNG and fixed epsilon.
    base_epsilon: float = 0.4
    epsilon_alpha: float = 7.0

    # Ten million decisions is only 40M emulator frames and is too short for
    # R2D2 on Ms. Pac-Man. This is still far below the paper's compute budget,
    # but is a more meaningful workstation run.
    total_transitions: int = 40_000_000
    # The paper/reference uses roughly 50K sequences, but 50K * 88 * 84 * 84
    # uint8 frames is about 28.9 GiB before Python/object/learner overhead.
    # 25K sequences is a safer local compromise: 1M learning transitions and
    # about 14.5 GiB raw replay (roughly 17.3 GiB with 20% overhead).
    replay_capacity_sequences: int = 25_000
    # This field is measured in learning transitions, not replay sequence slots.
    # 50K transitions matches the warmup used by the reference implementation.
    learning_starts: int = 50_000
    learner_batch_size: int = 64
    updates_per_sequence: float = 1/8 # 每拿到若干个新sequence就执行一次更新
    gamma: float = 0.997
    n_step: int = 5
    burn_in_steps: int = 40
    learning_steps: int = 40
    forward_steps: int = 5
    hidden_size: int = 512
    learning_rate: float = 1.0e-4
    adam_epsilon: float = 1.0e-3
    gradient_clip_norm: float = 40.0
    target_sync_interval_updates: int = 2_500
    parameter_broadcast_interval_updates: int = 4

    prioritized_replay_alpha: float = 0.9
    importance_sampling_beta: float = 0.6
    priority_mix: float = 0.9
    priority_epsilon: float = 1.0e-6

    tensorboard_interval_transitions: int = 200_000
    # Preserve roughly ten expensive checkpoint evaluations over the longer
    # default run instead of creating an evaluator backlog every 1M steps.
    checkpoint_interval_transitions: int = 2_000_000
    evaluation_enabled: bool = True
    evaluation_episodes: int = 30
    training_max_episode_steps: int = 30_000
    evaluation_max_episode_steps: int = 30_000
    evaluation_seed: int = 20_260_000
    shutdown_timeout_seconds: float = 30.0
    evaluation_shutdown_timeout_seconds: float = 3_600.0
    learner_device: str = "cuda:0"
    # Keep outputs alongside DQN's project-level runs while retaining all
    # implementation code under _R2D2/.
    runs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "runs")
    checkpoints_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "checkpoint")
    resume_checkpoint: Path | None = None

    def __post_init__(self) -> None:
        positive = {
            "num_actors": self.num_actors,
            "actor_torch_threads": self.actor_torch_threads,
            "evaluator_torch_threads": self.evaluator_torch_threads,
            "actor_parameter_update_interval": self.actor_parameter_update_interval,
            "actor_sequence_chunk_size": self.actor_sequence_chunk_size,
            "rollout_queue_capacity": self.rollout_queue_capacity,
            "metrics_queue_capacity": self.metrics_queue_capacity,
            "total_transitions": self.total_transitions,
            "replay_capacity_sequences": self.replay_capacity_sequences,
            "learning_starts": self.learning_starts,
            "learner_batch_size": self.learner_batch_size,
            "n_step": self.n_step,
            "burn_in_steps": self.burn_in_steps,
            "learning_steps": self.learning_steps,
            "forward_steps": self.forward_steps,
            "hidden_size": self.hidden_size,
            "target_sync_interval_updates": self.target_sync_interval_updates,
            "parameter_broadcast_interval_updates": self.parameter_broadcast_interval_updates,
            "tensorboard_interval_transitions": self.tensorboard_interval_transitions,
            "checkpoint_interval_transitions": self.checkpoint_interval_transitions,
            "evaluation_episodes": self.evaluation_episodes,
            "training_max_episode_steps": self.training_max_episode_steps,
            "evaluation_max_episode_steps": self.evaluation_max_episode_steps,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.learning_starts > self.replay_capacity_sequences * self.learning_steps:
            raise ValueError(
                "learning_starts cannot exceed replay learning-transition capacity"
            )
        if self.learner_batch_size > self.replay_capacity_sequences:
            raise ValueError("learner_batch_size cannot exceed replay capacity")
        if self.actor_env.step_cost != 0.0 or self.actor_env.clip_training_reward:
            raise ValueError("R2D2 actor_env must return raw, unclipped rewards")
        if not 0.0 <= self.base_epsilon <= 1.0:
            raise ValueError("base_epsilon must be in [0, 1]")
        if self.epsilon_alpha < 0 or not math.isfinite(self.epsilon_alpha):
            raise ValueError("epsilon_alpha must be finite and non-negative")
        if not math.isfinite(self.updates_per_sequence) or self.updates_per_sequence <= 0:
            raise ValueError("updates_per_sequence must be finite and positive")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if self.learning_rate <= 0 or self.adam_epsilon <= 0:
            raise ValueError("optimizer parameters must be positive")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if not 0.0 <= self.prioritized_replay_alpha <= 1.0:
            raise ValueError("prioritized_replay_alpha must be in [0, 1]")
        if not 0.0 < self.importance_sampling_beta <= 1.0:
            raise ValueError("importance_sampling_beta must be in (0, 1]")
        if not 0.0 <= self.priority_mix <= 1.0:
            raise ValueError("priority_mix must be in [0, 1]")
        if self.priority_epsilon <= 0:
            raise ValueError("priority_epsilon must be positive")
        if self.queue_timeout_seconds <= 0 or self.initial_parameters_timeout_seconds <= 0:
            raise ValueError("queue timeouts must be positive")
        if self.shutdown_timeout_seconds <= 0 or self.evaluation_shutdown_timeout_seconds <= 0:
            raise ValueError("shutdown timeouts must be positive")
        if self.n_step != self.forward_steps:
            raise ValueError("forward_steps must equal n_step for R2D2 targets")
        if self.actor_env.frame_stack != 4:
            raise ValueError("R2D2 expects a four-frame observation stack")

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return (
            self.actor_env.frame_stack,
            self.actor_env.screen_size,
            self.actor_env.screen_size,
        )

    @property
    def action_count(self) -> int:
        # PacManEnv's packaged minimal action set has nine actions.
        return 9

    @property
    def replay_capacity(self) -> int:
        """Compatibility alias documenting that capacity is measured in sequences."""
        return self.replay_capacity_sequences

    @property
    def batch_size(self) -> int:
        """Compatibility alias for the recurrent learner minibatch size."""
        return self.learner_batch_size

    @property
    def checkpoint_dir(self) -> Path:
        """Singular-name alias for integrations that use ``checkpoint/``."""
        return self.checkpoints_dir

    @property
    def estimated_replay_memory_gib(self) -> float:
        """Conservative replay estimate including a 20% object overhead."""
        packed_frames_per_sequence = (
            self.burn_in_steps
            + self.learning_steps
            + self.forward_steps
            + self.actor_env.frame_stack
            - 1
        )
        raw_bytes = (
            self.replay_capacity_sequences
            * packed_frames_per_sequence
            * self.actor_env.screen_size**2
        )
        return raw_bytes * 1.2 / (1024**3)
