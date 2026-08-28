"""Configuration for one fixed curriculum-stage DQN run."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrainConfig:
    run_dir: str = "curri_DQN/runs/hero_dddqn"
    checkpoint_root: str = "curri_DQN/chkpt"
    hero_checkpoint_dir: str = "HeroEnv/checkpoints" # 人工玩teacher.py保存的课程起始点
    target_stage: int = 1
    total_transitions: int = 3_000_000
    num_actors: int = 16
    seed: int = 7
    gpu: int = 0

    train_current_stage_fraction: float = 0.8
    eval_current_stage_fraction: float = 0.5

    queue_size: int = 8_192
    replay_capacity: int = 500_000
    batch_size: int = 64
    learning_starts: int = 10_000
    gamma: float = 0.99
    decision_step_penalty: float = 0.002
    wall_event_reward: float = 0.5
    creature_event_reward: float = 0.5
    miner_event_reward: float = 10.0
    learning_rate: float = 1e-4
    adam_eps: float = 1.5e-4
    max_grad_norm: float = 10.0
    update_ratio: float = 0.15
    max_updates_per_cycle: int = 256
    target_update_interval: int = 10_000
    publish_interval: int = 250
    actor_weight_sync_interval: int = 100

    epsilon_start: float = 0.9
    epsilon_end: float = 0.05
    epsilon_decay_transitions: int = 500_000

    max_curriculum_stage: int = 512
    action_repeat: int = 4
    episode_timeout_decisions: int = 500
    frame_stack: int = 4
    screen_size: int = 84
    sticky_action_probability: float = 0.25
    timeout_terminal_reward: float = -10.0
    life_lost_terminal_reward: float = -10.0

    checkpoint_interval: int = 500_000
    eval_interval: int = 100_000
    eval_episodes: int = 50
    eval_epsilon: float = 0.001
    log_interval_seconds: float = 10.0
    episode_window: int = 100
    save_replay: bool = True
    load_checkpoint: str | None = None
    resume: str | None = None
    after_curri: bool = False

    def validate(self) -> None:
        if not 1 <= self.target_stage <= self.max_curriculum_stage:
            raise ValueError("target_stage is out of range")
        if self.num_actors < 1 or self.total_transitions < 1:
            raise ValueError("num_actors and total_transitions must be positive")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be at least one batch")
        if not 0 < self.update_ratio <= 1:
            raise ValueError("update_ratio must be in (0, 1]")
        if self.episode_timeout_decisions < 1:
            raise ValueError("episode_timeout_decisions must be positive")
        if self.timeout_terminal_reward >= 0:
            raise ValueError("timeout_terminal_reward must be negative")
        if self.life_lost_terminal_reward >= 0:
            raise ValueError("life_lost_terminal_reward must be negative")
        if self.wall_event_reward < 0 or self.creature_event_reward < 0:
            raise ValueError("event rewards must be non-negative")
        if self.miner_event_reward <= 0:
            raise ValueError("miner_event_reward must be positive")
        if not 0 <= self.decision_step_penalty < 1:
            raise ValueError("decision_step_penalty must be in [0, 1)")
        if not 0 <= self.train_current_stage_fraction <= 1:
            raise ValueError("train_current_stage_fraction must be in [0, 1]")
        if not 0 <= self.eval_current_stage_fraction <= 1:
            raise ValueError("eval_current_stage_fraction must be in [0, 1]")
        if self.epsilon_decay_transitions < 1:
            raise ValueError("epsilon_decay_transitions must be positive")
        if self.eval_interval < 1 or self.eval_episodes < 1:
            raise ValueError("evaluation interval and episodes must be positive")
        if self.load_checkpoint is not None and self.resume is not None:
            raise ValueError("load_checkpoint and resume are mutually exclusive")
        if self.after_curri and self.load_checkpoint is None and self.resume is None:
            raise ValueError("--after-curri requires --load-checkpoint or --resume")
        if self.frame_stack != 4 or self.screen_size != 84:
            raise ValueError("this network and transition codec require 4x84x84 observations")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def run_path(self) -> Path:
        base = Path(self.run_dir).resolve()
        if self.after_curri:
            suffix = "_afterCurri"
            if base.name.endswith(suffix):
                return base
            return base.with_name(f"{base.name}{suffix}")
        suffix = f"_stage-{self.target_stage:02d}"
        if base.name.endswith(suffix):
            return base
        return base.with_name(f"{base.name}{suffix}")

    @property
    def hero_checkpoint_path(self) -> Path:
        return Path(self.hero_checkpoint_dir).resolve()

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.checkpoint_root).resolve() / self.run_path.name
