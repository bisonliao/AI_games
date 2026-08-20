"""Low-overhead throughput instrumentation for the pixel SAC learner."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper


@dataclass
class TimingStats:
    """Accumulated wall-clock time; no per-step TensorBoard writes occur."""

    env_step: float = 0.0
    predict: float = 0.0
    replay_sample: float = 0.0
    train: float = 0.0
    env_steps: int = 0
    predict_calls: int = 0
    replay_samples: int = 0
    train_calls: int = 0

    def snapshot_and_reset(self) -> dict[str, float]:
        values = {
            "env_step": self.env_step / max(self.env_steps, 1),
            "predict": self.predict / max(self.predict_calls, 1),
            "replay_sample": self.replay_sample / max(self.replay_samples, 1),
            "train": self.train / max(self.train_calls, 1),
            "env_step_total": self.env_step,
            "predict_total": self.predict,
            "replay_sample_total": self.replay_sample,
            "train_total": self.train,
        }
        self.env_step = self.predict = self.replay_sample = self.train = 0.0
        self.env_steps = self.predict_calls = self.replay_samples = self.train_calls = 0
        return values


class TimingVecEnv(VecEnvWrapper):
    """Time the complete vector-env wait, including IPC and rendering."""

    def __init__(self, venv: VecEnv, stats: TimingStats) -> None:
        super().__init__(venv)
        self.stats = stats
        self._step_started = 0.0

    def reset(self):
        return self.venv.reset()

    def step_async(self, actions) -> None:
        self._step_started = time.perf_counter()
        self.venv.step_async(actions)

    def step_wait(self):
        result = self.venv.step_wait()
        self.stats.env_step += time.perf_counter() - self._step_started
        self.stats.env_steps += 1
        return result


class TimedSAC(SAC):
    """SB3 SAC with timing hooks around prediction, replay sampling and train."""

    def __init__(self, *args: Any, timing_stats: TimingStats, **kwargs: Any) -> None:
        self.timing_stats = timing_stats
        super().__init__(*args, **kwargs)
        original_sample = self.replay_buffer.sample

        def timed_sample(*sample_args: Any, **sample_kwargs: Any):
            started = time.perf_counter()
            result = original_sample(*sample_args, **sample_kwargs)
            self.timing_stats.replay_sample += time.perf_counter() - started
            self.timing_stats.replay_samples += 1
            return result

        # DictReplayBuffer does not expose a callback hook. An instance-level
        # wrapper keeps this instrumentation local to the training model and
        # adds only two perf_counter calls per gradient step.
        self.replay_buffer.sample = timed_sample

    def _sample_action(
        self,
        learning_starts: int,
        action_noise=None,
        n_envs: int = 1,
    ):
        # SB3 samples uniformly before learning_starts, so that path contains
        # no actor CNN inference and must not be labelled as prediction time.
        if self.num_timesteps < learning_starts and not (
            self.use_sde and self.use_sde_at_warmup
        ):
            return super()._sample_action(learning_starts, action_noise, n_envs)
        started = time.perf_counter()
        result = super()._sample_action(learning_starts, action_noise, n_envs)
        self.timing_stats.predict += time.perf_counter() - started
        self.timing_stats.predict_calls += 1
        return result

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        sample_started = self.timing_stats.replay_sample
        started = time.perf_counter()
        super().train(gradient_steps=gradient_steps, batch_size=batch_size)
        elapsed = time.perf_counter() - started
        # ``replay_sample`` is measured inside this call. Subtract it so
        # ``time/train`` represents the remaining SAC forward/backward/update
        # work rather than double-counting replay sampling.
        sampled = self.timing_stats.replay_sample - sample_started
        self.timing_stats.train += max(elapsed - sampled, 0.0)
        self.timing_stats.train_calls += 1
        # SB3 intentionally excludes n_updates from TensorBoard. For parallel
        # off-policy collection this hides an important distinction: one vec
        # step may add many transitions while doing only one update. Expose
        # both the count and cumulative update-to-data ratio explicitly.
        learned_transitions = max(self.num_timesteps - self.learning_starts, 1)
        self.logger.record("train/n_updates", self._n_updates)
        self.logger.record(
            "train/update_to_data_ratio",
            self._n_updates / learned_transitions,
        )


class TimingTensorBoardCallback(BaseCallback):
    """Aggregate timing and emit four ``time/*`` TB scalars infrequently."""

    def __init__(self, stats: TimingStats, log_freq: int = 5_000, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.stats = stats
        self.log_freq = int(log_freq)
        self._next_log = self.log_freq

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_log:
            return True
        values = self.stats.snapshot_and_reset()
        self.logger.record("time/env_step", values["env_step"])
        self.logger.record("time/predict", values["predict"])
        self.logger.record("time/replay_sample", values["replay_sample"])
        self.logger.record("time/train", values["train"])
        # A single low-frequency dump prevents the timing metrics from waiting
        # for an episode boundary. It is intentionally not called per step.
        self.logger.dump(self.num_timesteps)
        while self._next_log <= self.num_timesteps:
            self._next_log += self.log_freq
        return True

    def _on_training_end(self) -> None:
        if self.stats.env_steps or self.stats.predict_calls or self.stats.train_calls:
            values = self.stats.snapshot_and_reset()
            self.logger.record("time/env_step", values["env_step"])
            self.logger.record("time/predict", values["predict"])
            self.logger.record("time/replay_sample", values["replay_sample"])
            self.logger.record("time/train", values["train"])


__all__ = ["TimedSAC", "TimingStats", "TimingTensorBoardCallback", "TimingVecEnv"]
