"""Task metrics for visual SAC runs."""

from __future__ import annotations

from typing import Dict

import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger, TensorBoardOutputFormat
from stable_baselines3.common.utils import obs_as_tensor


def write_tensorboard_scalars(
    logger: Logger,
    values: Dict[str, float],
    step: int,
) -> bool:
    """Write scalars at an explicit x-axis step using SB3's sole TB writer.

    Async evaluation results arrive after the learner has advanced, so going
    through ``logger.record`` would assign the result to the completion time.
    Direct use of the learner-process writer preserves the checkpoint step and
    does not introduce a second process or event writer.
    """

    written = False
    for output_format in logger.output_formats:
        if not isinstance(output_format, TensorBoardOutputFormat):
            continue
        for name, value in values.items():
            output_format.writer.add_scalar(name, float(value), global_step=int(step))
        output_format.writer.flush()
        written = True
    return written


class PixelTaskMetricsCallback(BaseCallback):
    """Log completed-episode task metrics and failure categories."""

    def __init__(self, task: str, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        if task not in {"reach", "pick_place"}:
            raise ValueError("task must be 'reach' or 'pick_place'")
        self.task = task

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", ()):
            episode = info.get("episode")
            if episode is None:
                continue
            self.logger.record_mean("task/success_rate", float(episode["success"]))
            self.logger.record_mean("task/failure_rate", float(episode["failure"]))
            self.logger.record_mean(
                "task/truncation_rate", float(episode["time_limit_reached"])
            )
            if self.task == "reach":
                continue

            self.logger.record_mean("task/grasp_rate", float(episode["ever_grasped"]))
            self.logger.record_mean("task/lift_rate", float(episode["ever_lifted"]))
            self.logger.record_mean("task/final_stage", float(episode["stage_index"]))
            reason = str(episode.get("failure_reason", ""))
            self.logger.record_mean("task/grasp_lost_rate", float(reason == "grasp_lost"))
            self.logger.record_mean(
                "task/drop_rate",
                float(reason in {"object_dropped", "object_left_goal"}),
            )
            self.logger.record_mean(
                "task/stage_timeout_rate", float(reason.endswith("_timeout"))
            )
        return True


class RolloutActionEntropyCallback(BaseCallback):
    """Periodically estimate the squashed policy entropy on rollout actions.

    SAC's temperature (``train/ent_coef``) only controls the weight of the
    entropy term.  This callback instead records a Monte Carlo estimate of the
    current action distribution's differential entropy, ``-log pi(a|s)``,
    using the actions that were actually sampled for the current rollout
    states.  The estimate includes the tanh change-of-variables correction.
    """

    def __init__(self, log_freq: int = 5_000, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        if log_freq <= 0:
            raise ValueError("log_freq must be positive")
        self.log_freq = int(log_freq)
        self._next_log = self.log_freq

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_log:
            return True

        # The action that just reached the environment was chosen before
        # num_timesteps advanced by n_envs. Do not label uniformly sampled
        # learning-start warm-up actions as samples from the SAC policy.
        n_envs = int(self.training_env.num_envs)
        if self.num_timesteps - n_envs < self.model.learning_starts:
            return True

        observation = self.model._last_obs
        buffer_actions = self.locals.get("buffer_actions")
        if observation is None or buffer_actions is None:
            if self.verbose:
                print("rollout action entropy skipped: rollout data is unavailable")
            return True

        observation_tensor = obs_as_tensor(observation, self.model.device)
        action_tensor = th.as_tensor(buffer_actions, device=self.model.device)
        with th.inference_mode():
            mean_actions, log_std, kwargs = self.model.actor.get_action_dist_params(
                observation_tensor
            )
            distribution = self.model.actor.action_dist.proba_distribution(
                mean_actions,
                log_std,
                **kwargs,
            )
            entropy = float((-distribution.log_prob(action_tensor)).mean().item())

        step = int(self.num_timesteps)
        metrics = {"rollout/action_entropy": entropy}
        if not write_tensorboard_scalars(self.logger, metrics, step):
            self.logger.record("rollout/action_entropy", entropy)
        while self._next_log <= self.num_timesteps:
            self._next_log += self.log_freq
        return True


class VisualHealthCallback(BaseCallback):
    """Probe visual features once per regular checkpoint without rendering."""

    def __init__(self, check_freq: int, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        if check_freq <= 0:
            raise ValueError("check_freq must be positive")
        self.check_freq = int(check_freq)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True
        observation = self.locals.get("new_obs")
        if not isinstance(observation, dict) or "image" not in observation:
            if self.verbose:
                print("visual health probe skipped: rollout image is unavailable")
            return True

        actor_extractor = self.model.actor.features_extractor
        critic_extractor = self.model.critic.features_extractor
        if not hasattr(actor_extractor, "encode_visual"):
            if self.verbose:
                print("visual health probe skipped: extractor has no encode_visual")
            return True

        image = th.as_tensor(observation["image"], device=self.model.device)
        observation_tensor = {
            key: th.as_tensor(value, device=self.model.device)
            for key, value in observation.items()
        }
        with th.inference_mode():
            actor_metrics = self._feature_metrics(actor_extractor, image)
            critic_metrics = self._feature_metrics(critic_extractor, image)
            action_metrics = self._action_sensitivity_metrics(
                observation_tensor,
                image,
            )
        step = int(self.num_timesteps)
        metrics = {
            f"diagnostics/visual_{name}": value
            for name, value in actor_metrics.items()
        }
        metrics.update(
            {
                f"diagnostics/critic_visual_{name}": value
                for name, value in critic_metrics.items()
            }
        )
        metrics.update(
            {f"diagnostics/{name}": value for name, value in action_metrics.items()}
        )
        # The training entry point always enables TensorBoard. Keep a logger
        # fallback for isolated uses/tests where no TB output format exists.
        if not write_tensorboard_scalars(self.logger, metrics, step):
            for name, value in metrics.items():
                self.logger.record(name, value)
        if self.verbose:
            print(
                f"visual health: step={step} "
                f"actor_std={actor_metrics['batch_std']:.6g} "
                f"critic_std={critic_metrics['batch_std']:.6g} "
                f"action_delta={action_metrics.get('image_action_delta_mean', 0.0):.6g} "
                f"inactive={bool(actor_metrics['inactive'])}"
            )
        return True

    @staticmethod
    def _feature_metrics(extractor, image: th.Tensor) -> Dict[str, float]:
        visual = extractor.encode_visual(image)
        zero_fraction = float((visual.abs() <= 1e-8).float().mean().item())
        rms = float(visual.square().mean().sqrt().item())
        batch_std = float(
            visual.std(dim=0, unbiased=False).mean().item()
            if visual.shape[0] > 1
            else 0.0
        )
        inactive = float(
            zero_fraction >= 0.999
            or rms <= 1e-6
            or (visual.shape[0] > 1 and batch_std <= 1e-6)
        )
        return {
            "zero_fraction": zero_fraction,
            "batch_std": batch_std,
            "relative_batch_std": batch_std / max(rms, 1e-12),
            "rms": rms,
            "inactive": inactive,
        }

    def _action_sensitivity_metrics(
        self,
        observation: Dict[str, th.Tensor],
        image: th.Tensor,
    ) -> Dict[str, float]:
        # A one-environment run has no second image to exchange. The feature
        # health metrics remain valid, so only omit this counterfactual probe.
        if image.shape[0] <= 1:
            return {}
        shifted = dict(observation)
        shifted["image"] = th.roll(image, shifts=1, dims=0)
        original_action = self.model.actor(observation, deterministic=True)
        shifted_action = self.model.actor(shifted, deterministic=True)
        delta = (original_action - shifted_action).abs()
        delta_mean = float(delta.mean().item())
        return {
            "image_action_delta_mean": delta_mean,
            "image_action_delta_max": float(delta.max().item()),
            # The action range is [-1, 1]; a 1e-4 response is already only
            # 0.005% of that range. This is a warning, not a training stop.
            "image_action_insensitive": float(delta_mean < 1e-4),
        }


__all__ = [
    "PixelTaskMetricsCallback",
    "RolloutActionEntropyCallback",
    "VisualHealthCallback",
    "write_tensorboard_scalars",
]
