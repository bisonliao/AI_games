"""TensorBoard metrics beyond Stable-Baselines3's built-in SAC logs."""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback

from .env import STAGE_NAMES


class TaskMetricsCallback(BaseCallback):
    """Report task progress, stage occupancy, and episode outcomes to TB."""

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", ())
        for info in infos:
            stage_index = int(info.get("stage_index", 0))
            for index, name in enumerate(STAGE_NAMES):
                self.logger.record_mean(
                    f"task/stage_fraction_{name}", float(stage_index == index)
                )

            for name, value in info.get("reward_terms", {}).items():
                self.logger.record_mean(f"reward/{name}", float(value))

            # Monitor adds episode only to the terminal transition.  Recording
            # these values here therefore computes rates over completed episodes
            # rather than over environment steps.
            episode = info.get("episode")
            if episode is not None:
                self.logger.record_mean("task/success_rate", float(episode["success"]))
                self.logger.record_mean("task/failure_rate", float(episode["failure"]))
                self.logger.record_mean("task/grasp_rate", float(episode["ever_grasped"]))
                self.logger.record_mean("task/lift_rate", float(episode["ever_lifted"]))
                self.logger.record_mean("task/final_stage", float(episode["stage_index"]))
                self.logger.record_mean(
                    "task/truncation_rate", float(episode["time_limit_reached"])
                )
                failure_reason = str(episode.get("failure_reason", ""))
                self.logger.record_mean(
                    "task/grasp_lost_rate", float(failure_reason == "grasp_lost")
                )
                self.logger.record_mean(
                    "task/drop_rate",
                    float(failure_reason in {"object_dropped", "object_left_goal"}),
                )
                self.logger.record_mean(
                    "task/release_regression_rate",
                    float(failure_reason == "release_regressed"),
                )
                self.logger.record_mean(
                    "task/stage_timeout_rate", float(failure_reason.endswith("_timeout"))
                )
        return True


__all__ = ["TaskMetricsCallback"]
