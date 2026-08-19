"""Task metrics for visual SAC runs."""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class PixelTaskMetricsCallback(BaseCallback):
    """Log completed-episode task metrics and failure categories."""

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", ()):
            episode = info.get("episode")
            if episode is None:
                continue
            self.logger.record_mean("task/success_rate", float(episode["success"]))
            self.logger.record_mean("task/failure_rate", float(episode["failure"]))
            self.logger.record_mean("task/grasp_rate", float(episode["ever_grasped"]))
            self.logger.record_mean("task/lift_rate", float(episode["ever_lifted"]))
            self.logger.record_mean("task/final_stage", float(episode["stage_index"]))
            self.logger.record_mean(
                "task/truncation_rate", float(episode["time_limit_reached"])
            )
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


__all__ = ["PixelTaskMetricsCallback"]
