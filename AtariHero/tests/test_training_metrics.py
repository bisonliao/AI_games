from __future__ import annotations

from curri_DQN.config import TrainConfig
from curri_DQN.messages import EpisodeSummary
from curri_DQN.train import TrainingMetrics


def test_training_metrics_episode_return_is_rl_return() -> None:
    metrics = TrainingMetrics(TrainConfig())
    metrics.consume(
        EpisodeSummary(
            actor_id=0,
            reset_stage=1,
            task_id="task",
            checkpoint_id="task-V01",
            episode_return=100.96,
            ale_score_return=1150.0,
            episode_length=4,
            success=True,
            timeout=False,
            walls_destroyed=1,
            creatures_killed=1,
            miner_rescue_events=1,
            dynamite_bonus_sticks=3,
            unmapped_ale_reward=0.0,
            visited_levels=(1,),
            completed_levels=(1,),
            epsilon=0.1,
        ),
        target_stage=1,
    )
    assert list(metrics.returns) == [100.96]
    assert list(metrics.ale_score_returns) == [1150.0]
