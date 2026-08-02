"""TensorBoard business success-rate tag emission test."""

from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from dqn.evaluate import EvaluationResult
from dqn.train import TrainingMetrics


def test_business_success_metric_is_written(tmp_path: Path):
    writer = SummaryWriter(str(tmp_path), flush_secs=1)
    metrics = TrainingMetrics(writer)
    metrics.add_evaluation(
        EvaluationResult(
            env_steps=1000,
            success_rate=0.95,
            mean_return=1.5,
            mean_distance_m=60.0,
            mean_length=600,
            fall_rate=0.04,
            timeout_rate=0.01,
            roll_rms=0.1,
        )
    )
    writer.close()
    event_file = next(tmp_path.glob("events.*"))
    accumulator = EventAccumulator(str(event_file))
    accumulator.Reload()
    event = accumulator.Scalars("business/eval_success_rate_100")[-1]
    assert event.step == 1000
    assert abs(event.value - 0.95) < 1e-6
