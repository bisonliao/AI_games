from __future__ import annotations

import unittest

from QMIX.tb_logger import TensorBoardLogger


class _RecordingWriter:
    def __init__(self) -> None:
        self.scalars = []
        self.flush_count = 0
        self.close_count = 0

    def add_scalar(self, tag, value, step) -> None:
        self.scalars.append((tag, float(value), int(step)))

    def flush(self) -> None:
        self.flush_count += 1

    def close(self) -> None:
        self.close_count += 1


class TaskMetricLoggerTest(unittest.TestCase):
    def test_reports_maddpg_compatible_task_success_tags(self):
        writer = _RecordingWriter()
        logger = TensorBoardLogger(
            interval=10,
            initial_step=0,
            log_dir=None,
            writer=writer,
        )
        logger.record_task_metrics(
            {
                "covered_landmarks": 3.0,
                "coverage_ratio": 1.0,
                "episode_success": 1.0,
            }
        )
        logger.record_task_metrics(
            {
                "covered_landmarks": 1.0,
                "coverage_ratio": 1.0 / 3.0,
                "episode_success": 0.0,
            }
        )

        self.assertTrue(logger.maybe_flush(10))
        scalars = {tag: value for tag, value, _ in writer.scalars}
        self.assertEqual(scalars["task/covered_landmarks"], 2.0)
        self.assertAlmostEqual(scalars["task/coverage_ratio"], 2.0 / 3.0)
        self.assertEqual(scalars["task/episode_success"], 0.5)
        self.assertEqual(scalars["task/success_rate"], 0.5)
        self.assertEqual(scalars["task/success_rate_roll100"], 0.5)

        logger.record_task_metrics(
            {
                "covered_landmarks": 3.0,
                "coverage_ratio": 1.0,
                "episode_success": 1.0,
            }
        )
        self.assertTrue(logger.maybe_flush(20))
        latest = {
            tag: value
            for tag, value, step in writer.scalars
            if step == 20
        }
        self.assertAlmostEqual(latest["task/success_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(
            latest["task/success_rate_roll100"], 2.0 / 3.0
        )


if __name__ == "__main__":
    unittest.main()
