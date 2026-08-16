import unittest

from maddpg.common.tensorboard_logger_torch import TensorBoardIntervalLogger


class RecordingWriter:
    def __init__(self):
        self.calls = []
        self.flush_count = 0
        self.closed = False

    def add_scalar(self, name, value, step):
        self.calls.append((name, float(value), step))

    def flush(self):
        self.flush_count += 1

    def close(self):
        self.closed = True


def _update(q_loss, entropy):
    return {
        "q_loss": q_loss,
        "p_loss": q_loss + 1,
        "pg_loss": q_loss + 2,
        "p_reg": q_loss + 3,
        "mean_q": q_loss + 4,
        "mean_target_q": q_loss + 5,
        "mean_target_q_next": q_loss + 6,
        "std_target_q": q_loss + 7,
        "mean_batch_rew": q_loss + 8,
        "action_entropy": entropy,
        "action_std_mean": q_loss + 9,
        "q_grad_norm": q_loss + 10,
        "p_grad_norm": q_loss + 11,
    }


class TensorBoardIntervalLoggerTest(unittest.TestCase):
    def test_interval_aggregates_every_episode_and_training_update(self):
        writer = RecordingWriter()
        logger = TensorBoardIntervalLogger(
            interval=100, initial_step=0, writer=writer
        )
        logger.record_episode(1.0, 20, [2.0, 4.0], {"episode_success": 0})
        logger.record_episode(3.0, 30, [4.0, 8.0], {"episode_success": 1})
        logger.record_training_update([_update(1.0, 0.2), _update(3.0, 0.4)])
        logger.record_training_update([_update(5.0, 0.6), _update(7.0, 0.8)])

        self.assertFalse(logger.maybe_flush(99))
        self.assertTrue(logger.maybe_flush(100))
        metrics = {name: value for name, value, step in writer.calls}
        self.assertTrue(all(step == 100 for _, _, step in writer.calls))
        self.assertEqual(metrics["reward/episode_reward"], 2.0)
        self.assertEqual(metrics["env/episode_length"], 25.0)
        self.assertEqual(metrics["reward/agent0_episode_reward"], 3.0)
        self.assertEqual(metrics["reward/agent1_episode_reward"], 6.0)
        self.assertEqual(metrics["task/episode_success"], 0.5)
        self.assertEqual(metrics["task/success_rate"], 0.5)
        self.assertEqual(metrics["task/success_rate_roll100"], 0.5)
        # Agent mean per optimizer event is 2 then 6; interval mean is 4.
        self.assertEqual(metrics["loss/q_loss"], 4.0)
        self.assertEqual(metrics["loss/q_loss_agent0"], 3.0)
        self.assertEqual(metrics["loss/q_loss_agent1"], 5.0)
        self.assertEqual(metrics["policy/action_entropy"], 0.5)
        self.assertNotIn("reward/episode_reward_std_interval", metrics)
        self.assertNotIn("reward/episode_reward_min_interval", metrics)
        self.assertNotIn("reward/episode_reward_max_interval", metrics)
        self.assertNotIn("env/episodes_in_log_interval", metrics)
        self.assertNotIn("task/success_rate_interval", metrics)

    def test_close_forces_final_partial_interval(self):
        writer = RecordingWriter()
        logger = TensorBoardIntervalLogger(
            interval=1000, initial_step=0, writer=writer
        )
        logger.record_episode(7.0, 25, [7.0], {})
        logger.close(25)

        self.assertIn(("reward/episode_reward", 7.0, 25), writer.calls)
        self.assertEqual(writer.flush_count, 1)
        self.assertTrue(writer.closed)

    def test_disabled_logger_is_no_op(self):
        logger = TensorBoardIntervalLogger(interval=0)
        logger.record_episode(1.0, 1, [1.0], {"episode_success": 1})
        logger.record_training_update([_update(1.0, 0.5)])
        self.assertFalse(logger.maybe_flush(100, force=True))
        logger.close(100)


if __name__ == "__main__":
    unittest.main()
