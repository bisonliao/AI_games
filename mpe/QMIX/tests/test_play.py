from __future__ import annotations

import json
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from QMIX.env import LegacyMPEEnv
from QMIX.learner import LearnerConfig, QMIXLearner
from QMIX.play import parse_args, play, resolve_checkpoint
from QMIX.train import _checkpoint_payload, _metadata


class PlayArgumentsTest(unittest.TestCase):
    def test_render_and_environment_seed_are_explicit_options(self):
        options = parse_args(
            [
                "--checkpoint",
                "state_steps_10.pt",
                "--episodes",
                "3",
                "--env-seed",
                "42",
                "--render",
            ]
        )
        self.assertEqual(options.episodes, 3)
        self.assertEqual(options.env_seed, 42)
        self.assertTrue(options.render)

    def test_misspelled_render_option_is_rejected(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--checkpoint", "state_steps_10.pt", "--reander"])

    def test_checkpoint_directory_selects_greatest_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "state_steps_9.pt").touch()
            expected = root / "state_steps_100.pt"
            expected.touch()
            (root / "not_a_checkpoint.pt").touch()
            self.assertEqual(resolve_checkpoint(root), expected.resolve())


class PlayEvaluationTest(unittest.TestCase):
    def test_headless_evaluation_uses_checkpoint_metadata_and_seed_sequence(self):
        env = LegacyMPEEnv("simple_spread")
        try:
            config = LearnerConfig(
                n_agents=env.n_agents,
                obs_dim=env.observation_dims[0],
                state_dim=env.state_dim,
                n_actions=env.action_specs[0].n_actions,
                hidden_dim=17,
            )
            learner = QMIXLearner(config, torch.device("cpu"))
            training_args = SimpleNamespace(
                scenario="simple_spread",
                max_episode_len=1,
                reward_scale=1.0 / env.n_agents,
                bootstrap_time_limit=False,
            )
            payload = _checkpoint_payload(
                learner,
                _metadata(training_args, env, config),
                env_steps=123,
                completed_episodes=7,
                last_target_update_episode=0,
            )
        finally:
            env.close()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "state_steps_123.pt"
            report_path = root / "report.json"
            torch.save(payload, checkpoint_path)
            options = parse_args(
                [
                    "--checkpoint",
                    str(checkpoint_path),
                    "--episodes",
                    "2",
                    "--env-seed",
                    "73",
                    "--report-json",
                    str(report_path),
                    "--no-cuda",
                ]
            )
            with redirect_stdout(io.StringIO()) as output:
                report = play(options)

            self.assertEqual(report["saved_env_steps"], 123)
            self.assertEqual(report["saved_completed_episodes"], 7)
            self.assertEqual(report["evaluation_env_seed"], 73)
            self.assertEqual(
                [episode["seed"] for episode in report["episodes"]],
                [73, 74],
            )
            self.assertEqual(
                [episode["episode_length"] for episode in report["episodes"]],
                [1, 1],
            )
            self.assertEqual(report["task_success_count"], 0)
            self.assertEqual(report["task_success_rate"], 0.0)
            self.assertIn("task_landmark_center_success_count", report)
            self.assertIn("task_landmark_center_success_rate", report)
            self.assertGreaterEqual(
                report["task_landmark_center_success_rate"],
                report["task_success_rate"],
            )
            self.assertIn("landmark-center success (d < 0.15)", output.getvalue())
            self.assertEqual(report["metadata"]["learner_config"]["hidden_dim"], 17)
            self.assertTrue(report_path.is_file())
            saved_report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_report["evaluation_env_seed"], 73)


if __name__ == "__main__":
    unittest.main()
