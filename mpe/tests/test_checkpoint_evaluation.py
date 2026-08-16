import json
import io
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.train_torch import (
    _checkpoint_payload,
    _evaluation_tensorboard_metrics,
    _save_checkpoint_and_evaluate,
    evaluate_checkpoint,
    get_trainers,
    make_env,
)
from experiments.play_torch import parse_args as parse_play_args, play
from maddpg.common.env_adapters_torch import infer_action_specs
from maddpg.common.tensorboard_logger_torch import TensorBoardIntervalLogger
from maddpg.common.tf_util_torch import load_state, resolve_state_path, save_state


def _args():
    return SimpleNamespace(
        scenario="simple",
        env_backend="pettingzoo",
        policy_mode="official",
        target_init="copy",
        max_episode_len=3,
        num_adversaries=0,
        good_policy="maddpg",
        adv_policy="maddpg",
        num_units=16,
        lr=1e-2,
        batch_size=4,
        gamma=0.95,
        display=False,
        checkpoint_eval_episodes=3,
        checkpoint_eval_seed=7000,
    )


def _checkpoint_fixture(args):
    env = make_env(args.scenario, args)
    try:
        observations, _ = env.reset(seed=1)
        agent_list = list(observations)
        obs_shapes = [env.observation_space(a).shape for a in agent_list]
        action_spaces = [env.action_space(a) for a in agent_list]
        action_specs = infer_action_specs(env, agent_list, args.policy_mode)
        trainers = get_trainers(
            env,
            agent_list,
            obs_shapes,
            action_spaces,
            action_specs,
            min(len(agent_list), args.num_adversaries),
            args,
            torch.device("cpu"),
        )
        checkpoint = _checkpoint_payload(
            args, action_specs, trainers, train_step=123, completed_episodes=9
        )
        return checkpoint, action_specs, trainers
    finally:
        env.close()


class CheckpointEvaluationTest(unittest.TestCase):
    def test_simple_adversary_checkpoint_evaluation_reports_task_metrics(self):
        args = _args()
        args.scenario = "simple_adversary"
        args.env_backend = "legacy"
        args.num_adversaries = 1
        args.max_episode_len = 2
        args.checkpoint_eval_episodes = 3
        checkpoint, _, _ = _checkpoint_fixture(args)

        evaluation = evaluate_checkpoint(
            checkpoint, args, torch.device("cpu")
        )
        task_metrics = evaluation["task_metrics"]
        expected = {
            "mean_adv_goal_distance",
            "mean_nearest_good_goal_distance",
            "mean_distance_gap",
            "good_closer_rate",
            "adversary_closer_rate",
            "tie_rate",
        }
        self.assertEqual(set(task_metrics), expected)
        self.assertEqual(
            evaluation["agent_roles"], ["adversary", "good", "good"]
        )
        self.assertAlmostEqual(
            task_metrics["good_closer_rate"]
            + task_metrics["adversary_closer_rate"]
            + task_metrics["tie_rate"],
            1.0,
        )

    def test_checkpoint_directory_selects_greatest_numeric_step(self):
        with tempfile.TemporaryDirectory() as directory:
            save_state(
                str(Path(directory) / "state_steps_9.pt"), {"step": 9}
            )
            save_state(
                str(Path(directory) / "state_steps_10.pt"), {"step": 10}
            )

            self.assertEqual(
                Path(resolve_state_path(directory)).name,
                "state_steps_10.pt",
            )
            self.assertEqual(load_state(directory)["step"], 10)

    def test_play_entry_point_is_headless_by_default(self):
        options = parse_play_args(["--checkpoint", "unused.pt"])
        self.assertFalse(options.render)
        self.assertFalse(options.stochastic)

    def test_play_loads_self_describing_checkpoint_and_writes_report(self):
        args = _args()
        checkpoint, _, _ = _checkpoint_fixture(args)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = save_state(directory, checkpoint)
            report_path = Path(directory) / "play-report.json"
            options = parse_play_args(
                [
                    "--checkpoint",
                    checkpoint_path,
                    "--episodes",
                    "2",
                    "--seed",
                    "8123",
                    "--report-json",
                    str(report_path),
                    "--no-cuda",
                ]
            )
            with redirect_stdout(io.StringIO()) as output:
                evaluation = play(options)

            self.assertFalse(evaluation["render"])
            self.assertTrue(evaluation["deterministic"])
            self.assertEqual(evaluation["evaluation_episodes"], 2)
            self.assertEqual(evaluation["metadata"]["scenario"], "simple")
            self.assertIn("[Play 1/2]", output.getvalue())
            self.assertIn("[Summary]", output.getvalue())
            self.assertEqual(
                json.loads(report_path.read_text(encoding="utf-8")),
                evaluation,
            )

    def test_play_runs_legacy_checkpoint_without_gui(self):
        args = _args()
        args.env_backend = "legacy"
        args.max_episode_len = 2
        checkpoint, _, _ = _checkpoint_fixture(args)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = save_state(directory, checkpoint)
            options = parse_play_args(
                [
                    "--checkpoint",
                    checkpoint_path,
                    "--episodes",
                    "1",
                    "--no-cuda",
                ]
            )
            with redirect_stdout(io.StringIO()):
                evaluation = play(options)

            self.assertFalse(evaluation["render"])
            self.assertEqual(evaluation["metadata"]["env_backend"], "legacy")
            self.assertEqual(evaluation["episode_lengths"], [2])

    def test_play_simple_adversary_labels_roles_and_reports_metrics(self):
        args = _args()
        args.scenario = "simple_adversary"
        args.env_backend = "legacy"
        args.num_adversaries = 1
        args.max_episode_len = 2
        checkpoint, _, _ = _checkpoint_fixture(args)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = save_state(directory, checkpoint)
            options = parse_play_args(
                [
                    "--checkpoint",
                    checkpoint_path,
                    "--episodes",
                    "1",
                    "--no-cuda",
                ]
            )
            with redirect_stdout(io.StringIO()) as output:
                evaluation = play(options)

            self.assertEqual(
                evaluation["agent_roles"], ["adversary", "good", "good"]
            )
            self.assertIn("mean_distance_gap", evaluation["task_metrics"])
            self.assertIn("(adversary)=", output.getvalue())
            self.assertIn("(good)=", output.getvalue())

    def test_play_accepts_recent_version_2_checkpoint(self):
        args = _args()
        args.env_backend = "legacy"
        args.max_episode_len = 25
        checkpoint, _, _ = _checkpoint_fixture(args)
        checkpoint["checkpoint_version"] = 2
        checkpoint["metadata"] = {
            key: checkpoint["metadata"][key]
            for key in (
                "env_backend",
                "scenario",
                "policy_mode",
                "target_init",
                "action_specs",
            )
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = save_state(directory, checkpoint)
            options = parse_play_args(
                [
                    "--checkpoint",
                    checkpoint_path,
                    "--episodes",
                    "1",
                    "--no-cuda",
                ]
            )
            with redirect_stdout(io.StringIO()) as output:
                evaluation = play(options)

            self.assertEqual(evaluation["evaluation_episodes"], 1)
            self.assertEqual(evaluation["episode_lengths"], [25])
            self.assertIn("version 2 checkpoint", output.getvalue())

    def test_evaluation_is_deterministic_and_preserves_training_rng(self):
        args = _args()
        checkpoint, _, _ = _checkpoint_fixture(args)
        random.seed(31)
        np.random.seed(31)
        torch.manual_seed(31)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state().clone()

        first = evaluate_checkpoint(checkpoint, args, torch.device("cpu"))

        self.assertEqual(random.getstate(), python_state)
        restored_numpy_state = np.random.get_state()
        self.assertEqual(restored_numpy_state[0], numpy_state[0])
        np.testing.assert_array_equal(restored_numpy_state[1], numpy_state[1])
        self.assertEqual(restored_numpy_state[2:], numpy_state[2:])
        torch.testing.assert_close(torch.get_rng_state(), torch_state)

        second = evaluate_checkpoint(checkpoint, args, torch.device("cpu"))
        self.assertEqual(first, second)
        self.assertEqual(first["evaluation_episodes"], 3)
        self.assertEqual(first["train_step"], 123)
        self.assertEqual(first["completed_episodes"], 9)
        self.assertEqual(len(first["agent_episode_reward_mean"]), 1)

    def test_saved_checkpoint_gets_json_and_tensorboard_report(self):
        args = _args()
        checkpoint, action_specs, trainers = _checkpoint_fixture(args)

        class RecordingWriter:
            def __init__(self):
                self.calls = []
                self.flush_count = 0

            def add_scalar(self, name, value, step):
                self.calls.append((name, value, step))

            def flush(self):
                self.flush_count += 1

        writer = RecordingWriter()
        tb_logger = TensorBoardIntervalLogger(
            interval=1000,
            initial_step=0,
            writer=writer,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path, evaluation = _save_checkpoint_and_evaluate(
                directory,
                args,
                action_specs,
                trainers,
                checkpoint["train_step"],
                checkpoint["completed_episodes"],
                torch.device("cpu"),
                tb_logger,
            )

            self.assertTrue(Path(checkpoint_path).is_file())
            self.assertEqual(
                Path(checkpoint_path).name,
                "state_steps_123.pt",
            )
            report_path = Path(directory) / "evaluation.json"
            self.assertTrue(report_path.is_file())
            self.assertTrue(
                (Path(directory) / "evaluation_steps_123.json").is_file()
            )
            self.assertEqual(
                json.loads(report_path.read_text(encoding="utf-8")), evaluation
            )

        tensorboard_metrics = _evaluation_tensorboard_metrics(evaluation)
        self.assertIn("eval/episode_reward_mean", tensorboard_metrics)
        self.assertIn("eval/agent0_episode_reward_mean", tensorboard_metrics)
        self.assertEqual(writer.flush_count, 1)
        self.assertEqual(
            {name for name, _, _ in writer.calls}, set(tensorboard_metrics)
        )


if __name__ == "__main__":
    unittest.main()
