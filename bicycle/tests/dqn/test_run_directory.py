"""Sortable timestamp/algorithm/PID run-directory naming tests."""

from datetime import datetime
from pathlib import Path

from dqn.config import DQNConfig
from dqn.train import build_parser, create_run_dir


def test_default_run_name_contains_timestamp_algorithm_and_pid(tmp_path: Path):
    result = create_run_dir(
        tmp_path,
        now=datetime(2026, 8, 2, 12, 15, 30),
        pid=12345,
    )
    assert result == tmp_path / "20260802-121530_distributed-dqn_env1_pid12345"


def test_run_name_avoids_same_second_collision(tmp_path: Path):
    existing = tmp_path / "20260802-121530_distributed-dqn_env1_pid12345"
    existing.mkdir()
    result = create_run_dir(
        tmp_path,
        now=datetime(2026, 8, 2, 12, 15, 30),
        pid=12345,
    )
    assert result == tmp_path / "20260802-121530_distributed-dqn_env1_pid12345_01"


def test_run_name_contains_selected_environment(tmp_path: Path):
    result = create_run_dir(
        tmp_path,
        env_id=2,
        now=datetime(2026, 8, 2, 12, 15, 30),
        pid=12345,
    )
    assert result == tmp_path / "20260802-121530_distributed-dqn_env2_pid12345"


def test_fresh_train_cli_defaults_match_dqn_config():
    args = build_parser().parse_args([])
    config = DQNConfig()
    assert args.env_id == 1
    assert args.replay_capacity == config.replay_capacity
    assert args.warmup == config.replay_warmup
    assert args.batch_size == config.batch_size
    assert args.checkpoint_interval == config.checkpoint_interval
    assert args.evaluation_interval_steps == config.evaluation_interval_steps
    assert args.evaluation_episodes == config.evaluation_episodes


def test_train_cli_selects_environment_two():
    args = build_parser().parse_args(["--env_id", "2"])
    assert args.env_id == 2
