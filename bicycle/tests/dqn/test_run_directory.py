from datetime import datetime
from pathlib import Path

from dqn.train import create_run_dir


def test_default_run_name_contains_timestamp_algorithm_and_pid(tmp_path: Path):
    result = create_run_dir(
        tmp_path,
        now=datetime(2026, 8, 2, 12, 15, 30),
        pid=12345,
    )
    assert result == tmp_path / "20260802-121530_distributed-dqn_pid12345"


def test_run_name_avoids_same_second_collision(tmp_path: Path):
    existing = tmp_path / "20260802-121530_distributed-dqn_pid12345"
    existing.mkdir()
    result = create_run_dir(
        tmp_path,
        now=datetime(2026, 8, 2, 12, 15, 30),
        pid=12345,
    )
    assert result == tmp_path / "20260802-121530_distributed-dqn_pid12345_01"
