from __future__ import annotations

import os

from env.gui_keys import ESCAPE_KEY, SPACE_KEY, exit_requested
from env.play import build_parser as build_play_parser
from td3.cli import (
    DEFAULT_TRAINING_TRANSITIONS,
    build_parser as build_td3_parser,
)
from td3.eval import build_parser as build_eval_parser
from td3.eval import env_config_from_metadata, resolve_checkpoint


def test_checkpoint_can_be_resolved_by_file_or_run_name(tmp_path) -> None:
    older_run = tmp_path / "20260730-120000-pid1-demo"
    newer_run = tmp_path / "20260730-120001-pid2-demo"
    older_run.mkdir()
    newer_run.mkdir()
    older = older_run / "checkpoint.pt"
    newer = newer_run / "checkpoint.pt"
    older.write_bytes(b"old")
    newer.write_bytes(b"new")
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    assert resolve_checkpoint(str(older), str(tmp_path)) == older.resolve()
    assert resolve_checkpoint("demo", str(tmp_path)) == newer.resolve()
    assert resolve_checkpoint(newer_run.name, str(tmp_path)) == newer.resolve()


def test_gui_cli_arguments() -> None:
    play_args = build_play_parser().parse_args(["--speed", "0.5", "--episodes", "2"])
    assert play_args.speed == 0.5
    assert play_args.episodes == 2

    eval_args = build_eval_parser().parse_args(
        ["checkpoint.pt", "--speed", "0.25", "--episodes", "3"]
    )
    assert eval_args.checkpoint == "checkpoint.pt"
    assert eval_args.speed == 0.25
    assert eval_args.episodes == 3

    train_args = build_td3_parser().parse_args(["train", "--device", "cpu"])
    assert train_args.observation_mode == "rgb"
    assert train_args.observation_size == 64
    # None 表示让 CLI 根据观测模式选择默认预算：RGB 500k、vector 100k。
    assert train_args.transitions is None
    assert DEFAULT_TRAINING_TRANSITIONS == {"rgb": 500_000, "vector": 100_000}


def test_gui_keys_work_without_b3g_escape_constant() -> None:
    assert ESCAPE_KEY == 27
    assert SPACE_KEY == 32
    assert exit_requested({ESCAPE_KEY: 2})
    assert exit_requested({ord("q"): 2})


def test_old_checkpoint_metadata_implies_vector_observation() -> None:
    config = env_config_from_metadata({"env_config": {"region_size": 6.0}})
    assert config.observation_mode == "vector"
    assert config.charge_exponent == 1.0
