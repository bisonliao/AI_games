from pathlib import Path

import pytest

from DQN.play import checkpoint_sort_key, resolve_checkpoint, result_text


def test_default_checkpoint_uses_largest_sequence(tmp_path: Path) -> None:
    history = tmp_path / "experiment"
    history.mkdir()
    for name in ("experiment_000002.pt", "experiment_000010.pt", "experiment_000003.pt"):
        (history / name).touch()
    assert resolve_checkpoint(tmp_path, "experiment", None).name == "experiment_000010.pt"


def test_checkpoint_can_be_selected_by_number_or_name(tmp_path: Path) -> None:
    history = tmp_path / "experiment"
    history.mkdir()
    checkpoint = history / "experiment_000007.pt"
    checkpoint.touch()
    assert resolve_checkpoint(tmp_path, "experiment", "7") == checkpoint
    assert resolve_checkpoint(tmp_path, "experiment", "experiment_000007.pt") == checkpoint


def test_missing_history_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No .pt checkpoints"):
        resolve_checkpoint(tmp_path, "missing", None)


def test_empty_run_name_uses_legacy_history_root(tmp_path: Path) -> None:
    for name in ("000003.pt", "000012.pt", "000007.pt"):
        (tmp_path / name).touch()
    assert resolve_checkpoint(tmp_path, "", None) == tmp_path / "000012.pt"
    assert resolve_checkpoint(tmp_path, "", "7") == tmp_path / "000007.pt"
    assert resolve_checkpoint(tmp_path, "", "000003.pt") == tmp_path / "000003.pt"


def test_checkpoint_sort_key_handles_names_without_numbers() -> None:
    assert checkpoint_sort_key(Path("final.pt")) < checkpoint_sort_key(Path("000001.pt"))


def test_result_text_covers_all_outcomes() -> None:
    assert "你赢" in result_text(1)
    assert "agent" in result_text(-1)
    assert result_text(0) == "和棋。"
