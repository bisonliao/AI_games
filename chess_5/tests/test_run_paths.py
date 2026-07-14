from pathlib import Path

import pytest

from DQN.run_paths import checkpoint_filename, named_directory, validate_run_name


@pytest.mark.parametrize("name", ["baseline", "run_01", "五子棋-实验", "dqn.v2"])
def test_valid_run_names(name: str) -> None:
    assert validate_run_name(name) == name


@pytest.mark.parametrize("name", ["", " ", ".", "..", "a/b", "a\\b", "name with space"])
def test_unsafe_run_names_are_rejected(name: str) -> None:
    with pytest.raises(ValueError):
        validate_run_name(name)


def test_run_name_is_in_directory_and_checkpoint_filename(tmp_path: Path) -> None:
    assert named_directory(tmp_path, "trial") == tmp_path / "trial"
    assert checkpoint_filename("trial", 12) == "trial_000012.pt"
