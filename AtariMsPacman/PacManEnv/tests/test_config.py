from dataclasses import replace

import pytest

from PacManEnv import MsPacmanEnvConfig


@pytest.mark.parametrize("frame_skip", [1, 2, 4])
def test_supported_frame_skips(frame_skip: int) -> None:
    assert replace(MsPacmanEnvConfig(), frame_skip=frame_skip).frame_skip == frame_skip


@pytest.mark.parametrize("frame_skip", [0, 3, 5])
def test_rejects_unsupported_frame_skips(frame_skip: int) -> None:
    with pytest.raises(ValueError, match="frame_skip"):
        replace(MsPacmanEnvConfig(), frame_skip=frame_skip)


def test_rejects_invalid_reward_and_process_settings() -> None:
    with pytest.raises(ValueError, match="step_cost"):
        replace(MsPacmanEnvConfig(), step_cost=-0.1)
    with pytest.raises(ValueError, match="multiprocessing_context"):
        replace(MsPacmanEnvConfig(), multiprocessing_context="fork")
