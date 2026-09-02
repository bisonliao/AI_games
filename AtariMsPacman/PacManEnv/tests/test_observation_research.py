"""ROM-backed regression tests for observation and frame-skip decisions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ale_py.env import AtariEnv


NORMAL_GHOST_COLORS = {
    "orange": ((180, 122, 48), 131),
    "cyan": ((84, 184, 153), 151),
    "pink": ((198, 89, 179), 132),
    "red": ((200, 72, 72), 110),
}
EDIBLE_GHOST_COLOR = (66, 114, 194)


def make_raw_env(seed: int) -> AtariEnv:
    env = AtariEnv(
        game="ms_pacman",
        frameskip=1,
        repeat_action_probability=0.0,
        max_num_frames_per_episode=None,
        mode=0,
        difficulty=0,
    )
    env.reset(seed=seed)
    return env


def test_edible_ghost_state_is_preserved_in_ale_grayscale() -> None:
    env = make_raw_env(seed=1)
    try:
        # Start the game and follow a deterministic route to the lower-left pill.
        for action, count in ((3, 330), (4, 100), (3, 50)):
            for _ in range(count):
                env.step(action)

        normal_rgb = env.ale.getScreenRGB()
        normal_gray = env.ale.getScreenGrayscale()
        for color, expected_gray in NORMAL_GHOST_COLORS.values():
            mask = np.all(normal_rgb == color, axis=2)
            assert np.count_nonzero(mask) > 0
            assert np.unique(normal_gray[mask]).tolist() == [expected_gray]

        reward = 0.0
        for _ in range(10):
            _, reward_delta, _, _, _ = env.step(1)
            reward += reward_delta
        assert reward >= 50.0

        edible_rgb = env.ale.getScreenRGB()
        edible_gray = env.ale.getScreenGrayscale()
        edible_mask = np.all(edible_rgb == EDIBLE_GHOST_COLOR, axis=2)
        assert np.count_nonzero(edible_mask) > 0
        assert np.unique(edible_gray[edible_mask]).tolist() == [109]
    finally:
        env.close()


@dataclass(frozen=True)
class TurnCase:
    approach_action: int
    turn_action: int
    axis: str
    low: int
    high: int
    fixed_coordinate: int
    target_coordinate: int
    expected_passage_frames: int
    expected_turn_window: int


TURN_CASES = (
    TurnCase(3, 1, "x", 45, 61, 99, 53, 28, 14),
    TurnCase(2, 1, "x", 89, 105, 99, 97, 28, 16),
    TurnCase(1, 2, "y", 43, 59, 53, 51, 20, 10),
)


def player_position(env: AtariEnv) -> tuple[int, int]:
    ram = env.ale.getRAM()
    return int(ram[10]) - 13, int(ram[16]) + 1


def has_turned(
    before: tuple[int, int], after: tuple[int, int], action: int
) -> bool:
    x_before, y_before = before
    x_after, y_after = after
    return (
        (action == 1 and y_after < y_before)
        or (action == 2 and x_after > x_before)
        or (action == 3 and x_after < x_before)
        or (action == 4 and y_after > y_before)
    )


def test_typical_intersections_have_enough_window_for_frame_skip_four() -> None:
    for case in TURN_CASES:
        env = make_raw_env(seed=17)
        try:
            snapshots = []
            for _ in range(700):
                env.step(case.approach_action)
                x, y = player_position(env)
                moving_coordinate = x if case.axis == "x" else y
                fixed_coordinate = y if case.axis == "x" else x
                if (
                    case.low <= moving_coordinate <= case.high
                    and fixed_coordinate == case.fixed_coordinate
                ):
                    snapshots.append(
                        (env.ale.getEpisodeFrameNumber(), x, y, env.ale.cloneSystemState())
                    )

                passed = (
                    case.approach_action in (1, 3) and moving_coordinate < case.low
                ) or (
                    case.approach_action == 2 and moving_coordinate > case.high
                )
                if snapshots and passed:
                    break

            assert snapshots
            passage_frames = snapshots[-1][0] - snapshots[0][0] + 1
            assert passage_frames == case.expected_passage_frames

            successful_frames = []
            for source_frame, x, y, state in snapshots:
                env.ale.restoreSystemState(state)
                before = (x, y)
                turn_position = None
                for _ in range(40):
                    env.step(case.turn_action)
                    current = player_position(env)
                    if has_turned(before, current, case.turn_action):
                        turn_position = current
                        break
                if turn_position is not None:
                    turn_coordinate = (
                        turn_position[0] if case.axis == "x" else turn_position[1]
                    )
                    if turn_coordinate == case.target_coordinate:
                        successful_frames.append(source_frame)

            turn_window = successful_frames[-1] - successful_frames[0] + 1
            assert turn_window == case.expected_turn_window

            # Any alignment of a four-frame decision boundary after entering the
            # intersection still issues the turn before the measured window closes.
            entry_state = snapshots[0][3]
            for phase in range(4):
                env.ale.restoreSystemState(entry_state)
                for _ in range(phase):
                    env.step(case.approach_action)
                before = player_position(env)
                turn_position = None
                for _ in range(40):
                    env.step(case.turn_action)
                    current = player_position(env)
                    if has_turned(before, current, case.turn_action):
                        turn_position = current
                        break
                assert turn_position is not None
                turn_coordinate = (
                    turn_position[0] if case.axis == "x" else turn_position[1]
                )
                assert turn_coordinate == case.target_coordinate
        finally:
            env.close()
