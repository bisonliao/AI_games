import numpy as np

from DQN.returns import NStepAccumulator, Transition, board_potential, shaped_reward


def make_transition(reward: float, done: bool, marker: int) -> Transition:
    state = np.full((3, 5, 5), marker, dtype=np.float32)
    next_state = np.full((3, 5, 5), marker + 1, dtype=np.float32)
    return Transition(state, marker, reward, next_state, np.ones(25, dtype=np.bool_), done)


def test_five_step_accumulator_flushes_short_terminal_tail() -> None:
    accumulator = NStepAccumulator(num_envs=1, n_step=5, gamma=0.5)
    assert accumulator.add(0, make_transition(0.0, False, 0)) == []
    assert accumulator.add(0, make_transition(0.0, False, 1)) == []
    emitted = accumulator.add(0, make_transition(1.0, True, 2))

    assert len(emitted) == 3
    np.testing.assert_allclose([item.reward for item in emitted], [0.25, 0.5, 1.0])
    np.testing.assert_allclose([item.discount for item in emitted], [0.125, 0.25, 0.5])
    assert all(item.done for item in emitted)
    assert [item.action for item in emitted] == [0, 1, 2]


def test_accumulators_do_not_mix_vector_environments() -> None:
    accumulator = NStepAccumulator(num_envs=2, n_step=2, gamma=0.9)
    assert accumulator.add(0, make_transition(1.0, False, 10)) == []
    assert accumulator.add(1, make_transition(2.0, False, 20)) == []
    emitted = accumulator.add(0, make_transition(3.0, False, 11))
    assert len(emitted) == 1
    assert emitted[0].action == 10
    assert emitted[0].reward == 1.0 + 0.9 * 3.0


def test_board_potential_changes_sign_when_players_are_swapped() -> None:
    state = np.zeros((3, 5, 5), dtype=np.float32)
    state[0, 0, :4] = 1.0
    state[2] = 1.0 - state[0]
    swapped = state.copy()
    swapped[[0, 1]] = swapped[[1, 0]]
    assert board_potential(state) > 0
    np.testing.assert_allclose(board_potential(swapped), -board_potential(state))


def test_shaping_can_be_disabled_and_terminal_potential_is_zero() -> None:
    state = np.zeros((3, 5, 5), dtype=np.float32)
    state[0, 0, :4] = 1.0
    state[2] = 1.0 - state[0]
    terminal_state = np.zeros_like(state)
    assert shaped_reward(1.0, state, terminal_state, True, 0.99, 0.0) == 1.0
    expected = 1.0 - 0.02 * board_potential(state)
    np.testing.assert_allclose(
        shaped_reward(1.0, state, terminal_state, True, 0.99, 0.02), expected
    )
