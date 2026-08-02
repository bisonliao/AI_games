import numpy as np

from dqn.nstep import NStepAccumulator, StepTransition


def transition(index, reward=1.0, terminated=False, truncated=False):
    return StepTransition(
        np.full(5, index, np.float32),
        index % 3,
        reward,
        np.full(5, index + 1, np.float32),
        terminated,
        truncated,
    )


def test_three_step_return():
    accumulator = NStepAccumulator(3, 0.9)
    assert accumulator.add(transition(0)) == []
    assert accumulator.add(transition(1)) == []
    result = accumulator.add(transition(2))[0]
    assert result.reward == 1 + 0.9 + 0.9**2
    assert result.discount == 0.9**3
    np.testing.assert_array_equal(result.next_observation, np.full(5, 3))


def test_termination_disables_bootstrap_but_truncation_keeps_it():
    terminated = NStepAccumulator(3, 0.9)
    term_result = terminated.add(transition(0, terminated=True))[0]
    assert term_result.discount == 0
    truncated = NStepAccumulator(3, 0.9)
    trunc_result = truncated.add(transition(0, truncated=True))[0]
    assert trunc_result.discount == 0.9

