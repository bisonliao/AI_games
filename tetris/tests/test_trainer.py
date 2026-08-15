import numpy as np
import pytest

from DQN.replay import TransitionBatch
from DQN.gather import combine_actor_round
from DQN.trainer import _TensorBoardBuffer
from DQN.schedule import (
    decay_progress,
    epsilon_for_schedule,
    final_actor_epsilons,
    learning_rate_for_schedule,
)


class _Writer:
    def __init__(self):
        self.values = []

    def add_scalar(self, tag, value, step):
        self.values.append((tag, value, step))


def test_tensorboard_buffer_writes_only_on_flush():
    writer = _Writer()
    buffer = _TensorBoardBuffer(writer)
    buffer.mean("train/loss", 1.0)
    buffer.mean("train/loss", 3.0)
    buffer.latest("communication/wait", 5.0)
    assert writer.values == []
    buffer.flush(10_000)
    assert writer.values == [("train/loss", 2.0, 10_000), ("communication/wait", 5.0, 10_000)]


def test_epsilon_profile_linearly_reaches_one_greedy_actor():
    base = [0.05, 0.10, 0.20, 0.40]
    final = final_actor_epsilons(4, 0.01)
    assert final == [0.0, 0.01, 0.01, 0.01]
    assert [
        epsilon_for_schedule(base_epsilon, 0.0, final_epsilon)
        for base_epsilon, final_epsilon in zip(base, final)
    ] == base
    assert [
        epsilon_for_schedule(base_epsilon, 1.0, final_epsilon)
        for base_epsilon, final_epsilon in zip(base, final)
    ] == final


def test_fixed_linear_decay_values():
    assert decay_progress(0) == 0.0
    assert decay_progress(2_500_000) == 0.5
    assert decay_progress(5_000_000) == 1.0
    assert decay_progress(6_000_000) == 1.0

    assert epsilon_for_schedule(0.4, 0.5, 0.01) == pytest.approx(0.205)
    assert learning_rate_for_schedule(1e-4, 0.5) == pytest.approx(5.5e-5)


def test_final_profile_keeps_single_actor_greedy():
    assert final_actor_epsilons(1, 0.01) == [0.0]


def _actor_batch(actor_id: int, size: int = 2) -> TransitionBatch:
    values = np.full(size, actor_id, dtype=np.int64)
    obs = {"board": values[:, None]}
    return TransitionBatch(
        obs=obs,
        actions=values,
        rewards=values.astype(np.float32),
        next_obs={"board": obs["board"].copy()},
        terminated=np.zeros(size, dtype=np.bool_),
    )


def test_synchronous_round_combines_one_equal_batch_per_actor_in_actor_order():
    combined = combine_actor_round(
        [(7, _actor_batch(0)), (7, _actor_batch(1)), (7, _actor_batch(2))],
        expected_round=7,
        actor_batch_size=2,
    )
    assert combined.actions.tolist() == [0, 0, 1, 1, 2, 2]


def test_synchronous_round_rejects_stale_round_or_unequal_actor_batch():
    with pytest.raises(RuntimeError, match="returned round 6"):
        combine_actor_round(
            [(6, _actor_batch(0))], expected_round=7, actor_batch_size=2
        )
    with pytest.raises(RuntimeError, match="returned 1 transitions"):
        combine_actor_round(
            [(7, _actor_batch(0, size=1))], expected_round=7, actor_batch_size=2
        )
