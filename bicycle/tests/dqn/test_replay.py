import numpy as np

from dqn.nstep import NStepTransition
from dqn.replay import ReplayBuffer


def item(index):
    return NStepTransition(
        np.full(5, index, np.float32),
        index % 3,
        float(index),
        np.full(5, index + 1, np.float32),
        0.99,
        False,
        False,
    )


def test_uniform_sample_has_unique_valid_items():
    replay = ReplayBuffer(16, (5,))
    for index in range(8):
        replay.add(item(index))
    batch = replay.sample(8, np.random.default_rng(4))
    sampled = batch.observations[:, 0].astype(int)
    assert sorted(sampled.tolist()) == list(range(8))
    assert batch.actions.shape == (8,)
    assert batch.discounts.shape == (8,)


def test_uniform_sampling_is_reproducible_from_rng_seed():
    replay = ReplayBuffer(16, (5,))
    replay.extend(item(index) for index in range(16))
    first = replay.sample(6, np.random.default_rng(7)).observations
    second = replay.sample(6, np.random.default_rng(7)).observations
    np.testing.assert_array_equal(first, second)


def test_ring_overwrite_and_state_roundtrip():
    replay = ReplayBuffer(4, (5,))
    for index in range(7):
        replay.add(item(index))
    restored = ReplayBuffer(4, (5,))
    restored.load_state_dict(replay.state_dict())
    assert len(restored) == 4
    np.testing.assert_allclose(restored.observations, replay.observations)
    batch = restored.sample(4, np.random.default_rng(1))
    assert sorted(batch.observations[:, 0].tolist()) == [3, 4, 5, 6]


def test_legacy_prioritized_state_loads_as_uniform_buffer():
    replay = ReplayBuffer(4, (5,))
    replay.extend(item(index) for index in range(3))
    legacy_state = replay.state_dict()
    legacy_state.update(
        {
            "alpha": 0.6,
            "priority_epsilon": 1e-6,
            "tree": np.ones(8),
            "max_priority": 2.0,
        }
    )
    restored = ReplayBuffer(4, (5,))
    restored.load_state_dict(legacy_state)
    assert len(restored) == 3
