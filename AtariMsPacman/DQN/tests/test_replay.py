import numpy as np

from DQN.messages import TransitionChunk
from DQN.replay import PrioritizedReplayBuffer


def make_chunk(start: int, count: int, shape=(4, 8, 8)) -> TransitionChunk:
    values = np.arange(start, start + count, dtype=np.uint8)
    observations = np.broadcast_to(
        values[:, None, None, None], (count, *shape)
    ).copy()
    return TransitionChunk(
        actor_id=0,
        observations=observations,
        actions=np.arange(count, dtype=np.int64) % 3,
        rewards=np.arange(count, dtype=np.float32),
        next_observations=(observations + 1).astype(np.uint8),
        terminated=np.zeros(count, dtype=np.bool_),
        epsilon=0.5,
        policy_version=0,
    )


def test_prioritized_replay_add_sample_update_and_wrap() -> None:
    replay = PrioritizedReplayBuffer(
        8, (4, 8, 8), alpha=0.6, priority_epsilon=1.0e-6
    )
    replay.add(make_chunk(0, 6))
    assert len(replay) == 6
    sample = replay.sample(4, beta=0.4, rng=np.random.default_rng(1))
    assert sample.observations.shape == (4, 4, 8, 8)
    assert sample.weights.shape == (4,)
    assert np.all(sample.weights > 0)

    replay.update_priorities(
        sample.indices, np.linspace(1.0, 4.0, num=4, dtype=np.float64)
    )
    assert replay.max_priority == 4.0
    replay.add(make_chunk(10, 6))
    assert len(replay) == 8
    assert replay.position == 4
    assert np.isfinite(replay.sum_tree[1])
    assert replay.sum_tree[1] > 0
