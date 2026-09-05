from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from _R2D2.config import PROJECT_ROOT, R2D2Config
from _R2D2.learner import transformed_double_q_targets
from _R2D2.network import RecurrentDuelingQNetwork
from _R2D2.replay import SequenceReplay
from _R2D2.sequence import SequenceAssembler, mixed_priority


def _stack(step: int, shape=(84, 84)) -> np.ndarray:
    return np.stack(
        [np.full(shape, step + offset, dtype=np.uint8) for offset in range(4)]
    )


def _build_sequences(*, terminal_step: int = 11):
    assembler = SequenceAssembler(
        3, burn_in_steps=2, learning_steps=4, forward_steps=2, gamma=0.9
    )
    sequences = []
    hidden = (
        np.zeros((1, 5), dtype=np.float32),
        np.zeros((1, 5), dtype=np.float32),
    )
    for step in range(terminal_step + 1):
        sequences.extend(
            assembler.add(
                _stack(step),
                np.eye(3, dtype=np.float32)[step % 3],
                float(step),
                step % 3,
                1.0,
                _stack(step + 1),
                hidden,
                terminated=step == terminal_step,
            )
        )
    return sequences


def test_config_enforces_raw_four_frame_environment() -> None:
    config = R2D2Config()
    assert config.observation_shape == (4, 84, 84)
    assert config.action_count == 9
    assert config.runs_dir == PROJECT_ROOT / "runs"
    assert config.checkpoints_dir == PROJECT_ROOT / "checkpoint"
    assert config.num_actors == 8
    assert config.total_transitions == 40_000_000
    assert config.replay_capacity_sequences * config.learning_steps == 1_000_000
    assert config.learning_starts == 50_000
    assert config.estimated_replay_memory_gib == pytest.approx(17.35, rel=0.02)
    assert config.checkpoint_interval_transitions == 2_000_000
    with pytest.raises(ValueError, match="raw"):
        replace(config, actor_env=replace(config.actor_env, clip_training_reward=True))


def test_network_shapes_hidden_and_value_rescaling() -> None:
    model = RecurrentDuelingQNetwork(hidden_size=32)
    q_values, hidden = model.step(
        torch.zeros(2, 4, 84, 84, dtype=torch.uint8),
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2),
    )
    assert q_values.shape == (2, 9)
    assert hidden[0].shape == (1, 2, 32)
    sequence_q, _ = model.unroll(
        torch.zeros(2, 5, 4, 84, 84, dtype=torch.uint8),
        torch.zeros(2, 5, dtype=torch.long),
        torch.zeros(2, 5),
        burn_in_steps=torch.tensor([2, 3]),
        lengths=torch.tensor([5, 3]),
        valid_mask=torch.tensor(
            [[True, True, True, True, True], [True, True, True, False, False]]
        ),
    )
    assert sequence_q.shape == (2, 5, 9)
    assert torch.count_nonzero(sequence_q[1, 3:]) == 0
    values = torch.linspace(-1_000, 1_000, 101, dtype=torch.float64)
    restored = model.inverse_value_rescale(model.value_rescale(values))
    torch.testing.assert_close(restored, values, rtol=1e-8, atol=1e-8)


def test_sequence_overlap_burn_in_tail_and_n_step() -> None:
    sequences = _build_sequences()
    assert len(sequences) == 3
    assert [s.learning_steps for s in sequences] == [4, 4, 4]
    assert [s.burn_in_steps for s in sequences] == [0, 2, 2]
    assert [s.start_transition for s in sequences] == [0, 4, 8]
    assert all(s.unpack_observations().shape[1:] == (4, 84, 84) for s in sequences)
    np.testing.assert_allclose(sequences[0].n_step_rewards, 1.9)
    np.testing.assert_allclose(sequences[-1].discounts[-2:], 0.0)
    assert sequences[-1].terminated[-1]


def test_replay_wrap_weights_and_stale_generation_guard() -> None:
    sequences = _build_sequences()
    replay = SequenceReplay(2, alpha=0.9, beta=0.6)
    index, generation = replay.add(sequences[0])
    replay.add(sequences[1])
    replay.add(sequences[2])
    before = replay._tree.sum.copy()
    replay.update_priorities(
        np.array([index]), np.array([99.0]), np.array([generation])
    )
    np.testing.assert_array_equal(replay._tree.sum, before)
    sample = replay.sample(2, np.random.default_rng(1))
    assert np.all(sample.weights > 0)
    assert np.all(sample.weights <= 1.0)
    assert sample.weights.max() == pytest.approx(1.0)
    assert mixed_priority(np.array([1.0, 3.0]), 0.9) == pytest.approx(2.9)


def test_is_weights_are_normalized_by_sampled_batch_minimum() -> None:
    class FixedRng:
        @staticmethod
        def random(size: int) -> np.ndarray:
            assert size == 2
            return np.array([0.5, 0.5])

    sequences = _build_sequences()
    replay = SequenceReplay(3, alpha=1.0, beta=0.6)
    replay.add(sequences[0], priority=1.0e-9)
    replay.add(sequences[1], priority=1.0)
    replay.add(sequences[2], priority=1.0)
    sample = replay.sample(2, FixedRng())
    # The tiny global-replay minimum was not sampled and must not suppress
    # this batch. Both sampled priorities are equal, so both weights are one.
    np.testing.assert_allclose(sample.weights, np.ones(2, dtype=np.float32))


def test_replay_counts_learning_transitions_including_short_tails() -> None:
    sequences = _build_sequences(terminal_step=9)
    replay = SequenceReplay(2)
    replay.add(sequences[0])
    replay.add(sequences[-1])
    assert replay.learning_transitions == sum(len(item) for item in (sequences[0], sequences[-1]))
    replay.add(sequences[1])
    assert replay.learning_transitions == sum(len(item) for item in (sequences[-1], sequences[1]))


def test_double_q_target_selects_online_action_and_honors_terminal_discount() -> None:
    online_next_q = torch.tensor([[0.0, 3.0, 2.0], [5.0, 1.0, 0.0]])
    # The target network's own argmax intentionally disagrees with online.
    target_unscaled = torch.tensor([[20.0, 7.0, 9.0], [4.0, 30.0, 2.0]])
    target_next_q = RecurrentDuelingQNetwork.value_rescale(target_unscaled)
    rewards = torch.tensor([2.0, 11.0])
    discounts = torch.tensor([0.5, 0.0])
    result = transformed_double_q_targets(
        rewards, discounts, online_next_q, target_next_q
    )
    expected = RecurrentDuelingQNetwork.value_rescale(torch.tensor([5.5, 11.0]))
    torch.testing.assert_close(result, expected)
