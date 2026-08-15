from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch

from DQN.learner import Learner, double_dqn_next_values
from DQN.replay import TransitionBatch


def _learner_config(**overrides):
    values = dict(
        piece_placed_reward=0.01,
        line_clear_reward=0.75,
        terminal_penalty=1.0,
        replay_capacity=64,
        batch_size=2,
        learning_starts=4,
        update_every=2,
        target_update_every=100,
        learning_rate=1e-4,
        final_epsilon=0.01,
        gamma=0.99,
        gradient_clip_norm=10.0,
    )
    values.update(overrides)
    return SimpleNamespace(
        **values,
        to_dict=lambda: dict(values),
    )


def _batch(size: int) -> TransitionBatch:
    obs = {
        "board": np.zeros((size, 20, 10), dtype=np.uint8),
        "active": np.zeros((size, 20, 10), dtype=np.uint8),
        "current_piece": np.zeros((size, 7), dtype=np.int8),
        "next_piece": np.zeros((size, 7), dtype=np.int8),
        "rotation": np.zeros((size, 4), dtype=np.int8),
        "position": np.zeros((size, 2), dtype=np.float32),
    }
    obs["current_piece"][:, 0] = 1
    obs["next_piece"][:, 1] = 1
    obs["rotation"][:, 0] = 1
    return TransitionBatch(
        obs=obs,
        actions=np.zeros(size, dtype=np.int64),
        rewards=np.ones(size, dtype=np.float32),
        next_obs={key: value.copy() for key, value in obs.items()},
        terminated=np.zeros(size, dtype=np.bool_),
    )


def test_learner_catches_up_updates_for_large_ipc_batch():
    learner = Learner(_learner_config(learning_rate=5e-5), device="cpu", seed=0)
    learner.add(_batch(10))
    updates = 0
    assert learner.update()
    updates += 1
    while learner.update():
        updates += 1
    assert updates == 4
    assert learner.gradient_updates == 4
    assert learner._next_update_transition == 12
    stats = learner.pop_training_stats()
    assert stats["lr"] == pytest.approx(learner.learning_rate_at())
    assert stats["replay_size"] == 10
    assert all(np.isfinite(stats[key]) for key in ("loss", "q_mean", "target_mean", "gradient_norm"))
    assert learner.pop_training_stats() == {}


def test_update_count_is_independent_of_ipc_batch_partitioning():
    config = _learner_config()
    one_batch = Learner(config, device="cpu", seed=0)
    one_batch.add(_batch(10))
    while one_batch.update():
        pass

    partitioned = Learner(config, device="cpu", seed=0)
    for size in (2, 2, 2, 2, 2):
        partitioned.add(_batch(size))
        while partitioned.update():
            pass

    assert one_batch.transitions == partitioned.transitions == 10
    assert one_batch.gradient_updates == partitioned.gradient_updates == 4
    assert one_batch._next_update_transition == partitioned._next_update_transition == 12


def test_double_dqn_selects_online_action_and_evaluates_it_with_target():
    online_q = torch.tensor([[1.0, 5.0, 3.0], [4.0, 9.0, 8.0]])
    target_q = torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    mask = torch.tensor([[True, True, True], [True, False, True]])

    values = double_dqn_next_values(online_q, target_q, mask)

    # Row 0 proves this is not max(target_q), and row 1 proves selection is masked.
    assert values.tolist() == [20.0, 60.0]


def test_linear_decay_uses_absolute_first_five_million_transitions():
    learner = Learner(
        _learner_config(),
        device="cpu",
        seed=0,
    )
    assert learner.decay_progress_at(0) == 0.0
    assert learner.learning_rate_at(0) == pytest.approx(1e-4)
    assert learner.decay_progress_at(2_500_000) == 0.5
    assert learner.learning_rate_at(2_500_000) == pytest.approx(5.5e-5)
    assert learner.decay_progress_at(5_000_000) == 1.0
    assert learner.learning_rate_at(5_000_000) == pytest.approx(1e-5)


def test_linear_decay_checkpoint_resumes_at_absolute_progress(tmp_path: Path):
    config = _learner_config(replay_capacity=8)
    checkpoint = tmp_path / "linear.pt"
    source = Learner(config, device="cpu", seed=0)
    source.transitions = 2_500_000
    source._apply_learning_rate()
    source.checkpoint(checkpoint)

    restored = Learner(config, device="cpu", seed=1)
    restored.load_checkpoint(checkpoint, total_transitions=6_000_000)

    assert restored.decay_progress_at() == pytest.approx(0.5)
    assert restored.learning_rate_at() == pytest.approx(5.5e-5)
    assert restored.optimizer.param_groups[0]["lr"] == pytest.approx(5.5e-5)


def _trained_checkpoint(path: Path, config) -> Learner:
    learner = Learner(config, device="cpu", seed=7)
    learner.add(_batch(10))
    while learner.update():
        pass
    learner.checkpoint(path)
    return learner


def test_resume_restores_training_state_and_rebuilds_full_replay(tmp_path: Path):
    config = _learner_config(replay_capacity=8)
    checkpoint = tmp_path / "source.pt"
    source = _trained_checkpoint(checkpoint, config)
    restored = Learner(config, device="cpu", seed=99)

    restored.load_checkpoint(
        checkpoint,
        total_transitions=100,
    )

    assert restored.transitions == source.transitions
    assert restored.gradient_updates == source.gradient_updates
    assert restored.replay_warming_up is True
    for source_value, restored_value in zip(
        source.online.state_dict().values(), restored.online.state_dict().values()
    ):
        assert torch.equal(source_value, restored_value)
    for source_value, restored_value in zip(
        source.target.state_dict().values(), restored.target.state_dict().values()
    ):
        assert torch.equal(source_value, restored_value)
    source_optimizer = source.optimizer.state_dict()
    restored_optimizer = restored.optimizer.state_dict()
    assert restored_optimizer["param_groups"][0]["lr"] == pytest.approx(
        source.learning_rate_at()
    )
    for key, value in source_optimizer["param_groups"][0].items():
        if key != "lr":
            assert restored_optimizer["param_groups"][0][key] == value
    for parameter_id, source_state in source_optimizer["state"].items():
        for key, value in source_state.items():
            restored_value = restored_optimizer["state"][parameter_id][key]
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, restored_value)
            else:
                assert value == restored_value

    restored.add(_batch(4))
    assert restored.replay_warming_up is True
    assert restored.update() is False
    restored.add(_batch(4))
    assert restored.replay_warming_up is False
    assert restored._next_update_transition == restored.transitions

    updates_before = restored.gradient_updates
    assert restored.update() is True
    # The cursor advances from the post-warmup step; no historical debt remains.
    assert restored.update() is False
    assert restored.gradient_updates == updates_before + 1


def test_resume_rejects_incompatible_reward_configuration(tmp_path: Path):
    checkpoint = tmp_path / "source.pt"
    source_config = _learner_config()
    Learner(source_config, device="cpu", seed=0).checkpoint(checkpoint)
    restored = Learner(
        _learner_config(line_clear_reward=0.5), device="cpu", seed=0
    )

    with pytest.raises(ValueError, match="line_clear_reward"):
        restored.load_checkpoint(checkpoint, total_transitions=100)
