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
        schedule_trigger_mean_lines=100.0,
        schedule_trigger_mean_survival_pieces=300.0,
        schedule_trigger_patience=2,
        schedule_force_transition=10_000_000,
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
    assert stats["lr"] == 5e-5
    assert stats["replay_size"] == 10
    assert all(np.isfinite(stats[key]) for key in ("loss", "q_mean", "target_mean", "gradient_norm"))
    assert learner.pop_training_stats() == {}


def test_double_dqn_selects_online_action_and_evaluates_it_with_target():
    online_q = torch.tensor([[1.0, 5.0, 3.0], [4.0, 9.0, 8.0]])
    target_q = torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    mask = torch.tensor([[True, True, True], [True, False, True]])

    values = double_dqn_next_values(online_q, target_q, mask)

    # Row 0 proves this is not max(target_q), and row 1 proves selection is masked.
    assert values.tolist() == [20.0, 60.0]


def test_schedule_requires_consecutive_capability_evaluations():
    learner = Learner(
        _learner_config(update_every=4),
        device="cpu",
        seed=0,
    )
    assert learner.learning_rate_at() == pytest.approx(1e-4)
    assert learner.updates_per_transition_at(1_000_000) == pytest.approx(0.25)
    assert learner.gamma == pytest.approx(0.99)

    assert not learner.observe_schedule_evaluation(
        mean_lines=120.0,
        mean_survival_pieces=350.0,
        checkpoint_transition=100,
    )
    assert learner.schedule_consecutive_qualifying_evals == 1
    assert not learner.observe_schedule_evaluation(
        mean_lines=99.0,
        mean_survival_pieces=500.0,
        checkpoint_transition=200,
    )
    assert learner.schedule_consecutive_qualifying_evals == 0
    assert not learner.observe_schedule_evaluation(
        mean_lines=100.0,
        mean_survival_pieces=300.0,
        checkpoint_transition=300,
    )
    learner.transitions = 375
    assert learner.observe_schedule_evaluation(
        mean_lines=101.0,
        mean_survival_pieces=301.0,
        checkpoint_transition=350,
    )
    assert learner.schedule_triggered
    assert learner.schedule_trigger_source == "capability"
    assert learner.schedule_trigger_checkpoint_transition == 350
    assert learner.schedule_applied_transition == 375
    assert learner.learning_rate_at() == pytest.approx(1e-5)
    assert learner.optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)

    # The transition is permanent: later weak evaluations cannot undo it.
    assert not learner.observe_schedule_evaluation(
        mean_lines=0.0,
        mean_survival_pieces=0.0,
        checkpoint_transition=400,
    )
    assert learner.schedule_triggered
    assert learner.learning_rate_at() == pytest.approx(1e-5)


def test_schedule_is_forced_at_transition_limit_without_capability_trigger():
    learner = Learner(
        _learner_config(schedule_force_transition=100),
        device="cpu",
        seed=0,
    )
    learner.transitions = 99
    learner.add(_batch(2))

    assert learner.transitions == 101
    assert learner.schedule_triggered
    assert learner.schedule_trigger_source == "transition_limit"
    assert learner.schedule_trigger_checkpoint_transition is None
    assert learner.schedule_applied_transition == 101
    assert learner.optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)


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
    assert restored_optimizer["param_groups"][0]["lr"] == pytest.approx(1e-4)
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


def test_checkpoint_restores_capability_schedule_state(tmp_path: Path):
    config = _learner_config(replay_capacity=8, update_every=4)
    checkpoint = tmp_path / "pending_schedule.pt"
    source = Learner(config, device="cpu", seed=0)
    source.add(_batch(10))
    assert not source.observe_schedule_evaluation(
        mean_lines=110.0,
        mean_survival_pieces=320.0,
        checkpoint_transition=10,
    )
    source.checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert "stability_controls" not in payload
    assert payload["schedule_state"] == {
        "triggered": False,
        "trigger_source": None,
        "consecutive_qualifying_evals": 1,
        "trigger_checkpoint_transition": None,
        "applied_transition": None,
    }

    restored = Learner(config, device="cpu", seed=1)
    restored.load_checkpoint(
        checkpoint,
        total_transitions=100,
    )

    assert restored.transitions == 10
    assert not restored.schedule_triggered
    assert restored.schedule_consecutive_qualifying_evals == 1
    assert restored.learning_rate_at() == pytest.approx(1e-4)
    assert restored.observe_schedule_evaluation(
        mean_lines=100.0,
        mean_survival_pieces=300.0,
        checkpoint_transition=10,
    )
    assert restored.schedule_triggered
    assert restored.schedule_trigger_source == "capability"
    assert restored.schedule_trigger_checkpoint_transition == 10
    assert restored.schedule_applied_transition == 10
    assert restored.optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)

    triggered_checkpoint = tmp_path / "triggered_schedule.pt"
    restored.checkpoint(triggered_checkpoint)
    triggered_restore = Learner(config, device="cpu", seed=2)
    triggered_restore.load_checkpoint(triggered_checkpoint, total_transitions=100)
    assert triggered_restore.schedule_triggered
    assert triggered_restore.schedule_trigger_source == "capability"
    assert triggered_restore.schedule_consecutive_qualifying_evals == 2
    assert triggered_restore.schedule_trigger_checkpoint_transition == 10
    assert triggered_restore.schedule_applied_transition == 10
    assert triggered_restore.learning_rate_at() == pytest.approx(1e-5)
    assert triggered_restore.optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)
    # Replay warmup temporarily reports zero update rate.
    triggered_restore._resume_replay_warmup = False
    assert triggered_restore.updates_per_transition_at() == pytest.approx(0.25)
    assert triggered_restore.gamma == pytest.approx(0.99)


def test_resume_past_force_transition_triggers_old_checkpoint(tmp_path: Path):
    config = _learner_config(
        replay_capacity=8,
        schedule_force_transition=10,
    )
    checkpoint = tmp_path / "legacy_untriggered.pt"
    source = Learner(config, device="cpu", seed=0)
    # Simulate a checkpoint written before hard-limit support existed.
    source.transitions = 10
    source.checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload.pop("schedule_state")
    torch.save(payload, checkpoint)

    restored = Learner(config, device="cpu", seed=1)
    restored.load_checkpoint(checkpoint, total_transitions=100)

    assert restored.schedule_triggered
    assert restored.schedule_trigger_source == "transition_limit"
    assert restored.schedule_trigger_checkpoint_transition is None
    assert restored.schedule_applied_transition == 10
    assert restored.optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)


def test_resume_rejects_incompatible_reward_configuration(tmp_path: Path):
    checkpoint = tmp_path / "source.pt"
    source_config = _learner_config()
    Learner(source_config, device="cpu", seed=0).checkpoint(checkpoint)
    restored = Learner(
        _learner_config(line_clear_reward=0.5), device="cpu", seed=0
    )

    with pytest.raises(ValueError, match="line_clear_reward"):
        restored.load_checkpoint(checkpoint, total_transitions=100)
