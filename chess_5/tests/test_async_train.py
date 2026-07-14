from __future__ import annotations

import numpy as np
import torch
import queue

from DQN.agent import DQNAgent
from DQN.async_train import (
    InferencePolicy,
    Transition,
    _queue_put,
    cpu_state_dict,
    pack_transitions,
)


def test_weight_snapshot_round_trip_does_not_use_torch_tensors() -> None:
    agent = DQNAgent(
        5, hidden_channels=8, num_res_blocks=1, replay_size=8,
        min_replay_size=1, batch_size=1, device="cpu",
    )
    snapshot = cpu_state_dict(agent.online_net)
    assert snapshot
    assert all(isinstance(value, np.ndarray) for value in snapshot.values())

    policy = InferencePolicy(
        5, agent.model_kwargs, snapshot, seed=123, device="cpu"
    )
    for expected, actual in zip(agent.online_net.parameters(), policy.net.parameters()):
        torch.testing.assert_close(expected, actual)


def test_transition_batch_preserves_terminal_boundaries() -> None:
    zero_state = np.zeros((3, 5, 5), dtype=np.float32)
    zero_mask = np.zeros(25, dtype=np.bool_)
    terminal = Transition(zero_state, 7, 1.0, zero_state, zero_mask, True)
    batch = pack_transitions(
        [terminal], [(1, 8, 1.0)], actor_id=2, policy_version=5000,
        blocked_seconds=0.25,
    )
    assert batch["states"].shape == (1, 3, 5, 5)
    assert batch["next_masks"].shape == (1, 25)
    assert batch["dones"].tolist() == [True]
    assert not batch["next_masks"][0].any()
    assert batch["episodes"] == [(1, 8, 1.0)]
    assert batch["actor_id"] == 2
    assert batch["policy_version"] == 5000


def test_forced_updates_bypass_legacy_train_frequency() -> None:
    agent = DQNAgent(
        5, hidden_channels=8, num_res_blocks=1, replay_size=8,
        min_replay_size=1, batch_size=1, train_freq=100, device="cpu",
    )
    state = np.zeros((3, 5, 5), dtype=np.float32)
    mask = np.ones(25, dtype=np.bool_)
    agent.add_transition(state, 0, 0.0, state, mask, False)
    assert agent.train_step() is None
    assert agent.train_step(force=True) is not None
    assert agent.update_steps == 1


def test_training_can_skip_expensive_metric_collection() -> None:
    agent = DQNAgent(
        5, hidden_channels=8, num_res_blocks=1, replay_size=8,
        min_replay_size=1, batch_size=1, device="cpu",
    )
    state = np.zeros((3, 5, 5), dtype=np.float32)
    mask = np.ones(25, dtype=np.bool_)
    agent.add_transition(state, 0, 0.0, state, mask, False)
    assert agent.train_step(force=True, collect_metrics=False) is None
    assert agent.update_steps == 1


def test_queue_put_reports_only_first_timeout_for_one_batch() -> None:
    class ResultQueue:
        def __init__(self) -> None:
            self.calls = 0

        def put(self, message, timeout: float) -> None:
            del message, timeout
            self.calls += 1
            if self.calls <= 2:
                raise queue.Full

    class StatusQueue:
        def __init__(self) -> None:
            self.events = []

        def put_nowait(self, event) -> None:
            self.events.append(event)

    class StopEvent:
        def is_set(self) -> bool:
            return False

    result_queue = ResultQueue()
    status_queue = StatusQueue()
    blocked = _queue_put(result_queue, status_queue, {"type": "batch"}, StopEvent(), 3)
    assert blocked >= 0
    assert result_queue.calls == 3
    assert len(status_queue.events) == 1
    assert status_queue.events[0]["actor_id"] == 3
