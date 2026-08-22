from __future__ import annotations

import queue
import threading

import numpy as np

from afterBC.actor import WhiteBatchRollout
from afterBC.common import GatherBatch, GatherPermit, Transition, pack_transitions
from afterBC.gather import gather_worker


class SequencePolicy:
    def __init__(self, actions: list[int]) -> None:
        self.actions = actions

    def select_actions(self, boards, players, action_masks, *, epsilon=0.0):
        del epsilon
        results = []
        for board, player, mask in zip(boards, players, action_masks):
            own_count = int(np.count_nonzero(board == int(player)))
            action = self.actions[own_count]
            assert mask[action]
            results.append(action)
        return np.asarray(results, dtype=np.int64)


def test_white_transition_closes_after_black_winning_reply() -> None:
    rollout = WhiteBatchRollout(
        num_envs=1, board_size=9,
        black_policy=SequencePolicy([0, 1, 2, 3, 4]),
        white_policy=SequencePolicy([9, 10, 11, 12, 13]),
        n_step=1, gamma=0.99, seed=1,
    )
    try:
        transitions = []
        for _ in range(4):
            transitions.extend(rollout.advance(0.0))
        assert len(transitions) == 4
        terminal = transitions[-1]
        assert terminal.done
        assert terminal.reward == -1.0
        assert not terminal.next_state.any()
        assert not terminal.next_mask.any()
        assert rollout.take_completed() == [(1, 4, -1.0)]
        assert int(rollout.envs[0].current_player) == -1
    finally:
        rollout.close()


def test_white_win_ends_before_an_extra_black_reply() -> None:
    rollout = WhiteBatchRollout(
        num_envs=1, board_size=9,
        black_policy=SequencePolicy([9, 11, 13, 15, 17, 19]),
        white_policy=SequencePolicy([0, 1, 2, 3, 4]),
        n_step=1, gamma=0.99, seed=2,
    )
    try:
        transitions = []
        for _ in range(5):
            transitions.extend(rollout.advance(0.0))
        terminal = transitions[-1]
        assert terminal.done and terminal.reward == 1.0
        assert rollout.take_completed() == [(-1, 5, 1.0)]
        # A new game has exactly its opening black stone, not an extra reply.
        assert np.count_nonzero(rollout.envs[0].board == 1) == 1
        assert np.count_nonzero(rollout.envs[0].board == -1) == 0
    finally:
        rollout.close()


def _packet(actor_id: int, count: int):
    transitions = [
        Transition(
            np.zeros((5, 5), dtype=np.int8), actor_id * 2 + index, 0.0,
            np.zeros((5, 5), dtype=np.int8), np.ones(25, dtype=np.bool_), False, 0.99,
        )
        for index in range(count)
    ]
    return pack_transitions(
        transitions, actor_id=actor_id, policy_version=actor_id,
        epsilon=0.1, schedule_step=0,
    )


def test_gather_reads_actor_queues_in_order_and_stops_at_full_round() -> None:
    actor_queues = [queue.Queue(maxsize=1), queue.Queue(maxsize=1)]
    permit_queues = [queue.Queue(maxsize=1), queue.Queue(maxsize=1)]
    learner_queue = queue.Queue(maxsize=1)
    error_queue = queue.Queue()
    stop_event = threading.Event()
    actor_queues[0].put(_packet(0, 2))
    actor_queues[1].put(_packet(1, 2))
    thread = threading.Thread(
        target=gather_worker,
        args=(actor_queues, permit_queues, learner_queue, error_queue, stop_event),
        kwargs={"actor_batch_size": 2, "start_global_step": 0, "target_global_step": 4},
    )
    thread.start()
    batch = learner_queue.get(timeout=5)
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert isinstance(batch, GatherBatch) and batch.final and batch.global_step == 4
    assert batch.packet.source_actor_ids.tolist() == [0, 0, 1, 1]
    assert batch.packet.actions.tolist() == [0, 1, 2, 3]
    assert [item.kind for item in (permit_queues[0].get(), permit_queues[1].get())] == [
        "stop", "stop",
    ]
    assert error_queue.empty()
