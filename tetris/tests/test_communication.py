import multiprocessing as mp
import queue
import threading

import numpy as np

from DQN.actor import (
    _latest_weight_message,
    _put_latest_weight,
    _put_with_wait,
    _vector_info_value,
)
from DQN.replay import TransitionBatch, concatenate_transition_batches


def test_put_with_wait_reports_full_queue_timeouts():
    ctx = mp.get_context("spawn")
    full = ctx.Queue(maxsize=1)
    full.put("occupied")
    stop = ctx.Event()
    threading.Timer(0.04, stop.set).start()
    waited, timeouts, sent = _put_with_wait(full, "blocked", poll_timeout=0.01, stop_event=stop)
    assert sent is False
    assert timeouts >= 1
    assert waited >= 0.01
    full.close()


def test_put_with_wait_retries_until_transition_is_accepted():
    ctx = mp.get_context("spawn")
    full = ctx.Queue(maxsize=1)
    full.put("occupied")
    stop = ctx.Event()

    def release_queue():
        full.get()

    threading.Timer(0.03, release_queue).start()
    waited, timeouts, sent = _put_with_wait(full, "transition", poll_timeout=0.01, stop_event=stop)
    assert sent is True
    assert full.get(timeout=1.0) == "transition"
    assert timeouts >= 1
    assert waited >= 0.03
    full.close()


def test_weight_mailbox_replaces_stale_snapshot_without_dropping_latest():
    ctx = mp.get_context("spawn")
    mailbox = ctx.Queue(maxsize=1)
    mailbox.put((1, "old"))
    assert _put_latest_weight(mailbox, (2, "new"), poll_timeout=0.01)
    assert mailbox.get(timeout=1.0) == (2, "new")
    mailbox.close()


def test_actor_drains_weight_messages_and_selects_highest_version():
    mailbox = queue.Queue()
    mailbox.put((3, "newest"))
    mailbox.put((2, "stale"))
    assert _latest_weight_message(mailbox) == (3, "newest")


def test_actor_merges_vector_batches_before_ipc():
    def make(value):
        obs = {"board": np.full((2, 20, 10), value, dtype=np.uint8)}
        return TransitionBatch(
            obs=obs,
            actions=np.asarray([value, value + 1]),
            rewards=np.asarray([value, value + 1], dtype=np.float32),
            next_obs={"board": obs["board"].copy()},
            terminated=np.asarray([False, True]),
        )

    merged = concatenate_transition_batches([make(1), make(3)])
    assert merged.obs["board"].shape == (4, 20, 10)
    assert merged.actions.tolist() == [1, 2, 3, 4]
    assert merged.terminated.tolist() == [False, True, False, True]


def test_actor_reads_same_step_final_info_for_terminated_env():
    infos = {
        "survival_pieces": np.asarray([0, 4]),
        "_survival_pieces": np.asarray([True, True]),
        "final_info": {
            "survival_pieces": np.asarray([12, 0]),
            "_survival_pieces": np.asarray([True, False]),
        },
    }
    assert _vector_info_value(infos, "survival_pieces", 0, terminated=True, default=-1) == 12
    assert _vector_info_value(infos, "survival_pieces", 1, terminated=False, default=-1) == 4
