"""Bulk-synchronous actor collection decoupled from learner updates."""
from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
import queue
import time
import traceback

from .actor import _put_metric, _put_with_wait, actor_process
from .replay import TransitionBatch, concatenate_transition_batches


@dataclass
class GatheredRound:
    """One complete, actor-balanced round delivered reliably to the learner."""

    round_id: int
    batch: TransitionBatch
    is_final: bool


def combine_actor_round(
    results: list[tuple[int, TransitionBatch]],
    *,
    expected_round: int,
    actor_batch_size: int,
) -> TransitionBatch:
    """Validate one result per actor and concatenate in stable actor order."""
    if not results:
        raise ValueError("a synchronous round requires at least one actor result")
    batches: list[TransitionBatch] = []
    for actor_id, (round_id, batch) in enumerate(results):
        if int(round_id) != expected_round:
            raise RuntimeError(
                f"actor {actor_id} returned round {round_id}, expected {expected_round}"
            )
        if len(batch.actions) != actor_batch_size:
            raise RuntimeError(
                f"actor {actor_id} returned {len(batch.actions)} transitions, "
                f"expected {actor_batch_size}"
            )
        batches.append(batch)
    return concatenate_transition_batches(batches)


def gather_process(
    *,
    actors_count: int,
    env_count: int,
    base_epsilons: list[float],
    final_epsilons: list[float],
    seed: int,
    weight_queues,
    metric_queue,
    gathered_round_queue,
    error_queue,
    decay_progress,
    stop_event,
    start_transition: int,
    total_transitions: int,
    transition_batch_size: int,
    transition_put_poll_timeout: float,
    learner_idle_sleep: float,
    actor_stats_every: int,
    gamma: float,
    piece_placed_reward: float,
    line_clear_reward: float,
    terminal_penalty: float,
) -> None:
    """Collect equal actor batches and pipeline complete rounds to the learner."""
    ctx = mp.get_context("spawn")
    transition_queues = [ctx.Queue(maxsize=1) for _ in range(actors_count)]
    command_queues = [ctx.Queue(maxsize=1) for _ in range(actors_count)]
    actors = []
    enqueued_transitions = int(start_transition)
    batch_queue_wait_seconds = 0.0
    batch_queue_wait_timeouts = 0
    rounds_enqueued = 0

    try:
        for actor_id in range(actors_count):
            actor_seed = seed + actor_id * 1_000_000
            process = ctx.Process(
                target=actor_process,
                args=(
                    actor_id,
                    env_count,
                    actor_seed,
                    base_epsilons[actor_id],
                    transition_queues[actor_id],
                    metric_queue,
                    command_queues[actor_id],
                    weight_queues[actor_id],
                    decay_progress,
                    stop_event,
                    transition_put_poll_timeout,
                    actor_stats_every,
                    transition_batch_size,
                    gamma,
                    piece_placed_reward,
                    line_clear_reward,
                    terminal_penalty,
                    final_epsilons[actor_id],
                ),
                daemon=True,
            )
            process.start()
            actors.append(process)

        # Gather timing is low-priority telemetry. Critical rounds use their
        # own reliable queue and are never subject to this cancellation.
        metric_queue.cancel_join_thread()

        round_id = 0
        while not stop_event.is_set():
            round_started = time.perf_counter()
            for command_queue in command_queues:
                command_queue.put(round_id)

            round_results: list[tuple[int, TransitionBatch] | None] = [None] * actors_count
            actor_arrival_seconds = [0.0] * actors_count
            while any(result is None for result in round_results) and not stop_event.is_set():
                made_progress = False
                for actor_id, transition_queue in enumerate(transition_queues):
                    if round_results[actor_id] is not None:
                        continue
                    try:
                        result = transition_queue.get_nowait()
                    except queue.Empty:
                        continue
                    round_results[actor_id] = result
                    actor_arrival_seconds[actor_id] = time.perf_counter() - round_started
                    made_progress = True
                if made_progress:
                    continue
                dead_actor_ids = [
                    actor_id
                    for actor_id, process in enumerate(actors)
                    if round_results[actor_id] is None and not process.is_alive()
                ]
                if dead_actor_ids:
                    raise RuntimeError(
                        "actor processes exited before completing synchronous round "
                        f"{round_id}: {dead_actor_ids}"
                    )
                if learner_idle_sleep > 0:
                    time.sleep(learner_idle_sleep)

            if stop_event.is_set():
                break
            complete_results = [result for result in round_results if result is not None]
            batch = combine_actor_round(
                complete_results,
                expected_round=round_id,
                actor_batch_size=transition_batch_size,
            )
            round_collection_seconds = time.perf_counter() - round_started
            enqueued_transitions += len(batch.actions)
            is_final = enqueued_transitions >= total_transitions
            waited, timed_out, sent = _put_with_wait(
                gathered_round_queue,
                GatheredRound(round_id=round_id, batch=batch, is_final=is_final),
                poll_timeout=transition_put_poll_timeout,
                stop_event=stop_event,
            )
            batch_queue_wait_seconds += waited
            batch_queue_wait_timeouts += timed_out
            if not sent:
                break
            rounds_enqueued += 1
            _put_metric(
                metric_queue,
                {
                    "kind": "gather_communication",
                    "round_id": round_id,
                    "rounds_enqueued": rounds_enqueued,
                    "round_collection_seconds": round_collection_seconds,
                    "actor_arrival_seconds": tuple(actor_arrival_seconds),
                    "batch_queue_wait_seconds": batch_queue_wait_seconds,
                    "batch_queue_wait_timeouts": batch_queue_wait_timeouts,
                },
            )
            if is_final:
                # Keep the gather process alive until the learner consumes the
                # final message and initiates coordinated shutdown.
                while not stop_event.wait(timeout=transition_put_poll_timeout):
                    pass
                break
            round_id += 1
    except BaseException as exc:
        try:
            error_queue.put_nowait(
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        except queue.Full:
            pass
        stop_event.set()
    finally:
        stop_event.set()
        for process in actors:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        for actor_queue in transition_queues:
            actor_queue.close()
        for command_queue in command_queues:
            command_queue.close()
