"""Fair round-robin barrier gatherer for per-actor transition queues."""

from __future__ import annotations

import queue
import traceback
from typing import Any, Sequence

from .common import (
    GatherBatch,
    GatherPermit,
    TransitionPacket,
    blocking_put,
    concatenate_packets,
)


def _blocking_get(source_queue: Any, stop_event: Any) -> Any:
    while True:
        try:
            return source_queue.get(timeout=1.0)
        except queue.Empty:
            if stop_event.is_set():
                raise RuntimeError("pipeline stopped while gather was waiting for an actor")


def gather_worker(
    actor_queues: Sequence[Any],
    permit_queues: Sequence[Any],
    learner_queue: Any,
    error_queue: Any,
    stop_event: Any,
    *,
    actor_batch_size: int,
    start_global_step: int,
    target_global_step: int,
) -> None:
    try:
        global_step = int(start_global_step)
        while global_step < target_global_step and not stop_event.is_set():
            packets: list[TransitionPacket] = []
            for actor_id, actor_queue in enumerate(actor_queues):
                packet = _blocking_get(actor_queue, stop_event)
                if not isinstance(packet, TransitionPacket):
                    raise TypeError(f"actor {actor_id} sent {type(packet)!r}, expected TransitionPacket")
                if packet.actor_id != actor_id:
                    raise ValueError(f"expected actor {actor_id}, got packet from {packet.actor_id}")
                if len(packet) != actor_batch_size:
                    raise ValueError(
                        f"actor {actor_id} sent {len(packet)} transitions, expected {actor_batch_size}"
                    )
                packets.append(packet)

            combined = concatenate_packets(packets)
            global_step += len(combined)
            final = global_step >= target_global_step
            permit_kind = "stop" if final else "continue"
            for permit_queue in permit_queues:
                blocking_put(
                    permit_queue, GatherPermit(permit_kind, global_step), stop_event
                )
            blocking_put(
                learner_queue,
                GatherBatch(packet=combined, global_step=global_step, final=final),
                stop_event,
            )
            if final:
                return
    except BaseException:
        try:
            error_queue.put({"worker": "gather", "traceback": traceback.format_exc()})
        finally:
            stop_event.set()
