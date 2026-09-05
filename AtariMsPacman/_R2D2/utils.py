"""Scheduling, queue, serialization, and checkpoint helpers."""

from __future__ import annotations

import io
from multiprocessing.synchronize import Event
from queue import Empty, Full
import time
from pathlib import Path
import zlib

import numpy as np
import torch

from _R2D2.config import R2D2Config


def actor_epsilon(actor_id: int, config: R2D2Config) -> float:
    if not 0 <= actor_id < config.num_actors:
        raise ValueError("actor_id is outside the configured range")
    if config.num_actors == 1:
        exponent = 1.0
    else:
        exponent = 1.0 + actor_id / (config.num_actors - 1) * config.epsilon_alpha
    return float(config.base_epsilon**exponent)


def linear_epsilon(_global_transitions: int, config: R2D2Config, actor_id: int = 0) -> float:
    """Return the actor's fixed Ape-X/R2D2 exploration epsilon.

    R2D2 uses a fixed epsilon ladder rather than a single global linear decay;
    the transition argument is accepted so callers can share a scheduler API
    with non-recurrent DQN trainers.
    """
    return actor_epsilon(actor_id, config)


def linear_beta(_global_transitions: int, config: R2D2Config) -> float:
    """Return the paper's fixed importance-sampling exponent (0.6 by default)."""
    return float(config.importance_sampling_beta)


def actor_environment_seed(config: R2D2Config, actor_id: int) -> int:
    if not 0 <= actor_id < config.num_actors:
        raise ValueError("actor_id is outside the configured range")
    return int(config.seed + actor_id * 100_000)


def actor_policy_seed(config: R2D2Config, actor_id: int) -> int:
    return actor_environment_seed(config, actor_id) + 50_000


def increment_counter(counter, amount: int) -> int:
    with counter.get_lock():
        counter.value += int(amount)
        return int(counter.value)


def read_counter(counter) -> int:
    with counter.get_lock():
        return int(counter.value)


def put_reliably(queue, item, stop_event: Event, timeout: float) -> tuple[bool, float]:
    started = time.monotonic()
    while True:
        try:
            queue.put(item, timeout=timeout)
            return True, time.monotonic() - started
        except Full:
            if stop_event.is_set():
                return False, time.monotonic() - started


def put_latest(queue, item) -> None:
    while True:
        try:
            queue.put_nowait(item)
            return
        except Full:
            try:
                queue.get_nowait()
            except Empty:
                return


def drain_latest(queue):
    latest = None
    while True:
        try:
            latest = queue.get_nowait()
        except Empty:
            return latest


def serialize_state_dict(model: torch.nn.Module) -> bytes:
    buffer = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, buffer)
    return buffer.getvalue()


def load_state_dict_bytes(payload: bytes) -> dict[str, torch.Tensor]:
    return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)


def unique_observation_fraction(observations: np.ndarray) -> float:
    if len(observations) == 0:
        return 0.0
    checksums = {
        zlib.crc32(np.ascontiguousarray(item).view(np.uint8)) for item in observations
    }
    return len(checksums) / len(observations)


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
