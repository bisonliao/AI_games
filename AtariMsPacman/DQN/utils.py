"""Shared scheduling, serialization, queue, and seeding helpers."""

from __future__ import annotations

import io
from multiprocessing.synchronize import Event
from pathlib import Path
from queue import Empty, Full
import time
import zlib

import numpy as np
import torch

from DQN.config import DQNConfig


def linear_epsilon(global_transitions: int, config: DQNConfig) -> float:
    fraction = min(max(global_transitions, 0) / config.epsilon_decay_transitions, 1.0)
    return config.epsilon_start + fraction * (
        config.epsilon_end - config.epsilon_start
    )


def linear_beta(global_transitions: int, config: DQNConfig) -> float:
    fraction = min(
        max(global_transitions, 0) / config.prioritized_replay_beta_transitions,
        1.0,
    )
    return config.prioritized_replay_beta_start + fraction * (
        config.prioritized_replay_beta_end - config.prioritized_replay_beta_start
    )


def actor_environment_seed(config: DQNConfig, actor_id: int) -> int:
    if not 0 <= actor_id < config.num_actors:
        raise ValueError("actor_id is outside the configured actor range")
    return config.seed + actor_id * config.actor_seed_stride


def actor_policy_seed(config: DQNConfig, actor_id: int) -> int:
    return actor_environment_seed(config, actor_id) + config.actor_seed_stride // 2


def increment_counter(counter, amount: int) -> int:
    """Atomically increment a multiprocessing Value and return the new value."""
    with counter.get_lock():
        counter.value += amount
        return int(counter.value)


def read_counter(counter) -> int:
    with counter.get_lock():
        return int(counter.value)


def put_reliably(queue, item, stop_event: Event, timeout: float) -> tuple[bool, float]:
    """Apply queue backpressure until the item is sent or shutdown is requested."""
    wait_started = time.monotonic()
    while True:
        try:
            queue.put(item, timeout=timeout)
            return True, time.monotonic() - wait_started
        except Full:
            if stop_event.is_set():
                return False, time.monotonic() - wait_started


def put_latest(queue, item) -> None:
    """Replace a stale parameter message; rollout data never uses this helper."""
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
    cpu_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(cpu_state, buffer)
    return buffer.getvalue()


def load_state_dict_bytes(payload: bytes) -> dict[str, torch.Tensor]:
    return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)


def unique_observation_fraction(observations: np.ndarray) -> float:
    if observations.shape[0] == 0:
        return 0.0
    checksums = {
        zlib.crc32(np.ascontiguousarray(observation).view(np.uint8))
        for observation in observations
    }
    return len(checksums) / observations.shape[0]


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)
