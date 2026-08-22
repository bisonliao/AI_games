"""Shared data contracts and small utilities for the after-BC training pipeline."""

from __future__ import annotations

import hashlib
import os
import queue
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_BC_CHECKPOINT = PACKAGE_ROOT / "bootstrap" / "BC_BEST.pt"
DEFAULT_RUN_ROOT = PACKAGE_ROOT / "runs"
EXPECTED_BC_SHA256 = "525e396d1ad97310101d3cd19f9eb2b647043cdfe785eb037928948c3d1df428"


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    next_mask: np.ndarray
    done: bool
    discount: float = 1.0


@dataclass
class TransitionPacket:
    actor_id: int
    policy_version: int
    epsilon: float
    schedule_step: int
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    next_masks: np.ndarray
    dones: np.ndarray
    discounts: np.ndarray
    source_actor_ids: np.ndarray
    blocked_seconds: float = 0.0

    def __len__(self) -> int:
        return int(len(self.actions))


@dataclass
class GatherBatch:
    packet: TransitionPacket
    global_step: int
    final: bool


@dataclass(frozen=True)
class GatherPermit:
    kind: str
    global_step: int


@dataclass
class WeightSnapshot:
    version: int
    state_dict: dict[str, np.ndarray]


@dataclass
class RolloutSummary:
    actor_id: int
    policy_version: int
    epsilon: float
    schedule_step: int
    transitions: int
    episodes: int
    white_wins: int
    white_losses: int
    draws: int
    return_sum: float
    white_move_sum: int
    collection_seconds: float
    blocked_seconds: float


@dataclass
class EvaluationResult:
    checkpoint: str
    step: int
    result: dict[str, Any] | None = None
    error: str | None = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_bc_checkpoint(path: Path, *, expected_hash: str = EXPECTED_BC_SHA256) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"BC_BEST checkpoint does not exist: {path}")
    actual = file_sha256(path)
    if expected_hash and actual != expected_hash:
        raise ValueError(
            f"BC_BEST SHA256 mismatch: expected {expected_hash}, got {actual} ({path})"
        )
    return actual


def encode_boards(boards: np.ndarray, players: np.ndarray | Sequence[int] | int) -> np.ndarray:
    boards = np.asarray(boards, dtype=np.int8)
    if boards.ndim == 2:
        boards = boards[None, ...]
    player_array = np.asarray(players, dtype=np.int8).reshape(-1, 1, 1)
    if len(player_array) == 1 and len(boards) != 1:
        player_array = np.repeat(player_array, len(boards), axis=0)
    if len(player_array) != len(boards):
        raise ValueError("players and boards must have the same batch size")
    return np.stack(
        (boards == player_array, boards == -player_array, boards == 0), axis=1
    ).astype(np.float32)


def random_legal_actions(action_masks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    masks = np.asarray(action_masks, dtype=np.bool_)
    if masks.ndim == 1:
        masks = masks[None, :]
    result = np.empty(len(masks), dtype=np.int64)
    for index, mask in enumerate(masks):
        legal = np.flatnonzero(mask)
        if not len(legal):
            raise RuntimeError("No legal action is available")
        result[index] = int(rng.choice(legal))
    return result


def actor_initial_epsilon(actor_id: int, num_actors: int) -> float:
    if num_actors < 1:
        raise ValueError("num_actors must be positive")
    if not 0 <= actor_id < num_actors:
        raise ValueError("actor_id must be in [0, num_actors)")
    if num_actors == 1:
        return 0.4
    return 0.4 - 0.3 * actor_id / (num_actors - 1)


def actor_epsilon(
    actor_id: int,
    num_actors: int,
    global_step: int,
    *,
    decay_steps: int = 1_000_000,
    final_epsilon: float = 0.05,
) -> float:
    initial = actor_initial_epsilon(actor_id, num_actors)
    if decay_steps <= 0:
        return float(final_epsilon)
    fraction = min(1.0, max(0.0, int(global_step) / float(decay_steps)))
    if fraction >= 1.0:
        return float(final_epsilon)
    return float(initial + fraction * (final_epsilon - initial))


def pack_transitions(
    transitions: Sequence[Transition],
    *,
    actor_id: int,
    policy_version: int,
    epsilon: float,
    schedule_step: int,
    blocked_seconds: float = 0.0,
) -> TransitionPacket:
    if not transitions:
        raise ValueError("cannot pack an empty transition batch")
    count = len(transitions)
    return TransitionPacket(
        actor_id=int(actor_id),
        policy_version=int(policy_version),
        epsilon=float(epsilon),
        schedule_step=int(schedule_step),
        states=np.stack([item.state for item in transitions]).astype(np.int8, copy=False),
        actions=np.asarray([item.action for item in transitions], dtype=np.int16),
        rewards=np.asarray([item.reward for item in transitions], dtype=np.float32),
        next_states=np.stack([item.next_state for item in transitions]).astype(np.int8, copy=False),
        next_masks=np.stack([item.next_mask for item in transitions]).astype(np.bool_, copy=False),
        dones=np.asarray([item.done for item in transitions], dtype=np.bool_),
        discounts=np.asarray([item.discount for item in transitions], dtype=np.float32),
        source_actor_ids=np.full(count, actor_id, dtype=np.int16),
        blocked_seconds=float(blocked_seconds),
    )


def concatenate_packets(packets: Sequence[TransitionPacket]) -> TransitionPacket:
    if not packets:
        raise ValueError("cannot concatenate an empty packet list")
    fields = (
        "states", "actions", "rewards", "next_states", "next_masks", "dones",
        "discounts", "source_actor_ids",
    )
    values = {name: np.concatenate([getattr(packet, name) for packet in packets]) for name in fields}
    return TransitionPacket(
        actor_id=-1,
        policy_version=max(packet.policy_version for packet in packets),
        epsilon=float("nan"),
        schedule_step=max(packet.schedule_step for packet in packets),
        blocked_seconds=sum(packet.blocked_seconds for packet in packets),
        **values,
    )


def cpu_state_dict(module: torch.nn.Module) -> dict[str, np.ndarray]:
    return {
        name: value.detach().cpu().numpy().copy()
        for name, value in module.state_dict().items()
    }


def load_numpy_state_dict(module: torch.nn.Module, state: Mapping[str, np.ndarray]) -> None:
    module.load_state_dict({name: torch.as_tensor(value) for name, value in state.items()})


def put_latest(target_queue: Any, item: Any) -> None:
    while True:
        try:
            target_queue.put_nowait(item)
            return
        except queue.Full:
            try:
                target_queue.get_nowait()
            except queue.Empty:
                time.sleep(0.001)


def blocking_put(target_queue: Any, item: Any, stop_event: Any, *, timeout: float = 1.0) -> float:
    started = time.perf_counter()
    while True:
        try:
            target_queue.put(item, timeout=timeout)
            return time.perf_counter() - started
        except queue.Full:
            if stop_event.is_set():
                raise RuntimeError("pipeline stopped while waiting to enqueue data")


def validate_run_name(value: str) -> str:
    if not value or value in {".", ".."} or not re.fullmatch(r"[\w.-]+", value):
        raise ValueError("run name may contain only letters, numbers, underscore, dot, and dash")
    return value


def atomic_torch_save(payload: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
