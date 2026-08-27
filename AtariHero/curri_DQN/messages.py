"""Serializable actor, learner, and evaluator messages."""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np


OBSERVATION_SHAPE = (4, 84, 84)
OBSERVATION_BYTES = int(np.prod(OBSERVATION_SHAPE))


@dataclass(slots=True)
class PackedTransition:
    observations: bytes
    action: int
    reward: float
    terminated: bool
    stage: int
    actor_id: int


@dataclass(slots=True)
class EpisodeSummary:
    actor_id: int
    reset_stage: int
    task_id: str
    checkpoint_id: str
    episode_return: float
    ale_score_return: float
    episode_length: int
    success: bool
    timeout: bool
    walls_destroyed: int
    creatures_killed: int
    miner_rescue_events: int
    dynamite_bonus_sticks: int
    unmapped_ale_reward: float
    visited_levels: tuple[int, ...]
    completed_levels: tuple[int, ...]
    epsilon: float


@dataclass(slots=True)
class SuccessfulEpisode:
    """Legacy pickle shim; new training never creates or consumes this type.

    Format-v7/v8 checkpoints may contain instances inside their obsolete
    success-replay payload.  ``torch.load`` must resolve the original module
    and class name even when ``--load-checkpoint`` only needs ``online_model``.
    """

    reset_stage: int
    task_id: str
    checkpoint_id: str
    transitions: tuple[PackedTransition, ...]


@dataclass(slots=True)
class StageEvaluationResult:
    checkpoint_step: int
    reset_stages: tuple[int, ...]
    task_ids: tuple[str, ...]
    checkpoint_ids: tuple[str, ...]
    episode_returns: tuple[float, ...]
    ale_score_returns: tuple[float, ...]
    episode_lengths: tuple[int, ...]
    successes: tuple[bool, ...]
    timeouts: tuple[bool, ...]
    walls_destroyed: tuple[int, ...]
    creatures_killed: tuple[int, ...]
    miner_rescue_events: tuple[int, ...]
    unmapped_ale_rewards: tuple[float, ...]


@dataclass(slots=True)
class WorkerFailure:
    worker: str
    traceback: str


def pack_transition(
    observation: np.ndarray,
    next_observation: np.ndarray,
    *,
    action: int,
    reward: float,
    terminated: bool,
    stage: int,
    actor_id: int,
) -> PackedTransition:
    if observation.shape != OBSERVATION_SHAPE:
        raise ValueError(f"unexpected observation shape: {observation.shape}")
    raw = observation.tobytes(order="C") + next_observation.tobytes(order="C")
    return PackedTransition(
        observations=zlib.compress(raw, level=1),
        action=action,
        reward=reward,
        terminated=terminated,
        stage=stage,
        actor_id=actor_id,
    )


def unpack_observations(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    raw = zlib.decompress(payload)
    expected = OBSERVATION_BYTES * 2
    if len(raw) != expected:
        raise ValueError(f"invalid packed observation size: {len(raw)} != {expected}")
    values = np.frombuffer(raw, dtype=np.uint8)
    observation = values[:OBSERVATION_BYTES].reshape(OBSERVATION_SHAPE)
    next_observation = values[OBSERVATION_BYTES:].reshape(OBSERVATION_SHAPE)
    return observation, next_observation
