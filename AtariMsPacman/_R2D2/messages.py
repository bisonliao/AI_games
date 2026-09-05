"""Pickle-friendly messages exchanged by R2D2 processes."""

from __future__ import annotations

from dataclasses import dataclass, field

from _R2D2.sequence import Sequence


@dataclass(slots=True)
class SequenceChunk:
    actor_id: int
    sequences: list[Sequence]
    transitions: int
    epsilon: float
    policy_version: int


@dataclass(slots=True)
class ActorReport:
    actor_id: int
    transitions: int
    collection_seconds: float
    queue_wait_seconds: float
    epsilon: float
    policy_version: int
    unique_observation_fraction: float
    episode_lengths: list[int] = field(default_factory=list)
    episode_returns: list[float] = field(default_factory=list)
    episode_raw_scores: list[float] = field(default_factory=list)


@dataclass(slots=True)
class ParameterUpdate:
    version: int
    global_transitions: int
    state_dict_bytes: bytes


@dataclass(slots=True)
class EvaluationRequest:
    checkpoint_path: str
    checkpoint_transition: int


@dataclass(slots=True)
class EvaluationResult:
    checkpoint_transition: int
    checkpoint_path: str
    episode_lengths: list[int]
    episode_returns: list[float]
    episode_raw_scores: list[float]
    capped_episodes: int
    elapsed_seconds: float


@dataclass(slots=True)
class EvaluatorStop:
    pass


@dataclass(slots=True)
class ProcessError:
    process_name: str
    traceback: str

