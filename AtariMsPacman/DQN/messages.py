"""Pickle-friendly messages exchanged by training processes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class TransitionChunk:
    actor_id: int
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminated: np.ndarray
    epsilon: float
    policy_version: int

    def __len__(self) -> int:
        return int(self.actions.shape[0])


@dataclass(slots=True)
class ActorReport:
    actor_id: int
    transitions: int
    collection_seconds: float
    queue_wait_seconds: float
    epsilon: float
    policy_version: int
    unique_observation_fraction: float
    episode_lengths: list[int]
    episode_returns: list[float]
    episode_raw_scores: list[float]


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
