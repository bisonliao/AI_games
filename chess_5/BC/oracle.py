"""Versioned contract and compact encodings for the frozen BC expert oracle."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from .heuristic_agent import ExpertDecision


ORACLE_VERSION = "heuristic-v1"
DATA_FORMAT_VERSION = 3
DEFAULT_ORACLE_TOP_K = 4

REASONS = ("opening_center", "win", "block", "own_fork", "block_fork", "normal")
REASON_TO_ID = {reason: index for index, reason in enumerate(REASONS)}
ID_TO_REASON = dict(enumerate(REASONS))

SOURCES = (
    "expert_selfplay",
    "perturbed_opening",
    "epsilon_expert",
    "policy_expert",
    "policy_selfplay",
    "policy_history",
    "epsilon_policy",
    "challenge",
)
SOURCE_TO_ID = {source: index for index, source in enumerate(SOURCES)}
ID_TO_SOURCE = dict(enumerate(SOURCES))


def oracle_identity(*, max_candidates: int = 12,
                    top_k: int = DEFAULT_ORACLE_TOP_K) -> dict[str, Any]:
    config = {
        "version": ORACLE_VERSION,
        "max_candidates": int(max_candidates),
        "top_k": int(top_k),
    }
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return {**config, "sha256": hashlib.sha256(encoded).hexdigest()}


def validate_oracle_identity(actual: dict[str, Any] | None, expected: dict[str, Any]) -> None:
    if not isinstance(actual, dict) or actual.get("sha256") != expected.get("sha256"):
        raise ValueError("dataset/checkpoint oracle identity is incompatible")


def encode_decision(decision: ExpertDecision,
                    top_k: int = DEFAULT_ORACLE_TOP_K) -> tuple[np.ndarray, int, int, bool]:
    candidates = np.full(int(top_k), -1, dtype=np.int16)
    count = min(len(decision.actions), int(top_k))
    candidates[:count] = np.asarray(decision.actions[:count], dtype=np.int16)
    try:
        reason = REASON_TO_ID[decision.reason]
    except KeyError as exc:
        raise ValueError(f"unknown oracle reason: {decision.reason}") from exc
    return candidates, count, reason, bool(decision.tactical)


def rank_target(candidate_count: int, *, top_k: int = DEFAULT_ORACLE_TOP_K,
                temperature: float = 1.0) -> np.ndarray:
    """Return a padded rank-soft target distribution."""
    result = np.zeros(int(top_k), dtype=np.float32)
    count = min(max(0, int(candidate_count)), int(top_k))
    if count == 0:
        return result
    logits = -np.arange(count, dtype=np.float64) / float(temperature)
    probabilities = np.exp(logits - logits.max())
    result[:count] = (probabilities / probabilities.sum()).astype(np.float32)
    return result
