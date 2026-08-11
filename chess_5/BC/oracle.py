"""冻结专家的版本契约，以及写入 NPZ 的紧凑原因/来源编码。

数据集、checkpoint 与 challenge bank 都保存 oracle identity。只要专家版本、候选
规模或 top-k 语义不同，训练和评测就拒绝混用，避免同一状态出现两套监督答案。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from .heuristic_agent import ExpertDecision


# 修改任何会改变专家落子排序的逻辑时，必须升级版本并重新生成全部相关数据。
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
    """生成可序列化的专家身份及其配置 hash。"""
    config = {
        "version": ORACLE_VERSION,
        "max_candidates": int(max_candidates),
        "top_k": int(top_k),
    }
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return {**config, "sha256": hashlib.sha256(encoded).hexdigest()}


def validate_oracle_identity(actual: dict[str, Any] | None, expected: dict[str, Any]) -> None:
    """校验数据/checkpoint 的专家身份，防止跨 oracle 版本混用。"""
    if not isinstance(actual, dict) or actual.get("sha256") != expected.get("sha256"):
        raise ValueError("dataset/checkpoint oracle identity is incompatible")


def encode_decision(decision: ExpertDecision,
                    top_k: int = DEFAULT_ORACLE_TOP_K) -> tuple[np.ndarray, int, int, bool]:
    """把变长专家候选编码为定宽动作数组、有效数量、原因 ID 和战术标记。"""
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
    """把有效候选数转换为按排名衰减、尾部补零的 soft target。"""
    result = np.zeros(int(top_k), dtype=np.float32)
    count = min(max(0, int(candidate_count)), int(top_k))
    if count == 0:
        return result
    logits = -np.arange(count, dtype=np.float64) / float(temperature)
    probabilities = np.exp(logits - logits.max())
    result[:count] = (probabilities / probabilities.sum()).astype(np.float32)
    return result
