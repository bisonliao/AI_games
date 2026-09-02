"""Training-side reward shaping applied after the raw Gym environment step."""

from __future__ import annotations

import numpy as np

from DQN.config import DQNConfig


def shape_rewards(
    raw_rewards: np.ndarray,
    life_lost: np.ndarray,
    game_over: np.ndarray,
    config: DQNConfig,
) -> np.ndarray:
    """Log-scale scores, subtract decision cost, then apply death penalties."""
    raw_rewards = np.asarray(raw_rewards, dtype=np.float32)
    life_lost = np.asarray(life_lost, dtype=np.bool_)
    game_over = np.asarray(game_over, dtype=np.bool_)
    if raw_rewards.shape != life_lost.shape or raw_rewards.shape != game_over.shape:
        raise ValueError(
            "raw_rewards, life_lost, and game_over must have identical shapes"
        )
    non_negative_rewards = np.maximum(raw_rewards, np.float32(0.0))
    score_rewards = np.log1p(
        non_negative_rewards / np.float32(config.reward_log_scale)
    )
    shaped = np.clip(
        score_rewards, config.reward_clip_min, config.reward_clip_max
    ).astype(np.float32, copy=False)
    shaped = shaped - np.float32(config.decision_step_cost)
    shaped[life_lost] = np.float32(config.lost_life_penalty)
    shaped[game_over] = np.float32(config.game_over_penalty)
    return shaped


def shape_reward(
    raw_reward: float,
    life_lost: bool,
    game_over: bool,
    config: DQNConfig,
) -> float:
    values = shape_rewards(
        np.asarray([raw_reward], dtype=np.float32),
        np.asarray([life_lost], dtype=np.bool_),
        np.asarray([game_over], dtype=np.bool_),
        config,
    )
    return float(values[0])
