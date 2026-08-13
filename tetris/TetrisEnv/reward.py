"""Reward shaping based on event rewards and a potential function."""
from __future__ import annotations

from .board import BoardFeatures


def potential(features: BoardFeatures) -> float:
    """Score board quality on roughly normalized structural features.

    Holes deliberately carry more weight than in the original reward: one new
    hole now changes the potential by 0.02 instead of 0.002.  Aggregate height
    distinguishes a broadly filled board from one narrow tower, while max height
    still captures immediate top-out risk.
    """
    return -(
        0.45 * (features.aggregate_height / 200.0)
        + 0.20 * (features.max_height / 20.0)
        + 4.00 * (features.holes / 200.0)
        + 0.30 * (features.bumpiness / 180.0)
        + 0.30 * (features.wells / 200.0)
    )


def shaped_reward(
    *,
    piece_placed: bool,
    lines_cleared: int,
    terminated: bool,
    previous: BoardFeatures,
    current: BoardFeatures,
    gamma: float = 0.99,
    apply_potential: bool = True,
    piece_placed_reward: float = 0.01,
    line_clear_reward: float = 0.75,
    terminal_penalty: float = 1.0,
) -> float:
    reward = (
        piece_placed_reward * float(piece_placed)
        + line_clear_reward * float(lines_cleared)
        - terminal_penalty * float(terminated)
    )
    if apply_potential:
        # Keep the real terminal board potential. Treating it as zero would turn
        # escape from a bad (negative-potential) board into a positive shaping
        # event at exactly the transition where the agent tops out.
        reward += gamma * potential(current) - potential(previous)
    return float(reward)
