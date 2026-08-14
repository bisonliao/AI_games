"""Training values selected by the evaluation-driven schedule state."""
from __future__ import annotations


POST_SWITCH_EPSILON = 0.05
POST_SWITCH_LR_MULTIPLIER = 0.1


def epsilon_for_schedule(base_epsilon: float, triggered: bool) -> float:
    """Return an actor's exploration probability for the current schedule state."""
    if triggered:
        return POST_SWITCH_EPSILON
    return float(base_epsilon)


def learning_rate_for_schedule(
    base_learning_rate: float,
    triggered: bool,
) -> float:
    """Return the optimizer learning rate for the current schedule state."""
    if triggered:
        return float(base_learning_rate) * POST_SWITCH_LR_MULTIPLIER
    return float(base_learning_rate)
