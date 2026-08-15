"""The fixed epsilon and learning-rate schedules used by training."""
from __future__ import annotations


LINEAR_DECAY_TRANSITIONS = 5_000_000
FINAL_LR_MULTIPLIER = 0.1


def decay_progress(transition: int | float) -> float:
    """Return progress through the first five million transitions."""
    return min(max(float(transition) / LINEAR_DECAY_TRANSITIONS, 0.0), 1.0)


def final_actor_epsilons(
    num_actors: int,
    final_epsilon: float,
) -> list[float]:
    """Keep one greedy actor and give the others the final exploration rate."""
    if num_actors < 1:
        raise ValueError("num_actors must be positive")
    if not 0 <= final_epsilon <= 1:
        raise ValueError("final_epsilon must be in [0, 1]")
    return [0.0] + [float(final_epsilon)] * (int(num_actors) - 1)


def epsilon_for_schedule(
    base_epsilon: float,
    progress: float,
    final_epsilon: float,
) -> float:
    """Linearly interpolate an actor's exploration probability."""
    progress = min(max(float(progress), 0.0), 1.0)
    if progress == 0.0:
        return float(base_epsilon)
    if progress == 1.0:
        return float(final_epsilon)
    return float(base_epsilon) + progress * (
        float(final_epsilon) - float(base_epsilon)
    )


def learning_rate_for_schedule(
    base_learning_rate: float,
    progress: float,
) -> float:
    """Linearly interpolate the optimizer learning rate to one tenth."""
    progress = min(max(float(progress), 0.0), 1.0)
    if progress == 0.0:
        return float(base_learning_rate)
    if progress == 1.0:
        return float(base_learning_rate) * FINAL_LR_MULTIPLIER
    multiplier = 1.0 + progress * (FINAL_LR_MULTIPLIER - 1.0)
    return float(base_learning_rate) * multiplier
