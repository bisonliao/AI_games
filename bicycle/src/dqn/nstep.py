"""Transition records and per-environment n-step return accumulation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class StepTransition:
    """Raw one-control-step transition produced by one environment slot."""

    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    terminated: bool
    truncated: bool


@dataclass(slots=True)
class NStepTransition:
    """Replay-ready discounted transition spanning one to n environment steps."""

    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    discount: float
    terminated: bool
    truncated: bool


class NStepAccumulator:
    """Convert one-step experience into n-step replay transitions.

    True terminations disable bootstrap. Time-limit truncations end the local
    sequence but retain a discount so the learner bootstraps from final_obs.
    """

    def __init__(self, n_step: int, gamma: float):
        if n_step <= 0 or not 0 < gamma <= 1:
            raise ValueError("invalid n-step parameters")
        self.n_step = n_step
        self.gamma = gamma
        self._steps: deque[StepTransition] = deque()

    def add(self, transition: StepTransition) -> list[NStepTransition]:
        """Append one step and emit all transitions made ready by it."""
        self._steps.append(transition)
        emitted: list[NStepTransition] = []
        if transition.terminated or transition.truncated:
            while self._steps:
                emitted.append(self._build(min(self.n_step, len(self._steps))))
                self._steps.popleft()
        elif len(self._steps) >= self.n_step:
            emitted.append(self._build(self.n_step))
            self._steps.popleft()
        return emitted

    def clear(self) -> None:
        self._steps.clear()

    def _build(self, count: int) -> NStepTransition:
        reward = 0.0
        last: StepTransition | None = None
        used = 0
        for index, step in enumerate(self._steps):
            if index >= count:
                break
            reward += self.gamma**index * step.reward
            last = step
            used += 1
            if step.terminated or step.truncated:
                break
        assert last is not None
        discount = 0.0 if last.terminated else self.gamma**used
        first = self._steps[0]
        return NStepTransition(
            observation=np.asarray(first.observation, dtype=np.float32).copy(),
            action=int(first.action),
            reward=float(reward),
            next_observation=np.asarray(last.next_observation, dtype=np.float32).copy(),
            discount=float(discount),
            terminated=bool(last.terminated),
            truncated=bool(last.truncated),
        )
