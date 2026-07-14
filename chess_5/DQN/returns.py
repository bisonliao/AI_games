from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    next_mask: np.ndarray
    done: bool
    discount: float = 1.0


class NStepAccumulator:
    """Build one replay item per raw transition while preserving env boundaries."""

    def __init__(self, num_envs: int, n_step: int, gamma: float) -> None:
        if n_step < 1:
            raise ValueError("n_step must be at least 1")
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.buffers: List[Deque[Transition]] = [deque() for _ in range(num_envs)]

    def add(self, env_index: int, transition: Transition) -> List[Transition]:
        buffer = self.buffers[env_index]
        buffer.append(transition)
        emitted: List[Transition] = []
        if transition.done:
            while buffer:
                emitted.append(self._aggregate(buffer))
                buffer.popleft()
        elif len(buffer) >= self.n_step:
            emitted.append(self._aggregate(buffer))
            buffer.popleft()
        return emitted

    def _aggregate(self, buffer: Deque[Transition]) -> Transition:
        first = buffer[0]
        reward = 0.0
        last: Optional[Transition] = None
        steps = 0
        for item in buffer:
            reward += (self.gamma ** steps) * item.reward
            last = item
            steps += 1
            if item.done or steps >= self.n_step:
                break
        assert last is not None
        return Transition(
            state=first.state,
            action=first.action,
            reward=reward,
            next_state=last.next_state,
            next_mask=last.next_mask,
            done=last.done,
            discount=self.gamma ** steps,
        )


def board_potential(encoded_state: np.ndarray) -> float:
    """Return a bounded black-perspective potential from all length-five lines."""
    state = np.asarray(encoded_state)
    if state.shape[0] != 3 or not np.any(state):
        return 0.0
    own = state[0].astype(np.int8, copy=False)
    opponent = state[1].astype(np.int8, copy=False)
    size = own.shape[0]
    weights = np.asarray([0.0, 0.02, 0.08, 0.25, 0.65, 1.0], dtype=np.float32)
    total = 0.0
    lines = 0
    for row in range(size):
        for col in range(size):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                end_row = row + 4 * dr
                end_col = col + 4 * dc
                if not (0 <= end_row < size and 0 <= end_col < size):
                    continue
                own_count = 0
                opponent_count = 0
                for offset in range(5):
                    own_count += int(own[row + offset * dr, col + offset * dc])
                    opponent_count += int(opponent[row + offset * dr, col + offset * dc])
                if opponent_count == 0:
                    total += float(weights[own_count])
                if own_count == 0:
                    total -= float(weights[opponent_count])
                lines += 1
    return total / max(1, lines)


def shaped_reward(
    reward: float,
    state: np.ndarray,
    next_state: np.ndarray,
    done: bool,
    gamma: float,
    scale: float,
) -> float:
    if scale <= 0:
        return float(reward)
    current_potential = board_potential(state)
    next_potential = 0.0 if done else board_potential(next_state)
    return float(reward + scale * (gamma * next_potential - current_potential))
