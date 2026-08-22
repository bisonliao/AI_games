"""Per-environment n-step return aggregation for white decision intervals."""

from __future__ import annotations

from collections import deque
from typing import Deque

from .common import Transition


class NStepAccumulator:
    def __init__(self, num_envs: int, n_step: int, gamma: float) -> None:
        if num_envs < 1 or n_step < 1:
            raise ValueError("num_envs and n_step must be positive")
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.buffers: list[Deque[Transition]] = [deque() for _ in range(num_envs)]

    def add(self, env_index: int, transition: Transition) -> list[Transition]:
        buffer = self.buffers[int(env_index)]
        buffer.append(transition)
        emitted: list[Transition] = []
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
        last = first
        steps = 0
        for item in buffer:
            reward += (self.gamma ** steps) * float(item.reward)
            last = item
            steps += 1
            if item.done or steps >= self.n_step:
                break
        return Transition(
            state=first.state,
            action=first.action,
            reward=reward,
            next_state=last.next_state,
            next_mask=last.next_mask,
            done=last.done,
            discount=self.gamma ** steps,
        )

    def pending(self) -> int:
        return sum(len(buffer) for buffer in self.buffers)
