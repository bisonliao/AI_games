from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


Transition = tuple[
    np.ndarray,
    np.ndarray,
    int,
    float,
    np.ndarray,
    np.ndarray,
    bool,
    float,
]


@dataclass
class _OneStepTransition:
    state: tuple[np.ndarray, np.ndarray]
    action: int
    reward: float
    next_state: tuple[np.ndarray, np.ndarray]
    done: bool


class NStepAccumulator:
    def __init__(self, n_step: int, gamma: float) -> None:
        self.n_step = n_step
        self.gamma = gamma
        self._buffer: deque[_OneStepTransition] = deque()

    def append(
        self,
        state: tuple[np.ndarray, np.ndarray],
        action: int,
        reward: float,
        next_state: tuple[np.ndarray, np.ndarray],
        done: bool,
    ) -> list[Transition]:
        self._buffer.append(_OneStepTransition(state, action, reward, next_state, done))
        output: list[Transition] = []
        if len(self._buffer) >= self.n_step:
            output.append(self._build_transition())
            self._buffer.popleft()
        if done:
            while self._buffer:
                output.append(self._build_transition())
                self._buffer.popleft()
        return output

    def _build_transition(self) -> Transition:
        total_reward = 0.0
        discount = 1.0
        final_next_state = self._buffer[0].next_state
        final_done = False
        for item in list(self._buffer)[: self.n_step]:
            total_reward += discount * item.reward
            discount *= self.gamma
            final_next_state = item.next_state
            final_done = item.done
            if item.done:
                discount = 0.0
                break
        first = self._buffer[0]
        return (
            first.state[0],
            first.state[1],
            first.action,
            total_reward,
            final_next_state[0],
            final_next_state[1],
            final_done,
            discount,
        )


class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        priority_epsilon: float = 1e-6,
        seed: int = 0,
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.priority_epsilon = priority_epsilon
        self.rng = np.random.default_rng(seed)
        self.size = 0
        self.position = 0
        self.max_priority = 1.0

        tree_capacity = 1
        while tree_capacity < capacity:
            tree_capacity *= 2
        self.tree_capacity = tree_capacity
        self.sum_tree = np.zeros(2 * tree_capacity, dtype=np.float64)

        self.pixels: np.ndarray | None = None
        self.macros: np.ndarray | None = None
        self.next_pixels: np.ndarray | None = None
        self.next_macros: np.ndarray | None = None
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.bool_)
        self.discounts = np.empty(capacity, dtype=np.float32)

    def __len__(self) -> int:
        return self.size

    def add(self, transition: Transition) -> None:
        state_pixels, state_macro, action, reward, next_pixels, next_macro, done, discount = transition
        if self.pixels is None:
            self.pixels = np.empty((self.capacity, *state_pixels.shape), dtype=np.uint8)
            self.next_pixels = np.empty((self.capacity, *next_pixels.shape), dtype=np.uint8)
            self.macros = np.empty((self.capacity, *state_macro.shape), dtype=np.float32)
            self.next_macros = np.empty((self.capacity, *next_macro.shape), dtype=np.float32)

        index = self.position
        self.pixels[index] = state_pixels
        self.macros[index] = state_macro
        self.next_pixels[index] = next_pixels
        self.next_macros[index] = next_macro
        self.actions[index] = action
        self.rewards[index] = reward
        self.dones[index] = done
        self.discounts[index] = discount

        self._set_priority(index, self.max_priority)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float) -> dict[str, Any]:
        if self.size < batch_size or self.pixels is None:
            raise ValueError("Not enough replay data")
        total = self.sum_tree[1]
        if total <= 0:
            raise RuntimeError("Replay priority tree is empty")

        segment = total / batch_size
        values = (np.arange(batch_size) + self.rng.random(batch_size)) * segment
        indices = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)
        for i, value in enumerate(values):
            tree_index = 1
            while tree_index < self.tree_capacity:
                left = tree_index * 2
                if value <= self.sum_tree[left]:
                    tree_index = left
                else:
                    value -= self.sum_tree[left]
                    tree_index = left + 1
            data_index = tree_index - self.tree_capacity
            if data_index >= self.size:
                data_index = int(self.rng.integers(self.size))
                tree_index = data_index + self.tree_capacity
            indices[i] = data_index
            priorities[i] = self.sum_tree[tree_index]

        probabilities = priorities / total
        weights = np.power(self.size * probabilities, -beta)
        weights /= weights.max()
        return {
            "pixels": self.pixels[indices],
            "macro": self.macros[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_pixels": self.next_pixels[indices],
            "next_macro": self.next_macros[indices],
            "dones": self.dones[indices],
            "discounts": self.discounts[indices],
            "weights": weights.astype(np.float32),
            "indices": indices,
        }

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        for index, error in zip(indices, td_errors, strict=True):
            raw_priority = float(abs(error) + self.priority_epsilon)
            priority = raw_priority**self.alpha
            self.max_priority = max(self.max_priority, priority)
            self._set_priority(int(index), priority)

    def _set_priority(self, data_index: int, priority: float) -> None:
        tree_index = data_index + self.tree_capacity
        difference = priority - self.sum_tree[tree_index]
        while tree_index >= 1:
            self.sum_tree[tree_index] += difference
            tree_index //= 2
