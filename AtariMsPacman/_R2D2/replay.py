"""Memory-bounded sequence prioritized replay."""

from __future__ import annotations

from dataclasses import dataclass
import threading

import numpy as np

from _R2D2.sequence import Sequence, mixed_priority


class _SegmentTree:
    def __init__(self, capacity: int) -> None:
        size = 1
        while size < capacity:
            size <<= 1
        self.size = size
        self.sum = np.zeros(2 * size, dtype=np.float64)
        self.minimum = np.full(2 * size, np.inf, dtype=np.float64)

    def set(self, index: int, value: float) -> None:
        node = self.size + int(index)
        self.sum[node] = value
        self.minimum[node] = value
        node //= 2
        while node:
            self.sum[node] = self.sum[2 * node] + self.sum[2 * node + 1]
            self.minimum[node] = min(self.minimum[2 * node], self.minimum[2 * node + 1])
            node //= 2

    def find(self, mass: float) -> int:
        node = 1
        while node < self.size:
            left = 2 * node
            if mass < self.sum[left]:
                node = left
            else:
                mass -= self.sum[left]
                node = left + 1
        return node - self.size


@dataclass(slots=True)
class ReplaySample:
    sequences: list[Sequence]
    indices: np.ndarray
    generations: np.ndarray
    weights: np.ndarray


class SequenceReplay:
    """Ring-buffer replay with sequence-level proportional priorities."""

    def __init__(
        self,
        capacity: int,
        *,
        alpha: float = 0.9,
        beta: float = 0.6,
        priority_epsilon: float = 1.0e-6,
        priority_mix: float = 0.9,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if not 0.0 <= alpha <= 1.0 or not 0.0 < beta <= 1.0:
            raise ValueError("invalid priority exponents")
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.priority_epsilon = float(priority_epsilon)
        self.priority_mix = float(priority_mix)
        self._items: list[Sequence | None] = [None] * self.capacity
        self._generations = np.zeros(self.capacity, dtype=np.int64)
        self._tree = _SegmentTree(self.capacity)
        self._position = 0
        self._size = 0
        self._learning_transitions = 0
        self._max_priority = 1.0
        self._lock = threading.RLock()

    def __len__(self) -> int:
        with self._lock:
            return self._size

    @property
    def position(self) -> int:
        return self._position

    @property
    def learning_transitions(self) -> int:
        with self._lock:
            return self._learning_transitions

    @property
    def generations(self) -> np.ndarray:
        return self._generations.copy()

    def add(self, sequence: Sequence, priority: float | None = None) -> tuple[int, int]:
        with self._lock:
            index = self._position
            previous = self._items[index]
            if previous is not None:
                self._learning_transitions -= len(previous)
            self._items[index] = sequence
            self._generations[index] += 1
            value = self._max_priority if priority is None else max(float(priority), self.priority_epsilon)
            self._max_priority = max(self._max_priority, value)
            self._tree.set(index, value**self.alpha)
            self._position = (index + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)
            self._learning_transitions += len(sequence)
            return index, int(self._generations[index])

    def add_many(self, sequences: list[Sequence]) -> None:
        for sequence in sequences:
            self.add(sequence)

    def sample(self, batch_size: int, rng: np.random.Generator) -> ReplaySample:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        with self._lock:
            if self._size < batch_size:
                raise ValueError("not enough sequences in replay")
            total = self._tree.sum[1]
            if total <= 0:
                raise RuntimeError("replay priority sum must be positive")
            segment = total / batch_size
            masses = (np.arange(batch_size) + rng.random(batch_size)) * segment
            indices = np.asarray(
                [min(self._tree.find(float(m)), self.capacity - 1) for m in masses],
                dtype=np.int64,
            )
            priorities = self._tree.sum[self._tree.size + indices]
            probabilities = priorities / total
            # Match the reference implementation: normalize by the minimum
            # priority in this sampled batch, so the largest IS weight is 1.
            sampled_minimum = max(float(priorities.min()), np.finfo(np.float64).tiny)
            weights = (priorities / sampled_minimum) ** (-self.beta)
            return ReplaySample(
                sequences=[self._items[int(i)] for i in indices],  # type: ignore[misc]
                indices=indices,
                generations=self._generations[indices].copy(),
                weights=weights.astype(np.float32),
            )

    def update_priorities(
        self,
        indices: np.ndarray,
        priorities: np.ndarray,
        generations: np.ndarray | None = None,
    ) -> None:
        indices = np.asarray(indices, dtype=np.int64)
        priorities = np.asarray(priorities, dtype=np.float64)
        if indices.shape != priorities.shape:
            raise ValueError("indices and priorities must have the same shape")
        if np.any(~np.isfinite(priorities)) or np.any(priorities <= 0):
            raise ValueError("priorities must be finite and positive")
        if generations is not None and np.asarray(generations).shape != indices.shape:
            raise ValueError("generations must match indices")
        with self._lock:
            for offset, (index, priority) in enumerate(zip(indices, priorities, strict=True)):
                index = int(index)
                if not 0 <= index < self.capacity:
                    continue
                if generations is not None and int(generations[offset]) != int(self._generations[index]):
                    continue
                if self._items[index] is None:
                    continue
                value = max(float(priority), self.priority_epsilon)
                self._max_priority = max(self._max_priority, value)
                self._tree.set(index, value**self.alpha)

    def priority_for_errors(self, td_errors: np.ndarray) -> float:
        return max(mixed_priority(td_errors, self.priority_mix), self.priority_epsilon)


# Familiar short name for small standalone experiments.
ReplayBuffer = SequenceReplay
