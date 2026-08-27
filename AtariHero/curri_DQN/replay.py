"""One uniform transition replay ring."""

from __future__ import annotations

from typing import Any

import numpy as np

from .messages import PackedTransition


class ReplayBuffer:
    def __init__(self, *, capacity: int, seed: int) -> None:
        self.capacity = capacity
        self.items: list[PackedTransition | None] = [None] * capacity
        self.position = 0
        self.size = 0
        self.inserted = 0
        self.stage_counts: dict[int, int] = {}
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(self, transition: PackedTransition) -> None:
        replaced = self.items[self.position]
        if replaced is not None:
            old_count = self.stage_counts[replaced.stage] - 1
            if old_count:
                self.stage_counts[replaced.stage] = old_count
            else:
                del self.stage_counts[replaced.stage]
        self.items[self.position] = transition
        self.stage_counts[transition.stage] = (
            self.stage_counts.get(transition.stage, 0) + 1
        )
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.inserted += 1

    def sample(self, batch_size: int) -> list[PackedTransition]:
        if batch_size < 0:
            raise ValueError("batch_size must be non-negative")
        if batch_size == 0:
            return []
        if self.size == 0:
            raise ValueError("cannot sample an empty replay buffer")
        indices = self.rng.integers(self.size, size=batch_size)
        result = []
        for index in indices:
            transition = self.items[int(index)]
            assert transition is not None
            result.append(transition)
        return result

    def stage_sizes(self) -> dict[int, int]:
        return dict(self.stage_counts)

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "items": self.items,
            "position": self.position,
            "size": self.size,
            "inserted": self.inserted,
            "stage_counts": self.stage_counts,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["capacity"]) != self.capacity:
            raise ValueError("replay capacity differs from checkpoint")
        self.items = state["items"]
        self.position = int(state["position"])
        self.size = int(state["size"])
        self.inserted = int(state["inserted"])
        self.stage_counts = {
            int(stage): int(count)
            for stage, count in state["stage_counts"].items()
        }
        self.rng.bit_generator.state = state["rng_state"]
