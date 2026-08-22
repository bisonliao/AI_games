"""Proportional prioritized replay and Gomoku D4 augmentation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .common import TransitionPacket


def transform_board(board: np.ndarray, transform: int) -> np.ndarray:
    transform = int(transform)
    if not 0 <= transform < 8:
        raise ValueError("transform must be in [0, 8)")
    value = np.asarray(board)
    if transform >= 4:
        value = np.flip(value, axis=-1)
    return np.rot90(value, k=transform % 4, axes=(-2, -1)).copy()


def transform_action(action: int, board_size: int, transform: int) -> int:
    marker = np.zeros((board_size, board_size), dtype=np.bool_)
    row, col = divmod(int(action), board_size)
    marker[row, col] = True
    return int(np.flatnonzero(transform_board(marker, transform).reshape(-1))[0])


def augment_batch(states: np.ndarray, actions: np.ndarray, next_states: np.ndarray,
                  next_masks: np.ndarray, transforms: np.ndarray) -> tuple[np.ndarray, ...]:
    states = np.asarray(states)
    actions = np.asarray(actions).copy()
    next_states = np.asarray(next_states)
    next_masks = np.asarray(next_masks)
    transforms = np.asarray(transforms, dtype=np.int8)
    count, size, _ = states.shape
    out_states = np.empty_like(states)
    out_next_states = np.empty_like(next_states)
    out_masks = np.empty_like(next_masks)
    for index in range(count):
        transform = int(transforms[index])
        out_states[index] = transform_board(states[index], transform)
        out_next_states[index] = transform_board(next_states[index], transform)
        actions[index] = transform_action(int(actions[index]), size, transform)
        out_masks[index] = transform_board(
            next_masks[index].reshape(size, size), transform
        ).reshape(-1)
    return out_states, actions, out_next_states, out_masks


class SumMinTree:
    def __init__(self, capacity: int) -> None:
        size = 1
        while size < capacity:
            size *= 2
        self.capacity = int(capacity)
        self.size = size
        self.sums = np.zeros(2 * size, dtype=np.float64)
        self.minimums = np.full(2 * size, np.inf, dtype=np.float64)

    def update(self, indices: np.ndarray, values: np.ndarray) -> None:
        for raw_index, raw_value in zip(indices, values):
            node = self.size + int(raw_index)
            value = float(raw_value)
            self.sums[node] = value
            self.minimums[node] = value
            node //= 2
            while node:
                self.sums[node] = self.sums[2 * node] + self.sums[2 * node + 1]
                self.minimums[node] = min(self.minimums[2 * node], self.minimums[2 * node + 1])
                node //= 2

    @property
    def total(self) -> float:
        return float(self.sums[1])

    @property
    def minimum(self) -> float:
        return float(self.minimums[1])

    def find_prefix(self, masses: np.ndarray) -> np.ndarray:
        result = np.empty(len(masses), dtype=np.int64)
        total = self.total
        for offset, raw_mass in enumerate(masses):
            mass = min(max(0.0, float(raw_mass)), np.nextafter(total, 0.0))
            node = 1
            while node < self.size:
                left = node * 2
                if mass < self.sums[left]:
                    node = left
                else:
                    mass -= self.sums[left]
                    node = left + 1
            result[offset] = node - self.size
        return result


@dataclass
class ReplaySample:
    indices: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    next_masks: np.ndarray
    dones: np.ndarray
    discounts: np.ndarray
    weights: np.ndarray


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, board_size: int, *, alpha: float = 0.6,
                 priority_epsilon: float = 1e-6, seed: int = 0) -> None:
        if capacity < 1 or board_size < 5 or not 0.0 <= alpha <= 1.0:
            raise ValueError("invalid prioritized replay configuration")
        self.capacity = int(capacity)
        self.board_size = int(board_size)
        self.action_dim = board_size * board_size
        self.alpha = float(alpha)
        self.priority_epsilon = float(priority_epsilon)
        self.rng = np.random.default_rng(seed)
        self.states = np.zeros((capacity, board_size, board_size), dtype=np.int8)
        self.actions = np.zeros(capacity, dtype=np.int16)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros_like(self.states)
        self.next_masks = np.zeros((capacity, self.action_dim), dtype=np.bool_)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.discounts = np.zeros(capacity, dtype=np.float32)
        self.tree = SumMinTree(capacity)
        self.position = 0
        self.length = 0
        self.max_priority = 1.0

    def __len__(self) -> int:
        return self.length

    def add_packet(self, packet: TransitionPacket) -> None:
        count = len(packet)
        if count > self.capacity:
            start = count - self.capacity
            packet = TransitionPacket(**{
                **packet.__dict__,
                **{name: getattr(packet, name)[start:] for name in (
                    "states", "actions", "rewards", "next_states", "next_masks",
                    "dones", "discounts", "source_actor_ids",
                )},
            })
            count = len(packet)
        indices = (self.position + np.arange(count)) % self.capacity
        self.states[indices] = packet.states
        self.actions[indices] = packet.actions
        self.rewards[indices] = packet.rewards
        self.next_states[indices] = packet.next_states
        self.next_masks[indices] = packet.next_masks
        self.dones[indices] = packet.dones
        self.discounts[indices] = packet.discounts
        scaled = np.full(count, self.max_priority ** self.alpha, dtype=np.float64)
        self.tree.update(indices, scaled)
        self.position = int((self.position + count) % self.capacity)
        self.length = min(self.capacity, self.length + count)

    def sample(self, batch_size: int, beta: float, *, augment: bool = True) -> ReplaySample:
        if self.length < batch_size:
            raise ValueError("not enough replay entries for the requested batch")
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        total = self.tree.total
        segment = total / batch_size
        masses = (np.arange(batch_size) + self.rng.random(batch_size)) * segment
        indices = self.tree.find_prefix(masses)
        probabilities = self.tree.sums[self.tree.size + indices] / total
        minimum_probability = self.tree.minimum / total
        maximum_weight = (self.length * minimum_probability) ** (-beta)
        weights = (self.length * probabilities) ** (-beta) / maximum_weight
        states = self.states[indices].copy()
        actions = self.actions[indices].astype(np.int64, copy=True)
        next_states = self.next_states[indices].copy()
        next_masks = self.next_masks[indices].copy()
        if augment:
            transforms = self.rng.integers(0, 8, size=batch_size, dtype=np.int8)
            states, actions, next_states, next_masks = augment_batch(
                states, actions, next_states, next_masks, transforms
            )
        return ReplaySample(
            indices=indices,
            states=states,
            actions=actions,
            rewards=self.rewards[indices].copy(),
            next_states=next_states,
            next_masks=next_masks,
            dones=self.dones[indices].copy(),
            discounts=self.discounts[indices].copy(),
            weights=weights.astype(np.float32),
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        priorities = np.abs(np.asarray(td_errors, dtype=np.float64)) + self.priority_epsilon
        self.max_priority = max(self.max_priority, float(priorities.max(initial=0.0)))
        self.tree.update(np.asarray(indices, dtype=np.int64), priorities ** self.alpha)
