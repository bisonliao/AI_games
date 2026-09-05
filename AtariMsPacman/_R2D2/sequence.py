"""Trajectory windows used by recurrent prioritized replay."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class Sequence:
    """One replay item.

    ``observations`` contains the burn-in prefix, learning states, and the
    lookahead state needed by the n-step target.  Its first state is paired
    with ``initial_hidden``.
    """

    observations: np.ndarray
    previous_actions: np.ndarray
    previous_rewards: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    n_step_rewards: np.ndarray
    discounts: np.ndarray
    terminated: np.ndarray
    initial_hidden: tuple[np.ndarray, np.ndarray]
    burn_in_steps: int
    learning_steps: int
    forward_steps: int
    start_transition: int = 0

    def __len__(self) -> int:
        return int(self.learning_steps)

    @property
    def observation_steps(self) -> int:
        """Number of recurrent observations represented by packed storage."""
        return int(self.previous_actions.shape[0])

    def unpack_observations(self, stack_size: int = 4) -> np.ndarray:
        """Return a zero-copy ``(T, stack, H, W)`` sliding-window view."""
        if self.observations.ndim == 4:
            # Compatibility with hand-built samples and older checkpoints.
            return self.observations
        if self.observations.ndim != 3:
            raise ValueError("packed observations must have shape (T+stack-1, H, W)")
        windows = np.lib.stride_tricks.sliding_window_view(
            self.observations, window_shape=stack_size, axis=0
        )
        return np.moveaxis(windows, -1, 1)


@dataclass(slots=True)
class _Record:
    observation: np.ndarray
    previous_action: np.ndarray
    previous_reward: float
    action: int
    reward: float
    next_observation: np.ndarray
    hidden_before: tuple[np.ndarray, np.ndarray]
    terminated: bool


def mixed_priority(td_errors: np.ndarray, mix: float = 0.9) -> float:
    """R2D2's max/mean sequence priority mixture."""
    values = np.abs(np.asarray(td_errors, dtype=np.float32))
    if values.size == 0:
        return 0.0
    return float(mix * values.max() + (1.0 - mix) * values.mean())


class SequenceAssembler:
    """Build overlapping windows without crossing an episode boundary."""

    def __init__(
        self,
        action_count: int,
        *,
        burn_in_steps: int = 40,
        learning_steps: int = 40,
        forward_steps: int = 5,
        gamma: float = 0.997,
    ) -> None:
        if min(action_count, learning_steps, forward_steps) <= 0:
            raise ValueError("action_count, learning_steps and forward_steps must be positive")
        if burn_in_steps < 0 or not 0.0 <= gamma <= 1.0:
            raise ValueError("invalid burn-in or discount")
        self.action_count = int(action_count)
        self.burn_in_steps = int(burn_in_steps)
        self.learning_steps = int(learning_steps)
        self.forward_steps = int(forward_steps)
        self.gamma = float(gamma)
        self.records: list[_Record] = []
        self.next_start = 0
        self._base_transition = 0

    def reset(self) -> None:
        self.records.clear()
        self.next_start = 0
        self._base_transition = 0

    def add(
        self,
        observation: np.ndarray,
        previous_action: np.ndarray,
        previous_reward: float,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        hidden_before: tuple[np.ndarray, np.ndarray],
        *,
        terminated: bool = False,
    ) -> list[Sequence]:
        """Append one environment decision and return newly complete windows."""
        previous_action = np.asarray(previous_action, dtype=np.float32).copy()
        if previous_action.shape != (self.action_count,):
            raise ValueError("previous_action must be a one-hot vector")
        hidden = (
            np.asarray(hidden_before[0], dtype=np.float32).copy(),
            np.asarray(hidden_before[1], dtype=np.float32).copy(),
        )
        self.records.append(
            _Record(
                np.asarray(observation, dtype=np.uint8).copy(),
                previous_action,
                float(previous_reward),
                int(action),
                float(reward),
                np.asarray(next_observation, dtype=np.uint8).copy(),
                hidden,
                bool(terminated),
            )
        )
        emitted: list[Sequence] = []
        while len(self.records) >= self.next_start + self.learning_steps + self.forward_steps:
            emitted.append(self._build(self.next_start, self.learning_steps))
            self.next_start += self.learning_steps
        if terminated:
            # Flush every remaining tail.  No item can include a record after
            # this terminal transition, so windows never cross game-over.
            while self.next_start < len(self.records):
                learning = min(self.learning_steps, len(self.records) - self.next_start)
                emitted.append(
                    self._build(self.next_start, learning)
                )
                self.next_start += self.learning_steps
        self._compact()
        return emitted

    def flush(self, *, terminated: bool = False) -> list[Sequence]:
        """Flush a partially collected trajectory (used at an actor cap)."""
        emitted: list[Sequence] = []
        while self.next_start < len(self.records):
            learning = min(self.learning_steps, len(self.records) - self.next_start)
            emitted.append(self._build(self.next_start, learning))
            self.next_start += self.learning_steps
        self._compact()
        if terminated:
            self.records.clear()
            self.next_start = 0
        return emitted

    def _compact(self) -> None:
        """Discard records that no future overlapping window can reference."""
        drop = max(0, self.next_start - self.burn_in_steps)
        if drop:
            self.records = self.records[drop:]
            self.next_start -= drop
            self._base_transition += drop

    def _build(self, start: int, learning: int) -> Sequence:
        burn = min(self.burn_in_steps, start)
        context_start = start - burn
        end = start + learning + self.forward_steps
        observations: list[np.ndarray] = []
        previous_actions: list[np.ndarray] = []
        previous_rewards: list[float] = []
        for index in range(context_start, end):
            if index < len(self.records):
                record = self.records[index]
                observations.append(record.observation)
                previous_actions.append(record.previous_action)
                previous_rewards.append(record.previous_reward)
            else:
                # The only possible out-of-range state is the final next
                # observation.  Carrying the last action/reward is harmless
                # because terminal tails have zero bootstrap discount.
                last = self.records[-1]
                observations.append(last.next_observation)
                previous_actions.append(
                    np.eye(self.action_count, dtype=np.float32)[last.action]
                )
                previous_rewards.append(last.reward)

        actions = np.asarray([r.action for r in self.records[start : start + learning]], dtype=np.int64)
        rewards = np.asarray([r.reward for r in self.records[start : start + learning]], dtype=np.float32)
        terminated = np.asarray(
            [r.terminated for r in self.records[start : start + learning]], dtype=np.bool_
        )
        n_step_rewards = np.zeros(learning, dtype=np.float32)
        discounts = np.zeros(learning, dtype=np.float32)
        for offset in range(learning):
            total = 0.0
            discount = 1.0
            terminal = False
            horizon = min(self.forward_steps, len(self.records) - (start + offset))
            for k in range(horizon):
                record = self.records[start + offset + k]
                total += discount * record.reward
                if record.terminated:
                    terminal = True
                    discount = 0.0
                    break
                discount *= self.gamma
            n_step_rewards[offset] = total
            if not terminal and horizon == self.forward_steps:
                discounts[offset] = discount
        initial = self.records[context_start].hidden_before
        stacked_observations = np.stack(observations).astype(np.uint8, copy=False)
        if stacked_observations.shape[1] > 1 and np.array_equal(
            stacked_observations[1:, :-1], stacked_observations[:-1, 1:]
        ):
            packed_observations = np.concatenate(
                (stacked_observations[0], stacked_observations[1:, -1]), axis=0
            )
        else:
            # Keep a fully stacked representation when callers provide
            # synthetic/non-sliding observations (useful for diagnostics).
            packed_observations = stacked_observations
        return Sequence(
            observations=packed_observations,
            previous_actions=np.stack(previous_actions).astype(np.float32, copy=False),
            previous_rewards=np.asarray(previous_rewards, dtype=np.float32),
            actions=actions,
            rewards=rewards,
            n_step_rewards=n_step_rewards,
            discounts=discounts,
            terminated=terminated,
            initial_hidden=(initial[0].copy(), initial[1].copy()),
            burn_in_steps=burn,
            learning_steps=learning,
            forward_steps=self.forward_steps,
            start_transition=self._base_transition + start,
        )
