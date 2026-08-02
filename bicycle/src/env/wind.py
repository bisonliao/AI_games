"""Deterministic smooth cross-wind gust process driven by Gymnasium RNG."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .config import WindConfig


@dataclass(frozen=True, slots=True)
class WindState:
    """Instantaneous force and current gust metadata returned to the environment."""

    force_y_n: float = 0.0
    peak_force_n: float = 0.0
    direction: int = 0
    active: bool = False
    gust_count: int = 0


class SmoothGustGenerator:
    """Seeded cross-wind generator with smooth sin-squared gust envelopes."""

    def __init__(self, config: WindConfig):
        self.config = config
        self._rng: np.random.Generator | None = None
        self._start_s = math.inf
        self._end_s = math.inf
        self._peak_n = 0.0
        self._direction = 0
        self._gust_count = 0

    def reset(self, rng: np.random.Generator) -> WindState:
        """Reset gust scheduling and retain the environment-owned seeded RNG."""
        self._rng = rng
        self._gust_count = 0
        self._peak_n = 0.0
        self._direction = 0
        if self.config.enabled:
            self._start_s = self._sample_interval(0.0)
        else:
            self._start_s = math.inf
        self._end_s = math.inf
        return WindState()

    def value(self, time_s: float) -> WindState:
        """Return wind at simulation time, lazily scheduling future gusts."""
        if not self.config.enabled or self._rng is None:
            return WindState(gust_count=self._gust_count)

        if time_s >= self._end_s:
            self._start_s = self._sample_interval(self._end_s)
            self._end_s = math.inf
            self._peak_n = 0.0
            self._direction = 0

        if time_s >= self._start_s and not math.isfinite(self._end_s):
            self._peak_n = float(
                self._rng.uniform(self.config.min_force_n, self.config.max_force_n)
            )
            self._direction = 1 if int(self._rng.integers(0, 2)) else -1
            duration = float(
                self._rng.uniform(
                    self.config.min_duration_s, self.config.max_duration_s
                )
            )
            self._end_s = self._start_s + duration
            self._gust_count += 1

        if self._start_s <= time_s < self._end_s:
            phase = (time_s - self._start_s) / (self._end_s - self._start_s)
            envelope = math.sin(math.pi * phase) ** 2
            return WindState(
                force_y_n=self._direction * self._peak_n * envelope,
                peak_force_n=self._peak_n,
                direction=self._direction,
                active=True,
                gust_count=self._gust_count,
            )
        return WindState(gust_count=self._gust_count)

    def _sample_interval(self, after_s: float) -> float:
        assert self._rng is not None
        return after_s + float(
            self._rng.uniform(
                self.config.min_interval_s, self.config.max_interval_s
            )
        )
