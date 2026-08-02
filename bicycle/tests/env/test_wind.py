"""Seeded smooth-gust timing, bounds, and envelope tests."""

import numpy as np

from env import WindConfig
from env.wind import SmoothGustGenerator


def collect(seed: int):
    generator = SmoothGustGenerator(
        WindConfig(
            min_force_n=5,
            max_force_n=15,
            min_interval_s=1,
            max_interval_s=1,
            min_duration_s=2,
            max_duration_s=2,
        )
    )
    generator.reset(np.random.default_rng(seed))
    return [generator.value(t) for t in np.linspace(0, 3.1, 32)]


def test_wind_is_seeded_and_bounded():
    first = collect(7)
    second = collect(7)
    assert first == second
    active = [state for state in first if state.active]
    assert active
    assert all(5 <= state.peak_force_n <= 15 for state in active)
    assert all(state.direction in (-1, 1) for state in active)
    assert max(abs(state.force_y_n) for state in active) <= 15


def test_smooth_envelope_is_zero_at_boundaries():
    states = collect(3)
    assert states[10].force_y_n == 0.0  # t=1.0, gust start
    assert abs(states[20].force_y_n) == states[20].peak_force_n
    assert states[30].force_y_n == 0.0  # t=3.0, gust end
