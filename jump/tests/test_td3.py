from __future__ import annotations

import numpy as np
import pytest
import torch

from td3.agent import BanditTD3, resolve_device
from td3.replay import ReplayBuffer


def _batch(next_value: float = 0.0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(3)
    return {
        "observations": rng.random((32, 1), dtype=np.float32),
        "actions": rng.uniform(-1, 1, size=(32, 1)).astype(np.float32),
        "rewards": rng.normal(size=32).astype(np.float32),
        "next_observations": np.full((32, 1), next_value, dtype=np.float32),
        "terminated": np.ones(32, dtype=np.bool_),
        "truncated": np.zeros(32, dtype=np.bool_),
    }


def test_replay_wraps_and_samples() -> None:
    replay = ReplayBuffer(capacity=16, seed=0)
    replay.add_batch(_batch())
    assert len(replay) == 16
    sample = replay.sample(8)
    assert sample["observations"].shape == (8, 1)
    assert sample["actions"].dtype == np.float32


def test_critic_target_does_not_depend_on_next_observation() -> None:
    torch.manual_seed(5)
    first = BanditTD3(hidden_dim=32)
    torch.manual_seed(5)
    second = BanditTD3(hidden_dim=32)
    first_result = first.update(_batch(next_value=-100.0))
    second_result = second.update(_batch(next_value=100.0))
    assert first_result.critic_loss == pytest.approx(second_result.critic_loss)
    for first_parameter, second_parameter in zip(
        first.critic1.parameters(), second.critic1.parameters(), strict=True
    ):
        torch.testing.assert_close(first_parameter, second_parameter)


def test_device_resolution_preserves_requested_type() -> None:
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("cuda").type == "cuda"
