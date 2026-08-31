from __future__ import annotations

import numpy as np
import torch

from DQN.cli import build_train_parser
from DQN.model import DuelingDQN, NetworkSpec
from DQN.preprocessing import encode_observation
from DQN.replay import PrioritizedReplayBuffer
from DQN.runtime import (
    TrainConfig,
    _sample_low_target,
    low_epsilon_at,
    piecewise_epsilon,
)


def test_dueling_network_shapes() -> None:
    for spec in (NetworkSpec(1, 0, 19), NetworkSpec(2, 45, 6)):
        model = DuelingDQN(spec)
        images = torch.zeros(3, spec.input_channels, 84, 84)
        macro = torch.zeros(3, spec.macro_dim)
        assert model(images, macro).shape == (3, spec.num_actions)


def test_low_preprocessing_concatenates_45_conditioning_features() -> None:
    observation = {
        "current": np.zeros((210, 160, 3), dtype=np.uint8),
        "goal": np.zeros((210, 160, 3), dtype=np.uint8),
        "macro": np.arange(21, dtype=np.float32),
        "current_answer": np.arange(22, dtype=np.float32) + 100,
        "cursor": np.array([200, 201], dtype=np.float32),
    }
    pixels, conditioning = encode_observation(observation, goal_conditioned=True)
    assert pixels.shape == (2, 84, 84)
    assert conditioning.shape == (45,)
    np.testing.assert_array_equal(conditioning[:21], observation["macro"])
    np.testing.assert_array_equal(conditioning[21:43], observation["current_answer"])
    np.testing.assert_array_equal(conditioning[43:], observation["cursor"])


def test_prioritized_replay_round_trip() -> None:
    replay = PrioritizedReplayBuffer(capacity=32, seed=3)
    for index in range(16):
        pixels = np.full((2, 84, 84), index, dtype=np.uint8)
        macro = np.zeros(45, dtype=np.float32)
        replay.add((pixels, macro, index % 6, 0.0, pixels, macro, False, 0.99))

    batch = replay.sample(batch_size=8, beta=0.4)
    assert batch["pixels"].shape == (8, 2, 84, 84)
    assert batch["macro"].shape == (8, 45)
    replay.update_priorities(batch["indices"], np.ones(8, dtype=np.float32))


def test_cli_defaults_come_from_train_config() -> None:
    config = TrainConfig(task="high")
    parsed_defaults = vars(build_train_parser("high").parse_args([]))
    for field_name, parsed_value in parsed_defaults.items():
        assert parsed_value == getattr(config, field_name)

    overridden = build_train_parser("high").parse_args(
        ["--learning-rate", "0.0003", "--actors", "3"]
    )
    assert overridden.learning_rate == 0.0003
    assert overridden.num_actors == 3


def test_low_global_epsilon_schedule() -> None:
    config = TrainConfig(task="low")
    assert piecewise_epsilon(0.0, config.low_epsilon_fraction_points) == 0.9
    assert low_epsilon_at(config, 0) == 0.9
    assert low_epsilon_at(config, 1_000_000) == 0.05
    assert low_epsilon_at(config, 5_000_000) == 0.01
    assert low_epsilon_at(config, 20_000_000) == 0.001


def test_low_train_parser_is_single_stage() -> None:
    parsed = build_train_parser("low").parse_args([])
    assert parsed.total_transitions == 20_000_000
    assert parsed.low_distance_reward_scale == 0.1
    assert not hasattr(parsed, "low_stage")
    assert not hasattr(parsed, "initial_checkpoint")

    overridden = build_train_parser("low").parse_args(
        ["--total-steps", "30000000", "--distance-reward-scale", "0"]
    )
    assert overridden.total_transitions == 30_000_000
    assert overridden.low_distance_reward_scale == 0.0


def test_low_target_sampling_covers_all_answers_without_stages() -> None:
    config = TrainConfig(task="low")
    rng = np.random.default_rng(17)
    sampled = {_sample_low_target(config, rng) for _ in range(2_000)}
    assert sampled == set(range(19))
