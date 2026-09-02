from dataclasses import replace

import numpy as np

from DQN.config import DQNConfig
from DQN.rewards import shape_rewards
from PacManEnv import MsPacmanEnvConfig, make_env


def test_reward_shaping_order_and_lost_life_override() -> None:
    config = DQNConfig()
    raw = np.asarray(
        [-10.0, 0.0, 10.0, 50.0, 200.0, 400.0, 800.0, 1600.0, 0.0, 0.0],
        dtype=np.float32,
    )
    life_lost = np.asarray(
        [False, False, False, False, False, False, False, False, True, True]
    )
    game_over = np.asarray(
        [False, False, False, False, False, False, False, False, False, True]
    )
    shaped = shape_rewards(raw, life_lost, game_over, config)
    expected = [
        -0.01,
        -0.01,
        np.log1p(1.0) - 0.01,
        np.log1p(5.0) - 0.01,
        np.log1p(20.0) - 0.01,
        np.log1p(40.0) - 0.01,
        np.log1p(80.0) - 0.01,
        4.99,
        -5.0,
        -10.0,
    ]
    np.testing.assert_allclose(shaped, expected, rtol=1.0e-6)


def test_actor_environment_configuration_is_raw() -> None:
    config = DQNConfig()
    assert config.actor_env.step_cost == 0.0
    assert config.actor_env.clip_training_reward is False


def test_lost_life_raw_reward_is_zero_for_current_rom() -> None:
    config = replace(
        MsPacmanEnvConfig(),
        num_envs=1,
        frame_skip=4,
        noop_max=0,
        repeat_action_probability=0.0,
        step_cost=0.0,
        clip_training_reward=False,
    )
    env = make_env(config)
    rng = np.random.default_rng(1_000)
    lost_life_rewards: list[float] = []
    try:
        env.reset(seed=0)
        for _ in range(5_000):
            _, reward, terminated, truncated, info = env.step(
                int(rng.integers(env.action_space.n))
            )
            assert not truncated
            assert reward == info["raw_reward"]
            if info["life_lost"]:
                lost_life_rewards.append(float(info["raw_reward"]))
            if terminated:
                break
    finally:
        env.close()
    assert len(lost_life_rewards) == 3
    assert lost_life_rewards == [0.0, 0.0, 0.0]
