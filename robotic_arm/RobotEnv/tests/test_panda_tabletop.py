import numpy as np

from RobotEnv import PandaTabletopEnv


def test_reach_env_reset_and_step():
    env = PandaTabletopEnv(task="reach", max_episode_steps=5, action_repeat=1)
    try:
        observation, info = env.reset(seed=7)
        assert observation.shape == (28,)
        assert observation.dtype == np.float32
        assert info["task"] == "reach"
        next_observation, reward, terminated, truncated, step_info = env.step(
            np.zeros(4, dtype=np.float32)
        )
        assert next_observation.shape == (28,)
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert step_info["step"] == 1
    finally:
        env.close()


def test_pick_place_env_is_seeded_and_renders_rgb():
    env = PandaTabletopEnv(task="pick_place", render_mode="rgb_array", action_repeat=1)
    try:
        first, _ = env.reset(seed=11)
        first_object = env.object_position.copy()
        first_goal = env.goal_position.copy()
        second, _ = env.reset(seed=11)
        np.testing.assert_allclose(first, second)
        np.testing.assert_allclose(first_object, env.object_position)
        np.testing.assert_allclose(first_goal, env.goal_position)
        image = env.render()
        assert image.shape == (256, 256, 3)
        assert image.dtype == np.uint8
    finally:
        env.close()
