import numpy as np

from SAC_VecObs.env import (
    PickPlaceStage,
    SACVectorTaskEnv,
    VIOLATION_GRACE_STEPS,
)


def test_reach_has_three_actions_and_seeded_vector_observation():
    env = SACVectorTaskEnv(task="reach", max_episode_steps=5, action_repeat=1)
    try:
        first, _ = env.reset(seed=23)
        second, _ = env.reset(seed=23)
        assert first.shape == (52,)
        assert first.dtype == np.float32
        assert env.action_space.shape == (3,)
        np.testing.assert_allclose(first, second)
        _, reward, terminated, truncated, info = env.step(
            np.zeros(3, dtype=np.float32)
        )
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert info["stage"] == "approach"
    finally:
        env.close()


def test_pick_place_script_exercises_full_phase_machine():
    env = SACVectorTaskEnv(task="pick_place", max_episode_steps=180, action_repeat=8)
    try:
        env.reset(seed=0)
        visited = set()
        final_info = None
        for _ in range(180):
            action = env.base_env.heuristic_action()
            _, _, terminated, truncated, final_info = env.step(action)
            visited.add(final_info["stage"])
            if terminated or truncated:
                break
        assert final_info is not None
        assert visited == {"approach", "grasp", "transport", "place", "release"}
        assert final_info["success"] is True
        assert final_info["ever_grasped"] is True
        assert final_info["ever_lifted"] is True
    finally:
        env.close()


def test_pick_place_time_limit_is_not_bypassed_by_base_success():
    """A pushed object must not create an infinite wrapper episode.

    RobotEnv's demo task accepts an object near the goal without requiring a
    prior lift.  The SAC task deliberately rejects that shortcut, but must
    still truncate at its own horizon when the base task reports terminated.
    """
    env = SACVectorTaskEnv(task="pick_place", max_episode_steps=5, action_repeat=1)
    try:
        env.reset(seed=3)
        env.base_env._set_goal_position(env.base_env.object_position)
        env.base_env.step_count = 4
        _, _, terminated, truncated, info = env.step(
            np.zeros(4, dtype=np.float32)
        )
        assert terminated is False
        assert truncated is True
        assert info["success"] is False
        assert info["time_limit_reached"] is True
        assert env.ever_lifted is False
    finally:
        env.close()


def _fake_pick_place_metrics(*, contact: bool, lifted: bool = False, z: float = 0.025):
    object_position = np.array([0.55, 0.0, z], dtype=np.float32)
    goal_position = np.array([0.70, 0.0, 0.025], dtype=np.float32)
    return {
        "ee_position": np.array([0.55, 0.0, 0.10], dtype=np.float32),
        "object_position": object_position,
        "goal_position": goal_position,
        "linear_velocity": np.zeros(3, dtype=np.float32),
        "angular_velocity": np.zeros(3, dtype=np.float32),
        "ee_object_distance": 0.075,
        "object_goal_distance": 0.15,
        "object_goal_xy_distance": 0.15,
        "gripper_width": 0.05 if contact else 0.08,
        "finger_contact": contact,
        "lifted": lifted,
    }


def test_pick_place_confirms_grasp_and_fails_regression_without_stage_loop():
    env = SACVectorTaskEnv(task="pick_place", max_episode_steps=150, action_repeat=1)
    try:
        env.reset(seed=4)
        metrics = _fake_pick_place_metrics(contact=True)
        assert env._update_stage(metrics) == ""
        assert env.stage.name.lower() == "approach"
        assert env._update_stage(metrics) == ""
        assert env.stage.name.lower() == "grasp"

        env.ever_grasped = True
        env._stage_steps = 0
        no_contact = _fake_pick_place_metrics(contact=False)
        for _ in range(VIOLATION_GRACE_STEPS - 1):
            _, _, failure = env._pick_place_reward(metrics, no_contact, np.zeros(4))
            assert failure is False
            assert env.stage.name.lower() == "grasp"
        reward, _, failure = env._pick_place_reward(
            metrics, no_contact, np.zeros(4)
        )
        assert failure is True
        assert env.stage.name.lower() == "grasp"
        assert env._failure_reason == "grasp_lost"
        assert reward < -4.0
    finally:
        env.close()


def test_pick_place_grasp_progress_is_signed_not_an_oscillation_bonus():
    env = SACVectorTaskEnv(task="pick_place", max_episode_steps=150, action_repeat=1)
    try:
        env.reset(seed=5)
        env.stage = PickPlaceStage.GRASP
        env.ever_grasped = True
        lower = _fake_pick_place_metrics(contact=True, z=0.035)
        higher = _fake_pick_place_metrics(contact=True, z=0.045)
        env._pick_place_reward(lower, higher, np.zeros(4))
        upward_progress = env._last_reward_terms["progress"]
        env._pick_place_reward(higher, lower, np.zeros(4))
        downward_progress = env._last_reward_terms["progress"]
        assert upward_progress > 0.0
        assert downward_progress < 0.0
        np.testing.assert_allclose(upward_progress + downward_progress, 0.0)
    finally:
        env.close()
