import unittest
from unittest.mock import patch

import numpy as np
from robot_arm.RobotEnv import PandaTabletopEnv
from robot_arm.SAC_VecObs.env import SACVectorTaskEnv, PickPlaceStage, STAGE_STEP_LIMITS
from robot_arm.SAC_PixelObs.env import PixelTaskEnv, PIXEL_STAGE_STEP_LIMITS, CAMERA_VIEW_NAMES


class EnvironmentTests(unittest.TestCase):
    def test_five_stage_reward_regression_and_timeouts(self):
        pixel = PixelTaskEnv(task="pick_place", frame_stack=2, action_repeat=1)
        vector = SACVectorTaskEnv(task="pick_place", action_repeat=1)
        previous = dict(ee_object_distance=0.3, object_goal_xy_distance=0.3,
                        object_position=np.array([0.5, 0, 0.2]))
        current = dict(ee_object_distance=0.2, object_goal_xy_distance=0.2,
                       object_position=np.array([0.5, 0, 0.1]))
        expected = [0.19, -0.41, 0.19, 0.19, -0.01]
        try:
            pixel.reset(seed=42)
            vector.reset(seed=42)
            for task, limits in [(pixel.task_env, PIXEL_STAGE_STEP_LIMITS), (vector, STAGE_STEP_LIMITS)]:
                task.ever_grasped = task.ever_lifted = False
                task.place_bonus_given = task.release_bonus_given = True
                with patch.object(task, "_update_stage", return_value=""), patch.object(task, "_settled_at_goal", return_value=False):
                    for stage in PickPlaceStage:
                        task.stage = stage
                        task._stage_steps = 0
                        reward, success, failure = task._pick_place_reward(previous, current, np.zeros(4))
                        self.assertAlmostEqual(reward, expected[int(stage)], places=6)
                        self.assertFalse(success or failure)
                        task._stage_steps = limits[stage] - 1
                        _, _, failure = task._pick_place_reward(previous, current, np.zeros(4))
                        self.assertTrue(failure)
                        self.assertEqual(task._failure_reason, stage.name.lower() + "_timeout")
        finally:
            pixel.close()
            vector.close()

    def test_base_and_vector(self):
        for factory in [PandaTabletopEnv, SACVectorTaskEnv]:
            env = factory(task="pick_place", action_repeat=1, max_episode_steps=3)
            try:
                env.reset(seed=42)
                observation, reward, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
                self.assertTrue(np.isfinite(reward))
                self.assertEqual(observation.ndim, 1)
            finally:
                env.close()

    def test_pixel_history_and_reward_semantics(self):
        env = PixelTaskEnv(task="pick_place", frame_stack=2, action_repeat=1, max_episode_steps=3)
        try:
            first, info = env.reset(seed=42)
            second, reward, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
            self.assertEqual(first["image"].shape, (18, 96, 96))
            self.assertEqual(first["proprio"].shape, (26,))
            np.testing.assert_array_equal(first["image"][9:], second["image"][:9])
            self.assertEqual(CAMERA_VIEW_NAMES, ("xy", "xz", "yz"))
            self.assertEqual(len(PIXEL_STAGE_STEP_LIMITS), 5)
            self.assertTrue(all(value == 100 for value in PIXEL_STAGE_STEP_LIMITS.values()))
            self.assertEqual(list(STAGE_STEP_LIMITS.values()), [50, 30, 75, 100, 20])
            self.assertAlmostEqual(reward, sum(info["reward_terms"].values()), places=6)
            self.assertEqual(info["stage_index"], 0)
            env.task_env._enter_stage(PickPlaceStage.GRASP)
            with self.assertRaises(RuntimeError):
                env.task_env._enter_stage(PickPlaceStage.APPROACH)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
