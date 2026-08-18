import unittest
from types import SimpleNamespace

import numpy as np

from maddpg.common.scenario_metrics import (
    SIMPLE_ADVERSARY_TIE_DISTANCE,
    SIMPLE_SPREAD_LANDMARK_CENTER_RADIUS,
    get_scenario_metric_plugin,
    simple_adversary_metrics,
    simple_spread_metrics,
)


def _entity(position):
    return SimpleNamespace(state=SimpleNamespace(p_pos=np.asarray(position)))


def _env(agent_positions, landmark_positions):
    world = SimpleNamespace(
        agents=[_entity(position) for position in agent_positions],
        landmarks=[_entity(position) for position in landmark_positions],
    )
    return SimpleNamespace(unwrapped=SimpleNamespace(world=world))


def _adversary_env(adversary_positions, good_positions, goal_position=(0, 0)):
    goal = _entity(goal_position)

    def agent(position, adversary):
        value = _entity(position)
        value.adversary = adversary
        value.goal_a = goal
        return value

    world = SimpleNamespace(
        agents=[agent(position, True) for position in adversary_positions]
        + [agent(position, False) for position in good_positions],
        landmarks=[goal],
    )
    return SimpleNamespace(unwrapped=SimpleNamespace(world=world))


class SimpleSpreadMetricsTest(unittest.TestCase):
    def test_success_requires_all_landmarks_covered(self):
        env = _env(
            agent_positions=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
            landmark_positions=[(0.05, 0.0), (1.05, 0.0), (2.05, 0.0)],
        )

        self.assertEqual(
            simple_spread_metrics(env),
            {
                "covered_landmarks": 3.0,
                "coverage_ratio": 1.0,
                "episode_success": 1.0,
                "landmark_center_success": 1.0,
            },
        )

    def test_one_agent_is_not_counted_twice(self):
        env = _env(
            agent_positions=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
            landmark_positions=[(0.02, 0.0), (0.04, 0.0), (1.05, 0.0)],
        )

        self.assertEqual(
            simple_spread_metrics(env),
            {
                "covered_landmarks": 2.0,
                "coverage_ratio": 2.0 / 3.0,
                "episode_success": 0.0,
                "landmark_center_success": 0.0,
            },
        )

    def test_landmark_center_success_uses_agent_radius(self):
        offset = SIMPLE_SPREAD_LANDMARK_CENTER_RADIUS - 0.01
        env = _env(
            agent_positions=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
            landmark_positions=[
                (offset, 0.0),
                (1.0 + offset, 0.0),
                (2.0 + offset, 0.0),
            ],
        )

        metrics = simple_spread_metrics(env)

        self.assertEqual(metrics["episode_success"], 0.0)
        self.assertEqual(metrics["landmark_center_success"], 1.0)


class SimpleAdversaryMetricsTest(unittest.TestCase):
    def test_good_closer_metrics(self):
        metrics = simple_adversary_metrics(
            _adversary_env([(0.4, 0.0)], [(0.1, 0.0), (0.8, 0.0)])
        )

        self.assertAlmostEqual(metrics["mean_adv_goal_distance"], 0.4)
        self.assertAlmostEqual(
            metrics["mean_nearest_good_goal_distance"], 0.1
        )
        self.assertAlmostEqual(metrics["mean_distance_gap"], 0.3)
        self.assertEqual(metrics["good_closer_rate"], 1.0)
        self.assertEqual(metrics["adversary_closer_rate"], 0.0)
        self.assertEqual(metrics["tie_rate"], 0.0)

    def test_adversary_closer_metrics(self):
        metrics = simple_adversary_metrics(
            _adversary_env([(0.1, 0.0)], [(0.4, 0.0), (0.8, 0.0)])
        )

        self.assertAlmostEqual(metrics["mean_distance_gap"], -0.3)
        self.assertEqual(metrics["good_closer_rate"], 0.0)
        self.assertEqual(metrics["adversary_closer_rate"], 1.0)
        self.assertEqual(metrics["tie_rate"], 0.0)

    def test_tie_rate_uses_configured_distance_tolerance(self):
        gap_inside_tolerance = SIMPLE_ADVERSARY_TIE_DISTANCE * 0.8
        metrics = simple_adversary_metrics(
            _adversary_env(
                [(0.1 + gap_inside_tolerance, 0.0)],
                [(0.1, 0.0), (0.8, 0.0)],
            )
        )

        self.assertEqual(metrics["good_closer_rate"], 0.0)
        self.assertEqual(metrics["adversary_closer_rate"], 0.0)
        self.assertEqual(metrics["tie_rate"], 1.0)
        self.assertEqual(
            metrics["good_closer_rate"]
            + metrics["adversary_closer_rate"]
            + metrics["tie_rate"],
            1.0,
        )

    def test_plugin_is_conditional_on_simple_adversary(self):
        self.assertIs(
            get_scenario_metric_plugin("simple_adversary"),
            simple_adversary_metrics,
        )
        self.assertIs(
            get_scenario_metric_plugin("simple_adversary_v3"),
            simple_adversary_metrics,
        )


if __name__ == "__main__":
    unittest.main()
