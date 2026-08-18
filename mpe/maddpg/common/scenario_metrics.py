"""Scenario-specific business metrics for Legacy and PettingZoo MPE.

Metric plugins are intentionally kept outside the generic training loop.  A
plugin reads the terminal environment state and returns scalar values that the
caller can report to TensorBoard.
"""

from collections.abc import Callable
from typing import Dict, Optional

import numpy as np


ScenarioMetricPlugin = Callable[[object], Dict[str, float]]

# PettingZoo simple_spread's benchmark_data() considers a landmark occupied
# when its nearest agent is strictly closer than 0.1.
SIMPLE_SPREAD_COVERAGE_RADIUS = 0.1
SIMPLE_SPREAD_LANDMARK_CENTER_RADIUS = 0.15
SIMPLE_ADVERSARY_TIE_DISTANCE = 0.05


def _maximum_bipartite_matches(within_coverage: np.ndarray) -> int:
    """Return the maximum number of one-to-one landmark/agent matches."""

    if within_coverage.ndim != 2:
        raise ValueError("within_coverage must be a 2-D matrix")

    num_agents, num_landmarks = within_coverage.shape
    agent_to_landmark = [-1] * num_agents

    def assign_landmark(landmark_index: int, seen_agents: set[int]) -> bool:
        for agent_index in range(num_agents):
            if not within_coverage[agent_index, landmark_index]:
                continue
            if agent_index in seen_agents:
                continue
            seen_agents.add(agent_index)

            previous_landmark = agent_to_landmark[agent_index]
            if previous_landmark == -1 or assign_landmark(
                previous_landmark, seen_agents
            ):
                agent_to_landmark[agent_index] = landmark_index
                return True
        return False

    matches = 0
    for landmark_index in range(num_landmarks):
        if assign_landmark(landmark_index, set()):
            matches += 1
    return matches


def simple_spread_metrics(env: object) -> Dict[str, float]:
    """Compute terminal one-agent-per-landmark coverage for simple_spread.

    Strict full coverage uses distance < 0.1, matching PettingZoo's own
    benchmark definition.  Landmark-center coverage additionally uses
    distance < 0.15, the simple_spread agent radius.  Maximum bipartite
    matching prevents one agent from being counted as covering two nearby
    landmarks.
    """

    raw_env = getattr(env, "unwrapped", env)
    world = getattr(raw_env, "world", None)
    if world is None:
        raise ValueError("simple_spread metric plugin requires env.unwrapped.world")

    agents = list(world.agents)
    landmarks = list(world.landmarks)
    if not agents or not landmarks:
        return {
            "covered_landmarks": 0.0,
            "coverage_ratio": 0.0,
            "episode_success": 0.0,
            "landmark_center_success": 0.0,
        }

    agent_positions = np.asarray([agent.state.p_pos for agent in agents])
    landmark_positions = np.asarray(
        [landmark.state.p_pos for landmark in landmarks]
    )
    distances = np.linalg.norm(
        agent_positions[:, np.newaxis, :] - landmark_positions[np.newaxis, :, :],
        axis=-1,
    )
    covered_landmarks = _maximum_bipartite_matches(
        distances < SIMPLE_SPREAD_COVERAGE_RADIUS
    )
    landmark_centers_covered = _maximum_bipartite_matches(
        distances < SIMPLE_SPREAD_LANDMARK_CENTER_RADIUS
    )
    total_landmarks = len(landmarks)

    return {
        "covered_landmarks": float(covered_landmarks),
        "coverage_ratio": float(covered_landmarks / total_landmarks),
        "episode_success": float(covered_landmarks == total_landmarks),
        "landmark_center_success": float(
            landmark_centers_covered == total_landmarks
        ),
    }


def simple_adversary_metrics(env: object) -> Dict[str, float]:
    """Compute terminal goal-distance and closer-side metrics.

    The scenario contains one adversary, but taking the adversary mean keeps
    the metric definition explicit and robust to compatible scenario variants.
    A positive distance gap means the nearest good agent is closer to the goal.
    """

    raw_env = getattr(env, "unwrapped", env)
    world = getattr(raw_env, "world", None)
    if world is None:
        raise ValueError(
            "simple_adversary metric plugin requires env.unwrapped.world"
        )

    adversaries = [
        agent for agent in world.agents if getattr(agent, "adversary", False)
    ]
    good_agents = [
        agent for agent in world.agents if not getattr(agent, "adversary", False)
    ]
    if not adversaries or not good_agents:
        raise ValueError(
            "simple_adversary metrics require adversary and good agents"
        )

    adversary_distances = []
    for agent in adversaries:
        goal = getattr(agent, "goal_a", None)
        if goal is None:
            raise ValueError("simple_adversary adversary has no goal landmark")
        adversary_distances.append(
            float(np.linalg.norm(agent.state.p_pos - goal.state.p_pos))
        )

    good_distances = []
    for agent in good_agents:
        goal = getattr(agent, "goal_a", None)
        if goal is None:
            raise ValueError("simple_adversary good agent has no goal landmark")
        good_distances.append(
            float(np.linalg.norm(agent.state.p_pos - goal.state.p_pos))
        )

    mean_adv_distance = float(np.mean(adversary_distances))
    nearest_good_distance = float(np.min(good_distances))
    distance_gap = mean_adv_distance - nearest_good_distance
    good_is_closer = distance_gap > SIMPLE_ADVERSARY_TIE_DISTANCE
    adversary_is_closer = distance_gap < -SIMPLE_ADVERSARY_TIE_DISTANCE
    tie = not good_is_closer and not adversary_is_closer

    return {
        "mean_adv_goal_distance": mean_adv_distance,
        "mean_nearest_good_goal_distance": nearest_good_distance,
        "mean_distance_gap": float(distance_gap),
        "good_closer_rate": float(good_is_closer),
        "adversary_closer_rate": float(adversary_is_closer),
        "tie_rate": float(tie),
    }


_SCENARIO_METRIC_PLUGINS: Dict[str, ScenarioMetricPlugin] = {
    "simple_spread": simple_spread_metrics,
    "simple_spread_v3": simple_spread_metrics,
    "simple_adversary": simple_adversary_metrics,
    "simple_adversary_v3": simple_adversary_metrics,
}


def get_scenario_metric_plugin(
    scenario_name: str,
) -> Optional[ScenarioMetricPlugin]:
    """Return a terminal metric plugin for a scenario, if one is registered."""

    return _SCENARIO_METRIC_PLUGINS.get(scenario_name)
