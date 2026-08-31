from __future__ import annotations

import time
from collections import Counter

import numpy as np
import torch

from MathEnv import BasicMathEnv

from .model import DuelingDQN, NetworkSpec
from .preprocessing import EncodedObservation, encode_observation, observation_to_tensors


def _greedy_action(model: DuelingDQN, observation: EncodedObservation) -> int:
    images, macro = observation_to_tensors(observation, "cpu")
    with torch.inference_mode():
        return int(model(images, macro).argmax(dim=1).item())


def evaluate_combined(
    high_model: DuelingDQN,
    low_model: DuelingDQN,
    episodes: int,
    seed: int,
    gui: bool = False,
    fps: float = 0.0,
) -> dict[str, float | dict[int, int]]:
    """Evaluate frozen high/low policies through raw ALE actions only."""
    expected_high_spec = NetworkSpec(input_channels=1, macro_dim=0, num_actions=19)
    expected_low_spec = NetworkSpec(input_channels=2, macro_dim=45, num_actions=6)
    if high_model.spec != expected_high_spec:
        raise ValueError(
            f"High-level checkpoint network {high_model.spec} does not match "
            f"expected {expected_high_spec}"
        )
    if low_model.spec != expected_low_spec:
        if low_model.spec.macro_dim == 21:
            raise ValueError(
                "Legacy low-level checkpoints have 21 conditioning features; "
                "the current combined evaluator requires a retrained 45-feature policy"
            )
        raise ValueError(
            f"Low-level checkpoint network {low_model.spec} does not match "
            f"expected {expected_low_spec}"
        )
    torch.set_num_threads(1)
    high_model = high_model.cpu().eval()
    low_model = low_model.cpu().eval()
    env = BasicMathEnv(
        action_mode="raw",
        goal_conditioned=True,
        render_mode="human" if gui else "rgb_array",
    )

    executor_successes: list[float] = []
    game_successes: list[float] = []
    hierarchy_successes: list[float] = []
    primitive_steps: list[float] = []
    timeouts: list[float] = []
    selected_macros: Counter[int] = Counter()
    problem_pairs: set[tuple[int, int]] = set()
    started = time.monotonic()

    try:
        for episode in range(episodes):
            observation, reset_info = env.reset(
                seed=seed if episode == 0 else None
            )
            operands = reset_info.get("problem_operands")
            if operands is not None:
                problem_pairs.add((int(operands[0]), int(operands[1])))

            high_observation = encode_observation(observation["current"], False)
            macro_action = _greedy_action(high_model, high_observation)
            selected_macros[macro_action] += 1

            observation = env.set_target_macro_action(macro_action)
            low_observation = encode_observation(observation, True)
            done = False
            info: dict[str, object] = {}
            while not done:
                raw_action = _greedy_action(low_model, low_observation)
                observation, _, terminated, truncated, info = env.step(raw_action)
                low_observation = encode_observation(observation, True)
                done = bool(terminated or truncated)
                if fps > 0:
                    time.sleep(1.0 / fps)

            executor_success = float(bool(info.get("success", False)))
            game_success = float(float(info.get("game_reward", 0.0)) > 0.0)
            executor_successes.append(executor_success)
            game_successes.append(game_success)
            hierarchy_successes.append(executor_success * game_success)
            primitive_steps.append(float(info.get("primitive_steps", 0)))
            timeouts.append(float(bool(info.get("timeout", False))))
    finally:
        env.close()

    elapsed = max(time.monotonic() - started, 1e-6)
    successful_executions = int(sum(executor_successes))
    conditional_game_success = (
        float(sum(hierarchy_successes) / successful_executions)
        if successful_executions
        else 0.0
    )
    return {
        "executor_success_rate": float(np.mean(executor_successes)),
        "game_success_rate": float(np.mean(game_successes)),
        "hierarchy_success_rate": float(np.mean(hierarchy_successes)),
        "game_success_given_execution": conditional_game_success,
        "mean_primitive_steps": float(np.mean(primitive_steps)),
        "timeout_rate": float(np.mean(timeouts)),
        "episodes_per_second": episodes / elapsed,
        "unique_problem_count": float(len(problem_pairs)),
        "selected_macro_counts": dict(sorted(selected_macros.items())),
    }
