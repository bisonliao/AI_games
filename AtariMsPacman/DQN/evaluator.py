"""Asynchronous greedy checkpoint evaluator running exclusively on CPU."""

from __future__ import annotations

from dataclasses import replace
import time
import traceback

import numpy as np
import torch

from DQN.config import DQNConfig
from DQN.messages import (
    EvaluationRequest,
    EvaluationResult,
    EvaluatorStop,
    ProcessError,
)
from DQN.network import DuelingQNetwork
from DQN.rewards import shape_reward
from PacManEnv import make_env


def evaluator_process(
    config: DQNConfig,
    evaluation_request_queue,
    evaluation_result_queue,
    error_queue,
) -> None:
    process_name = "evaluator"
    try:
        evaluation_result_queue.cancel_join_thread()
        torch.set_num_threads(config.evaluator_torch_threads)
        torch.set_num_interop_threads(1)
        model = DuelingQNetwork(config.observation_shape, config.action_count).cpu()
        model.eval()
        eval_env_config = replace(config.actor_env, num_envs=1)
        env = make_env(eval_env_config)
        try:
            while True:
                request = evaluation_request_queue.get()
                if isinstance(request, EvaluatorStop):
                    break
                if not isinstance(request, EvaluationRequest):
                    raise TypeError(f"Unexpected evaluator message: {type(request)}")

                checkpoint = torch.load(
                    request.checkpoint_path,
                    map_location="cpu",
                    weights_only=False,
                )
                model.load_state_dict(checkpoint["online_state_dict"])
                model.eval()
                episode_lengths: list[int] = []
                episode_returns: list[float] = []
                episode_raw_scores: list[float] = []
                capped_episodes = 0
                started = time.monotonic()

                for episode_index in range(config.evaluation_episodes):
                    observation, _ = env.reset(
                        seed=config.evaluation_seed + episode_index
                    )
                    episode_length = 0
                    episode_return = 0.0
                    while True:
                        with torch.inference_mode():
                            q_values = model(torch.from_numpy(observation[None]))
                            action = int(q_values.argmax(dim=1).item())
                        observation, reward, terminated, truncated, info = env.step(
                            action
                        )
                        if truncated:
                            raise RuntimeError("Evaluator received a forbidden truncation")
                        episode_length += 1
                        if not np.isclose(reward, float(info["raw_reward"])):
                            raise RuntimeError(
                                "Evaluator environment must return raw rewards"
                            )
                        episode_return += shape_reward(
                            float(info["raw_reward"]),
                            bool(info["life_lost"]),
                            bool(info["game_over"]),
                            config,
                        )
                        if terminated:
                            episode_lengths.append(episode_length)
                            episode_returns.append(episode_return)
                            episode_raw_scores.append(float(info["raw_score"]))
                            break
                        if episode_length >= config.evaluation_max_episode_steps:
                            episode_lengths.append(episode_length)
                            episode_returns.append(episode_return)
                            episode_raw_scores.append(float(info["raw_score"]))
                            capped_episodes += 1
                            break

                evaluation_result_queue.put(
                    EvaluationResult(
                        checkpoint_transition=request.checkpoint_transition,
                        checkpoint_path=request.checkpoint_path,
                        episode_lengths=episode_lengths,
                        episode_returns=episode_returns,
                        episode_raw_scores=episode_raw_scores,
                        capped_episodes=capped_episodes,
                        elapsed_seconds=time.monotonic() - started,
                    )
                )
        finally:
            env.close()
    except BaseException:
        error_queue.put(
            ProcessError(process_name=process_name, traceback=traceback.format_exc())
        )
        raise
