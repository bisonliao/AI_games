"""Asynchronous greedy recurrent evaluator running on CPU."""

from __future__ import annotations

import traceback
import time

import numpy as np
import torch

from PacManEnv import MsPacmanEnvConfig, make_env

from _R2D2.config import R2D2Config
from _R2D2.messages import EvaluationRequest, EvaluationResult, EvaluatorStop, ProcessError
from _R2D2.network import RecurrentDuelingQNetwork


def evaluator_process(config: R2D2Config, request_queue, result_queue, error_queue) -> None:
    try:
        result_queue.cancel_join_thread()
        torch.set_num_threads(config.evaluator_torch_threads)
        torch.set_num_interop_threads(1)
        model = RecurrentDuelingQNetwork(
            config.observation_shape, config.action_count, config.hidden_size
        ).cpu()
        model.eval()
        env_config = MsPacmanEnvConfig(
            num_envs=1,
            frame_skip=config.actor_env.frame_skip,
            frame_stack=config.actor_env.frame_stack,
            screen_size=config.actor_env.screen_size,
            repeat_action_probability=config.actor_env.repeat_action_probability,
            noop_max=config.actor_env.noop_max,
            mode=config.actor_env.mode,
            difficulty=config.actor_env.difficulty,
            step_cost=0.0,
            clip_training_reward=False,
            include_ram_metrics=False,
            multiprocessing_context=config.actor_env.multiprocessing_context,
        )
        env = make_env(env_config)
        try:
            while True:
                request = request_queue.get()
                if isinstance(request, EvaluatorStop):
                    break
                if not isinstance(request, EvaluationRequest):
                    raise TypeError(f"unexpected evaluator message: {type(request)}")
                checkpoint = torch.load(request.checkpoint_path, map_location="cpu", weights_only=False)
                model.load_state_dict(checkpoint["online_state_dict"])
                model.eval()
                lengths: list[int] = []
                returns: list[float] = []
                scores: list[float] = []
                capped = 0
                started = time.monotonic()
                for episode in range(config.evaluation_episodes):
                    observation, _ = env.reset(seed=config.evaluation_seed + episode)
                    hidden = model.initial_hidden(1, device="cpu")
                    previous_action = torch.zeros(1, config.action_count)
                    previous_action[:, 0] = 1.0
                    previous_reward = torch.zeros(1)
                    episode_return = 0.0
                    episode_length = 0
                    while True:
                        with torch.inference_mode():
                            q_values, hidden = model.step(
                                torch.from_numpy(observation[None]),
                                previous_action,
                                previous_reward,
                                hidden,
                            )
                        action = int(q_values.argmax(dim=1).item())
                        observation, reward, terminated, truncated, info = env.step(action)
                        if truncated:
                            raise RuntimeError("evaluator received a forbidden truncation")
                        raw_reward = float(info["raw_reward"])
                        if not np.isclose(float(reward), raw_reward):
                            raise RuntimeError("evaluator environment must return raw rewards")
                        episode_return += raw_reward
                        episode_length += 1
                        previous_action.zero_()
                        previous_action[0, action] = 1.0
                        previous_reward[0] = raw_reward
                        if terminated:
                            lengths.append(episode_length)
                            returns.append(episode_return)
                            scores.append(float(info["raw_score"]))
                            break
                        if episode_length >= config.evaluation_max_episode_steps:
                            lengths.append(episode_length)
                            returns.append(episode_return)
                            scores.append(float(info["raw_score"]))
                            capped += 1
                            break
                result_queue.put(
                    EvaluationResult(
                        request.checkpoint_transition,
                        request.checkpoint_path,
                        lengths,
                        returns,
                        scores,
                        capped,
                        time.monotonic() - started,
                    )
                )
        finally:
            env.close()
    except BaseException:
        error_queue.put(ProcessError("evaluator", traceback.format_exc()))
        raise

