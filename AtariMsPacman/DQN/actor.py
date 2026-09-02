"""CPU rollout Actor for the Ape-X-style training topology."""

from __future__ import annotations

from queue import Empty
import time
import traceback

import numpy as np
import torch

from DQN.config import DQNConfig
from DQN.messages import ActorReport, ParameterUpdate, ProcessError, TransitionChunk
from DQN.network import DuelingQNetwork
from DQN.rewards import shape_rewards
from DQN.utils import (
    actor_environment_seed,
    actor_policy_seed,
    drain_latest,
    linear_epsilon,
    load_state_dict_bytes,
    put_reliably,
    read_counter,
    unique_observation_fraction,
)
from PacManEnv import make_vector_env


def select_epsilon_greedy_actions(
    model: DuelingQNetwork,
    observations: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    action_count: int,
) -> np.ndarray:
    with torch.inference_mode():
        q_values = model(torch.from_numpy(observations))
        actions = q_values.argmax(dim=1).cpu().numpy().astype(np.int64, copy=False)
    explore = rng.random(observations.shape[0]) < epsilon
    random_actions = rng.integers(
        0, action_count, size=observations.shape[0], dtype=np.int64
    )
    actions[explore] = random_actions[explore]
    return actions


def actor_process(
    actor_id: int,
    config: DQNConfig,
    rollout_queue,
    metrics_queue,
    parameter_queue,
    global_transition_counter,
    stop_event,
    error_queue,
) -> None:
    process_name = f"actor-{actor_id}"
    try:
        # The learner can stop consuming before an Actor's final feeder buffer is
        # empty. Do not let shutdown wait forever for already-obsolete tail data.
        rollout_queue.cancel_join_thread()
        metrics_queue.cancel_join_thread()
        torch.set_num_threads(config.actor_torch_threads)
        torch.set_num_interop_threads(1)
        torch.manual_seed(config.seed)
        rng = np.random.default_rng(actor_policy_seed(config, actor_id))

        model = DuelingQNetwork(config.observation_shape, config.action_count).cpu()
        model.eval()
        try:
            initial_update: ParameterUpdate = parameter_queue.get(
                timeout=config.initial_parameters_timeout_seconds
            )
        except Empty as exc:
            raise RuntimeError("Timed out waiting for initial learner parameters") from exc
        model.load_state_dict(load_state_dict_bytes(initial_update.state_dict_bytes))
        policy_version = initial_update.version

        env = make_vector_env(config.actor_env)
        try:
            observations, _ = env.reset(seed=actor_environment_seed(config, actor_id))
            env_count = config.actor_env.num_envs
            vector_steps_per_chunk = config.actor_transition_batch_size // env_count
            episode_lengths = np.zeros(env_count, dtype=np.int64)
            episode_returns = np.zeros(env_count, dtype=np.float64)

            while not stop_event.is_set():
                observation_steps: list[np.ndarray] = []
                action_steps: list[np.ndarray] = []
                reward_steps: list[np.ndarray] = []
                next_observation_steps: list[np.ndarray] = []
                terminated_steps: list[np.ndarray] = []
                completed_lengths: list[int] = []
                completed_returns: list[float] = []
                completed_raw_scores: list[float] = []
                chunk_started = time.monotonic()
                epsilon = linear_epsilon(
                    read_counter(global_transition_counter), config
                )

                for _ in range(vector_steps_per_chunk):
                    if stop_event.is_set():
                        break
                    global_transitions = read_counter(global_transition_counter)
                    epsilon = linear_epsilon(global_transitions, config)
                    actions = select_epsilon_greedy_actions(
                        model, observations, epsilon, rng, config.action_count
                    )
                    (
                        next_observations,
                        rewards,
                        terminated,
                        truncated,
                        infos,
                    ) = env.step(actions)
                    if np.any(truncated):
                        raise RuntimeError("Actor received a forbidden truncation")
                    raw_rewards = np.asarray(infos["raw_reward"], dtype=np.float32)
                    if not np.allclose(rewards, raw_rewards):
                        raise RuntimeError(
                            "Actor environment must return unshaped raw rewards"
                        )
                    shaped_rewards = shape_rewards(
                        raw_rewards,
                        np.asarray(infos["life_lost"], dtype=np.bool_),
                        np.asarray(infos["game_over"], dtype=np.bool_),
                        config,
                    )

                    observation_steps.append(observations)
                    action_steps.append(actions)
                    reward_steps.append(shaped_rewards)
                    next_observation_steps.append(next_observations)
                    terminated_steps.append(terminated)

                    episode_lengths += 1
                    episode_returns += shaped_rewards
                    for env_index in np.flatnonzero(terminated):
                        completed_lengths.append(int(episode_lengths[env_index]))
                        completed_returns.append(float(episode_returns[env_index]))
                        completed_raw_scores.append(
                            float(infos["raw_score"][env_index])
                        )
                        episode_lengths[env_index] = 0
                        episode_returns[env_index] = 0.0

                    observations = next_observations
                    if np.any(terminated):
                        observations, _ = env.reset(
                            options={
                                "reset_mask": terminated.astype(np.bool_, copy=False)
                            }
                        )

                if len(observation_steps) != vector_steps_per_chunk:
                    break

                chunk_observations = np.concatenate(observation_steps, axis=0)
                chunk = TransitionChunk(
                    actor_id=actor_id,
                    observations=chunk_observations,
                    actions=np.concatenate(action_steps, axis=0),
                    rewards=np.concatenate(reward_steps, axis=0),
                    next_observations=np.concatenate(next_observation_steps, axis=0),
                    terminated=np.concatenate(terminated_steps, axis=0),
                    epsilon=epsilon,
                    policy_version=policy_version,
                )
                collection_seconds = time.monotonic() - chunk_started
                sent, queue_wait_seconds = put_reliably(
                    rollout_queue,
                    chunk,
                    stop_event,
                    config.queue_retry_timeout_seconds,
                )
                if not sent:
                    break

                latest_update = drain_latest(parameter_queue)
                if latest_update is not None:
                    model.load_state_dict(
                        load_state_dict_bytes(latest_update.state_dict_bytes)
                    )
                    policy_version = latest_update.version

                report = ActorReport(
                    actor_id=actor_id,
                    transitions=len(chunk),
                    collection_seconds=collection_seconds,
                    queue_wait_seconds=queue_wait_seconds,
                    epsilon=epsilon,
                    policy_version=policy_version,
                    unique_observation_fraction=unique_observation_fraction(
                        chunk_observations
                    ),
                    episode_lengths=completed_lengths,
                    episode_returns=completed_returns,
                    episode_raw_scores=completed_raw_scores,
                )
                sent, _ = put_reliably(
                    metrics_queue,
                    report,
                    stop_event,
                    config.queue_retry_timeout_seconds,
                )
                if not sent:
                    break
        finally:
            env.close()
    except BaseException:
        error_queue.put(
            ProcessError(process_name=process_name, traceback=traceback.format_exc())
        )
        stop_event.set()
        raise
