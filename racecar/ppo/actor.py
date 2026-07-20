"""CPU rollout workers for the synchronous actor--learner protocol."""

from __future__ import annotations

import traceback
from time import perf_counter

import numpy as np
import torch

from env.actor_env import DirectRaceCarVectorEnv
from ppo.model import ActorCritic, actions_to_steering, shape_rewards


def actor_process(actor_id: int, config: dict, command_queue, rollout_queue) -> None:
    """Collect one on-policy rollout, then wait for the next learner command."""
    torch.set_num_threads(1)
    # Actor streams are distinct, and a resumed run does not replay the same
    # action/environment RNG stream that was used from step zero.
    seed = (int(config["seed"]) + int(config.get("resume_global_steps", 0))
            + actor_id * 1_000_003)
    np.random.seed(seed)
    torch.manual_seed(seed)
    environments = None

    try:
        num_envs = int(config["envs_per_actor"])
        environments = DirectRaceCarVectorEnv(
            num_envs, fps=int(config["fps"]),
            max_steps=int(config["max_episode_steps"]),
            start_position_noise=float(config["start_position_noise"]),
            start_heading_noise=float(config["start_heading_noise"]),
        )
        model = ActorCritic().cpu().eval()
        observations, _ = environments.reset(
            [seed + index for index in range(num_envs)]
        )
        episode_returns = np.zeros(num_envs, dtype=np.float64)
        episode_lengths = np.zeros(num_envs, dtype=np.int64)

        while True:
            command = command_queue.get()
            if command["type"] == "stop":
                break
            if command["type"] != "weights":
                raise ValueError(f"unknown learner command: {command['type']}")

            version = int(command["version"])
            model.load_state_dict({name: torch.from_numpy(value)
                                   for name, value in command["state_dict"].items()})
            rollout_started = perf_counter()
            rollout = _collect_rollout(
                model=model,
                environments=environments,
                observations=observations,
                episode_returns=episode_returns,
                episode_lengths=episode_lengths,
                rollout_steps=int(config["rollout_steps"]),
            )
            rollout["rollout_collect_seconds"] = perf_counter() - rollout_started
            observations = rollout.pop("next_observations")
            rollout.update(type="rollout", actor_id=actor_id, version=version)
            rollout_queue.put(rollout)
    except BaseException:
        rollout_queue.put({
            "type": "error",
            "actor_id": actor_id,
            "traceback": traceback.format_exc(),
        })
    finally:
        if environments is not None:
            environments.close()


def _collect_rollout(model: ActorCritic, environments: DirectRaceCarVectorEnv,
                     observations: np.ndarray, episode_returns: np.ndarray,
                     episode_lengths: np.ndarray, rollout_steps: int) -> dict:
    num_envs, observation_size = observations.shape
    obs_buffer = np.empty((rollout_steps, num_envs, observation_size), np.float32)
    action_buffer = np.empty((rollout_steps, num_envs), np.int64)
    logprob_buffer = np.empty((rollout_steps, num_envs), np.float32)
    reward_buffer = np.empty((rollout_steps, num_envs), np.float32)
    value_buffer = np.empty((rollout_steps, num_envs), np.float32)
    terminated_buffer = np.empty((rollout_steps, num_envs), bool)
    truncated_buffer = np.empty((rollout_steps, num_envs), bool)
    timeout_value_buffer = np.zeros((rollout_steps, num_envs), np.float32)
    completed_episodes = []
    action_counts = np.zeros(3, dtype=np.int64)

    for step in range(rollout_steps):
        obs_buffer[step] = observations
        with torch.no_grad():
            obs_tensor = torch.from_numpy(observations)
            actions, logprobs, _, values = model.get_action_and_value(obs_tensor)
        action_numpy = actions.numpy()
        action_buffer[step] = action_numpy
        logprob_buffer[step] = logprobs.numpy()
        value_buffer[step] = values.numpy()
        action_counts += np.bincount(action_numpy, minlength=3)

        raw_next_obs, _, terminated, truncated, infos = environments.step(
            actions_to_steering(action_numpy)
        )
        rewards = shape_rewards(observations, raw_next_obs, terminated, truncated, infos)
        reward_buffer[step] = rewards
        terminated_buffer[step] = terminated
        truncated_buffer[step] = truncated

        # A time-limit is not an MDP terminal: save V(s') before reset so the
        # learner can bootstrap its delta without propagating GAE into a new episode.
        timeout_indices = np.flatnonzero(truncated & ~terminated)
        if len(timeout_indices):
            with torch.no_grad():
                timeout_values = model.get_value(
                    torch.from_numpy(raw_next_obs[timeout_indices])
                ).numpy()
            timeout_value_buffer[step, timeout_indices] = timeout_values

        episode_returns += rewards
        episode_lengths += 1
        done = terminated | truncated
        for index in np.flatnonzero(done):
            info = infos[index]
            terminal_position = np.asarray(info["position"], dtype=np.float32)
            terminal_distance = float(np.linalg.norm(
                terminal_position - np.asarray((14.5, 7.5), dtype=np.float32)
            ))
            completed_episodes.append({
                "return": float(episode_returns[index]),
                "length": int(episode_lengths[index]),
                "success": bool(info.get("is_success", False)),
                "reason": info.get("termination_reason", "unknown"),
                "terminal_distance": terminal_distance,
            })
            episode_returns[index] = 0.0
            episode_lengths[index] = 0

        observations = raw_next_obs
        reset_results = environments.reset_at(np.flatnonzero(done))
        for index, (reset_observation, _) in reset_results.items():
            observations[index] = reset_observation

    with torch.no_grad():
        last_values = model.get_value(torch.from_numpy(observations)).numpy()

    return {
        "observations": obs_buffer,
        "actions": action_buffer,
        "logprobs": logprob_buffer,
        "rewards": reward_buffer,
        "values": value_buffer,
        "terminated": terminated_buffer,
        "truncated": truncated_buffer,
        "timeout_values": timeout_value_buffer,
        "last_values": last_values.astype(np.float32),
        "next_observations": observations,
        "episodes": completed_episodes,
        "action_counts": action_counts,
    }
