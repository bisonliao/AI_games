"""CPU actors that collect recurrent overlapping sequences."""

from __future__ import annotations

from queue import Empty
import time
import traceback

import numpy as np
import torch

from PacManEnv import make_vector_env

from _R2D2.config import R2D2Config
from _R2D2.messages import ActorReport, ParameterUpdate, ProcessError, SequenceChunk
from _R2D2.network import RecurrentDuelingQNetwork
from _R2D2.sequence import SequenceAssembler
from _R2D2.utils import (
    actor_epsilon,
    actor_environment_seed,
    actor_policy_seed,
    drain_latest,
    load_state_dict_bytes,
    put_reliably,
    unique_observation_fraction,
)


def _select_actions(
    model: RecurrentDuelingQNetwork,
    observations: np.ndarray,
    previous_actions: torch.Tensor,
    previous_rewards: torch.Tensor,
    hidden: tuple[torch.Tensor, torch.Tensor],
    epsilon: float,
    rng: np.random.Generator,
    action_count: int,
) -> tuple[np.ndarray, tuple[torch.Tensor, torch.Tensor]]:
    with torch.inference_mode():
        q_values, new_hidden = model.step(
            torch.from_numpy(observations), previous_actions, previous_rewards, hidden
        )
    actions = q_values.argmax(dim=1).cpu().numpy().astype(np.int64)
    explore = rng.random(len(actions)) < epsilon
    actions[explore] = rng.integers(0, action_count, size=int(explore.sum()))
    return actions, new_hidden


def actor_process(
    actor_id: int,
    config: R2D2Config,
    rollout_queue,
    metrics_queue,
    parameter_queue,
    global_transition_counter,
    stop_event,
    error_queue,
) -> None:
    name = f"actor-{actor_id}"
    try:
        rollout_queue.cancel_join_thread()
        metrics_queue.cancel_join_thread()
        torch.set_num_threads(config.actor_torch_threads)
        torch.set_num_interop_threads(1)
        rng = np.random.default_rng(actor_policy_seed(config, actor_id))
        epsilon = actor_epsilon(actor_id, config)
        model = RecurrentDuelingQNetwork(
            config.observation_shape, config.action_count, config.hidden_size
        ).cpu()
        model.eval()
        try:
            update: ParameterUpdate = parameter_queue.get(
                timeout=config.initial_parameters_timeout_seconds
            )
        except Empty as exc:
            raise RuntimeError("timed out waiting for initial learner parameters") from exc
        model.load_state_dict(load_state_dict_bytes(update.state_dict_bytes))
        policy_version = update.version

        env = make_vector_env(config.actor_env)
        try:
            observations, _ = env.reset(seed=actor_environment_seed(config, actor_id))
            env_count = config.actor_env.num_envs
            previous_actions = torch.zeros(env_count, config.action_count)
            previous_actions[:, 0] = 1.0
            previous_rewards = torch.zeros(env_count)
            hidden = model.initial_hidden(env_count, device="cpu")
            assemblers = [
                SequenceAssembler(
                    config.action_count,
                    burn_in_steps=config.burn_in_steps,
                    learning_steps=config.learning_steps,
                    forward_steps=config.forward_steps,
                    gamma=config.gamma,
                )
                for _ in range(env_count)
            ]
            episode_lengths = np.zeros(env_count, dtype=np.int64)
            episode_returns = np.zeros(env_count, dtype=np.float64)
            transitions_since_update = 0
            pending_sequences: list = []
            while not stop_event.is_set():
                started = time.monotonic()
                chunk_episode_lengths: list[int] = []
                chunk_episode_returns: list[float] = []
                chunk_raw_scores: list[float] = []
                chunk_transitions = 0
                chunk_observations: list[np.ndarray] = []
                while chunk_transitions < max(1, config.actor_sequence_chunk_size * config.learning_steps):
                    if stop_event.is_set():
                        break
                    epsilon_now = epsilon
                    old_hidden = (
                        hidden[0].detach().clone(),
                        hidden[1].detach().clone(),
                    )
                    actions, new_hidden = _select_actions(
                        model,
                        observations,
                        previous_actions,
                        previous_rewards,
                        hidden,
                        epsilon_now,
                        rng,
                        config.action_count,
                    )
                    next_observations, rewards, terminated, truncated, infos = env.step(actions)
                    if np.any(truncated):
                        raise RuntimeError("actor received a forbidden truncation")
                    raw_rewards = np.asarray(infos["raw_reward"], dtype=np.float32)
                    if not np.allclose(rewards, raw_rewards):
                        raise RuntimeError("actor environment must return raw rewards")
                    # R2D2 caps training games at 30 minutes of emulator time,
                    # which is 30,000 frame-skip decisions here. A cap is an
                    # artificial episode boundary for replay bootstrapping.
                    effective_terminated = np.asarray(terminated, dtype=np.bool_).copy()
                    for index in range(env_count):
                        episode_step = int(episode_lengths[index]) + 1
                        if episode_step >= config.training_max_episode_steps:
                            effective_terminated[index] = True
                        before = (old_hidden[0][:, index], old_hidden[1][:, index])
                        emitted = assemblers[index].add(
                            observations[index],
                            previous_actions[index].numpy(),
                            float(previous_rewards[index]),
                            int(actions[index]),
                            float(raw_rewards[index]),
                            next_observations[index],
                            (before[0].numpy(), before[1].numpy()),
                            terminated=bool(effective_terminated[index]),
                        )
                        pending_sequences.extend(emitted)
                        chunk_observations.append(observations[index])
                        episode_lengths[index] += 1
                        episode_returns[index] += float(raw_rewards[index])
                        if effective_terminated[index]:
                            chunk_episode_lengths.append(int(episode_lengths[index]))
                            chunk_episode_returns.append(float(episode_returns[index]))
                            chunk_raw_scores.append(float(infos["raw_score"][index]))
                            episode_lengths[index] = 0
                            episode_returns[index] = 0.0
                            assemblers[index].reset()
                    chunk_transitions += env_count
                    observations = next_observations
                    previous_actions = torch.zeros_like(previous_actions)
                    previous_actions.scatter_(1, torch.from_numpy(actions)[:, None], 1.0)
                    previous_rewards = torch.from_numpy(raw_rewards.astype(np.float32))
                    hidden = new_hidden
                    if np.any(effective_terminated):
                        # Store terminal transition before resetting these rows.
                        observations, _ = env.reset(
                            options={"reset_mask": effective_terminated}
                        )
                        for index in np.flatnonzero(effective_terminated):
                            previous_actions[index].zero_()
                            previous_actions[index, 0] = 1.0
                            previous_rewards[index] = 0.0
                            with torch.inference_mode():
                                hidden[0][:, index].zero_()
                                hidden[1][:, index].zero_()
                    # The reference interval is measured in vector-environment
                    # decisions, not in the number of parallel rows.
                    transitions_since_update += 1
                    if transitions_since_update >= config.actor_parameter_update_interval:
                        latest = drain_latest(parameter_queue)
                        if latest is not None:
                            model.load_state_dict(load_state_dict_bytes(latest.state_dict_bytes))
                            policy_version = latest.version
                        transitions_since_update = 0
                    if len(pending_sequences) >= config.actor_sequence_chunk_size:
                        break
                if chunk_transitions == 0:
                    continue
                # Empty sequence chunks are intentional during the first
                # trajectory warm-up: they advance the global transition
                # counter while replay waits for its first complete window.
                message = SequenceChunk(
                    actor_id=actor_id,
                    sequences=pending_sequences,
                    transitions=chunk_transitions,
                    epsilon=epsilon,
                    policy_version=policy_version,
                )
                sent, queue_wait = put_reliably(
                    rollout_queue, message, stop_event, config.queue_timeout_seconds
                )
                if not sent:
                    break
                report = ActorReport(
                    actor_id=actor_id,
                    transitions=chunk_transitions,
                    collection_seconds=max(time.monotonic() - started, 1.0e-9),
                    queue_wait_seconds=queue_wait,
                    epsilon=epsilon,
                    policy_version=policy_version,
                    unique_observation_fraction=unique_observation_fraction(
                        np.stack(chunk_observations)
                    ) if chunk_observations else 0.0,
                    episode_lengths=chunk_episode_lengths,
                    episode_returns=chunk_episode_returns,
                    episode_raw_scores=chunk_raw_scores,
                )
                put_reliably(metrics_queue, report, stop_event, config.queue_timeout_seconds)
                pending_sequences = []
        finally:
            env.close()
    except BaseException:
        error_queue.put(ProcessError(name, traceback.format_exc()))
        stop_event.set()
        raise
