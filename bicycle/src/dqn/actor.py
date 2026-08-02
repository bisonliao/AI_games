"""CPU actor process: parallel environment sampling and metric aggregation."""

from __future__ import annotations

from dataclasses import dataclass
import os
import queue
import time
from typing import Any

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, AutoresetMode
import numpy as np
import torch

from env import ENV_ID  # Registers the environment in every spawned process.

from .config import DQNConfig
from .network import DuelingQNetwork, load_numpy_state_dict
from .nstep import NStepAccumulator, NStepTransition, StepTransition


@dataclass(slots=True)
class ExperienceChunk:
    """Serialized n-step batch and collection metadata sent to the learner."""

    actor_id: int
    policy_version: int
    transitions: list[NStepTransition]
    environment_steps: int
    collection_seconds: float


@dataclass(slots=True)
class EpisodeMetric:
    """One completed episode's business, physics, wind, and action metrics."""

    actor_id: int
    policy_version: int
    outcome: str
    episode_return: float
    episode_length: int
    progress_m: float
    roll_rms: float
    max_abs_roll: float
    lateral_drift_m: float
    peak_wind_n: float
    gust_count: int
    saturation_fraction: float
    mean_abs_speed_error_mps: float
    positive_wind_fraction: float
    negative_wind_fraction: float
    action_counts: tuple[int, int, int]


def actor_process(
    actor_id: int,
    actor_count: int,
    env_count: int,
    epsilon: float,
    config: DQNConfig,
    base_seed: int,
    experience_queue: Any,
    metric_queue: Any,
    weight_queue: Any,
    stop_event: Any,
) -> None:
    """Collect n-step experience from one actor's vector of environments.

    Each actor owns a CPU copy of the Q-network and an AsyncVectorEnv whose
    PyBullet environments run in separate subprocesses. The actor periodically
    consumes the newest learner weights, selects epsilon-greedy actions in one
    batched inference call, converts one-step transitions to n-step records, and
    sends bounded chunks to the learner. Episode summaries travel through a
    separate best-effort metrics queue so logging can never stall experience.

    Gymnasium SAME_STEP autoreset returns the next episode's initial observation
    in the main observation array. For terminal transitions we explicitly use
    ``final_obs`` instead, which is required for correct timeout bootstrapping.
    """
    # Actor processes are deliberately CPU-only; the learner is the sole CUDA
    # owner. Restricting BLAS threads also prevents N actors from oversubscribing
    # the host with inference worker threads.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    rng = np.random.default_rng(base_seed + actor_id * 100_003)
    network = DuelingQNetwork(
        config.observation_dim, config.action_dim, config.hidden_dim
    ).cpu()
    # Block until the learner publishes a fully initialized policy. Subsequent
    # refreshes are non-blocking and discard stale intermediate versions.
    version, state = weight_queue.get()
    load_numpy_state_dict(network, state)
    network.eval()
    env_fns = [make_registered_env for _ in range(env_count)]
    vector_env = AsyncVectorEnv(
        env_fns,
        context="spawn",
        shared_memory=True,
        daemon=True,
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    seeds = [base_seed + actor_id * 10_000 + i for i in range(env_count)]
    observations, _ = vector_env.reset(seed=seeds)
    # N-step buffers and episode statistics are independent per vector slot.
    accumulators = [NStepAccumulator(config.n_step, config.gamma) for _ in range(env_count)]
    returns = np.zeros(env_count, dtype=np.float64)
    lengths = np.zeros(env_count, dtype=np.int64)
    roll_squares = np.zeros(env_count, dtype=np.float64)
    max_rolls = np.zeros(env_count, dtype=np.float64)
    peak_winds = np.zeros(env_count, dtype=np.float64)
    saturation_steps = np.zeros(env_count, dtype=np.int64)
    speed_error_sums = np.zeros(env_count, dtype=np.float64)
    positive_wind_steps = np.zeros(env_count, dtype=np.int64)
    negative_wind_steps = np.zeros(env_count, dtype=np.int64)
    action_counts = np.zeros((env_count, config.action_dim), dtype=np.int64)
    pending: list[NStepTransition] = []
    chunk_environment_steps = 0
    chunk_started = time.perf_counter()
    try:
        while not stop_event.is_set():
            version = _refresh_weights(weight_queue, network, version)
            actions = _epsilon_greedy(network, observations, epsilon, rng)
            next_observations, rewards, terminated, truncated, infos = vector_env.step(actions)
            chunk_environment_steps += env_count
            done = np.logical_or(terminated, truncated)
            for env_index in range(env_count):
                # SAME_STEP autoreset replaces next_observations on done slots;
                # retain the real final state for replay and timeout bootstrap.
                terminal_observation = next_observations[env_index]
                if done[env_index] and infos.get("_final_obs", np.zeros(env_count, bool))[env_index]:
                    terminal_observation = np.asarray(infos["final_obs"][env_index], dtype=np.float32)
                pending.extend(
                    accumulators[env_index].add(
                        StepTransition(
                            observation=observations[env_index],
                            action=int(actions[env_index]),
                            reward=float(rewards[env_index]),
                            next_observation=terminal_observation,
                            terminated=bool(terminated[env_index]),
                            truncated=bool(truncated[env_index]),
                        )
                    )
                )
                returns[env_index] += rewards[env_index]
                lengths[env_index] += 1
                action_counts[env_index, actions[env_index]] += 1
                info = _vector_info_at(infos, env_index, final=bool(done[env_index]))
                roll = abs(float(info.get("roll_rad", 0.0)))
                roll_squares[env_index] += roll * roll
                max_rolls[env_index] = max(max_rolls[env_index], roll)
                peak_winds[env_index] = max(
                    peak_winds[env_index], abs(float(info.get("wind_peak_force_n", 0.0)))
                )
                saturation_steps[env_index] += int(
                    bool(info.get("reaction_wheel_saturated", False))
                )
                speed_error_sums[env_index] += abs(
                    float(info.get("forward_speed_mps", 0.0)) - 2.0
                )
                wind_force = float(info.get("wind_force_y_n", 0.0))
                positive_wind_steps[env_index] += int(wind_force > 0)
                negative_wind_steps[env_index] += int(wind_force < 0)
                if done[env_index]:
                    metric = EpisodeMetric(
                        actor_id=actor_id,
                        policy_version=version,
                        outcome=str(info.get("outcome", "unknown")),
                        episode_return=float(returns[env_index]),
                        episode_length=int(lengths[env_index]),
                        progress_m=float(info.get("progress_m", 0.0)),
                        roll_rms=float(
                            np.sqrt(roll_squares[env_index] / max(1, lengths[env_index]))
                        ),
                        max_abs_roll=float(max_rolls[env_index]),
                        lateral_drift_m=float(info.get("lateral_drift_m", 0.0)),
                        peak_wind_n=float(peak_winds[env_index]),
                        gust_count=int(info.get("gust_count", 0)),
                        saturation_fraction=float(
                            saturation_steps[env_index] / max(1, lengths[env_index])
                        ),
                        mean_abs_speed_error_mps=float(
                            speed_error_sums[env_index] / max(1, lengths[env_index])
                        ),
                        positive_wind_fraction=float(
                            positive_wind_steps[env_index] / max(1, lengths[env_index])
                        ),
                        negative_wind_fraction=float(
                            negative_wind_steps[env_index] / max(1, lengths[env_index])
                        ),
                        action_counts=tuple(int(x) for x in action_counts[env_index]),
                    )
                    _put_metric(metric_queue, metric)
                    returns[env_index] = 0
                    lengths[env_index] = 0
                    roll_squares[env_index] = 0
                    max_rolls[env_index] = 0
                    peak_winds[env_index] = 0
                    saturation_steps[env_index] = 0
                    speed_error_sums[env_index] = 0
                    positive_wind_steps[env_index] = 0
                    negative_wind_steps[env_index] = 0
                    action_counts[env_index] = 0
            observations = next_observations
            # Chunking amortizes multiprocessing serialization and provides
            # natural backpressure through the bounded experience queue.
            if len(pending) >= config.actor_chunk_size:
                chunk = ExperienceChunk(
                    actor_id=actor_id,
                    policy_version=version,
                    transitions=pending,
                    environment_steps=chunk_environment_steps,
                    collection_seconds=time.perf_counter() - chunk_started,
                )
                experience_queue.put(chunk)
                pending = []
                chunk_environment_steps = 0
                chunk_started = time.perf_counter()
    finally:
        vector_env.close(terminate=True)


def make_registered_env() -> gym.Env:
    """Pickle-friendly factory used by spawned AsyncVectorEnv workers."""
    return gym.make(ENV_ID)


def _epsilon_greedy(
    network: DuelingQNetwork,
    observations: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select a batched epsilon-greedy action on CPU."""
    with torch.no_grad():
        greedy = network(torch.as_tensor(observations, dtype=torch.float32)).argmax(1).numpy()
    explore = rng.random(len(observations)) < epsilon
    random_actions = rng.integers(0, network.advantage[-1].out_features, len(observations))
    return np.where(explore, random_actions, greedy).astype(np.int64)


def _refresh_weights(weight_queue: Any, network: DuelingQNetwork, version: int) -> int:
    """Drain queued policies and install only the newest available version."""
    latest: tuple[int, dict[str, np.ndarray]] | None = None
    while True:
        try:
            latest = weight_queue.get_nowait()
        except queue.Empty:
            break
    if latest is not None:
        version, state = latest
        load_numpy_state_dict(network, state)
    return version


def _vector_info_at(infos: dict[str, Any], index: int, final: bool) -> dict[str, Any]:
    """Extract one environment's normal or final info from vectorized arrays."""
    source = infos.get("final_info") if final else infos
    if not isinstance(source, dict):
        source = infos
    result: dict[str, Any] = {}
    for key, values in source.items():
        if key.startswith("_"):
            continue
        try:
            result[key] = values[index]
        except (IndexError, KeyError, TypeError):
            continue
    return result


def _put_metric(metric_queue: Any, metric: EpisodeMetric) -> None:
    try:
        metric_queue.put_nowait(metric)
    except queue.Full:
        pass
