"""CPU actor for one fixed curriculum-stage run."""

from __future__ import annotations

import queue
import traceback
from typing import Any

import numpy as np
import torch

from .config import TrainConfig
from .envs import make_training_env, sample_reset_stage
from .messages import (
    EpisodeSummary,
    WorkerFailure,
    pack_transition,
)
from .model import DuelingDQN


def epsilon_at(config: TrainConfig, global_transition: int) -> float:
    fraction = min(1.0, max(0, global_transition) / config.epsilon_decay_transitions)
    return config.epsilon_start + fraction * (
        config.epsilon_end - config.epsilon_start
    )


def compute_training_reward(
    config: TrainConfig,
    environment_reward: float,
    *,
    time_limit_reached: bool = False,
    life_lost: bool = False,
    walls_destroyed: int = 0,
    creatures_killed: int = 0,
    miner_rescued_events: int = 0,
) -> float:
    """Compute the documented event reward for one DQN decision."""
    if life_lost:
        return config.life_lost_terminal_reward - config.decision_step_penalty
    if time_limit_reached:
        return config.timeout_terminal_reward - config.decision_step_penalty
    return (
        config.wall_event_reward * walls_destroyed
        + config.creature_event_reward * creatures_killed
        + config.miner_event_reward * miner_rescued_events
        - config.decision_step_penalty
    )


def _put_lossless(output_queue: Any, item: Any, stop_event: Any) -> bool:
    while not stop_event.is_set():
        try:
            output_queue.put(item, block=True, timeout=0.5)
            return True
        except queue.Full:
            continue
    return False


def _synchronize_weights(
    local_model: DuelingDQN,
    shared_model: DuelingDQN,
    weight_lock: Any,
    weight_version: Any,
) -> int:
    with weight_lock:
        local_model.load_state_dict(shared_model.state_dict())
        return int(weight_version.value)


def actor_process(
    actor_id: int,
    config: TrainConfig,
    shared_model: DuelingDQN,
    weight_lock: Any,
    weight_version: Any,
    global_transition_count: Any,
    transition_queue: Any,
    metrics_queue: Any,
    stop_event: Any,
) -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    rng = np.random.default_rng(config.seed + 10_000 + actor_id)
    env = None
    try:
        env = make_training_env(config, config.target_stage)
        action_count = int(env.action_space.n)
        model = DuelingDQN(action_count, config.frame_stack).cpu().eval()
        local_version = _synchronize_weights(
            model, shared_model, weight_lock, weight_version
        )
        local_steps_since_sync = 0
        episode_index = 0
        collection_finished = False

        while not stop_event.is_set() and not collection_finished:
            if config.after_curri:
                reset_stage = 0
                start_level = 1 if rng.random() < 0.5 else 2
                start_ids = env.checkpoint_ids_for_level_start(start_level)
                if not start_ids:
                    raise RuntimeError(
                        f"after-curri requires a Level {start_level} room-1 checkpoint"
                    )
                checkpoint_id = start_ids[int(rng.integers(len(start_ids)))]
            else:
                reset_stage = sample_reset_stage(
                    rng,
                    config.target_stage,
                    config.train_current_stage_fraction,
                )
                env.set_curriculum_stage(reset_stage)
            reset_options = None
            if config.after_curri:
                reset_options = {
                    "curriculum_stage": None,
                    "checkpoint_id": checkpoint_id,
                }
            observation, info = env.reset(
                seed=config.seed + actor_id * 1_000_003 + episode_index,
                options=reset_options,
            )
            if config.after_curri:
                actual_reset_stage = 0
                task_id = f"after_curri_level{start_level}"
            else:
                actual_reset_stage = int(info["hero_curriculum_stage"])
                task_id = str(info["hero_task_id"])
                checkpoint_id = str(info["hero_checkpoint_id"])
            episode_return = 0.0
            ale_score_return = 0.0
            walls_destroyed_total = 0
            creatures_killed_total = 0
            miner_rescue_events_total = 0
            dynamite_bonus_sticks_total = 0
            unmapped_ale_reward_total = 0.0
            episode_length = 0
            visited_levels = {int(info.get("hero_level", 1))}
            completed_levels: set[int] = set()
            previous_level = int(info.get("hero_level", 1))
            last_epsilon = config.epsilon_start
            while not stop_event.is_set():
                if (
                    local_steps_since_sync >= config.actor_weight_sync_interval
                    and int(weight_version.value) != local_version
                ):
                    local_version = _synchronize_weights(
                        model, shared_model, weight_lock, weight_version
                    )
                    local_steps_since_sync = 0

                with global_transition_count.get_lock():
                    if global_transition_count.value >= config.total_transitions:
                        collection_finished = True
                        break
                    global_step = int(global_transition_count.value)
                    global_transition_count.value += 1
                last_epsilon = epsilon_at(config, global_step)
                if rng.random() < last_epsilon:
                    action = int(rng.integers(action_count))
                else:
                    tensor = torch.from_numpy(observation).unsqueeze(0)
                    with torch.inference_mode():
                        action = int(model(tensor).argmax(dim=1).item())

                next_observation, reward, terminated, truncated, next_info = env.step(
                    action
                )
                # DQNAtariWrapper is the sole authority for event reward.
                # Reconstructing it here used to pay the miner reward again
                # when Level transition followed an earlier +1000 event.
                training_reward = float(next_info["hero_rl_reward"])
                if not np.isclose(training_reward, float(reward)):
                    raise RuntimeError(
                        "environment reward differs from hero_rl_reward: "
                        f"{reward} != {training_reward}"
                    )
                transition = pack_transition(
                    observation,
                    next_observation,
                    action=action,
                    reward=training_reward,
                    terminated=bool(terminated),
                    stage=actual_reset_stage,
                    actor_id=actor_id,
                )
                if not _put_lossless(transition_queue, transition, stop_event):
                    break

                episode_return += float(training_reward)
                ale_score_return += float(next_info.get("hero_ale_reward", reward))
                walls_destroyed_total += int(
                    next_info.get("hero_walls_destroyed", 0)
                )
                creatures_killed_total += int(
                    next_info.get("hero_creatures_killed", 0)
                )
                miner_rescue_events_total += int(
                    next_info.get("hero_miner_rescued_events", 0)
                )
                dynamite_bonus_sticks_total += int(
                    next_info.get("hero_dynamite_bonus_sticks", 0)
                )
                unmapped_ale_reward_total += float(
                    next_info.get("hero_unmapped_ale_reward", 0.0)
                )
                episode_length += 1
                local_steps_since_sync += 1
                observation = next_observation
                info = next_info
                level = int(info.get("hero_level", previous_level))
                visited_levels.add(min(level, 4))
                if level > previous_level:
                    completed_levels.update(range(previous_level, min(level, 5)))
                if bool(info.get("hero_level_cap_reached", False)):
                    completed_levels.add(4)
                previous_level = level

                if terminated or truncated:
                    success = bool(info.get("is_success", False))
                    timed_out = bool(info.get("hero_time_limit_reached", False))
                    summary = EpisodeSummary(
                        actor_id=actor_id,
                        reset_stage=actual_reset_stage,
                        task_id=task_id,
                        checkpoint_id=checkpoint_id,
                        episode_return=episode_return,
                        ale_score_return=ale_score_return,
                        episode_length=episode_length,
                        success=success,
                        timeout=timed_out,
                        walls_destroyed=walls_destroyed_total,
                        creatures_killed=creatures_killed_total,
                        miner_rescue_events=miner_rescue_events_total,
                        dynamite_bonus_sticks=dynamite_bonus_sticks_total,
                        unmapped_ale_reward=unmapped_ale_reward_total,
                        visited_levels=tuple(sorted(visited_levels)),
                        completed_levels=tuple(sorted(completed_levels)),
                        epsilon=last_epsilon,
                    )
                    if not _put_lossless(metrics_queue, summary, stop_event):
                        break
                    episode_index += 1
                    break
    except Exception:
        metrics_queue.put(
            WorkerFailure(worker=f"actor-{actor_id}", traceback=traceback.format_exc())
        )
        stop_event.set()
    finally:
        if env is not None:
            env.close()
