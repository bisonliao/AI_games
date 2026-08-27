"""Independent mixed-stage frozen-policy evaluator."""

from __future__ import annotations

import queue
import traceback
from typing import Any

import numpy as np
import torch

from .config import TrainConfig
from .envs import make_training_env
from .messages import StageEvaluationResult, WorkerFailure
from .model import DuelingDQN


def _balanced_checkpoint_rows(
    rows: list[tuple[int, str]], desired_count: int
) -> list[tuple[int, str]]:
    if not rows:
        return []
    count = max(desired_count, len(rows))
    ordered = sorted(rows)
    return [ordered[index % len(ordered)] for index in range(count)]


def build_eval_checkpoint_plan(
    config: TrainConfig,
    env: Any,
) -> list[tuple[int, str, int]]:
    """Build a checkpoint-balanced matrix that is fixed across evaluations."""
    if getattr(config, "after_curri", False):
        level1_ids = env.checkpoint_ids_for_level_start(1)
        level2_ids = env.checkpoint_ids_for_level_start(2)
        if not level1_ids or not level2_ids:
            raise ValueError(
                "after-curri evaluation requires Level 1 and Level 2 room-1 checkpoints"
            )
        level1_count = config.eval_episodes // 2
        level2_count = config.eval_episodes - level1_count
        rows = []
        rows.extend(
            (0, level1_ids[index % len(level1_ids)])
            for index in range(level1_count)
        )
        rows.extend(
            (0, level2_ids[index % len(level2_ids)])
            for index in range(level2_count)
        )
        np.random.default_rng(config.seed + 959_000).shuffle(rows)
        return [
            (stage, checkpoint_id, config.seed + 960_000 + index)
            for index, (stage, checkpoint_id) in enumerate(rows)
        ]
    if config.target_stage == 1:
        current_count = config.eval_episodes
        earlier_count = 0
    else:
        current_count = round(
            config.eval_episodes * config.eval_current_stage_fraction
        )
        earlier_count = config.eval_episodes - current_count

    current_rows = [
        (config.target_stage, identifier)
        for identifier in env.checkpoint_ids_for_stage(config.target_stage)
    ]
    earlier_rows = [
        (stage, identifier)
        for stage in range(1, config.target_stage)
        for identifier in env.checkpoint_ids_for_stage(stage)
    ]
    rows = _balanced_checkpoint_rows(current_rows, current_count)
    rows.extend(_balanced_checkpoint_rows(earlier_rows, earlier_count))
    if not rows:
        raise ValueError("evaluation checkpoint matrix is empty")
    return [
        (stage, identifier, config.seed + 960_000 + index)
        for index, (stage, identifier) in enumerate(rows)
    ]


def stage_evaluator_process(
    config: TrainConfig,
    action_count: int,
    shared_model: DuelingDQN,
    weight_lock: Any,
    job_queue: Any,
    result_queue: Any,
    stop_event: Any,
) -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    env = None
    try:
        env = make_training_env(config, config.target_stage)
        model = DuelingDQN(action_count, config.frame_stack).cpu().eval()
        while not stop_event.is_set():
            try:
                checkpoint_step = job_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if checkpoint_step is None:
                break
            with weight_lock:
                model.load_state_dict(shared_model.state_dict())

            reset_stages = []
            task_ids = []
            checkpoint_ids = []
            returns = []
            ale_score_returns = []
            lengths = []
            successes = []
            timeouts = []
            walls_destroyed = []
            creatures_killed = []
            miner_rescue_events = []
            unmapped_ale_rewards = []
            checkpoint_plan = build_eval_checkpoint_plan(config, env)
            for reset_stage, checkpoint_id, episode_seed in checkpoint_plan:
                policy_rng = np.random.default_rng(episode_seed + 10_000)
                if getattr(config, "after_curri", False):
                    observation, info = env.reset(
                        seed=episode_seed,
                        options={
                            "curriculum_stage": None,
                            "checkpoint_id": checkpoint_id,
                        },
                    )
                    actual_reset_stage = 0
                else:
                    env.set_curriculum_stage(reset_stage)
                    observation, info = env.reset(
                        seed=episode_seed,
                        options={
                            "curriculum_stage": reset_stage,
                            "checkpoint_id": checkpoint_id,
                        },
                    )
                    actual_reset_stage = int(info["hero_curriculum_stage"])
                episode_return = 0.0
                ale_score_return = 0.0
                episode_length = 0
                episode_walls_destroyed = 0
                episode_creatures_killed = 0
                episode_miner_rescue_events = 0
                episode_unmapped_ale_reward = 0.0
                while True:
                    if policy_rng.random() < config.eval_epsilon:
                        action = int(policy_rng.integers(action_count))
                    else:
                        tensor = torch.from_numpy(observation).unsqueeze(0)
                        with torch.inference_mode():
                            action = int(model(tensor).argmax(dim=1).item())
                    observation, reward, terminated, truncated, info = env.step(action)
                    episode_return += float(reward)
                    ale_score_return += float(info.get("hero_ale_reward", 0.0))
                    episode_walls_destroyed += int(
                        info.get("hero_walls_destroyed", 0)
                    )
                    episode_creatures_killed += int(
                        info.get("hero_creatures_killed", 0)
                    )
                    episode_miner_rescue_events += int(
                        info.get("hero_miner_rescued_events", 0)
                    )
                    episode_unmapped_ale_reward += float(
                        info.get("hero_unmapped_ale_reward", 0.0)
                    )
                    episode_length += 1
                    if terminated or truncated:
                        break
                reset_stages.append(actual_reset_stage)
                if config.after_curri:
                    task_ids.append(f"after_curri_level{info['hero_level']}")
                    checkpoint_ids.append(str(info["hero_checkpoint_id"]))
                else:
                    task_ids.append(str(info["hero_task_id"]))
                    checkpoint_ids.append(str(info["hero_checkpoint_id"]))
                returns.append(episode_return)
                ale_score_returns.append(ale_score_return)
                lengths.append(episode_length)
                successes.append(bool(info.get("is_success", False)))
                timeouts.append(bool(info.get("hero_time_limit_reached", False)))
                walls_destroyed.append(episode_walls_destroyed)
                creatures_killed.append(episode_creatures_killed)
                miner_rescue_events.append(episode_miner_rescue_events)
                unmapped_ale_rewards.append(episode_unmapped_ale_reward)

            result_queue.put(
                StageEvaluationResult(
                    checkpoint_step=int(checkpoint_step),
                    reset_stages=tuple(reset_stages),
                    task_ids=tuple(task_ids),
                    checkpoint_ids=tuple(checkpoint_ids),
                    episode_returns=tuple(returns),
                    ale_score_returns=tuple(ale_score_returns),
                    episode_lengths=tuple(lengths),
                    successes=tuple(successes),
                    timeouts=tuple(timeouts),
                    walls_destroyed=tuple(walls_destroyed),
                    creatures_killed=tuple(creatures_killed),
                    miner_rescue_events=tuple(miner_rescue_events),
                    unmapped_ale_rewards=tuple(unmapped_ale_rewards),
                )
            )
    except Exception:
        result_queue.put(
            WorkerFailure(worker="stage-evaluator", traceback=traceback.format_exc())
        )
        stop_event.set()
    finally:
        if env is not None:
            env.close()
