"""CPU actor process for vectorized Tetris interaction."""
from __future__ import annotations

import queue
import time
from typing import Any

import numpy as np
import torch

from TetrisEnv.vector_runner import make_sync_vector_env

from .model import DuelingDQN, masked_q_values, observations_to_torch
from .replay import TransitionBatch, concatenate_transition_batches
from .schedule import epsilon_for_schedule


def _get_with_stop(item_queue, *, poll_timeout: float, stop_event):
    """Wait for a round command while still responding to global shutdown."""
    if poll_timeout <= 0:
        raise ValueError("poll_timeout must be positive")
    while not stop_event.is_set():
        try:
            return item_queue.get(timeout=poll_timeout)
        except queue.Empty:
            pass
    return None


def _latest_weight_message(weight_queue):
    """Return the newest queued weight message without waiting."""
    latest = None
    while True:
        try:
            candidate = weight_queue.get_nowait()
        except queue.Empty:
            break
        if latest is None or int(candidate[0]) >= int(latest[0]):
            latest = candidate
    return latest


def _put_latest_weight(
    weight_queue,
    item,
    *,
    poll_timeout: float,
    stop_event=None,
) -> bool:
    """Reliably replace a one-slot mailbox so the newest snapshot wins."""
    if poll_timeout <= 0:
        raise ValueError("poll_timeout must be positive")
    while stop_event is None or not stop_event.is_set():
        try:
            weight_queue.put_nowait(item)
            return True
        except queue.Full:
            try:
                weight_queue.get(timeout=poll_timeout)
            except queue.Empty:
                pass
    return False


def _put_with_wait(item_queue, item, *, poll_timeout: float, stop_event) -> tuple[float, int, bool]:
    """把 transition 放入有界队列，并在队列满时持续等待。

    ``poll_timeout`` 只是一次 ``put`` 尝试的轮询间隔，不是“放弃该 transition”
    的截止时间。队列满导致的 ``queue.Full`` 会被计数后重试，所以正常训练期间
    不会因为 timeout 丢样本；只有 learner 发出全局停止信号时才提前返回。
    """
    if poll_timeout <= 0:
        raise ValueError("poll_timeout must be positive")
    started = time.perf_counter()
    timeouts = 0
    while not stop_event.is_set():
        try:
            item_queue.put(item, timeout=poll_timeout)
            return time.perf_counter() - started, timeouts, True
        except queue.Full:
            timeouts += 1
    return time.perf_counter() - started, timeouts, False


def _put_metric(metric_queue, metric: dict[str, Any]) -> None:
    """尽力发送低优先级指标，绝不让指标通道阻塞 transition 采样。

    metrics 丢失只会降低监控细节，不影响 replay 数据；transition 则走另一条
    必须可靠入队的路径。
    """
    try:
        metric_queue.put_nowait(metric)
    except queue.Full:
        pass


def _vector_info_value(
    infos: dict[str, Any],
    key: str,
    index: int,
    *,
    terminated: bool,
    default: Any,
) -> Any:
    """Read one sub-environment's value, including SameStep final_info."""
    source: dict[str, Any] = infos
    if terminated:
        final_info = infos.get("final_info")
        if isinstance(final_info, dict):
            source = final_info
    if key not in source:
        return default
    present = source.get(f"_{key}")
    if present is not None:
        try:
            if not bool(np.asarray(present)[index]):
                return default
        except (IndexError, TypeError):
            return default
    try:
        return np.asarray(source[key])[index]
    except (IndexError, TypeError):
        return default


def actor_process(
    actor_id: int,
    envs_per_actor: int,
    seed: int,
    epsilon: float,
    transition_queue,
    metric_queue,
    command_queue,
    weight_queue,
    decay_progress,
    stop_event,
    transition_put_poll_timeout: float = 1.0,
    stats_every: int = 1_000,
    transition_batch_size: int = 256,
    gamma: float = 0.99,
    piece_placed_reward: float = 0.01,
    line_clear_reward: float = 0.75,
    terminal_penalty: float = 1.0,
    final_epsilon: float = 0.01,
) -> None:
    """运行一个 actor 进程中的多个同步 Tetris 环境。

    每轮先等待 gather 的 round command，并在轮首非阻塞加载 learner mailbox
    中的最新权重，然后严格产生 ``transition_batch_size`` 条 transition。
    batch 被放入这个 actor 独占的结果队列后，actor 停止采样，直到 gather
    收齐全部 actor 并统一放行下一轮。权重通道不参与这个 barrier。

    ``next_obs`` 是执行动作后的观测；某个子环境 terminated 时，SameStep
    autoreset 会返回下一局的初始观测。terminated mask 会在 learner 计算 Double
    DQN target 时阻止 bootstrap，因此不会跨 episode 传播价值。
    """
    if envs_per_actor < 1:
        raise ValueError("envs_per_actor must be positive")
    if transition_batch_size < envs_per_actor:
        raise ValueError("transition_batch_size must be at least envs_per_actor")
    if transition_batch_size % envs_per_actor:
        raise ValueError("transition_batch_size must be divisible by envs_per_actor")

    # Metrics are explicitly best-effort. Do not let this child wait for its
    # queue feeder to flush stale log records during coordinated shutdown.
    metric_queue.cancel_join_thread()

    # actor 永远只在 CPU 上执行环境和前向推理；GPU 参数更新由主进程 learner
    # 独占。限制线程数可以避免多个 actor 进程各自创建过多 BLAS/PyTorch 线程。
    torch.set_num_threads(1)
    num_actions = 40
    model = DuelingDQN().cpu().eval()
    initial_weight = _get_with_stop(
        weight_queue,
        poll_timeout=transition_put_poll_timeout,
        stop_event=stop_event,
    )
    if initial_weight is None:
        return
    weight_version, weight_state = initial_weight
    model.load_state_dict(
        {key: torch.as_tensor(value) for key, value in weight_state.items()}
    )

    # 同一个 actor 内的每个环境使用不同 seed；不同 actor 的 seed 基址由 trainer
    # 分配成足够大的步长。这样环境的 7-bag 和随机探索流都不会意外重合。
    env_seeds = tuple(seed + index for index in range(envs_per_actor))
    env = make_sync_vector_env(
        envs_per_actor,
        seeds=env_seeds,
        gravity_period=2,
        gamma=gamma,
        piece_placed_reward=piece_placed_reward,
        line_clear_reward=line_clear_reward,
        terminal_penalty=terminal_penalty,
    )
    rng = np.random.default_rng(seed + 7919)
    obs, _ = env.reset(seed=list(env_seeds))

    # 下面四个数组只在 actor 进程内维护，用于 episode 结束时发送监控指标；
    # 它们不进入 transition，因此不会增加 learner 的 replay 数据量。
    episode_returns = np.zeros(envs_per_actor, dtype=np.float32)
    episode_lengths = np.zeros(envs_per_actor, dtype=np.int64)
    episode_lines = np.zeros(envs_per_actor, dtype=np.int64)
    episode_pieces = np.zeros(envs_per_actor, dtype=np.int64)
    transitions_sent = 0
    transition_messages_sent = 0
    transition_put_wait_seconds = 0.0
    transition_put_poll_timeouts = 0
    last_stats_transitions = 0
    action_counts = np.zeros(num_actions, dtype=np.int64)
    line_clear_transitions = 0
    terminal_transitions = 0
    try:
        while not stop_event.is_set():
            command = _get_with_stop(
                command_queue,
                poll_timeout=transition_put_poll_timeout,
                stop_event=stop_event,
            )
            if command is None:
                break
            round_id = int(command)
            latest_weight = _latest_weight_message(weight_queue)
            if latest_weight is not None and int(latest_weight[0]) >= int(weight_version):
                weight_version, weight_state = latest_weight
                model.load_state_dict(
                    {key: torch.as_tensor(value) for key, value in weight_state.items()}
                )
            current_epsilon = epsilon_for_schedule(
                epsilon,
                float(decay_progress.value),
                final_epsilon,
            )
            round_batches: list[TransitionBatch] = []
            round_transitions = 0
            while round_transitions < transition_batch_size and not stop_event.is_set():
                # Placement 模式用 observation 中的 mask 同时约束贪心和随机动作。
                with torch.inference_mode():
                    torch_obs = observations_to_torch(obs, "cpu")
                    q_values = masked_q_values(model(torch_obs), torch_obs.get("action_mask"))
                    actions = q_values.argmax(dim=-1).numpy()
                explore = rng.random(envs_per_actor) < current_epsilon
                for index in np.flatnonzero(explore):
                    legal_actions = np.flatnonzero(np.asarray(obs["action_mask"])[index])
                    actions[index] = int(rng.choice(legal_actions))

                next_obs, rewards, terminated, truncated, infos = env.step(actions)
                action_counts += np.bincount(actions, minlength=num_actions)
                ended = np.logical_or(terminated, truncated)
                cleared = np.asarray(
                    [
                        _vector_info_value(
                            infos, "lines_cleared", i, terminated=bool(ended[i]), default=0
                        )
                        for i in range(envs_per_actor)
                    ],
                    dtype=np.int64,
                )
                placed = np.asarray(
                    [
                        _vector_info_value(
                            infos, "piece_placed", i, terminated=bool(ended[i]), default=False
                        )
                        for i in range(envs_per_actor)
                    ],
                    dtype=np.int64,
                )
                line_clear_transitions += int(np.count_nonzero(cleared))
                terminal_transitions += int(np.count_nonzero(terminated))
                round_batches.append(
                    TransitionBatch(
                        obs={k: np.asarray(v).copy() for k, v in obs.items()},
                        actions=np.asarray(actions, dtype=np.int64),
                        rewards=np.asarray(rewards, dtype=np.float32),
                        next_obs={k: np.asarray(v).copy() for k, v in next_obs.items()},
                        terminated=np.asarray(terminated, dtype=np.bool_),
                    )
                )
                round_transitions += envs_per_actor

                episode_returns += rewards
                episode_lengths += 1
                episode_lines += cleared
                episode_pieces += placed
                for i in range(envs_per_actor):
                    if bool(ended[i]):
                        def final_value(key: str, default: Any) -> Any:
                            return _vector_info_value(
                                infos, key, i, terminated=True, default=default
                            )

                        _put_metric(metric_queue, {
                            "actor_id": actor_id,
                            "return": float(episode_returns[i]),
                            "length": int(episode_lengths[i]),
                            "lines": int(episode_lines[i]),
                            "survival_pieces": int(final_value("survival_pieces", episode_pieces[i])),
                            "max_height": int(final_value("board_height", 0)),
                            "aggregate_height": int(final_value("aggregate_height", 0)),
                            "holes": int(final_value("holes", 0)),
                            "bumpiness": int(final_value("bumpiness", 0)),
                            "wells": int(final_value("wells", 0)),
                        })
                        episode_returns[i] = 0
                        episode_lengths[i] = 0
                        episode_lines[i] = 0
                        episode_pieces[i] = 0
                obs = next_obs

            if stop_event.is_set():
                break
            batch = concatenate_transition_batches(round_batches)
            waited, timed_out, sent = _put_with_wait(
                transition_queue,
                (int(round_id), batch),
                poll_timeout=transition_put_poll_timeout,
                stop_event=stop_event,
            )
            transition_put_wait_seconds += waited
            transition_put_poll_timeouts += timed_out
            if not sent:
                break
            transitions_sent += len(batch.actions)
            transition_messages_sent += 1

            # 只有 batch 已经可靠进入 transition_queue 后才增加 transitions_sent。
            # 因此这个数字表示 learner 可能已经收到的样本数，而不是 actor
            # 本地尚未发送的 pending 数量。
            if transitions_sent - last_stats_transitions >= stats_every:
                last_stats_transitions = transitions_sent
                _put_metric(
                    metric_queue,
                    {
                        "kind": "actor_communication",
                        "scene": "transition_put",
                        "actor_id": actor_id,
                        "transitions": transitions_sent,
                        "messages_sent": transition_messages_sent,
                        "queue_wait_seconds": transition_put_wait_seconds,
                        "queue_wait_timeouts": transition_put_poll_timeouts,
                        "action_counts": action_counts.copy(),
                        "line_clear_transitions": line_clear_transitions,
                        "terminal_transitions": terminal_transitions,
                    },
                )
    finally:
        _put_metric(metric_queue, {
            "kind": "actor_communication",
            "scene": "transition_put",
            "actor_id": actor_id,
            "transitions": transitions_sent,
            "messages_sent": transition_messages_sent,
            "queue_wait_seconds": transition_put_wait_seconds,
            "queue_wait_timeouts": transition_put_poll_timeouts,
            "action_counts": action_counts.copy(),
            "line_clear_transitions": line_clear_transitions,
            "terminal_transitions": terminal_transitions,
        })
        env.close()
