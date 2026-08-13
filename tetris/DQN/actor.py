"""CPU actor process for vectorized Tetris interaction."""
from __future__ import annotations

import queue
import time
from typing import Any

import numpy as np
import torch

from TetrisEnv.vector_runner import make_sync_vector_env

from .model import DuelingDQN, masked_q_values, observations_to_torch
from .replay import TransitionBatch


def _latest_weights(weight_queue, model: DuelingDQN) -> None:
    """只加载最新一份 learner 权重，丢弃 actor 尚未消费的旧快照。

    权重同步不是 transition 数据通道：旧权重即使被跳过也不会造成训练样本
    丢失，只是让 actor 在短时间内继续使用上一版策略。因此这里使用
    ``get_nowait`` 清空队列，避免 actor 因等待权重而停止采样。
    """
    latest = None
    while True:
        try:
            latest = weight_queue.get_nowait()
        except queue.Empty:
            break
    if latest is not None:
        model.load_state_dict({key: torch.as_tensor(value) for key, value in latest.items()})


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


def _merge_transition_batches(batches: list[TransitionBatch]) -> TransitionBatch:
    """按时间顺序把多个 vector-step batch 合并为一个 IPC batch。"""
    if not batches:
        raise ValueError("cannot merge an empty transition batch list")
    return TransitionBatch(
        obs={
            key: np.concatenate([item.obs[key] for item in batches], axis=0)
            for key in batches[0].obs
        },
        actions=np.concatenate([item.actions for item in batches], axis=0),
        rewards=np.concatenate([item.rewards for item in batches], axis=0),
        next_obs={
            key: np.concatenate([item.next_obs[key] for item in batches], axis=0)
            for key in batches[0].next_obs
        },
        terminated=np.concatenate([item.terminated for item in batches], axis=0),
    )


def actor_process(
    actor_id: int,
    envs_per_actor: int,
    seed: int,
    epsilon: float,
    transition_queue,
    metric_queue,
    weight_queue,
    stop_event,
    transition_put_poll_timeout: float = 1.0,
    stats_every: int = 1_000,
    transition_batch_size: int = 256,
    transition_batch_max_wait: float = 0.1,
    gamma: float = 0.99,
    piece_placed_reward: float = 0.01,
    line_clear_reward: float = 0.75,
    terminal_penalty: float = 1.0,
) -> None:
    """运行一个 actor 进程中的多个同步 Tetris 环境。

    一个 actor 的工作循环可以拆成四个阶段：

    1. 从本地 weight queue 非阻塞地取最新网络参数；
    2. 对当前 ``envs_per_actor`` 个环境做一次同步 ``step``，得到一批并行
       transition；
    3. 在 actor 本地累积多个 vector step，并把它们拼成一个较大的 batch，
       再发送给 learner 的 transition queue；
    4. 更新 episode 统计并把低频监控消息尽力发送出去。

    因而这里的“一个环境 step”和“向 learner 发送一次 IPC 消息”不是一一对应
    的：当前 vector step 的样本先进入 ``pending_batches``，达到
    ``transition_batch_size`` 后才发送。比如 8 个环境、batch size 256 时，
    大约每 32 个 vector step 发送一次，单条消息包含 256 条 transition。

    ``next_obs`` 是执行动作后的观测；某个子环境 terminated 时，SameStep
    autoreset 会返回下一局的初始观测。terminated mask 会在 learner 计算 Double
    DQN target 时阻止 bootstrap，因此不会跨 episode 传播价值。
    """
    if envs_per_actor < 1:
        raise ValueError("envs_per_actor must be positive")
    if transition_batch_size < envs_per_actor:
        raise ValueError("transition_batch_size must be at least envs_per_actor")
    if transition_batch_max_wait <= 0:
        raise ValueError("transition_batch_max_wait must be positive")

    # actor 永远只在 CPU 上执行环境和前向推理；GPU 参数更新由主进程 learner
    # 独占。限制线程数可以避免多个 actor 进程各自创建过多 BLAS/PyTorch 线程。
    torch.set_num_threads(1)
    num_actions = 40
    model = DuelingDQN().cpu().eval()
    _latest_weights(weight_queue, model)

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
    # 每个元素是一次 vector step 的 TransitionBatch，内部包含 E 条 transition。
    # 使用 list 而不是逐条发送，是为了降低 multiprocessing.Queue 的消息和
    # pickle 次数；真正入队前再按字段拼接成一个连续的大 batch。
    pending_batches: list[TransitionBatch] = []
    pending_transitions = 0
    pending_started_at: float | None = None
    transitions_sent = 0
    transition_messages_sent = 0
    transition_put_wait_seconds = 0.0
    transition_put_poll_timeouts = 0
    last_stats_transitions = 0
    action_counts = np.zeros(num_actions, dtype=np.int64)
    line_clear_transitions = 0
    terminal_transitions = 0
    def flush_pending() -> bool:
        """拼接并可靠发送本地累计 batch，返回是否成功入队。

        这里的拼接顺序保持 vector step 顺序、再保持环境索引顺序。对 DQN 的
        off-policy replay 来说不要求跨 actor 的全局顺序，但保持本地顺序便于
        复现和调试。flush 成功后才清空 pending，避免 put 失败时样本被丢掉。
        """
        nonlocal pending_transitions, pending_started_at, transition_put_wait_seconds, transition_put_poll_timeouts, transitions_sent, transition_messages_sent
        if not pending_batches:
            return True
        batch = _merge_transition_batches(pending_batches)
        waited, timed_out, sent = _put_with_wait(
            transition_queue,
            batch,
            poll_timeout=transition_put_poll_timeout,
            stop_event=stop_event,
        )
        transition_put_wait_seconds += waited
        transition_put_poll_timeouts += timed_out
        if sent:
            transitions_sent += len(batch.actions)
            transition_messages_sent += 1
            pending_batches.clear()
            pending_transitions = 0
            pending_started_at = None
        return sent
    try:
        while not stop_event.is_set():
            # 权重同步是 best-effort：如果 learner 连续广播多次，旧快照可以跳过，
            # 因为 actor 只需要尽快使用一份较新的策略，不需要逐版本执行。
            _latest_weights(weight_queue, model)

            # Placement 模式用 observation 中的 mask 同时约束贪心和随机动作。
            # 随机探索在合法落点上均匀采样，避免不可执行动作进入 replay。
            with torch.inference_mode():
                torch_obs = observations_to_torch(obs, "cpu")
                q_values = masked_q_values(model(torch_obs), torch_obs.get("action_mask"))
                actions = q_values.argmax(dim=-1).numpy()
            explore = rng.random(envs_per_actor) < epsilon
            for index in np.flatnonzero(explore):
                legal_actions = np.flatnonzero(np.asarray(obs["action_mask"])[index])
                actions[index] = int(rng.choice(legal_actions))

            # SyncVectorEnv.step 一次推进所有 E 个环境，因此这里得到的是 E 条
            # transition。它们先复制到 pending_batches，避免每个 tick 都发生 IPC。
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
            transition = TransitionBatch(
                obs={k: np.asarray(v).copy() for k, v in obs.items()},
                actions=np.asarray(actions, dtype=np.int64),
                rewards=np.asarray(rewards, dtype=np.float32),
                next_obs={k: np.asarray(v).copy() for k, v in next_obs.items()},
                terminated=np.asarray(terminated, dtype=np.bool_),
            )
            pending_batches.append(transition)
            pending_transitions += envs_per_actor
            if pending_started_at is None:
                pending_started_at = time.perf_counter()

            # 满足任一条件就 flush：
            #   1. 累计数量达到 transition_batch_size，优先吞吐；
            #   2. 等待超过 transition_batch_max_wait，限制端到端延迟。
            # flush 内部只有在 put 成功后才清空 pending，所以队列满不会丢样本。
            batch_ready = pending_transitions >= transition_batch_size
            batch_timed_out = time.perf_counter() - pending_started_at >= transition_batch_max_wait
            if (batch_ready or batch_timed_out) and not flush_pending():
                break

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
            # episode 统计放在 transition 入队之后；即使 metric queue 满了，
            # _put_metric 也只丢监控消息，不会影响下一轮环境交互。
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
    finally:
        # 训练正常运行时 pending 会在达到阈值后 flush。停止信号到来时不再
        # 强行等待队列，因此尾部不足一个 batch 的样本可能留在本地；这只发生
        # 在训练收尾阶段，不会发生在正常的 timeout 轮询中。
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
