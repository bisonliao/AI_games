from __future__ import annotations

import queue
import time
import traceback
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch

from env import make_vector_env

try:
    from .agent import encode_boards, random_legal_actions, slice_obs
    from .network import DuelingGomokuQNet
    from .returns import NStepAccumulator, Transition, shaped_reward
except ImportError:
    from agent import encode_boards, random_legal_actions, slice_obs
    from network import DuelingGomokuQNet
    from returns import NStepAccumulator, Transition, shaped_reward


StateDict = Mapping[str, np.ndarray]
QUEUE_PUT_TIMEOUT_SECONDS = 2.0


def cpu_state_dict(module: torch.nn.Module) -> Dict[str, np.ndarray]:
    """Return a snapshot that avoids torch's FD-based queue transport."""
    return {
        name: value.detach().cpu().numpy().copy()
        for name, value in module.state_dict().items()
    }


class InferencePolicy:
    """Network-only policy used by actors (no replay, target net, or optimizer)."""

    def __init__(
        self,
        board_size: int,
        model_kwargs: Mapping[str, Any],
        state_dict: StateDict,
        *,
        seed: int,
        device: str,
    ) -> None:
        self.board_size = int(board_size)
        self.action_dim = self.board_size * self.board_size
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.net = DuelingGomokuQNet(**dict(model_kwargs)).to(self.device)
        self.load_state_dict(state_dict)

    def load_state_dict(self, state_dict: StateDict) -> None:
        tensors = {name: torch.as_tensor(value) for name, value in state_dict.items()}
        self.net.load_state_dict(tensors)
        self.net.eval()

    def select_actions(
        self,
        boards: np.ndarray,  # [B, H, W]: board values in {-1, 0, 1}.
        current_players: np.ndarray,  # [B] or [B, 1]: player to move per board.
        action_masks: np.ndarray,  # [B, A]: legal-action flags; A = H * W.
        epsilon: float = 0.0,  # Probability that each board keeps its random legal action.
    ) -> np.ndarray:  # [B]: one flattened board-position action per board.
        boards = np.asarray(boards)  # Normalize input to a NumPy batch; shape [B, H, W].
        masks = np.asarray(action_masks).reshape((boards.shape[0], self.action_dim)).astype(bool)  # [B, A].
        actions = random_legal_actions(masks, self.rng)  # Initialize [B] epsilon-exploration actions.
        greedy = np.flatnonzero(self.rng.random(boards.shape[0]) >= float(epsilon))  # [G] batch rows using greedy actions.
        if len(greedy) == 0:
            return actions  # Every row explores; no network inference is needed.
        states = encode_boards(boards[greedy], np.asarray(current_players).reshape(-1)[greedy])  # [G, 3, H, W].
        with torch.inference_mode():
            q_values = self.net(torch.as_tensor(states, device=self.device))  # [G, A]: Q-value per action.
            mask = torch.as_tensor(masks[greedy], device=self.device)  # [G, A]: legal actions for greedy rows.
            actions[greedy] = q_values.masked_fill(~mask, -1e9).argmax(1).cpu().numpy()  # Replace G rows with legal argmax actions.
        return actions  # [B]: mixed epsilon-random and greedy actions.


class RandomPolicy:
    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def select_actions(
        self,
        boards: np.ndarray,
        current_players: np.ndarray,
        action_masks: np.ndarray,
        epsilon: float = 0.0,
    ) -> np.ndarray:
        del boards, current_players, epsilon
        return random_legal_actions(action_masks, self.rng)


def pack_transitions(
    transitions: List[Transition],
    episodes: List[Tuple[int, int, float]],
    *,
    actor_id: int,
    policy_version: int,
    blocked_seconds: float,
) -> Dict[str, Any]:
    return {
        "type": "batch",
        "actor_id": actor_id,
        "policy_version": policy_version,
        "states": np.stack([item.state for item in transitions]),
        "actions": np.asarray([item.action for item in transitions], dtype=np.int64),
        "rewards": np.asarray([item.reward for item in transitions], dtype=np.float32),
        "next_states": np.stack([item.next_state for item in transitions]),
        "next_masks": np.stack([item.next_mask for item in transitions]),
        "dones": np.asarray([item.done for item in transitions], dtype=np.bool_),
        "discounts": np.asarray([item.discount for item in transitions], dtype=np.float32),
        "episodes": episodes,
        "blocked_seconds": blocked_seconds,
    }


def _queue_put(
    result_queue: Any,
    status_queue: Any,
    message: Dict[str, Any],
    stop_event: Any,
    actor_id: int,
) -> float:
    started = time.monotonic()
    timeout_reported = False
    while not stop_event.is_set():
        try:
            result_queue.put(message, timeout=QUEUE_PUT_TIMEOUT_SECONDS)
            return time.monotonic() - started
        except queue.Full:
            if not timeout_reported:
                status_queue.put_nowait({
                    "type": "queue_put_timeout",
                    "actor_id": actor_id,
                    "timeout_seconds": QUEUE_PUT_TIMEOUT_SECONDS,
                    "event_time": time.time(),
                })
                timeout_reported = True
            continue
    return time.monotonic() - started


def _apply_commands(
    control_queue: Any,
    black: InferencePolicy,
    opponent: Any,
    board_size: int,
    actor_seed: int,
    actor_device: str,
    policy_version: int,
) -> Tuple[Any, int]:
    while True:
        try:
            command = control_queue.get_nowait()
        except queue.Empty:
            break
        kind = command[0]
        if kind == "policy":
            _, policy_version, state_dict = command
            black.load_state_dict(state_dict)
        elif kind == "opponent_random":
            opponent = RandomPolicy(actor_seed + 10_000)
        elif kind == "opponent":
            _, state_dict, model_kwargs = command
            opponent = InferencePolicy(
                board_size, model_kwargs, state_dict,
                seed=actor_seed + 10_000, device=actor_device,
            )
        elif kind == "stop":
            break
    return opponent, policy_version


def actor_worker(
    actor_id: int,  # 当前 actor 的唯一编号，用于种子派生、日志和消息标识。
    config: Dict[str, Any],  # 主进程传入的只读采样配置。
    initial_policy: StateDict,  # 初始黑棋网络权重；每项都是 CPU NumPy 数组。
    model_kwargs: Mapping[str, Any],  # 重建黑棋推理网络所需的结构参数。
    initial_opponent: Any,  # 带类型标签的初始白棋配置，不直接跨进程传递agent对象。
    result_queue: Any,  # actor -> learner：发送批量 transition 或异常信息。
    status_queue: Any,  # actor -> learner：发送队列写入超时等状态事件。
    control_queue: Any,  # learner -> 当前 actor：接收黑棋/白棋权重更新与停止命令。
    stop_event: Any,  # 所有进程共享的停止标志。
    generated_steps: Any,  # 所有 actor 共享的黑棋生成步数计数器。
) -> None:  # 子进程入口；正常退出时不返回训练数据。
    envs = None  # 提前定义，确保初始化中途失败时 finally 仍可安全判断。
    try:
        torch.set_num_threads(1)  # 限制当前 actor 的算子线程数，避免多个 actor 争抢全部 CPU 核心。
        try:
            torch.set_num_interop_threads(1)  # 限制 PyTorch 算子之间的并行调度线程数。
        except RuntimeError:  # 某些环境在线程池初始化后不允许再次设置该参数。
            pass  # 保留已生效的线程设置并继续运行。
        # Derive disjoint deterministic seed ranges from the shared root seed:
        #   actor_seed(actor_id) = root_seed + actor_id * 100_000
        #   env_seed(env_index)   = actor_seed + env_index
        # With the default 16 envs/actor, actor 0 uses env seeds 0..15,
        # actor 1 uses 100000..100015, etc. Thus reproducibility is retained
        # without making different actors or vector environments identical.
        seed = int(config["seed"]) + actor_id * 100_000  # 为当前 actor 生成独立且可复现的根种子。
        np.random.seed(seed)  # 设置当前子进程的 NumPy 全局随机种子。
        torch.manual_seed(seed)  # 设置当前子进程的 PyTorch 随机种子。
        black = InferencePolicy(
            config["board_size"], model_kwargs, initial_policy,
            seed=seed, device=config["actor_device"],
        )  # 创建 actor 私有的黑棋推理网络，不含 replay、target net 和 optimizer。
        opponent_kind = initial_opponent[0]
        if opponent_kind == "random":  # 主进程未提供白棋 checkpoint 时使用随机陪练。
            opponent: Any = RandomPolicy(seed + 10_000)  # 使用与黑棋不同的随机数流。
        elif opponent_kind == "dqn":  # 根据主进程快照创建白棋 DQN 推理网络。
            opponent = InferencePolicy(
                config["board_size"], initial_opponent[2], initial_opponent[1],
                seed=seed + 10_000, device=config["actor_device"],
            )  # 白棋只负责选动作，不参与梯度更新。
        else:
            raise ValueError(f"Unsupported rollout-A opponent type: {opponent_kind!r}")

        num_envs = int(config["envs_per_actor"])
        board_size = int(config["board_size"])
        envs = make_vector_env(num_envs, board_size=board_size, asynchronous=False, seed=seed)  # 在本 actor 内创建同步向量环境。
        # Gymnasium accepts one reset seed per vector-environment slot. This is
        # the point where actor_num * envs_per_actor distinct seeds are applied.
        obs, _ = envs.reset(seed=[seed + index for index in range(num_envs)])  # obs: board[N,H,W]、player[N,1]、mask[N,A]。
        pending_states: List[Optional[np.ndarray]] = [None] * num_envs  # 每个环境待闭合的黑棋状态；单项形状 [3,H,W]。
        pending_actions: List[Optional[int]] = [None] * num_envs  # 与 pending_states 对应的黑棋动作。
        episode_steps = np.zeros(num_envs, dtype=np.int64)  # [N]：每个环境当前对局的黑棋决策次数。
        transitions: List[Transition] = []  # 暂存已经完成 n-step 聚合、等待发送给 learner 的样本。
        n_step_accumulator = NStepAccumulator(
            num_envs, int(config["n_step"]), float(config["gamma"])
        )  # 为每个环境维护独立轨迹缓存，防止不同对局的 n-step return 混合。
        episodes: List[Tuple[int, int, float]] = []  # 待上报的 (赢家, 黑棋步数, 黑棋回报) 列表。
        policy_version = 0  # 当前黑棋快照版本，用于 learner 统计策略滞后。
        blocked_seconds = 0.0  # 上一次向 result_queue 写入 batch 时的阻塞时长。

        while not stop_event.is_set():  # 持续采样，直到 learner 请求所有 actor 停止。
            players_now = obs["current_player"].reshape(-1)  # [N]：每个环境当前轮到的玩家，黑=1、白=-1。
            actions = np.zeros(num_envs, dtype=np.int64)  # [N]：本轮提交给所有向量环境的动作。
            # 由于并发环境的自动reset后继续回合导致黑,黑，所以并不确保黑白轮流给环境施加动作，
            # 也就是每次动作并不整齐划一的都是黑棋或者白棋，这是一个复杂度。
            black_indices = np.flatnonzero(players_now == 1)  # [B]：本轮需要黑棋决策的环境下标。
            white_indices = np.flatnonzero(players_now == -1)  # [W]：本轮需要白棋决策的环境下标。

            if len(black_indices):  # 只对当前轮到黑棋的环境进行一次批量推理。
                with generated_steps.get_lock():  # 对跨进程共享计数器加锁，避免多个 actor 同时覆盖更新。
                    epsilon_step = int(generated_steps.value)  # 读取本批黑棋动作对应的全局探索步数。
                    generated_steps.value += len(black_indices)  # 按本批实际生成的黑棋动作数推进计数。
                fraction = min(1.0, epsilon_step / max(1, config["epsilon_decay_steps"]))  # epsilon 衰减进度 [0,1]。
                epsilon = config["epsilon_start"] + fraction * (
                    config["epsilon_end"] - config["epsilon_start"]
                ) if config["epsilon_decay_steps"] > 0 else config["epsilon_end"]  # 计算本批共享的 epsilon。
                boards, players, masks = slice_obs(obs, black_indices)  # [B,H,W]、[B]、[B,A]。
                selected = black.select_actions(boards, players, masks, epsilon)  # [B]：黑棋 epsilon-greedy 动作。
                encoded = encode_boards(boards, players)  # [B,3,H,W]：黑棋视角的网络状态。
                actions[black_indices] = selected  # 把黑棋子批次动作写回完整动作数组 [N]。
                for row, env_index in enumerate(black_indices):  # 为每个黑棋环境登记尚待白棋回应的决策。
                    pending_states[env_index] = encoded[row]  # [3,H,W]：保存黑棋落子前状态。
                    pending_actions[env_index] = int(selected[row])  # 保存该状态下执行的动作。
                    episode_steps[env_index] += 1  # 当前对局的黑棋长度加一。

            if len(white_indices):  # 只对当前轮到白棋的环境进行一次批量推理。
                boards, players, masks = slice_obs(obs, white_indices)  # [W,H,W]、[W]、[W,A]。
                actions[white_indices] = opponent.select_actions(
                    boards, players, masks, config["opponent_epsilon"]
                )  # [W]：将白棋陪练动作写回完整动作数组 [N]。

            next_obs, _, terminated, truncated, infos = envs.step(actions)  # 推进 N 个环境；next_obs 各数组首维均为 N。
            dones = np.logical_or(terminated, truncated)  # [N]：合并自然终局与时间截断标志。
            for env_index in range(num_envs):  # 分别闭合每个环境中的黑棋 decision interval。
                acted_player = int(players_now[env_index])  # 记录刚才实际执行动作的棋子颜色。
                done = bool(dones[env_index])  # 当前环境是否已结束并被 SameStep 自动重置。
                if acted_player == 1 and done:  # 黑棋刚落子便终局，无需等待白棋回应。
                    reward = float(_info_value(infos, "reward_black", env_index, True, 0.0))  # 读取终局黑棋奖励。
                    next_state = np.zeros((3, board_size, board_size), dtype=np.float32)  # [3,H,W]：终局不 bootstrap。
                    reward = shaped_reward(
                        reward, pending_states[env_index], next_state, True,
                        config["gamma"], config["reward_shaping_scale"],
                    )  # 在稀疏终局奖励上加入低强度势能差奖励。
                    transitions.extend(n_step_accumulator.add(env_index, Transition(
                        pending_states[env_index], int(pending_actions[env_index]), reward,
                        next_state, np.zeros(board_size * board_size, dtype=np.bool_), True,
                    )))  # 写入原始 interval，并接收终局 flush 出来的一个或多个 n-step 样本。
                    pending_states[env_index] = None  # 当前黑棋决策已经闭合，不再等待回应。
                    pending_actions[env_index] = None  # 清除与 pending state 配套的动作。
                elif acted_player == 1:  # 非终局黑棋落子后，transition 必须等白棋回应再闭合。
                    continue  # 跳过本环境剩余逻辑，保留 pending state/action。
                elif pending_states[env_index] is not None:  # 白棋已回应此前保存的黑棋动作。
                    reward = float(_info_value(infos, "reward_black", env_index, done, 0.0))  # 读取白棋回应后的黑棋奖励。
                    if done:  # 白棋回应导致终局，不能连接到 SameStep 自动重置后的新对局。
                        next_state = np.zeros((3, board_size, board_size), dtype=np.float32)  # [3,H,W]：终局零状态。
                        next_mask = np.zeros(board_size * board_size, dtype=np.bool_)  # [A]：终局无合法后继动作。
                    else:  # 对局继续，后继状态已经重新轮到黑棋。
                        next_state = encode_boards(
                            next_obs["board"][env_index], next_obs["current_player"][env_index]
                        )[0]  # [3,H,W]：下一次黑棋决策状态；去掉 encode_boards 添加的批次维。
                        next_mask = next_obs["action_mask"][env_index].astype(np.bool_)  # [A]：后继合法动作掩码。
                    reward = shaped_reward(
                        reward, pending_states[env_index], next_state, done,
                        config["gamma"], config["reward_shaping_scale"],
                    )  # 计算当前黑棋 decision interval 的塑形奖励。
                    transitions.extend(n_step_accumulator.add(env_index, Transition(
                        pending_states[env_index], int(pending_actions[env_index]), reward,
                        next_state, next_mask, done,
                    )))  # 聚合并取回已经成熟或因终局 flush 的 n-step replay 样本。
                    pending_states[env_index] = None  # 白棋已回应，当前 interval 闭合完成。
                    pending_actions[env_index] = None  # 清除已消费的黑棋动作。

                if done:  # 汇总本局统计；终局信息必须从 infos.final_info 读取。
                    episodes.append((
                        int(_info_value(infos, "winner", env_index, True, 0)),
                        int(episode_steps[env_index]),
                        float(_info_value(infos, "reward_black", env_index, True, 0.0)),
                    ))  # 添加 (winner, black_length, black_return) 供 learner 汇总日志。
                    episode_steps[env_index] = 0  # SameStep 已开启下一局，重置该槽位的局长计数。

            obs = next_obs  # 推进 actor 当前观测；各数组仍为以 N 为首维的批次。
            if len(transitions) >= config["actor_batch_size"]:  # 累积足够样本后才跨进程发送，摊薄 IPC 成本。
                message = pack_transitions(
                    transitions, episodes, actor_id=actor_id,
                    policy_version=policy_version, blocked_seconds=blocked_seconds,
                )  # 将列表堆叠为 states[M,3,H,W]、actions[M]、masks[M,A] 等连续批量数组。
                blocked_seconds = _queue_put(
                    result_queue, status_queue, message, stop_event, actor_id
                )  # 有背压地发送给 learner，并记录本次 queue.put 的总等待时间。
                transitions = []  # message 已持有独立堆叠数组，可以清空本地样本列表。
                episodes = []  # 本批对局统计已随 message 发送，开始积累下一批。
                opponent, policy_version = _apply_commands(
                    control_queue, black, opponent, board_size, seed,
                    config["actor_device"], policy_version,
                )  # 在 batch 边界应用 learner 发来的最新黑棋/白棋权重或停止命令。
    except BaseException:  # 捕获子进程中的所有异常，尽量把原因传回 learner。
        message = {"type": "error", "actor_id": actor_id, "traceback": traceback.format_exc()}  # 序列化完整 traceback。
        try:
            result_queue.put(message, timeout=1.0)  # 通过正常数据通道通知 learner 抛出对应错误。
        except (queue.Full, ValueError):  # 队列已满或关闭时无法保证错误消息成功送达。
            pass  # learner 仍可通过子进程退出码发现 actor 异常退出。
        stop_event.set()  # 请求其他 actor 一起停止，避免 learner 失败后它们继续生产。
    finally:  # 无论正常停止还是异常，都释放当前子进程持有的环境资源。
        if envs is not None:  # 环境成功创建后才需要关闭。
            envs.close()  # 关闭所有向量环境及其相关资源。


def _info_value(infos: Dict[str, Any], key: str, index: int, done: bool, default: Any) -> Any:
    if done:
        mask = infos.get("_final_info")
        if mask is None or not mask[index]:
            raise RuntimeError("Done environment is missing final_info")
        final_infos = infos.get("final_info")
        if isinstance(final_infos, dict):
            key_mask = final_infos.get(f"_{key}")
            if key in final_infos and (key_mask is None or key_mask[index]):
                return final_infos[key][index]
        elif final_infos is not None and final_infos[index] is not None:
            return final_infos[index].get(key, default)
        return default
    if key not in infos:
        return default
    try:
        return infos[key][index]
    except Exception:
        return default
