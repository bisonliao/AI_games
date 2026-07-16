"""Rollout B旁路：独立运行DQN与启发式机器人对局，收集双方transition并管理多actor进程。"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
import traceback
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch

from env import make_vector_env

try:
    from .agent import ReplayBuffer, encode_boards, random_legal_actions, slice_obs
    from .heuristic_agent import HeuristicAgent
    from .network import DuelingGomokuQNet
    from .player_transitions import PlayerTransitionCollector
    from .returns import Transition
except ImportError:
    from agent import ReplayBuffer, encode_boards, random_legal_actions, slice_obs
    from heuristic_agent import HeuristicAgent
    from network import DuelingGomokuQNet
    from player_transitions import PlayerTransitionCollector
    from returns import Transition


StateDict = Mapping[str, np.ndarray]
QUEUE_PUT_TIMEOUT_SECONDS = 2.0


class SidecarInferencePolicy:
    """Private CPU inference policy; intentionally independent of rollout A."""

    def __init__(
        self,
        board_size: int,
        model_kwargs: Mapping[str, Any],
        state_dict: StateDict,
        *,
        seed: int,
    ) -> None:
        self.action_dim = int(board_size) ** 2
        self.rng = np.random.default_rng(seed)
        self.net = DuelingGomokuQNet(**dict(model_kwargs)).to("cpu")
        self.load_state_dict(state_dict)

    def load_state_dict(self, state_dict: StateDict) -> None:
        self.net.load_state_dict({
            name: torch.as_tensor(value) for name, value in state_dict.items()
        })
        self.net.eval()

    def select_actions(
        self,
        boards: np.ndarray,
        current_players: np.ndarray,
        action_masks: np.ndarray,
        epsilon: float = 0.0,
    ) -> np.ndarray:
        boards = np.asarray(boards)
        masks = np.asarray(action_masks).reshape(len(boards), self.action_dim).astype(bool)
        actions = random_legal_actions(masks, self.rng)
        greedy = np.flatnonzero(self.rng.random(len(boards)) >= float(epsilon))
        if not len(greedy):
            return actions
        states = encode_boards(
            boards[greedy], np.asarray(current_players).reshape(-1)[greedy]
        )
        with torch.inference_mode():
            q_values = self.net(torch.as_tensor(states))
            legal = torch.as_tensor(masks[greedy])
            actions[greedy] = q_values.masked_fill(~legal, -1e9).argmax(1).numpy()
        return actions


def _info_value(
    infos: Dict[str, Any], key: str, env_index: int, done: bool, default: Any,
) -> Any:
    if done:
        mask = infos.get("_final_info")
        if mask is None or not mask[env_index]:
            raise RuntimeError("Done sidecar environment is missing final_info")
        final_infos = infos.get("final_info")
        if isinstance(final_infos, dict):
            key_mask = final_infos.get(f"_{key}")
            if key in final_infos and (key_mask is None or key_mask[env_index]):
                return final_infos[key][env_index]
        elif final_infos is not None and final_infos[env_index] is not None:
            return final_infos[env_index].get(key, default)
        return default
    values = infos.get(key)
    if values is None:
        return default
    try:
        return values[env_index]
    except Exception:
        return default


def _pack_batch(
    transitions: Sequence[Transition],
    *,
    actor_id: int,
    policy_version: int,
    agent_wins: int,
    agent_losses: int,
    draws: int,
    blocked_seconds: float,
) -> Dict[str, Any]:
    return {
        "type": "sidecar_batch",
        "actor_id": actor_id,
        "policy_version": policy_version,
        "states": np.stack([item.state for item in transitions]),
        "actions": np.asarray([item.action for item in transitions], dtype=np.int64),
        "rewards": np.asarray([item.reward for item in transitions], dtype=np.float32),
        "next_states": np.stack([item.next_state for item in transitions]),
        "next_masks": np.stack([item.next_mask for item in transitions]),
        "dones": np.asarray([item.done for item in transitions], dtype=np.bool_),
        "discounts": np.asarray([item.discount for item in transitions], dtype=np.float32),
        "agent_wins": int(agent_wins),
        "agent_losses": int(agent_losses),
        "draws": int(draws),
        "blocked_seconds": float(blocked_seconds),
    }


def _put_batch(
    result_queue: Any,
    status_queue: Any,
    message: Dict[str, Any],
    stop_event: Any,
    actor_id: int,
) -> float:
    started = time.monotonic()
    reported = False
    while not stop_event.is_set():
        try:
            result_queue.put(message, timeout=QUEUE_PUT_TIMEOUT_SECONDS)
            return time.monotonic() - started
        except queue.Full:
            if not reported:
                status_queue.put({
                    "type": "queue_put_timeout",
                    "actor_id": actor_id,
                    "timeout_seconds": QUEUE_PUT_TIMEOUT_SECONDS,
                })
                reported = True
    return time.monotonic() - started


def _apply_policy_commands(
    control_queue: Any,
    agent: SidecarInferencePolicy,
    policy_version: int,
) -> int:
    while True:
        try:
            command = control_queue.get_nowait()
        except queue.Empty:
            return policy_version
        if command[0] == "policy":
            _, policy_version, state_dict = command
            agent.load_state_dict(state_dict)
        elif command[0] == "stop":
            return policy_version


def sidecar_actor_worker(
    actor_id: int,
    config: Dict[str, Any],
    initial_policy: StateDict,
    model_kwargs: Mapping[str, Any],
    result_queue: Any,
    status_queue: Any,
    control_queue: Any,
    stop_event: Any,
) -> None:
    """Collect both colors without importing or changing rollout-A actor code."""
    envs = None
    try:
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        seed = int(config["seed"]) + 1_000_000 + actor_id * 100_000
        np.random.seed(seed)
        torch.manual_seed(seed)
        num_envs = int(config["envs_per_actor"])
        board_size = int(config["board_size"])
        agent = SidecarInferencePolicy(
            board_size, model_kwargs, initial_policy, seed=seed,
        )
        heuristic = HeuristicAgent(seed=seed + 10_000)
        # Half the slots put the latest DQN on black and half put it on white.
        # actor_id shifts odd-sized batches so multiple actors remain balanced.
        agent_is_black = (np.arange(num_envs) + actor_id) % 2 == 0
        envs = make_vector_env(
            num_envs, board_size=board_size, asynchronous=False, seed=seed,
        )
        obs, _ = envs.reset(seed=[seed + index for index in range(num_envs)])
        collectors = {
            1: PlayerTransitionCollector(
                player=1, num_envs=num_envs, board_size=board_size,
                n_step=int(config["n_step"]), gamma=float(config["gamma"]),
                reward_shaping_scale=float(config["reward_shaping_scale"]),
            ),
            -1: PlayerTransitionCollector(
                player=-1, num_envs=num_envs, board_size=board_size,
                n_step=int(config["n_step"]), gamma=float(config["gamma"]),
                reward_shaping_scale=float(config["reward_shaping_scale"]),
            ),
        }
        transitions: List[Transition] = []
        policy_version = 0
        agent_wins = agent_losses = draws = 0
        blocked_seconds = 0.0

        while not stop_event.is_set():  # 持续采样rollout B，直到learner通过跨进程Event发出停止信号。
            players = obs["current_player"].reshape(-1)  # ndarray[N]：每个同步环境当前行棋方，黑棋=1、白棋=-1。
            actions = np.zeros(num_envs, dtype=np.int64)  # ndarray[N]：本轮将提交给N个环境的扁平动作编号。
            agent_turn = ((players == 1) & agent_is_black) | (  # ndarray[N] bool：标记本轮哪些环境由DQN agent行动。
                (players == -1) & ~agent_is_black  # ndarray[N] bool：DQN被分配为白棋且当前轮到白棋的环境。
            )  # 两个条件取并集，保证每个环境恰好由DQN或启发式机器人其中一方决策。
            agent_indices = np.flatnonzero(agent_turn)  # ndarray[Na]：本轮由DQN决策的环境下标，Na范围为0～N。
            heuristic_indices = np.flatnonzero(~agent_turn)  # ndarray[Nh]：本轮由启发式机器人决策的下标，Na+Nh=N。
            if len(agent_indices):  # 仅在至少有一个环境轮到DQN时执行一次批量网络推理。
                boards, current, masks = slice_obs(obs, agent_indices)  # ndarray分别为[Na,H,W]、[Na]、[Na,H*W]。
                actions[agent_indices] = agent.select_actions(  # 将DQN返回的ndarray[Na]动作写回完整动作数组[N]。
                    boards, current, masks, epsilon=0.0,  # DQN在sidecar中固定greedy，不加入epsilon随机探索。
                )  # 批量推理结束后，actions中Na个对应位置已经填入合法动作。
            if len(heuristic_indices):  # 仅在至少有一个环境轮到规则机器人时调用启发式批量接口。
                boards, current, masks = slice_obs(obs, heuristic_indices)  # ndarray分别为[Nh,H,W]、[Nh]、[Nh,H*W]。
                actions[heuristic_indices] = heuristic.select_actions(  # 将规则机器人返回的ndarray[Nh]动作写回actions[N]。
                    boards, current, masks, epsilon=0.0,  # 启发式机器人忽略epsilon，只在同分动作间按种子打破平局。
                )  # 至此actions[N]的每个环境位置都应当包含一个合法动作。

            for player in (1, -1):  # 黑、白分别交给两个独立PlayerTransitionCollector记录各自的决策起点。
                indices = np.flatnonzero(players == player)  # ndarray[Np]：本轮实际由指定颜色player行动的环境下标。
                if len(indices):  # 当前颜色至少在一个环境行动时，才记录其pre-action state/action。
                    collectors[player].record_actions(  # 为该颜色的Np个环境建立待对手回应后闭合的pending transition。
                        obs["board"][indices], indices, actions[indices]  # ndarray形状依次为[Np,H,W]、[Np]、[Np]。
                    )  # 黑白collector状态彼此独立，不会把两种玩家视角的pending数据混在一起。

            next_obs, _, terminated, truncated, infos = envs.step(actions)  # 推进N个环境；next_obs含[N,H,W]/[N,1]/[N,H*W]，终止数组为[N]。
            dones = np.logical_or(terminated, truncated)  # ndarray[N] bool：合并自然终局与时间截断标志。
            rewards = {  # 为黑白分别准备当前step后的玩家视角奖励数组，非终局位置保持0。
                1: np.zeros(num_envs, dtype=np.float32),  # ndarray[N]：每个环境对应的黑棋奖励。
                -1: np.zeros(num_envs, dtype=np.float32),  # ndarray[N]：每个环境对应的白棋奖励。
            }  # 两个数组会分别传给黑、白transition collector闭合本方样本。
            for env_index in np.flatnonzero(dones):  # ndarray[D]中的每个元素是本轮刚结束的环境下标，D<=N。
                index = int(env_index)  # 将NumPy标量下标转成普通Python int，便于访问vector info。
                rewards[1][index] = float(  # 给黑棋奖励数组[N]的当前终局位置写入标量reward_black。
                    _info_value(infos, "reward_black", index, True, 0.0)  # 从SameStep的final_info读取旧对局黑棋奖励。
                )  # 黑胜为+1、白胜为-1、和棋为0。
                rewards[-1][index] = float(  # 给白棋奖励数组[N]的当前终局位置写入标量reward_white。
                    _info_value(infos, "reward_white", index, True, 0.0)  # 从final_info读取与黑棋符号相反的白棋奖励。
                )  # 白胜为+1、黑胜为-1、和棋为0。
                winner = int(_info_value(infos, "winner", index, True, 0))  # 标量：终局赢家，黑=1、白=-1、和棋=0。
                agent_player = 1 if agent_is_black[index] else -1  # 标量：DQN在当前环境中被固定分配的棋色。
                if winner == agent_player:  # 终局赢家恰好是DQN所执棋色，计为agent胜局。
                    agent_wins += 1  # 累加当前待发送batch中的DQN胜局数。
                elif winner == -agent_player:  # 终局赢家是启发式机器人所执棋色，计为agent负局。
                    agent_losses += 1  # 累加当前待发送batch中的DQN负局数。
                else:  # winner=0时双方均未获胜。
                    draws += 1  # 累加当前待发送batch中的和棋数。

            for player in (1, -1):  # 黑白collector分别观察同一次环境step并尝试闭合自己的pending transition。
                transitions.extend(collectors[player].finish_step(  # 把新成熟的0～多个Transition追加到公共待发送列表。
                    acted_players=players,  # ndarray[N]：说明本step各环境究竟是哪种颜色执行了动作。
                    next_obs=next_obs,  # dict：非终局时包含board[N,H,W]、player[N,1]、mask[N,H*W]。
                    dones=dones,  # ndarray[N] bool：终局样本据此使用零next_state，避免连接SameStep重置棋盘。
                    rewards=rewards[player],  # ndarray[N]：当前collector所属玩家视角的奖励。
                ))  # finish_step内部还会按配置执行n-step聚合，因此返回数量不一定等于本轮环境数。
            obs = next_obs  # 更新当前观测；各ndarray仍以环境批次N为首维，终局槽位已是自动重置后的新局。

            if len(transitions) >= int(config["batch_size"]):  # 累积到发送阈值后才进行一次进程间通信，摊薄IPC成本。
                message = _pack_batch(  # 将M条Transition堆叠成连续ndarray消息，M可能因终局flush略大于阈值。
                    transitions, actor_id=actor_id, policy_version=policy_version,  # states[M,3,H,W]、actions[M]等并附带actor信息。
                    agent_wins=agent_wins, agent_losses=agent_losses, draws=draws,  # 附带本批次期间完成对局的DQN W/L/D计数。
                    blocked_seconds=blocked_seconds,  # 附带上一次queue.put阻塞时间，供learner监控sidecar背压。
                )  # message中的next_states为[M,3,H,W]、next_masks为[M,H*W]，其余训练字段为[M]。
                blocked_seconds = _put_batch(  # 有背压地把完整message发送到rollout B独立队列，并返回本次等待秒数。
                    result_queue, status_queue, message, stop_event, actor_id,  # 只阻塞当前sidecar，不会占用rollout A数据队列。
                )  # 若队列连续满2秒会经status_queue上报一次timeout，但仍继续尝试直到停止或成功。
                transitions = []  # 当前message已拥有独立堆叠数组，清空本地Transition列表以积累下一批。
                agent_wins = agent_losses = draws = 0  # 本批W/L/D已经随message发出，窗口计数归零。
                policy_version = _apply_policy_commands(  # 仅在batch边界应用learner发来的最新DQN权重，避免step中途换策略。
                    control_queue, agent, policy_version,  # 持续排空控制队列并返回最终生效的最新策略版本号。
                )  # 下一轮采样将使用更新后的CPU推理网络；启发式机器人本身无需同步参数。
    except BaseException:
        try:
            status_queue.put({
                "type": "error", "actor_id": actor_id,
                "traceback": traceback.format_exc(),
            })
        except Exception:
            pass
    finally:
        if envs is not None:
            envs.close()


class HeuristicSidecar:
    """Parent-side controller for isolated heuristic rollout-B processes."""

    def __init__(
        self,
        *,
        config: Dict[str, Any],
        initial_policy: StateDict,
        model_kwargs: Mapping[str, Any],
        num_actors: int,
        queue_size: int,
        context: Any | None = None,
    ) -> None:
        self.context = context or mp.get_context("spawn")
        self.result_queue = self.context.Queue(maxsize=queue_size)
        self.status_queue = self.context.Queue()
        # Rollout B only needs a pending policy snapshot, not every historical
        # snapshot.  A single-slot queue prevents large state dicts from
        # accumulating in multiprocessing.Queue's feeder/IPC buffers when the
        # sidecar produces batches more slowly than the learner synchronizes.
        self.control_queues = [
            self.context.Queue(maxsize=1) for _ in range(num_actors)
        ]
        self.stop_event = self.context.Event()
        self.processes = [
            self.context.Process(
                target=sidecar_actor_worker,
                name=f"gomoku-heuristic-sidecar-{actor_id}",
                args=(
                    actor_id, config, initial_policy, dict(model_kwargs),
                    self.result_queue, self.status_queue,
                    self.control_queues[actor_id], self.stop_event,
                ),
            )
            for actor_id in range(num_actors)
        ]
        self.actor_versions = [-1] * num_actors
        self.stats = {
            "transitions": 0, "batches": 0, "agent_wins": 0,
            "agent_losses": 0, "draws": 0, "blocked_seconds": 0.0,
            "queue_put_timeouts": 0, "failures": 0,
        }
        self.closed = False

    def start(self) -> None:
        for process in self.processes:
            process.start()

    def drain_into(self, replay: ReplayBuffer) -> int:
        added = 0
        while True:
            try:
                message = self.result_queue.get_nowait()
            except queue.Empty:
                break
            actor_id = int(message["actor_id"])
            self.actor_versions[actor_id] = int(message["policy_version"])
            count = len(message["actions"])
            for index in range(count):
                replay.add(
                    message["states"][index], int(message["actions"][index]),
                    float(message["rewards"][index]), message["next_states"][index],
                    message["next_masks"][index], bool(message["dones"][index]),
                    float(message["discounts"][index]),
                )
            added += count
            self.stats["transitions"] += count
            self.stats["batches"] += 1
            for key in ("agent_wins", "agent_losses", "draws"):
                self.stats[key] += int(message[key])
            self.stats["blocked_seconds"] += float(message["blocked_seconds"])
        self._report_status()
        return added

    def _report_status(self) -> None:
        while True:
            try:
                event = self.status_queue.get_nowait()
            except queue.Empty:
                return
            if event["type"] == "queue_put_timeout":
                self.stats["queue_put_timeouts"] += 1
                print(
                    "WARNING: rollout-B queue put timed out "
                    f"actor={event['actor_id']} "
                    f"timeout={event['timeout_seconds']:.1f}s",
                    flush=True,
                )
            elif event["type"] == "error":
                self.stats["failures"] += 1
                print(
                    f"WARNING: rollout-B actor {event['actor_id']} failed; "
                    f"rollout A will continue:\n{event['traceback']}",
                    flush=True,
                )

    def sync_policy(self, version: int, state_dict: StateDict) -> None:
        for process, control in zip(self.processes, self.control_queues):
            if process.is_alive():
                try:
                    control.put_nowait(("policy", int(version), state_dict))
                except queue.Full:
                    # The actor will consume the already-pending snapshot at
                    # its next batch boundary.  A later sync will then enqueue
                    # a fresh snapshot without building an obsolete backlog.
                    pass

    def queue_size(self) -> int:
        try:
            return max(0, int(self.result_queue.qsize()))
        except (AttributeError, NotImplementedError):
            return -1

    def max_policy_lag(self, current_step: int) -> int:
        versions = [value for value in self.actor_versions if value >= 0]
        return current_step - min(versions) if versions else current_step

    def close(self, replay: ReplayBuffer | None = None) -> None:
        if self.closed:
            return
        self.closed = True
        self.stop_event.set()
        for control in self.control_queues:
            try:
                control.put_nowait(("stop",))
            except queue.Full:
                pass
        for process in self.processes:
            process.join(timeout=10.0)
        for process in self.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        if replay is not None:
            self.drain_into(replay)
        else:
            self._report_status()
        self.result_queue.close()
        self.status_queue.close()
        for control in self.control_queues:
            control.close()
