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

        while not stop_event.is_set():
            players = obs["current_player"].reshape(-1)
            actions = np.zeros(num_envs, dtype=np.int64)
            agent_turn = ((players == 1) & agent_is_black) | (
                (players == -1) & ~agent_is_black
            )
            agent_indices = np.flatnonzero(agent_turn)
            heuristic_indices = np.flatnonzero(~agent_turn)
            if len(agent_indices):
                boards, current, masks = slice_obs(obs, agent_indices)
                actions[agent_indices] = agent.select_actions(
                    boards, current, masks, epsilon=0.0,
                )
            if len(heuristic_indices):
                boards, current, masks = slice_obs(obs, heuristic_indices)
                actions[heuristic_indices] = heuristic.select_actions(
                    boards, current, masks, epsilon=0.0,
                )

            for player in (1, -1):
                indices = np.flatnonzero(players == player)
                if len(indices):
                    collectors[player].record_actions(
                        obs["board"][indices], indices, actions[indices]
                    )

            next_obs, _, terminated, truncated, infos = envs.step(actions)
            dones = np.logical_or(terminated, truncated)
            rewards = {
                1: np.zeros(num_envs, dtype=np.float32),
                -1: np.zeros(num_envs, dtype=np.float32),
            }
            for env_index in np.flatnonzero(dones):
                index = int(env_index)
                rewards[1][index] = float(
                    _info_value(infos, "reward_black", index, True, 0.0)
                )
                rewards[-1][index] = float(
                    _info_value(infos, "reward_white", index, True, 0.0)
                )
                winner = int(_info_value(infos, "winner", index, True, 0))
                agent_player = 1 if agent_is_black[index] else -1
                if winner == agent_player:
                    agent_wins += 1
                elif winner == -agent_player:
                    agent_losses += 1
                else:
                    draws += 1

            for player in (1, -1):
                transitions.extend(collectors[player].finish_step(
                    acted_players=players,
                    next_obs=next_obs,
                    dones=dones,
                    rewards=rewards[player],
                ))
            obs = next_obs

            if len(transitions) >= int(config["batch_size"]):
                message = _pack_batch(
                    transitions, actor_id=actor_id, policy_version=policy_version,
                    agent_wins=agent_wins, agent_losses=agent_losses, draws=draws,
                    blocked_seconds=blocked_seconds,
                )
                blocked_seconds = _put_batch(
                    result_queue, status_queue, message, stop_event, actor_id,
                )
                transitions = []
                agent_wins = agent_losses = draws = 0
                policy_version = _apply_policy_commands(
                    control_queue, agent, policy_version,
                )
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
        self.control_queues = [self.context.Queue() for _ in range(num_actors)]
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
                control.put(("policy", int(version), state_dict))

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
