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
        boards: np.ndarray,
        current_players: np.ndarray,
        action_masks: np.ndarray,
        epsilon: float = 0.0,
    ) -> np.ndarray:
        boards = np.asarray(boards)
        masks = np.asarray(action_masks).reshape((boards.shape[0], self.action_dim)).astype(bool)
        actions = random_legal_actions(masks, self.rng)
        greedy = np.flatnonzero(self.rng.random(boards.shape[0]) >= float(epsilon))
        if len(greedy) == 0:
            return actions
        states = encode_boards(boards[greedy], np.asarray(current_players).reshape(-1)[greedy])
        with torch.inference_mode():
            q_values = self.net(torch.as_tensor(states, device=self.device))
            mask = torch.as_tensor(masks[greedy], device=self.device)
            actions[greedy] = q_values.masked_fill(~mask, -1e9).argmax(1).cpu().numpy()
        return actions


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
    actor_id: int,
    config: Dict[str, Any],
    initial_policy: StateDict,
    model_kwargs: Mapping[str, Any],
    initial_opponent: Optional[Tuple[StateDict, Mapping[str, Any]]],
    result_queue: Any,
    status_queue: Any,
    control_queue: Any,
    stop_event: Any,
    generated_steps: Any,
) -> None:
    envs = None
    try:
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        seed = int(config["seed"]) + actor_id * 100_000
        np.random.seed(seed)
        torch.manual_seed(seed)
        black = InferencePolicy(
            config["board_size"], model_kwargs, initial_policy,
            seed=seed, device=config["actor_device"],
        )
        if initial_opponent is None:
            opponent: Any = RandomPolicy(seed + 10_000)
        else:
            opponent = InferencePolicy(
                config["board_size"], initial_opponent[1], initial_opponent[0],
                seed=seed + 10_000, device=config["actor_device"],
            )

        num_envs = int(config["envs_per_actor"])
        board_size = int(config["board_size"])
        envs = make_vector_env(num_envs, board_size=board_size, asynchronous=False, seed=seed)
        obs, _ = envs.reset(seed=[seed + index for index in range(num_envs)])
        pending_states: List[Optional[np.ndarray]] = [None] * num_envs
        pending_actions: List[Optional[int]] = [None] * num_envs
        episode_steps = np.zeros(num_envs, dtype=np.int64)
        transitions: List[Transition] = []
        n_step_accumulator = NStepAccumulator(
            num_envs, int(config["n_step"]), float(config["gamma"])
        )
        episodes: List[Tuple[int, int, float]] = []
        policy_version = 0
        blocked_seconds = 0.0

        while not stop_event.is_set():
            players_now = obs["current_player"].reshape(-1)
            actions = np.zeros(num_envs, dtype=np.int64)
            black_indices = np.flatnonzero(players_now == 1)
            white_indices = np.flatnonzero(players_now == -1)

            if len(black_indices):
                with generated_steps.get_lock():
                    epsilon_step = int(generated_steps.value)
                    generated_steps.value += len(black_indices)
                fraction = min(1.0, epsilon_step / max(1, config["epsilon_decay_steps"]))
                epsilon = config["epsilon_start"] + fraction * (
                    config["epsilon_end"] - config["epsilon_start"]
                ) if config["epsilon_decay_steps"] > 0 else config["epsilon_end"]
                boards, players, masks = slice_obs(obs, black_indices)
                selected = black.select_actions(boards, players, masks, epsilon)
                encoded = encode_boards(boards, players)
                actions[black_indices] = selected
                for row, env_index in enumerate(black_indices):
                    pending_states[env_index] = encoded[row]
                    pending_actions[env_index] = int(selected[row])
                    episode_steps[env_index] += 1

            if len(white_indices):
                boards, players, masks = slice_obs(obs, white_indices)
                actions[white_indices] = opponent.select_actions(
                    boards, players, masks, config["opponent_epsilon"]
                )

            next_obs, _, terminated, truncated, infos = envs.step(actions)
            dones = np.logical_or(terminated, truncated)
            for env_index in range(num_envs):
                acted_player = int(players_now[env_index])
                done = bool(dones[env_index])
                if acted_player == 1 and done:
                    reward = float(_info_value(infos, "reward_black", env_index, True, 0.0))
                    next_state = np.zeros((3, board_size, board_size), dtype=np.float32)
                    reward = shaped_reward(
                        reward, pending_states[env_index], next_state, True,
                        config["gamma"], config["reward_shaping_scale"],
                    )
                    transitions.extend(n_step_accumulator.add(env_index, Transition(
                        pending_states[env_index], int(pending_actions[env_index]), reward,
                        next_state, np.zeros(board_size * board_size, dtype=np.bool_), True,
                    )))
                    pending_states[env_index] = None
                    pending_actions[env_index] = None
                elif acted_player == 1:
                    continue
                elif pending_states[env_index] is not None:
                    reward = float(_info_value(infos, "reward_black", env_index, done, 0.0))
                    if done:
                        next_state = np.zeros((3, board_size, board_size), dtype=np.float32)
                        next_mask = np.zeros(board_size * board_size, dtype=np.bool_)
                    else:
                        next_state = encode_boards(
                            next_obs["board"][env_index], next_obs["current_player"][env_index]
                        )[0]
                        next_mask = next_obs["action_mask"][env_index].astype(np.bool_)
                    reward = shaped_reward(
                        reward, pending_states[env_index], next_state, done,
                        config["gamma"], config["reward_shaping_scale"],
                    )
                    transitions.extend(n_step_accumulator.add(env_index, Transition(
                        pending_states[env_index], int(pending_actions[env_index]), reward,
                        next_state, next_mask, done,
                    )))
                    pending_states[env_index] = None
                    pending_actions[env_index] = None

                if done:
                    episodes.append((
                        int(_info_value(infos, "winner", env_index, True, 0)),
                        int(episode_steps[env_index]),
                        float(_info_value(infos, "reward_black", env_index, True, 0.0)),
                    ))
                    episode_steps[env_index] = 0

            obs = next_obs
            if len(transitions) >= config["actor_batch_size"]:
                message = pack_transitions(
                    transitions, episodes, actor_id=actor_id,
                    policy_version=policy_version, blocked_seconds=blocked_seconds,
                )
                blocked_seconds = _queue_put(
                    result_queue, status_queue, message, stop_event, actor_id
                )
                transitions = []
                episodes = []
                opponent, policy_version = _apply_commands(
                    control_queue, black, opponent, board_size, seed,
                    config["actor_device"], policy_version,
                )
    except BaseException:
        message = {"type": "error", "actor_id": actor_id, "traceback": traceback.format_exc()}
        try:
            result_queue.put(message, timeout=1.0)
        except (queue.Full, ValueError):
            pass
        stop_event.set()
    finally:
        if envs is not None:
            envs.close()


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
