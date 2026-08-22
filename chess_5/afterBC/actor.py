"""CPU rollout actors for white-vs-frozen-BC interaction."""

from __future__ import annotations

import queue
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from gomoku_env import GomokuEnv

from .common import (
    GatherPermit,
    RolloutSummary,
    Transition,
    WeightSnapshot,
    actor_epsilon,
    blocking_put,
    pack_transitions,
)
from .network import NetworkPolicy, make_bc_policy, make_dueling_from_bc
from .opponent import controlled_black_actions
from .returns import NStepAccumulator


class WhiteBatchRollout:
    """A batched environment whose externally visible decisions are all white moves."""

    def __init__(
        self,
        *,
        num_envs: int,
        board_size: int,
        black_policy: Any,
        white_policy: Any,
        n_step: int,
        gamma: float,
        seed: int,
        black_stochastic: bool = False,
    ) -> None:
        self.num_envs = int(num_envs)
        self.board_size = int(board_size)
        self.action_dim = self.board_size * self.board_size
        self.black_policy = black_policy
        self.white_policy = white_policy
        self.black_stochastic = bool(black_stochastic)
        self.black_rngs = [
            np.random.default_rng(seed + 50_000_003 + index * 1_000_003)
            for index in range(num_envs)
        ]
        self.envs = [
            GomokuEnv(board_size=board_size, starting_player="black", illegal_action_mode="raise")
            for _ in range(num_envs)
        ]
        self.n_step = NStepAccumulator(num_envs, n_step, gamma)
        self.episode_white_moves = np.zeros(num_envs, dtype=np.int32)
        self.completed: list[tuple[int, int, float]] = []
        for index, env in enumerate(self.envs):
            env.reset(seed=seed + index)
        self._play_opening_moves(range(num_envs))

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def _batch_position(self, indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.stack([self.envs[index].board for index in indices]),
            np.stack([(self.envs[index].board.reshape(-1) == 0) for index in indices]),
        )

    def _black_actions(self, indices: Sequence[int]) -> np.ndarray:
        index_list = list(indices)
        boards, masks = self._batch_position(index_list)
        if self.black_stochastic:
            return controlled_black_actions(
                self.black_policy, boards, masks,
                [self.black_rngs[index] for index in index_list], stochastic=True,
            )
        return self.black_policy.select_actions(
            boards, np.ones(len(index_list), dtype=np.int8), masks, epsilon=0.0
        )

    def _play_opening_moves(self, indices: Sequence[int]) -> None:
        index_list = list(indices)
        if not index_list:
            return
        actions = self._black_actions(index_list)
        for index, action in zip(index_list, actions):
            env = self.envs[index]
            if int(env.current_player) != 1:
                raise RuntimeError(f"env {index} is not waiting for a black opening move")
            _, _, terminated, truncated, _ = env.step(int(action))
            if terminated or truncated:
                raise RuntimeError("a Gomoku game cannot terminate on its first move")
            if int(env.current_player) != -1:
                raise RuntimeError(f"env {index} did not advance to white")

    def _finish_episode(self, env_index: int, winner: int, reward: float) -> None:
        self.completed.append((int(winner), int(self.episode_white_moves[env_index]), float(reward)))
        self.episode_white_moves[env_index] = 0

    def advance(self, epsilon: float) -> list[Transition]:
        for index, env in enumerate(self.envs):
            if int(env.current_player) != -1 or env._terminated:  # noqa: SLF001 - invariant check
                raise RuntimeError(f"env {index} is not ready for a white decision")

        states, masks = self._batch_position(range(self.num_envs))
        white_actions = self.white_policy.select_actions(
            states, -np.ones(self.num_envs, dtype=np.int8), masks, epsilon=epsilon
        )
        emitted: list[Transition] = []
        pending_black: list[int] = []
        reset_after_white: list[int] = []

        for index, action in enumerate(white_actions):
            env = self.envs[index]
            self.episode_white_moves[index] += 1
            _, _, terminated, truncated, info = env.step(int(action))
            done = bool(terminated or truncated)
            if done:
                reward = float(info["reward_white"])
                raw = Transition(
                    state=states[index].copy(), action=int(action), reward=reward,
                    next_state=np.zeros_like(states[index]),
                    next_mask=np.zeros(self.action_dim, dtype=np.bool_), done=True,
                )
                emitted.extend(self.n_step.add(index, raw))
                self._finish_episode(index, int(info["winner"]), reward)
                env.reset()
                reset_after_white.append(index)
            else:
                pending_black.append(index)

        black_indices = pending_black + reset_after_white
        black_actions = self._black_actions(black_indices)
        reset_after_black: list[int] = []
        pending_set = set(pending_black)
        state_by_index = {index: states[index] for index in pending_black}
        action_by_index = {index: int(white_actions[index]) for index in pending_black}
        for index, black_action in zip(black_indices, black_actions):
            env = self.envs[index]
            _, _, terminated, truncated, info = env.step(int(black_action))
            if index not in pending_set:
                if terminated or truncated:
                    raise RuntimeError("a reset game terminated on the black opening move")
                continue
            done = bool(terminated or truncated)
            if done:
                reward = float(info["reward_white"])
                next_state = np.zeros((self.board_size, self.board_size), dtype=np.int8)
                next_mask = np.zeros(self.action_dim, dtype=np.bool_)
                self._finish_episode(index, int(info["winner"]), reward)
                env.reset()
                reset_after_black.append(index)
            else:
                reward = 0.0
                next_state = env.board.copy()
                next_mask = (env.board.reshape(-1) == 0)
                if int(env.current_player) != -1:
                    raise RuntimeError("black response did not return control to white")
            raw = Transition(
                state=state_by_index[index].copy(),
                action=action_by_index[index],
                reward=reward,
                next_state=next_state,
                next_mask=next_mask,
                done=done,
            )
            emitted.extend(self.n_step.add(index, raw))

        self._play_opening_moves(reset_after_black)
        return emitted

    def take_completed(self) -> list[tuple[int, int, float]]:
        completed, self.completed = self.completed, []
        return completed


def _drain_latest_weights(weight_queue: Any, policy: NetworkPolicy,
                          current_version: int) -> int:
    latest: WeightSnapshot | None = None
    while True:
        try:
            candidate = weight_queue.get_nowait()
        except queue.Empty:
            break
        if not isinstance(candidate, WeightSnapshot):
            raise TypeError(f"unexpected weight message: {type(candidate)!r}")
        latest = candidate
    if latest is not None and latest.version >= current_version:
        policy.load_numpy_weights(latest.state_dict)
        return int(latest.version)
    return current_version


def _wait_for_initial_weights(weight_queue: Any, policy: NetworkPolicy,
                              stop_event: Any) -> int:
    while True:
        try:
            snapshot = weight_queue.get(timeout=1.0)
        except queue.Empty:
            if stop_event.is_set():
                raise RuntimeError("pipeline stopped before initial actor weights arrived")
            continue
        if not isinstance(snapshot, WeightSnapshot):
            raise TypeError(f"unexpected initial weight message: {type(snapshot)!r}")
        policy.load_numpy_weights(snapshot.state_dict)
        return int(snapshot.version)


def _wait_for_permit(permit_queue: Any, weight_queue: Any, policy: NetworkPolicy,
                     current_version: int, stop_event: Any) -> tuple[GatherPermit, int]:
    while True:
        current_version = _drain_latest_weights(weight_queue, policy, current_version)
        try:
            permit = permit_queue.get(timeout=1.0)
        except queue.Empty:
            if stop_event.is_set():
                return GatherPermit("stop", 0), current_version
            continue
        if not isinstance(permit, GatherPermit):
            raise TypeError(f"unexpected gather permit: {type(permit)!r}")
        current_version = _drain_latest_weights(weight_queue, policy, current_version)
        return permit, current_version


def actor_worker(
    actor_id: int,
    num_actors: int,
    num_envs: int,
    board_size: int,
    actor_batch_size: int,
    n_step: int,
    gamma: float,
    seed: int,
    torch_threads: int,
    bc_checkpoint: Path,
    transition_queue: Any,
    permit_queue: Any,
    weight_queue: Any,
    rollout_queue: Any,
    error_queue: Any,
    stop_event: Any,
) -> None:
    rollout: WhiteBatchRollout | None = None
    try:
        if torch_threads < 1:
            raise ValueError("actor torch_threads must be positive")
        torch.set_num_threads(torch_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        black_model, _ = make_bc_policy(bc_checkpoint, device="cpu")
        white_model, _ = make_dueling_from_bc(bc_checkpoint, device="cpu")
        black_policy = NetworkPolicy(black_model, board_size, device="cpu", seed=seed + 17)
        white_policy = NetworkPolicy(white_model, board_size, device="cpu", seed=seed + 31)
        rollout = WhiteBatchRollout(
            num_envs=num_envs, board_size=board_size,
            black_policy=black_policy, white_policy=white_policy,
            n_step=n_step, gamma=gamma, seed=seed, black_stochastic=True,
        )
        carry: list[Transition] = []
        # A resume snapshot may differ substantially from BC_BEST. Do not let
        # multiprocessing queue feeder latency produce one stale initial packet.
        policy_version = _wait_for_initial_weights(weight_queue, white_policy, stop_event)
        permit, policy_version = _wait_for_permit(
            permit_queue, weight_queue, white_policy, policy_version, stop_event
        )
        while permit.kind == "continue" and not stop_event.is_set():
            schedule_step = int(permit.global_step)
            epsilon = actor_epsilon(actor_id, num_actors, schedule_step)
            started = time.perf_counter()
            while len(carry) < actor_batch_size:
                carry.extend(rollout.advance(epsilon))
            selected = carry[:actor_batch_size]
            del carry[:actor_batch_size]
            collection_seconds = time.perf_counter() - started
            packet = pack_transitions(
                selected, actor_id=actor_id, policy_version=policy_version,
                epsilon=epsilon, schedule_step=schedule_step,
            )
            blocked_seconds = blocking_put(transition_queue, packet, stop_event)
            episodes = rollout.take_completed()
            summary = RolloutSummary(
                actor_id=actor_id,
                policy_version=policy_version,
                epsilon=epsilon,
                schedule_step=schedule_step,
                transitions=actor_batch_size,
                episodes=len(episodes),
                white_wins=sum(winner == -1 for winner, _, _ in episodes),
                white_losses=sum(winner == 1 for winner, _, _ in episodes),
                draws=sum(winner == 0 for winner, _, _ in episodes),
                return_sum=sum(reward for _, _, reward in episodes),
                white_move_sum=sum(length for _, length, _ in episodes),
                collection_seconds=collection_seconds,
                blocked_seconds=blocked_seconds,
            )
            blocking_put(rollout_queue, summary, stop_event)
            permit, policy_version = _wait_for_permit(
                permit_queue, weight_queue, white_policy, policy_version, stop_event
            )
            if permit.kind not in {"continue", "stop"}:
                raise ValueError(f"unknown gather permit: {permit.kind}")
    except BaseException:
        try:
            error_queue.put({"worker": f"actor-{actor_id}", "traceback": traceback.format_exc()})
        finally:
            stop_event.set()
    finally:
        if rollout is not None:
            rollout.close()
