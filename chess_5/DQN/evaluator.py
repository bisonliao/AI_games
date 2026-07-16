"""异步评测模块：用独立CPU进程按FIFO评测checkpoint，并将胜负、和棋与耗时返回训练主进程。"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from env import make_vector_env

try:
    from .agent import DQNAgent, slice_obs
    from .heuristic_agent import HeuristicAgent
except ImportError:
    from agent import DQNAgent, slice_obs
    from heuristic_agent import HeuristicAgent


def _terminal_info(infos: Dict[str, Any], key: str, index: int, default: Any) -> Any:
    mask = infos.get("_final_info")
    if mask is None or not mask[index]:
        return default
    final_infos = infos.get("final_info")
    if isinstance(final_infos, dict):
        key_mask = final_infos.get(f"_{key}")
        if key in final_infos and (key_mask is None or key_mask[index]):
            return final_infos[key][index]
    elif final_infos is not None and final_infos[index] is not None:
        return final_infos[index].get(key, default)
    return default


def evaluate_checkpoint(
    checkpoint: Path,
    *,
    board_size: int,
    num_games: int,
    seed: int,
    agent_player: int = 1,
) -> Dict[str, Any]:
    """Evaluate a greedy DQN checkpoint against a seeded heuristic opponent.

    ``agent_player`` uses the environment's player values: ``1`` means the DQN
    plays black and ``-1`` means it plays white.  Wins and losses in the returned
    result are always from the DQN checkpoint's perspective.
    """
    if num_games < 1:
        raise ValueError("num_games must be at least 1")
    if agent_player not in (1, -1):
        raise ValueError("agent_player must be 1 (black) or -1 (white)")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    agent = DQNAgent(
        board_size, replay_size=1, min_replay_size=1, batch_size=1,
        device="cpu", seed=seed,
    )
    metadata = agent.load_checkpoint(checkpoint, load_optimizer=False)
    checkpoint_board_size = int(metadata.get("board_size", board_size))
    if checkpoint_board_size != board_size:
        raise ValueError(
            f"Checkpoint board size {checkpoint_board_size} does not match {board_size}"
        )
    heuristic = HeuristicAgent(seed=seed + 100_000)
    envs = make_vector_env(
        num_games, board_size=board_size, asynchronous=False, seed=seed
    )
    wins = losses = draws = 0
    finished = np.zeros(num_games, dtype=np.bool_)
    started = time.perf_counter()
    try:
        obs, _ = envs.reset(seed=[seed + index for index in range(num_games)])
        while not np.all(finished):
            players = obs["current_player"].reshape(-1)
            actions = np.zeros(num_games, dtype=np.int64)
            agent_indices = np.flatnonzero(players == agent_player)
            heuristic_indices = np.flatnonzero(players == -agent_player)
            if len(agent_indices):
                boards, current, masks = slice_obs(obs, agent_indices)
                actions[agent_indices] = agent.select_actions(
                    boards, current, masks, epsilon=0.0
                )
            if len(heuristic_indices):
                boards, current, masks = slice_obs(obs, heuristic_indices)
                actions[heuristic_indices] = heuristic.select_actions(
                    boards, current, masks, epsilon=0.0
                )
            next_obs, _, terminated, truncated, infos = envs.step(actions)
            done = np.logical_or(terminated, truncated)
            for index in np.flatnonzero(done & ~finished):
                winner = int(_terminal_info(infos, "winner", int(index), 0))
                if winner == agent_player:
                    wins += 1
                elif winner == -agent_player:
                    losses += 1
                else:
                    draws += 1
                finished[index] = True
            obs = next_obs
    finally:
        envs.close()
    return {
        "type": "evaluation",
        "checkpoint": str(checkpoint),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "games": num_games,
        "agent_player": agent_player,
        "duration_seconds": time.perf_counter() - started,
    }


def evaluator_worker(
    task_queue: Any,
    result_queue: Any,
    board_size: int,
    num_games: int,
    seed: int,
) -> None:
    while True:
        task = task_queue.get()
        if task is None:
            return
        try:
            result = evaluate_checkpoint(
                Path(task["checkpoint"]), board_size=board_size,
                num_games=num_games, seed=seed,
            )
            result["step"] = int(task["step"])
        except BaseException:
            result = {
                "type": "evaluation_error",
                "checkpoint": str(task.get("checkpoint", "")),
                "step": int(task.get("step", 0)),
                "traceback": traceback.format_exc(),
            }
        result_queue.put(result)


class HeuristicEvaluator:
    """Own a spawned evaluator process without exposing IPC details to training."""

    def __init__(
        self,
        *,
        board_size: int,
        num_games: int = 16,
        seed: int = 0,
        context: Optional[Any] = None,
    ) -> None:
        self.context = context or mp.get_context("spawn")
        self.task_queue = self.context.Queue()
        self.result_queue = self.context.Queue()
        self.process = self.context.Process(
            target=evaluator_worker,
            name="gomoku-heuristic-evaluator",
            args=(self.task_queue, self.result_queue, board_size, num_games, seed),
        )
        self.closed = False
        self.process.start()

    def submit(self, checkpoint: Path, step: int) -> None:
        if self.closed:
            raise RuntimeError("Evaluator is already closed")
        self.task_queue.put({"checkpoint": str(checkpoint), "step": int(step)})

    def poll(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        while True:
            try:
                results.append(self.result_queue.get_nowait())
            except queue.Empty:
                return results

    def close(self, *, drain: bool = True, timeout: float = 300.0) -> List[Dict[str, Any]]:
        if self.closed:
            return self.poll()
        self.closed = True
        if drain:
            self.task_queue.put(None)
            self.process.join(timeout=timeout)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5.0)
        results = self.poll()
        self.task_queue.close()
        self.result_queue.close()
        return results
