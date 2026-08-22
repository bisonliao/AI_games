"""Independent CPU checkpoint evaluation against the frozen BC_BEST black player."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from gomoku_env import GomokuEnv

from .common import EvaluationResult
from .network import DuelingGomokuQNet, NetworkPolicy, make_bc_policy
from .opponent import controlled_black_actions, protocol_description


def _positions(envs: Sequence[GomokuEnv], indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.stack([envs[index].board for index in indices]),
        np.stack([envs[index].board.reshape(-1) == 0 for index in indices]),
    )


def play_games(
    white_policy: NetworkPolicy,
    black_policy: NetworkPolicy,
    *,
    board_size: int,
    games: int,
    seed: int,
    stochastic_black: bool,
) -> dict[str, Any]:
    if games < 1:
        raise ValueError("games must be positive")
    envs = [
        GomokuEnv(board_size=board_size, starting_player="black", illegal_action_mode="raise")
        for _ in range(games)
    ]
    rngs = [np.random.default_rng(seed + index * 1_000_003) for index in range(games)]
    results = np.full(games, 2, dtype=np.int8)
    move_counts = np.zeros(games, dtype=np.int16)
    try:
        for index, env in enumerate(envs):
            env.reset(seed=seed + index)
        active = list(range(games))
        boards, masks = _positions(envs, active)
        black_actions = controlled_black_actions(
            black_policy, boards, masks, rngs, stochastic=stochastic_black
        )
        for index, action in zip(active, black_actions):
            envs[index].step(int(action))
            move_counts[index] += 1

        while active:
            boards, masks = _positions(envs, active)
            white_actions = white_policy.select_actions(
                boards, -np.ones(len(active), dtype=np.int8), masks, epsilon=0.0
            )
            waiting_for_black: list[int] = []
            for index, action in zip(active, white_actions):
                _, _, terminated, truncated, info = envs[index].step(int(action))
                move_counts[index] += 1
                if terminated or truncated:
                    results[index] = int(info["winner"])
                else:
                    waiting_for_black.append(index)
            if not waiting_for_black:
                break
            boards, masks = _positions(envs, waiting_for_black)
            black_actions = controlled_black_actions(
                black_policy, boards, masks, [rngs[index] for index in waiting_for_black],
                stochastic=stochastic_black,
            )
            next_active: list[int] = []
            for index, action in zip(waiting_for_black, black_actions):
                _, _, terminated, truncated, info = envs[index].step(int(action))
                move_counts[index] += 1
                if terminated or truncated:
                    results[index] = int(info["winner"])
                else:
                    next_active.append(index)
            active = next_active
    finally:
        for env in envs:
            env.close()
    if np.any(results == 2):
        raise RuntimeError("evaluation ended with unfinished games")
    wins = int(np.count_nonzero(results == -1))
    losses = int(np.count_nonzero(results == 1))
    draws = int(np.count_nonzero(results == 0))
    return {
        "games": games,
        "white_wins": wins,
        "white_losses": losses,
        "draws": draws,
        "white_win_rate": wins / games,
        "white_loss_rate": losses / games,
        "draw_rate": draws / games,
        "white_score_rate": (wins + 0.5 * draws) / games,
        "mean_moves": float(move_counts.mean()),
    }


def load_white_policy(checkpoint_path: Path, *, board_size: int = 9) -> NetworkPolicy:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if int(checkpoint.get("board_size", -1)) != board_size:
        raise ValueError("evaluation checkpoint board size does not match")
    model = DuelingGomokuQNet(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["online_state_dict"])
    return NetworkPolicy(model, board_size, device="cpu", seed=0)


def evaluate_checkpoint(
    checkpoint_path: Path,
    bc_checkpoint: Path,
    *,
    board_size: int = 9,
    stochastic_games: int = 128,
    seed: int = 70_000,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    white = load_white_policy(checkpoint_path, board_size=board_size)
    black_model, _ = make_bc_policy(bc_checkpoint, device="cpu")
    black = NetworkPolicy(black_model, board_size, device="cpu", seed=seed)
    deterministic = play_games(
        white, black, board_size=board_size, games=1, seed=seed,
        stochastic_black=False,
    )
    stochastic = play_games(
        white, black, board_size=board_size, games=stochastic_games,
        seed=seed, stochastic_black=True,
    )
    deterministic_winner = (
        "white" if deterministic["white_wins"] else
        "black" if deterministic["white_losses"] else "draw"
    )
    return {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "board_size": board_size,
        "deterministic": {**deterministic, "winner": deterministic_winner},
        "stochastic": stochastic,
        "success": stochastic["white_score_rate"] > 0.5,
        "deterministic_success": deterministic_winner == "white",
        "protocol": {
            "white": "greedy checkpoint",
            "black_training": protocol_description(),
            "black_stochastic": protocol_description(),
            "deterministic_audit": "greedy BC_BEST",
            "seed": seed,
        },
    }


def evaluator_worker(task_queue: Any, result_queue: Any, bc_checkpoint: Path,
                     board_size: int, stochastic_games: int, seed: int) -> None:
    while True:
        task = task_queue.get()
        if task is None:
            return
        checkpoint, step = task
        try:
            result = evaluate_checkpoint(
                Path(checkpoint), bc_checkpoint, board_size=board_size,
                stochastic_games=stochastic_games, seed=seed,
            )
            result_queue.put(EvaluationResult(str(checkpoint), int(step), result=result))
        except BaseException:
            result_queue.put(EvaluationResult(
                str(checkpoint), int(step), error=traceback.format_exc()
            ))


class AsyncEvaluator:
    def __init__(self, bc_checkpoint: Path, *, board_size: int = 9,
                 stochastic_games: int = 128, seed: int = 70_000,
                 context: Any | None = None) -> None:
        context = context or mp.get_context("spawn")
        self.task_queue = context.Queue()
        self.result_queue = context.Queue()
        self.process = context.Process(
            target=evaluator_worker,
            name="afterbc-evaluator",
            args=(self.task_queue, self.result_queue, bc_checkpoint, board_size,
                  stochastic_games, seed),
        )
        self.closed = False
        self.process.start()

    def submit(self, checkpoint: Path, step: int) -> None:
        if self.closed:
            raise RuntimeError("evaluator is closed")
        self.task_queue.put((str(checkpoint), int(step)))

    def poll(self) -> list[EvaluationResult]:
        import queue
        results: list[EvaluationResult] = []
        while True:
            try:
                results.append(self.result_queue.get_nowait())
            except queue.Empty:
                return results

    def close(self, *, drain: bool = True, timeout: float = 3600.0) -> list[EvaluationResult]:
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


def write_evaluation_json(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)
