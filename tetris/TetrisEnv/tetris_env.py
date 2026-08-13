"""Gymnasium environment for a standard, frame-controlled Tetris game."""
from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .board import Board, BoardFeatures
from .pieces import PIECES, ActivePiece, one_hot, spawn_piece
from .reward import shaped_reward


class SevenBag:
    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self._queue: deque[str] = deque()

    def reset(self) -> None:
        self._queue.clear()

    def next(self) -> str:
        if not self._queue:
            bag = list(PIECES)
            self.rng.shuffle(bag)
            self._queue.extend(bag)
        return self._queue.popleft()


class TetrisEnv(gym.Env[dict[str, np.ndarray], int]):
    """A 10x20 SRS Tetris environment with six frame-level actions."""

    metadata = {"render_modes": ["ansi"]}
    ACTION_NOOP = 0
    ACTION_LEFT = 1
    ACTION_RIGHT = 2
    ACTION_ROTATE_CW = 3
    ACTION_SOFT_DROP = 4
    ACTION_HARD_DROP = 5

    def __init__(
        self,
        *,
        gravity_period: int = 2,
        gamma: float = 0.99,
        piece_placed_reward: float = 0.01,
        line_clear_reward: float = 0.75,
        terminal_penalty: float = 1.0,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if gravity_period < 1:
            raise ValueError("gravity_period must be positive")
        self.gravity_period = gravity_period
        self.gamma = gamma
        self.piece_placed_reward = float(piece_placed_reward)
        self.line_clear_reward = float(line_clear_reward)
        self.terminal_penalty = float(terminal_penalty)
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(0, 1, shape=(20, 10), dtype=np.uint8),
                "active": spaces.Box(0, 1, shape=(20, 10), dtype=np.uint8),
                "current_piece": spaces.MultiBinary(7),
                "next_piece": spaces.MultiBinary(7),
                "rotation": spaces.MultiBinary(4),
                "position": spaces.Box(
                    low=np.asarray([-4, -4], dtype=np.float32),
                    high=np.asarray([10, 20], dtype=np.float32),
                    dtype=np.float32,
                ),
            }
        )
        self.board = Board()
        self.rng = np.random.default_rng()
        self.bag = SevenBag(self.rng)
        self.current: ActivePiece | None = None
        self.next_kind = PIECES[0]
        self._ticks = 0
        self._episode_return = 0.0
        self._episode_length = 0
        self._survival_pieces = 0
        self._terminated = False

    def _spawn(self) -> bool:
        self.current = spawn_piece(self.next_kind)
        self.next_kind = self.bag.next()
        return not self.board.collides(self.current)

    def _features(self) -> BoardFeatures:
        return self.board.features()

    def _observation(self) -> dict[str, np.ndarray]:
        assert self.current is not None
        rotation = np.zeros(4, dtype=np.int8)
        rotation[self.current.rotation] = 1
        return {
            "board": self.board.grid.copy(),
            "active": self.board.active_mask(self.current),
            "current_piece": one_hot(self.current.kind),
            "next_piece": one_hot(self.next_kind),
            "rotation": rotation,
            "position": np.asarray([self.current.x, self.current.y], dtype=np.float32),
        }

    def _info(self, *, lines_cleared: int = 0, piece_placed: bool = False) -> dict[str, Any]:
        features = self._features()
        return {
            "lines_cleared": int(lines_cleared),
            "piece_placed": bool(piece_placed),
            "episode_return": float(self._episode_return),
            "episode_length": int(self._episode_length),
            "survival_pieces": int(self._survival_pieces),
            "aggregate_height": int(features.aggregate_height),
            "board_height": int(features.max_height),
            "holes": int(features.holes),
            "bumpiness": int(features.bumpiness),
            "wells": int(features.wells),
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        # Gymnasium assigns self.np_random during super().reset. Re-seed the bag from it.
        self.rng = np.random.default_rng(int(self.np_random.integers(0, 2**63 - 1)))
        self.bag = SevenBag(self.rng)
        self.bag.reset()
        self.board.reset()
        self._ticks = 0
        self._episode_return = 0.0
        self._episode_length = 0
        self._survival_pieces = 0
        self._terminated = False
        self.next_kind = self.bag.next()
        if not self._spawn():
            self._terminated = True
        return self._observation(), self._info()

    def _lock_current(self) -> tuple[int, bool]:
        assert self.current is not None
        visible = self.board.lock(self.current)
        lines = self.board.clear_lines()
        self._survival_pieces += 1
        if not visible or not self._spawn():
            self._terminated = True
        return lines, self._terminated

    def step(self, action: int):
        if self.current is None:
            raise RuntimeError("reset() must be called before step()")
        if self._terminated:
            raise RuntimeError("step() called after episode termination; call reset()")
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action}")
        previous_features = self._features()
        placed = False
        lines = 0

        if action == self.ACTION_LEFT:
            moved = self.board.try_move(self.current, dx=-1)
            if moved is not None:
                self.current = moved
        elif action == self.ACTION_RIGHT:
            moved = self.board.try_move(self.current, dx=1)
            if moved is not None:
                self.current = moved
        elif action == self.ACTION_ROTATE_CW:
            rotated = self.board.try_rotate_cw(self.current)
            if rotated is not None:
                self.current = rotated

        if action == self.ACTION_HARD_DROP:
            while True:
                moved = self.board.try_move(self.current, dy=1)
                if moved is None:
                    break
                self.current = moved
            lines, _ = self._lock_current()
            placed = True
        else:
            if action == self.ACTION_SOFT_DROP:
                moved = self.board.try_move(self.current, dy=1)
                if moved is not None:
                    self.current = moved
                else:
                    lines, _ = self._lock_current()
                    placed = True
            self._ticks += 1
            if not placed and self._ticks % self.gravity_period == 0:
                moved = self.board.try_move(self.current, dy=1)
                if moved is not None:
                    self.current = moved
                else:
                    lines, _ = self._lock_current()
                    placed = True

        self._episode_length += 1
        reward = shaped_reward(
            piece_placed=placed,
            lines_cleared=lines,
            terminated=self._terminated,
            previous=previous_features,
            current=self._features(),
            gamma=self.gamma,
            apply_potential=placed or self._terminated,
            piece_placed_reward=self.piece_placed_reward,
            line_clear_reward=self.line_clear_reward,
            terminal_penalty=self.terminal_penalty,
        )
        self._episode_return += reward
        info = self._info(lines_cleared=lines, piece_placed=placed)
        if self.render_mode == "ansi":
            info["render"] = self.render()
        return self._observation(), reward, self._terminated, False, info

    def render(self) -> str:
        visible = self.board.grid.copy()
        if self.current is not None:
            for x, y in self.current.absolute_cells():
                if 0 <= y < self.board.height and 0 <= x < self.board.width:
                    visible[y, x] = 2
        return "\n".join("".join("#" if cell == 1 else "@" if cell == 2 else "." for cell in row) for row in visible)

    def close(self) -> None:
        self.current = None
