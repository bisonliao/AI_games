"""One-tetromino-per-step training environment with masked placement actions."""
from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from .pieces import SHAPES, ActivePiece
from .reward import shaped_reward
from .tetris_env import TetrisEnv


class PlacementTetrisEnv(TetrisEnv):
    """Place a complete piece with one ``rotation x target-column`` action.

    There are always 40 network outputs.  Action ``rotation * 10 + column``
    requests a clockwise rotation state and the leftmost occupied board column.
    ``observation["action_mask"]`` marks placements that can be reached by
    rotating at spawn, moving horizontally at spawn, and then hard-dropping.

    Duplicate geometric rotations (O: three duplicates; I/S/Z: two duplicates)
    are masked so exploration is not diluted by equivalent actions.  The original
    :class:`TetrisEnv` remains unchanged as the frame-level interactive API.
    """

    ROTATIONS = 4
    TARGET_COLUMNS = 10
    NUM_ACTIONS = ROTATIONS * TARGET_COLUMNS

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.action_space = spaces.Discrete(self.NUM_ACTIONS)
        observation_spaces = dict(self.observation_space.spaces)
        observation_spaces["action_mask"] = spaces.MultiBinary(self.NUM_ACTIONS)
        self.observation_space = spaces.Dict(observation_spaces)

    @classmethod
    def encode_action(cls, rotation: int, target_column: int) -> int:
        if not 0 <= rotation < cls.ROTATIONS:
            raise ValueError(f"rotation must be in [0, {cls.ROTATIONS})")
        if not 0 <= target_column < cls.TARGET_COLUMNS:
            raise ValueError(f"target_column must be in [0, {cls.TARGET_COLUMNS})")
        return rotation * cls.TARGET_COLUMNS + target_column

    @classmethod
    def decode_action(cls, action: int) -> tuple[int, int]:
        action = int(action)
        if not 0 <= action < cls.NUM_ACTIONS:
            raise ValueError(f"invalid placement action {action}")
        return divmod(action, cls.TARGET_COLUMNS)

    @staticmethod
    def _normalized_shape(kind: str, rotation: int) -> tuple[tuple[int, int], ...]:
        cells = SHAPES[kind][rotation]
        min_x = min(x for x, _ in cells)
        min_y = min(y for _, y in cells)
        return tuple(sorted((x - min_x, y - min_y) for x, y in cells))

    def _is_canonical_rotation(self, kind: str, rotation: int) -> bool:
        shape = self._normalized_shape(kind, rotation)
        return all(self._normalized_shape(kind, prior) != shape for prior in range(rotation))

    def _planned_piece(self, action: int) -> ActivePiece | None:
        """Return the pre-drop piece for an executable action, otherwise None."""
        assert self.current is not None
        rotation, target_column = self.decode_action(action)
        if not self._is_canonical_rotation(self.current.kind, rotation):
            return None

        candidate = self.current
        for _ in range(rotation):
            rotated = self.board.try_rotate_cw(candidate)
            if rotated is None:
                return None
            candidate = rotated

        leftmost = min(x for x, _ in candidate.absolute_cells())
        distance = target_column - leftmost
        direction = 1 if distance > 0 else -1
        for _ in range(abs(distance)):
            moved = self.board.try_move(candidate, dx=direction)
            if moved is None:
                return None
            candidate = moved
        if min(x for x, _ in candidate.absolute_cells()) != target_column:
            return None
        return candidate

    def action_mask(self) -> np.ndarray:
        if self.current is None or self._terminated:
            return np.zeros(self.NUM_ACTIONS, dtype=np.int8)
        return np.asarray(
            [self._planned_piece(action) is not None for action in range(self.NUM_ACTIONS)],
            dtype=np.int8,
        )

    def _observation(self) -> dict[str, np.ndarray]:
        observation = super()._observation()
        observation["action_mask"] = self.action_mask()
        return observation

    def step(self, action: int):
        if self.current is None:
            raise RuntimeError("reset() must be called before step()")
        if self._terminated:
            raise RuntimeError("step() called after episode termination; call reset()")
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"invalid placement action {action}")

        planned = self._planned_piece(action)
        if planned is None:
            rotation, target_column = self.decode_action(action)
            raise ValueError(
                f"masked placement action {action}: rotation={rotation}, "
                f"target_column={target_column}"
            )

        previous_features = self._features()
        self.current = planned
        while True:
            moved = self.board.try_move(self.current, dy=1)
            if moved is None:
                break
            self.current = moved

        lines, _ = self._lock_current()
        self._episode_length += 1
        reward = shaped_reward(
            piece_placed=True,
            lines_cleared=lines,
            terminated=self._terminated,
            previous=previous_features,
            current=self._features(),
            gamma=self.gamma,
            apply_potential=True,
            piece_placed_reward=self.piece_placed_reward,
            line_clear_reward=self.line_clear_reward,
            terminal_penalty=self.terminal_penalty,
        )
        self._episode_return += reward
        rotation, target_column = self.decode_action(action)
        info = self._info(lines_cleared=lines, piece_placed=True)
        info.update(
            {
                "placement_rotation": rotation,
                "placement_target_column": target_column,
                "legal_action_count": int(self.action_mask().sum()),
            }
        )
        if self.render_mode == "ansi":
            info["render"] = self.render()
        return self._observation(), reward, self._terminated, False, info
