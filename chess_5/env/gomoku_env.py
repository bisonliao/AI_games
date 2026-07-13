from __future__ import annotations

import math
import weakref
from typing import Any, Callable, ClassVar, Dict, Literal, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv, VectorEnv


Player = Literal[-1, 1]
StartingPlayer = Literal["black", "white", "random", -1, 1]
IllegalActionMode = Literal["penalty", "raise"]
RenderMode = Literal["human", "rgb_array", "ansi"]


class GomokuEnv(gym.Env):
    """A compact Gymnasium environment for self-play Gomoku.

    Board encoding:
    -  1: black / first player
    - -1: white / second player
    -  0: empty

    The scalar reward returned by ``step`` is from the player that just acted.
    Per-color rewards are also exposed in ``info`` as ``reward_black`` and
    ``reward_white`` for agent-to-agent training loops.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "ansi"],
        "render_fps": 30,
    }

    _active_graphical_env: ClassVar[Optional[Any]] = None

    def __init__(
        self,
        board_size: int = 5,
        *,
        render_mode: Optional[RenderMode] = None,
        starting_player: StartingPlayer = "black",
        illegal_action_mode: IllegalActionMode = "penalty",
        cell_size: int = 64,
    ) -> None:
        super().__init__()

        if not isinstance(board_size, int):
            raise TypeError("board_size must be an integer.")
        if board_size < 5 or board_size > 9:
            raise ValueError("board_size must be between 5 and 9.")
        if render_mode not in (None, "human", "rgb_array", "ansi"):
            raise ValueError(f"Unsupported render_mode: {render_mode!r}")
        if illegal_action_mode not in ("penalty", "raise"):
            raise ValueError("illegal_action_mode must be 'penalty' or 'raise'.")

        if render_mode in ("human", "rgb_array"):
            self._reserve_graphical_slot()

        self.board_size = board_size
        self.win_length = 5
        self.render_mode = render_mode
        self.starting_player = starting_player
        self.illegal_action_mode = illegal_action_mode
        self.cell_size = int(cell_size)

        self.action_space = spaces.Discrete(board_size * board_size)
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(
                    low=-1,
                    high=1,
                    shape=(board_size, board_size),
                    dtype=np.int8,
                ),
                "current_player": spaces.Box(
                    low=-1,
                    high=1,
                    shape=(1,),
                    dtype=np.int8,
                ),
                "action_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(board_size * board_size,),
                    dtype=np.int8,
                ),
            }
        )

        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.current_player: Player = 1
        self.move_count = 0
        self.last_action: Optional[int] = None
        self.last_move: Optional[Tuple[int, int]] = None
        self.winner = 0
        self._terminated = False

        self._pygame: Optional[Any] = None
        self._screen: Optional[Any] = None
        self._surface: Optional[Any] = None
        self._clock: Optional[Any] = None
        self._font: Optional[Any] = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)

        self.board.fill(0)
        self.current_player = self._resolve_starting_player(
            None if options is None else options.get("starting_player")
        )
        self.move_count = 0
        self.last_action = None
        self.last_move = None
        self.winner = 0
        self._terminated = False

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), self._get_info()

    def step(
        self,
        action: int,
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        if self._terminated:
            raise RuntimeError("Cannot call step() after the episode is done. Call reset().")

        action = int(action)
        acting_player = self.current_player

        if not self.is_valid_action(action):
            if self.illegal_action_mode == "raise":
                raise ValueError(f"Illegal Gomoku action: {action}")
            self._terminated = True
            self.winner = -acting_player
            reward = -1.0
            info = self._get_info(
                invalid_action=True,
                reward_black=-1.0 if acting_player == 1 else 1.0,
                reward_white=-1.0 if acting_player == -1 else 1.0,
            )
            if self.render_mode == "human":
                self.render()
            return self._get_obs(), reward, True, False, info

        row, col = self.action_to_coord(action)
        self.board[row, col] = acting_player
        self.move_count += 1
        self.last_action = action
        self.last_move = (row, col)

        reward_black = 0.0
        reward_white = 0.0
        reward = 0.0

        if self._has_five_from(row, col, acting_player):
            self._terminated = True
            self.winner = acting_player
            reward = 1.0
            reward_black = 1.0 if acting_player == 1 else -1.0
            reward_white = 1.0 if acting_player == -1 else -1.0
        elif self.move_count == self.board_size * self.board_size:
            self._terminated = True
            self.winner = 0
        else:
            self.current_player = -acting_player  # type: ignore[assignment]

        if self.render_mode == "human":
            self.render()

        info = self._get_info(
            reward_black=reward_black,
            reward_white=reward_white,
        )
        return self._get_obs(), reward, self._terminated, False, info

    def render(self) -> Union[np.ndarray, str, None]:
        if self.render_mode is None:
            return None
        if self.render_mode == "ansi":
            return self.render_text()
        return self._render_pygame()

    def close(self) -> None:
        if self._pygame is not None:
            if self._screen is not None:
                self._pygame.display.quit()
            self._pygame.quit()

        self._pygame = None
        self._screen = None
        self._surface = None
        self._clock = None
        self._font = None

        active = self._active_graphical_env() if self._active_graphical_env else None
        if active is self:
            type(self)._active_graphical_env = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @property
    def legal_actions(self) -> np.ndarray:
        return np.flatnonzero(self.board.reshape(-1) == 0).astype(np.int64)

    def is_valid_action(self, action: int) -> bool:
        if action < 0 or action >= self.action_space.n:
            return False
        row, col = self.action_to_coord(action)
        return self.board[row, col] == 0

    def action_to_coord(self, action: int) -> Tuple[int, int]:
        return divmod(int(action), self.board_size)

    def coord_to_action(self, row: int, col: int) -> int:
        if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
            raise ValueError(f"Coordinate out of bounds: {(row, col)}")
        return row * self.board_size + col

    def wait_for_human_action(self) -> int:
        """Block until the user clicks a legal cell in a human-rendered window."""

        if self.render_mode != "human":
            raise RuntimeError("wait_for_human_action() requires render_mode='human'.")
        pygame = self._require_pygame()
        self.render()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    raise KeyboardInterrupt("Gomoku window closed.")
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    action = self._pixel_to_action(event.pos)
                    if action is not None and self.is_valid_action(action):
                        return action

            if self._clock is not None:
                self._clock.tick(self.metadata["render_fps"])

    def render_text(self) -> str:
        symbols = {1: "X", -1: "O", 0: "."}
        rows = ["  " + " ".join(str(i) for i in range(self.board_size))]
        for row in range(self.board_size):
            cells = " ".join(symbols[int(v)] for v in self.board[row])
            rows.append(f"{row} {cells}")
        if self._terminated:
            if self.winner == 1:
                rows.append("winner: black")
            elif self.winner == -1:
                rows.append("winner: white")
            else:
                rows.append("winner: draw")
        else:
            rows.append(f"current_player: {'black' if self.current_player == 1 else 'white'}")
        return "\n".join(rows)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        return {
            "board": self.board.copy(),
            "current_player": np.array([self.current_player], dtype=np.int8),
            "action_mask": self._action_mask(),
        }

    def _get_info(
        self,
        *,
        invalid_action: bool = False,
        reward_black: float = 0.0,
        reward_white: float = 0.0,
    ) -> Dict[str, Any]:
        # VectorEnv stacks values with the same key into one NumPy array, so
        # info values must keep a stable shape across games.  The public
        # ``legal_actions`` property is intentionally variable-length; pad its
        # info representation with -1 instead.
        legal_actions = np.full(self.action_space.n, -1, dtype=np.int64)
        current_legal_actions = self.legal_actions
        legal_actions[: current_legal_actions.size] = current_legal_actions
        return {
            "action_mask": self._action_mask(),
            "legal_actions": legal_actions,
            "current_player": int(self.current_player),
            "last_action": -1 if self.last_action is None else int(self.last_action),
            "last_move": np.array(
                [-1, -1] if self.last_move is None else self.last_move,
                dtype=np.int16,
            ),
            "winner": int(self.winner),
            "move_count": self.move_count,
            "invalid_action": invalid_action,
            "reward_black": reward_black,
            "reward_white": reward_white,
        }

    def _action_mask(self) -> np.ndarray:
        return (self.board.reshape(-1) == 0).astype(np.int8)

    def _resolve_starting_player(self, override: Optional[Any] = None) -> Player:
        value = self.starting_player if override is None else override
        if value in ("black", 1):
            return 1
        if value in ("white", -1):
            return -1
        if value == "random":
            return 1 if self.np_random.integers(0, 2) == 0 else -1
        raise ValueError("starting_player must be 'black', 'white', 'random', 1, or -1.")

    def _has_five_from(self, row: int, col: int, player: Player) -> bool:
        for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
            count = 1
            count += self._count_direction(row, col, dr, dc, player)
            count += self._count_direction(row, col, -dr, -dc, player)
            if count >= self.win_length:
                return True
        return False

    def _count_direction(
        self,
        row: int,
        col: int,
        dr: int,
        dc: int,
        player: Player,
    ) -> int:
        count = 0
        row += dr
        col += dc
        while (
            0 <= row < self.board_size
            and 0 <= col < self.board_size
            and self.board[row, col] == player
        ):
            count += 1
            row += dr
            col += dc
        return count

    def _reserve_graphical_slot(self) -> None:
        active_ref = type(self)._active_graphical_env
        active = active_ref() if active_ref is not None else None
        if active is not None:
            raise RuntimeError(
                "Only one graphical GomokuEnv can exist at a time. "
                "Use render_mode=None for parallel training."
            )
        type(self)._active_graphical_env = weakref.ref(self)

    def _require_pygame(self) -> Any:
        if self._pygame is None:
            try:
                import pygame
            except ImportError as exc:
                raise DependencyNotInstalled(
                    "pygame is required for render_mode='human' or 'rgb_array'."
                ) from exc
            self._pygame = pygame
        return self._pygame

    def _render_pygame(self) -> Optional[np.ndarray]:
        pygame = self._require_pygame()
        pygame.init()

        width, height = self._window_size()
        if self._surface is None:
            self._surface = pygame.Surface((width, height))
        if self.render_mode == "human" and self._screen is None:
            pygame.display.init()
            pygame.display.set_caption("Gomoku")
            self._screen = pygame.display.set_mode((width, height))
        if self._clock is None:
            self._clock = pygame.time.Clock()
        if self._font is None:
            self._font = pygame.font.SysFont("sans", 20)

        self._draw_board()

        if self.render_mode == "human":
            pygame.event.pump()
            self._screen.blit(self._surface, (0, 0))
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
            return None

        return np.transpose(
            pygame.surfarray.array3d(self._surface),
            axes=(1, 0, 2),
        )

    def _draw_board(self) -> None:
        pygame = self._require_pygame()
        surface = self._surface
        assert surface is not None

        bg = (238, 190, 116)
        line = (65, 45, 25)
        black = (15, 15, 15)
        white = (240, 240, 232)
        red = (190, 45, 45)
        text = (35, 30, 25)

        surface.fill(bg)

        status = self._status_text()
        if self._font is not None:
            label = self._font.render(status, True, text)
            surface.blit(label, (16, 14))

        origin_x, origin_y = self._board_origin()
        last_row, last_col = self.last_move if self.last_move is not None else (-1, -1)

        for i in range(self.board_size):
            x = origin_x + i * self.cell_size
            y = origin_y + i * self.cell_size
            pygame.draw.line(
                surface,
                line,
                (origin_x, y),
                (origin_x + (self.board_size - 1) * self.cell_size, y),
                2,
            )
            pygame.draw.line(
                surface,
                line,
                (x, origin_y),
                (x, origin_y + (self.board_size - 1) * self.cell_size),
                2,
            )

        if self.board_size >= 7:
            center = self.board_size // 2
            for row, col in ((center, center),):
                pos = (
                    origin_x + col * self.cell_size,
                    origin_y + row * self.cell_size,
                )
                pygame.draw.circle(surface, line, pos, 4)

        radius = max(12, math.floor(self.cell_size * 0.36))
        for row in range(self.board_size):
            for col in range(self.board_size):
                value = int(self.board[row, col])
                if value == 0:
                    continue
                center = (
                    origin_x + col * self.cell_size,
                    origin_y + row * self.cell_size,
                )
                color = black if value == 1 else white
                pygame.draw.circle(surface, color, center, radius)
                pygame.draw.circle(surface, line, center, radius, 1)
                if row == last_row and col == last_col:
                    pygame.draw.circle(surface, red, center, max(4, radius // 4))

    def _status_text(self) -> str:
        if self._terminated:
            if self.winner == 1:
                return "Black wins"
            if self.winner == -1:
                return "White wins"
            return "Draw"
        return "Black to move" if self.current_player == 1 else "White to move"

    def _window_size(self) -> Tuple[int, int]:
        margin = self._margin()
        status_height = self._status_height()
        board_span = (self.board_size - 1) * self.cell_size
        return (
            margin * 2 + board_span,
            status_height + margin * 2 + board_span,
        )

    def _board_origin(self) -> Tuple[int, int]:
        margin = self._margin()
        return margin, self._status_height() + margin

    def _margin(self) -> int:
        return max(32, self.cell_size // 2)

    def _status_height(self) -> int:
        return 44

    def _pixel_to_action(self, pos: Tuple[int, int]) -> Optional[int]:
        x, y = pos
        origin_x, origin_y = self._board_origin()
        col = round((x - origin_x) / self.cell_size)
        row = round((y - origin_y) / self.cell_size)
        if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
            return None

        cell_x = origin_x + col * self.cell_size
        cell_y = origin_y + row * self.cell_size
        threshold = self.cell_size * 0.42
        if abs(x - cell_x) > threshold or abs(y - cell_y) > threshold:
            return None
        return self.coord_to_action(row, col)


def make_gomoku_env(
    *,
    board_size: int = 5,
    render_mode: Optional[RenderMode] = None,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> Callable[[], GomokuEnv]:
    """Return a thunk suitable for Gymnasium vector environments."""

    def _factory() -> GomokuEnv:
        env = GomokuEnv(board_size=board_size, render_mode=render_mode, **kwargs)
        if seed is not None:
            env.reset(seed=seed)
        return env

    return _factory


def make_vector_env(
    num_envs: int,
    *,
    board_size: int = 5,
    asynchronous: bool = False,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> VectorEnv:
    """Create several non-rendering Gomoku environments for parallel training."""

    if num_envs < 1:
        raise ValueError("num_envs must be at least 1.")
    if kwargs.get("render_mode") is not None:
        raise ValueError("Vector Gomoku environments must use render_mode=None.")

    env_fns = [
        make_gomoku_env(
            board_size=board_size,
            seed=None if seed is None else seed + rank,
            render_mode=None,
            **kwargs,
        )
        for rank in range(num_envs)
    ]
    vector_cls = AsyncVectorEnv if asynchronous else SyncVectorEnv
    # The training loop consumes the terminal transition from ``final_info``
    # and immediately continues with the reset observation.  Gymnasium 1.x
    # defaults to NextStep, which exposes the terminal observation for one
    # extra iteration (with an empty action mask).
    return vector_cls(env_fns, autoreset_mode="SameStep")
