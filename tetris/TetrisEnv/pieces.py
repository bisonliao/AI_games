"""Tetromino definitions and SRS rotation kicks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

PIECES = ("I", "O", "T", "S", "Z", "J", "L")
PIECE_TO_ID = {name: i for i, name in enumerate(PIECES)}

# Coordinates are expressed in a 4x4 local box, with y increasing downwards.
# They follow the standard SRS orientation convention.
SHAPES: dict[str, tuple[tuple[tuple[int, int], ...], ...]] = {
    "I": (
        ((0, 1), (1, 1), (2, 1), (3, 1)),
        ((2, 0), (2, 1), (2, 2), (2, 3)),
        ((0, 2), (1, 2), (2, 2), (3, 2)),
        ((1, 0), (1, 1), (1, 2), (1, 3)),
    ),
    "O": (
        ((1, 0), (2, 0), (1, 1), (2, 1)),
        ((1, 0), (2, 0), (1, 1), (2, 1)),
        ((1, 0), (2, 0), (1, 1), (2, 1)),
        ((1, 0), (2, 0), (1, 1), (2, 1)),
    ),
    "T": (
        ((1, 0), (0, 1), (1, 1), (2, 1)),
        ((1, 0), (1, 1), (2, 1), (1, 2)),
        ((0, 1), (1, 1), (2, 1), (1, 2)),
        ((1, 0), (0, 1), (1, 1), (1, 2)),
    ),
    "J": (
        ((0, 0), (0, 1), (1, 1), (2, 1)),
        ((1, 0), (2, 0), (1, 1), (1, 2)),
        ((0, 1), (1, 1), (2, 1), (2, 2)),
        ((1, 0), (1, 1), (0, 2), (1, 2)),
    ),
    "L": (
        ((2, 0), (0, 1), (1, 1), (2, 1)),
        ((1, 0), (1, 1), (1, 2), (2, 2)),
        ((0, 1), (1, 1), (2, 1), (0, 2)),
        ((0, 0), (1, 0), (1, 1), (1, 2)),
    ),
    "S": (
        ((1, 0), (2, 0), (0, 1), (1, 1)),
        ((1, 0), (1, 1), (2, 1), (2, 2)),
        ((1, 1), (2, 1), (0, 2), (1, 2)),
        ((0, 0), (0, 1), (1, 1), (1, 2)),
    ),
    "Z": (
        ((0, 0), (1, 0), (1, 1), (2, 1)),
        ((2, 0), (1, 1), (2, 1), (1, 2)),
        ((0, 1), (1, 1), (1, 2), (2, 2)),
        ((1, 0), (0, 1), (1, 1), (0, 2)),
    ),
}

# SRS kicks for transitions (old rotation, new rotation). O does not kick.
JLSTZ_KICKS: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {
    (0, 1): ((0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)),
    (1, 0): ((0, 0), (1, 0), (1, 1), (0, -2), (1, -2)),
    (1, 2): ((0, 0), (1, 0), (1, 1), (0, -2), (1, -2)),
    (2, 1): ((0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)),
    (2, 3): ((0, 0), (1, 0), (1, -1), (0, 2), (1, 2)),
    (3, 2): ((0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)),
    (3, 0): ((0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)),
    (0, 3): ((0, 0), (1, 0), (1, -1), (0, 2), (1, 2)),
}
I_KICKS: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {
    (0, 1): ((0, 0), (-2, 0), (1, 0), (-2, 1), (1, -2)),
    (1, 0): ((0, 0), (2, 0), (-1, 0), (2, -1), (-1, 2)),
    (1, 2): ((0, 0), (-1, 0), (2, 0), (-1, -2), (2, 1)),
    (2, 1): ((0, 0), (1, 0), (-2, 0), (1, 2), (-2, -1)),
    (2, 3): ((0, 0), (2, 0), (-1, 0), (2, -1), (-1, 2)),
    (3, 2): ((0, 0), (-2, 0), (1, 0), (-2, 1), (1, -2)),
    (3, 0): ((0, 0), (1, 0), (-2, 0), (1, 2), (-2, -1)),
    (0, 3): ((0, 0), (-1, 0), (2, 0), (-1, -2), (2, 1)),
}


@dataclass(frozen=True)
class ActivePiece:
    kind: str
    rotation: int
    x: int
    y: int

    @property
    def cells(self) -> tuple[tuple[int, int], ...]:
        return SHAPES[self.kind][self.rotation % 4]

    def absolute_cells(self) -> tuple[tuple[int, int], ...]:
        return tuple((self.x + dx, self.y + dy) for dx, dy in self.cells)

    def moved(self, dx: int = 0, dy: int = 0, rotation: int | None = None) -> "ActivePiece":
        return ActivePiece(self.kind, self.rotation if rotation is None else rotation % 4, self.x + dx, self.y + dy)


def piece_id(kind: str) -> int:
    return PIECE_TO_ID[kind]


def one_hot(kind: str) -> np.ndarray:
    out = np.zeros(len(PIECES), dtype=np.int8)
    out[piece_id(kind)] = 1
    return out


def spawn_piece(kind: str) -> ActivePiece:
    # The y=-2 spawn position leaves room for the complete SRS box above the board.
    return ActivePiece(kind, 0, 3, -2)


def kick_tests(kind: str, old_rotation: int, new_rotation: int) -> Iterable[tuple[int, int]]:
    if kind == "O":
        return ((0, 0),)
    table = I_KICKS if kind == "I" else JLSTZ_KICKS
    return table[(old_rotation % 4, new_rotation % 4)]
