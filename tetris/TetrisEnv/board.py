"""Pure board operations used by the environment and tests."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pieces import ActivePiece, kick_tests


@dataclass(frozen=True)
class BoardFeatures:
    aggregate_height: int
    max_height: int
    holes: int
    bumpiness: int
    wells: int


class Board:
    width = 10
    height = 20

    def __init__(self) -> None:
        self.grid = np.zeros((self.height, self.width), dtype=np.uint8)

    def reset(self) -> None:
        self.grid.fill(0)

    def collides(self, piece: ActivePiece) -> bool:
        for x, y in piece.absolute_cells():
            if x < 0 or x >= self.width or y >= self.height:
                return True
            if y >= 0 and self.grid[y, x]:
                return True
        return False

    def lock(self, piece: ActivePiece) -> bool:
        """Lock a piece. Return False if any cell is above the visible board."""
        above = False
        for x, y in piece.absolute_cells():
            if y < 0:
                above = True
            elif 0 <= y < self.height and 0 <= x < self.width:
                self.grid[y, x] = 1
            else:
                above = True
        return not above

    def clear_lines(self) -> int:
        full = np.all(self.grid != 0, axis=1)
        count = int(full.sum())
        if count:
            remaining = self.grid[~full]
            self.grid[:] = 0
            self.grid[count:] = remaining
        return count

    def features(self) -> BoardFeatures:
        heights = np.zeros(self.width, dtype=np.int32)
        holes = 0
        for x in range(self.width):
            filled = np.flatnonzero(self.grid[:, x])
            if filled.size:
                top = int(filled[0])
                heights[x] = self.height - top
                holes += int(np.count_nonzero(self.grid[top:, x] == 0))

        # A well is a column lower than both neighbours.  Summing the triangular
        # depth makes a deep, hard-to-fill shaft more expensive than several
        # shallow notches.  The board walls act as height-20 neighbours.
        wells = 0
        for x, height in enumerate(heights):
            left = self.height if x == 0 else int(heights[x - 1])
            right = self.height if x == self.width - 1 else int(heights[x + 1])
            depth = max(0, min(left, right) - int(height))
            wells += depth * (depth + 1) // 2
        return BoardFeatures(
            aggregate_height=int(heights.sum()),
            max_height=int(heights.max(initial=0)),
            holes=holes,
            bumpiness=int(np.abs(np.diff(heights)).sum()),
            wells=int(wells),
        )

    def active_mask(self, piece: ActivePiece | None) -> np.ndarray:
        mask = np.zeros_like(self.grid)
        if piece is not None:
            for x, y in piece.absolute_cells():
                if 0 <= y < self.height and 0 <= x < self.width:
                    mask[y, x] = 1
        return mask

    def try_move(self, piece: ActivePiece, dx: int = 0, dy: int = 0) -> ActivePiece | None:
        candidate = piece.moved(dx=dx, dy=dy)
        return None if self.collides(candidate) else candidate

    def try_rotate_cw(self, piece: ActivePiece) -> ActivePiece | None:
        new_rotation = (piece.rotation + 1) % 4
        for dx, dy_srs in kick_tests(piece.kind, piece.rotation, new_rotation):
            # SRS tables use positive y upwards; board coordinates use downwards.
            candidate = piece.moved(dx=dx, dy=-dy_srs, rotation=new_rotation)
            if not self.collides(candidate):
                return candidate
        return None
