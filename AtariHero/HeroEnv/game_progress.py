"""H.E.R.O. level and room metadata decoded from Atari RAM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ROM_MD5 = "fca4a5be1251927027f2c24774a02160"
MAX_LEVEL = 20
RAM_LEVEL_INDEX = 0x75
RAM_ROOM_NUMBER = 0x1C

# The standard ROM stores the last zero-based room index for each level.
LEVEL_LAST_ROOM_INDICES = (1, 3, 5, 7, 7, 9, 11, 13) + (15,) * 12
LEVEL_ROOM_COUNTS = tuple(last_room + 1 for last_room in LEVEL_LAST_ROOM_INDICES)


@dataclass(frozen=True)
class GameProgress:
    level: int
    room: int
    total_rooms: int

    @property
    def rooms_after(self) -> int:
        return self.total_rooms - self.room

    @property
    def route_index(self) -> int:
        return sum(LEVEL_ROOM_COUNTS[: self.level - 1]) + self.room

    @property
    def lesson_id(self) -> str:
        return f"L{self.level:02d}-R{self.room:02d}"

    def as_dict(self) -> dict[str, int | str]:
        return {
            "level": self.level,
            "room": self.room,
            "total_rooms": self.total_rooms,
            "rooms_after": self.rooms_after,
            "route_index": self.route_index,
            "lesson_id": self.lesson_id,
        }


def decode_game_progress(ram: np.ndarray) -> GameProgress | None:
    level = int(ram[RAM_LEVEL_INDEX]) + 1
    if not 1 <= level <= MAX_LEVEL:
        return None

    room = int(ram[RAM_ROOM_NUMBER]) + 1
    total_rooms = LEVEL_ROOM_COUNTS[level - 1]
    if not 1 <= room <= total_rooms:
        return None

    return GameProgress(level=level, room=room, total_rooms=total_rooms)


def decode_level(ram: np.ndarray) -> int | None:
    level = int(ram[RAM_LEVEL_INDEX]) + 1
    return level if 1 <= level <= MAX_LEVEL else None
