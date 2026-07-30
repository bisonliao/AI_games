from __future__ import annotations

from collections.abc import Mapping

import pybullet


# Some PyBullet wheels expose B3G_SPACE but omit B3G_ESCAPE. Ordinary keys
# use their ASCII value in getKeyboardEvents(), so 27 is the portable fallback.
ESCAPE_KEY = int(getattr(pybullet, "B3G_ESCAPE", 27))
SPACE_KEY = int(getattr(pybullet, "B3G_SPACE", ord(" ")))
KEY_WAS_TRIGGERED = int(getattr(pybullet, "KEY_WAS_TRIGGERED", 2))
KEY_WAS_RELEASED = int(getattr(pybullet, "KEY_WAS_RELEASED", 4))


def was_triggered(events: Mapping[int, int], key: int) -> bool:
    return bool(events.get(key, 0) & KEY_WAS_TRIGGERED)


def was_released(events: Mapping[int, int], key: int) -> bool:
    return bool(events.get(key, 0) & KEY_WAS_RELEASED)


def exit_requested(events: Mapping[int, int]) -> bool:
    """Accept Escape, q, or Q so every PyBullet build has an exit key."""
    return any(was_triggered(events, key) for key in (ESCAPE_KEY, ord("q"), ord("Q")))
