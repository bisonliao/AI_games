"""Collect RAM transitions and screenshots for display-metric research."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np

from PacManEnv.config import MsPacmanEnvConfig
from PacManEnv.factory import make_env


def read_ram_fields(ram: np.ndarray) -> dict[str, Any]:
    """Return known and candidate display fields without deriving a level."""
    return {
        "maze_layout": int(ram[0]),
        "dots_eaten": int(ram[119]),
        "score_bcd": [int(value) for value in ram[120:123]],
        "life_and_fruit_state": int(ram[123]),
    }


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    import cv2

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise OSError(f"Failed to write screenshot to {path}")


def collect_diagnostics(
    output_dir: Path,
    *,
    seed: int,
    max_agent_steps: int,
    frame_skip: int,
) -> Path:
    """Run a deterministic random policy and record candidate maze transitions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "ram_events.jsonl"
    config = replace(
        MsPacmanEnvConfig(),
        num_envs=1,
        frame_skip=frame_skip,
        noop_max=0,
        repeat_action_probability=0.0,
        include_ram_metrics=True,
    )
    env = make_env(config)
    rng = np.random.default_rng(seed)
    recent_screens: deque[tuple[int, np.ndarray]] = deque(maxlen=4)
    pending_after_screens = 0

    try:
        _, info = env.reset(seed=seed)
        ale = env.unwrapped.ale
        fields = read_ram_fields(ale.getRAM())
        recent_screens.append((int(info["emulator_frames"]), ale.getScreenRGB()))

        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(
                json.dumps(
                    {"event": "reset", "agent_step": 0, **fields},
                    sort_keys=True,
                )
                + "\n"
            )

            for agent_step in range(1, max_agent_steps + 1):
                action = int(rng.integers(env.action_space.n))
                _, reward, terminated, truncated, info = env.step(action)
                if truncated:
                    raise RuntimeError("Diagnostic environment unexpectedly truncated")

                screen = ale.getScreenRGB()
                frame_number = int(info["emulator_frames"])
                recent_screens.append((frame_number, screen.copy()))
                next_fields = read_ram_fields(ale.getRAM())

                maze_candidate = (
                    next_fields["maze_layout"] != fields["maze_layout"]
                    or next_fields["dots_eaten"] < fields["dots_eaten"]
                )
                notable = maze_candidate or bool(info["life_lost"]) or bool(terminated)
                if notable:
                    event = {
                        "event": "maze_candidate" if maze_candidate else "life_event",
                        "agent_step": agent_step,
                        "action": action,
                        "reward": float(reward),
                        "raw_score": float(info["raw_score"]),
                        "lives": int(info["lives"]),
                        "game_over": bool(info["game_over"]),
                        "emulator_frames": frame_number,
                        "previous": fields,
                        "current": next_fields,
                    }
                    log_file.write(json.dumps(event, sort_keys=True) + "\n")
                    log_file.flush()

                if maze_candidate:
                    event_dir = output_dir / f"candidate_{agent_step:08d}"
                    event_dir.mkdir(exist_ok=True)
                    for nearby_frame, nearby_screen in recent_screens:
                        _save_rgb(
                            event_dir / f"frame_{nearby_frame:09d}.png",
                            nearby_screen,
                        )
                    pending_after_screens = 4
                elif pending_after_screens > 0:
                    _save_rgb(
                        output_dir / f"after_frame_{frame_number:09d}.png",
                        screen,
                    )
                    pending_after_screens -= 1

                fields = next_fields
                if terminated:
                    break
    finally:
        env.close()

    return log_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/ms_pacman_ram"),
        help="Directory for JSONL events and RGB screenshots",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--frame-skip", type=int, choices=(1, 2, 4), default=1)
    args = parser.parse_args()
    log_path = collect_diagnostics(
        args.output,
        seed=args.seed,
        max_agent_steps=args.steps,
        frame_skip=args.frame_skip,
    )
    print(log_path)


if __name__ == "__main__":
    main()
