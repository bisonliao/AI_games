"""Run a small smoke test with the hand-written controller.

Examples
--------
    python -m RobotEnv.scripts.run_scripted_demo --task reach --episodes 2
    python -m RobotEnv.scripts.run_scripted_demo --task pick_place --episodes 3 --gui
"""

from __future__ import annotations

import argparse
import time

from RobotEnv import PandaTabletopEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["reach", "pick_place"], default="pick_place")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--gui", action="store_true", help="open the PyBullet GUI")
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="GUI playback rate; only used with --gui (default: 30)",
    )
    args = parser.parse_args()
    if args.fps <= 0.0:
        parser.error("--fps must be positive")

    env = PandaTabletopEnv(
        task=args.task,
        render_mode="human" if args.gui else None,
        max_episode_steps=180,
    )
    try:
        for episode in range(args.episodes):
            env.reset(seed=episode)
            total_reward = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                frame_start = time.perf_counter()
                _, reward, terminated, truncated, info = env.step(env.heuristic_action())
                total_reward += reward
                if args.gui:
                    # PyBullet advances as fast as possible by default.  Sleep
                    # only for GUI playback, keeping headless runs fast.
                    remaining = (1.0 / args.fps) - (time.perf_counter() - frame_start)
                    if remaining > 0.0:
                        time.sleep(remaining)
            print(
                f"episode={episode} task={args.task} steps={info['step']} "
                f"success={info['success']} reward={total_reward:.3f}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
