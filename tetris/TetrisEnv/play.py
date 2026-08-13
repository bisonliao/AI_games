"""Play one graphical Tetris environment with the keyboard."""
from __future__ import annotations

import argparse

from .rendering import PygameRenderer
from .tetris_env import TetrisEnv


def play(*, seed: int | None = None, fps: int = 8) -> None:
    env = TetrisEnv()
    renderer = PygameRenderer(title="Tetris RL - Manual Play", fps=fps)
    pygame = renderer.pygame
    pygame.key.set_repeat(140, 55)
    _, info = env.reset(seed=seed)
    terminated = False
    renderer.draw(env, info)
    try:
        while not renderer.closed:
            action = env.ACTION_NOOP
            for event in renderer.events():
                if event.type != pygame.KEYDOWN:
                    continue
                if event.key == pygame.K_ESCAPE:
                    renderer.closed = True
                elif terminated and event.key in (pygame.K_RETURN, pygame.K_r):
                    _, info = env.reset(seed=None)
                    terminated = False
                elif not terminated:
                    if event.key == pygame.K_LEFT:
                        action = env.ACTION_LEFT
                    elif event.key == pygame.K_RIGHT:
                        action = env.ACTION_RIGHT
                    elif event.key == pygame.K_SPACE:
                        action = env.ACTION_ROTATE_CW
                    elif event.key == pygame.K_DOWN:
                        action = env.ACTION_HARD_DROP
            if renderer.closed:
                break
            if not terminated:
                _, _, terminated, _, info = env.step(action)
            renderer.draw(env, info, game_over=terminated)
            renderer.tick()
    finally:
        renderer.close()
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Play the Tetris Gymnasium environment")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()
    play(seed=args.seed, fps=args.fps)


if __name__ == "__main__":
    main()
