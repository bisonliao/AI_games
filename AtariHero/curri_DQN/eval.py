"""Greedy evaluation of a saved H.E.R.O. DQN checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from HeroEnv import make_hero_level_1_to_2_env
from .envs import DQNAtariWrapper
from .model import DuelingDQN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help=".pt checkpoint to evaluate"
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--gui", action="store_true", help="render the game while evaluating"
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--scale", type=int, default=3)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.fps < 1:
        parser.error("--fps must be positive")
    if args.scale < 1:
        parser.error("--scale must be positive")
    return args


def load_online_model(
    checkpoint: Path, action_count: int, frame_stack: int = 4
) -> DuelingDQN:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    payload = torch.load(checkpoint.resolve(), map_location="cpu", weights_only=False)
    state = payload.get("online_model", payload)
    model = DuelingDQN(action_count, frame_stack=frame_stack).cpu().eval()
    model.load_state_dict(state)
    return model


def _draw_frame(screen: object, frame: np.ndarray, scale: int) -> None:
    import pygame

    surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    width, height = surface.get_size()
    scaled = pygame.transform.scale(surface, (width * scale, height * scale))
    screen.blit(scaled, (0, 0))
    pygame.display.flip()


def _poll_gui_events() -> bool:
    import pygame

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
            return False
    return True


def evaluate(args: argparse.Namespace) -> int:
    env = DQNAtariWrapper(
        make_hero_level_1_to_2_env(
            training=True,
            checkpoint_dir=Path("HeroEnv/checkpoints"),
            curriculum_stage=1,
            checkpoint_reset_probability=1.0,
            include_easier_stages=False,
            frameskip=1,
            repeat_action_probability=0.25,
        ),
        action_repeat=4,
        frame_stack=4,
        screen_size=84,
    )
    model = load_online_model(args.checkpoint, int(env.action_space.n))
    screen = None
    clock = None
    if args.gui:
        import pygame

        pygame.display.init()
        pygame.font.init()
        screen = pygame.display.set_mode((160 * args.scale, 210 * args.scale))
        pygame.display.set_caption(f"H.E.R.O. greedy evaluation: {args.checkpoint.name}")
        clock = pygame.time.Clock()

    successes = 0
    stopped_by_user = False
    rng = np.random.default_rng(args.seed + 910_000)
    try:
        for episode in range(args.episodes):
            start_level = 1 if rng.random() < 0.5 else 2
            start_checkpoint_ids = env.checkpoint_ids_for_level_start(start_level)
            if not start_checkpoint_ids:
                raise RuntimeError(
                    f"no Level {start_level} Room 1 checkpoint in the frozen manifest"
                )
            checkpoint_id = start_checkpoint_ids[
                int(rng.integers(len(start_checkpoint_ids)))
            ]
            observation, info = env.reset(
                seed=args.seed + episode,
                options={
                    "curriculum_stage": None,
                    "checkpoint_id": checkpoint_id,
                },
            )
            episode_return = 0.0
            ale_score_return = 0.0
            decisions = 0
            while True:
                if screen is not None and not _poll_gui_events():
                    stopped_by_user = True
                    break
                tensor = torch.from_numpy(observation).unsqueeze(0)
                with torch.inference_mode():
                    action = int(model(tensor).argmax(dim=1).item())
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                ale_score_return += float(info.get("hero_ale_reward", 0.0))
                decisions += 1
                if screen is not None:
                    _draw_frame(screen, env.unwrapped.ale.getScreenRGB(), args.scale)
                    assert clock is not None
                    clock.tick(args.fps)
                if terminated or truncated:
                    break
            if stopped_by_user:
                break
            success = bool(info.get("is_success", False))
            successes += int(success)
            reason = info.get("hero_terminal_reason")
            if reason is None:
                reason = "terminated" if terminated else "truncated"
            print(
                f"episode={episode + 1}/{args.episodes} "
                f"start_level={start_level} checkpoint={checkpoint_id} "
                f"success={int(success)} return={episode_return:g} "
                f"ale_score_return={ale_score_return:g} "
                f"decisions={decisions} reason={reason}",
                flush=True,
            )
    finally:
        env.close()
        if args.gui:
            import pygame

            pygame.quit()

    completed = episode + 1 if not stopped_by_user else episode
    if stopped_by_user:
        print("Evaluation stopped by user", flush=True)
    else:
        print(
            f"summary: success={successes}/{completed} "
            f"rate={successes / completed:.3f}",
            flush=True,
        )
    return 0


def main() -> int:
    return evaluate(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
