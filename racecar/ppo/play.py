"""Render a trained PPO checkpoint in a continuously looping environment.

Run from the project root::

    python -m ppo.play checkpoints/<run>/checkpoint_final.pt

The policy samples actions from its categorical distribution by default. Focus
the PyBullet GUI and press ``q`` to exit; otherwise a new episode starts after
every success, collision, or time limit. Each episode samples a fresh start
pose using the checkpoint's training noise. Pass ``--deterministic`` to use
argmax actions instead.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pybullet as p
import torch

from env.RaceCarEnv import DEFAULT_MAX_STEPS, RaceCarEnv
from ppo.model import ACTION_NAMES, ActorCritic, actions_to_steering


def _quit_requested(physics_client: int) -> bool:
    events = p.getKeyboardEvents(physicsClientId=physics_client)
    for key in (ord("q"), ord("Q")):
        state = events.get(key, 0)
        if state & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
            return True
    return False


def _load_policy(checkpoint_path: Path) -> tuple[ActorCritic, dict]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        metadata = checkpoint
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        metadata = {}
    else:
        raise ValueError("checkpoint must be a state_dict or contain a 'model' state_dict")
    model = ActorCritic().cpu().eval()
    model.load_state_dict(state_dict)
    return model, metadata


def play(checkpoint_path: Path, *, fps: int | None = None,
         max_episode_steps: int | None = None, stochastic: bool = True,
         start_position_noise: float | None = None,
         start_heading_noise: float | None = None,
         seed: int | None = None) -> None:
    model, checkpoint = _load_policy(checkpoint_path)
    saved_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    fps = int(fps if fps is not None else saved_config.get("fps", 20))
    max_episode_steps = int(
        max_episode_steps if max_episode_steps is not None
        else saved_config.get("max_episode_steps", DEFAULT_MAX_STEPS)
    )
    start_position_noise = float(
        start_position_noise if start_position_noise is not None
        else saved_config.get("start_position_noise", 0.15)
    )
    start_heading_noise = float(
        start_heading_noise if start_heading_noise is not None
        else saved_config.get("start_heading_noise", 0.08)
    )
    if fps <= 0 or max_episode_steps <= 0:
        raise ValueError("fps and max_episode_steps must be positive")
    if start_position_noise < 0 or start_heading_noise < 0:
        raise ValueError("start pose noise must be non-negative")

    env = RaceCarEnv(
        render=True, fps=fps, max_steps=max_episode_steps,
        start_position_noise=start_position_noise,
        start_heading_noise=start_heading_noise,
    )
    episode_number = 1
    episode_return = 0.0
    try:
        # Seed only the first explicit reset. Later reset() calls advance this
        # environment's RNG, giving every episode a different but reproducible
        # start sequence when --seed is supplied.
        observation, _ = env.reset(seed=seed)
        global_steps = checkpoint.get("global_steps", "unknown")
        print(
            f"Loaded {checkpoint_path} (training steps={global_steps}).\n"
            f"Policy mode={'stochastic' if stochastic else 'deterministic'}, "
            f"episode limit={max_episode_steps}, position noise=±{start_position_noise}m, "
            f"heading noise=±{start_heading_noise}rad. Focus GUI and press q to quit."
        )

        while p.isConnected(env.physicsClient):
            frame_started = time.perf_counter()
            if _quit_requested(env.physicsClient):
                break

            with torch.no_grad():
                observation_tensor = torch.from_numpy(observation).unsqueeze(0)
                if stochastic:
                    action, _, _, _ = model.get_action_and_value(observation_tensor)
                else:
                    logits = model.policy(model.trunk(observation_tensor))
                    action = logits.argmax(dim=-1)
            action_id = int(action.item())
            steering = actions_to_steering(np.asarray([action_id], dtype=np.int64))[0]
            observation, reward, terminated, truncated, info = env.step(steering)
            episode_return += float(reward)

            if terminated or truncated:
                position = np.asarray(info["position"], dtype=np.float32)
                terminal_distance = float(np.linalg.norm(
                    position - np.asarray(env.finish_pos[:2], dtype=np.float32)
                ))
                print(
                    f"episode={episode_number} reason={info.get('termination_reason', 'unknown')} "
                    f"success={info.get('is_success', False)} steps={info['steps']} "
                    f"distance={terminal_distance:.2f} env_return={episode_return:.3f} "
                    f"last_action={ACTION_NAMES[action_id]}"
                )
                episode_number += 1
                episode_return = 0.0
                observation, _ = env.reset()

            elapsed = time.perf_counter() - frame_started
            frame_interval = 1.0 / fps
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a RaceCar PPO checkpoint")
    parser.add_argument("checkpoint", type=Path, help="path to checkpoint_*.pt")
    parser.add_argument("--fps", type=int, default=None,
                        help="override checkpoint simulation/display frequency")
    parser.add_argument("--max-episode-steps", type=int, default=None,
                        help="override checkpoint episode limit")
    policy_mode = parser.add_mutually_exclusive_group()
    policy_mode.add_argument("--stochastic", dest="stochastic", action="store_true",
                             help="sample policy actions (default)")
    policy_mode.add_argument("--deterministic", dest="stochastic", action="store_false",
                             help="choose the most likely action with argmax")
    parser.set_defaults(stochastic=True)
    parser.add_argument("--start-position-noise", type=float, default=None,
                        help="override checkpoint x/y start-position noise in meters")
    parser.add_argument("--start-heading-noise", type=float, default=None,
                        help="override checkpoint start-heading noise in radians")
    parser.add_argument("--seed", type=int, default=None,
                        help="reproduce the sequence of randomized episode starts")
    args = parser.parse_args()
    play(args.checkpoint, fps=args.fps,
         max_episode_steps=args.max_episode_steps, stochastic=args.stochastic,
         start_position_noise=args.start_position_noise,
         start_heading_noise=args.start_heading_noise, seed=args.seed)


if __name__ == "__main__":
    main()
