"""Asynchronous checkpoint evaluator used by the training coordinator."""
from __future__ import annotations

import queue
from pathlib import Path

import numpy as np
import torch

from TetrisEnv.placement_env import PlacementTetrisEnv

from .model import DuelingDQN, masked_q_values, observations_to_torch


def evaluate_checkpoint(
    checkpoint: str | Path,
    *,
    episodes: int,
    max_steps: int,
    seed: int,
    device: str = "cpu",
    render: bool = False,
    render_fps: int = 10,
) -> dict[str, float | int | str]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    checkpoint_config = payload.get("config", {})
    model = DuelingDQN().to(device).eval()
    model.load_state_dict(payload["online"])
    returns: list[float] = []
    pieces: list[int] = []
    lines: list[int] = []
    lengths: list[int] = []
    truncated_count = 0
    interrupted = False
    renderer = None
    if render:
        from TetrisEnv.rendering import PygameRenderer

        renderer = PygameRenderer(title="Tetris RL - Checkpoint Evaluation", fps=render_fps)
    try:
        for episode in range(episodes):
            env = PlacementTetrisEnv(
                gamma=float(checkpoint_config.get("gamma", 0.99)),
                piece_placed_reward=float(checkpoint_config.get("piece_placed_reward", 0.01)),
                line_clear_reward=float(checkpoint_config.get("line_clear_reward", 0.75)),
                terminal_penalty=float(checkpoint_config.get("terminal_penalty", 1.0)),
            )
            if renderer is not None:
                if renderer.closed:
                    break
            try:
                obs, _ = env.reset(seed=seed + episode)
                total_return = 0.0
                last_info = {"survival_pieces": 0, "lines_cleared": 0}
                episode_lines = 0
                episode_length = 0
                if renderer is not None:
                    renderer.draw(env, last_info, episode=episode + 1, total_episodes=episodes)
                for _ in range(max_steps):
                    if renderer is not None:
                        renderer.events()
                        if renderer.closed:
                            interrupted = True
                            break
                    with torch.inference_mode():
                        batched = {key: np.expand_dims(value, 0) for key, value in obs.items()}
                        torch_obs = observations_to_torch(batched, device)
                        q_values = masked_q_values(model(torch_obs), torch_obs.get("action_mask"))
                        action = int(q_values.argmax(dim=-1).item())
                    obs, reward, terminated, _, last_info = env.step(action)
                    total_return += reward
                    episode_lines += int(last_info.get("lines_cleared", 0))
                    episode_length += 1
                    if renderer is not None:
                        display_info = dict(last_info)
                        display_info["total_lines"] = episode_lines
                        renderer.draw(
                            env,
                            display_info,
                            episode=episode + 1,
                            total_episodes=episodes,
                            game_over=terminated,
                        )
                        renderer.tick()
                    if terminated:
                        break
                else:
                    truncated_count += 1
                if interrupted:
                    break
                returns.append(total_return)
                pieces.append(int(last_info["survival_pieces"]))
                lines.append(episode_lines)
                lengths.append(episode_length)
            finally:
                env.close()
            if interrupted:
                break
    finally:
        if renderer is not None:
            renderer.close()
    return {
        "checkpoint": str(checkpoint),
        "env_mode": "placement",
        "transition_step": int(payload.get("transitions", 0)),
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "mean_survival_pieces": float(np.mean(pieces)) if pieces else 0.0,
        "mean_lines": float(np.mean(lines)) if lines else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "truncated_episodes": int(truncated_count),
        "completed_episodes": len(returns),
        "interrupted": int(interrupted),
    }


def evaluator_process(eval_queue, result_queue, stop_event, *, episodes: int, max_steps: int, seed: int, device: str) -> None:
    """Process at most one checkpoint at a time; training never waits for it."""
    while not stop_event.is_set():
        try:
            checkpoint = eval_queue.get(timeout=0.25)
        except queue.Empty:
            continue
        if checkpoint is None:
            break
        try:
            result = evaluate_checkpoint(
                checkpoint,
                episodes=episodes,
                max_steps=max_steps,
                seed=seed,
                device=device,
            )
        except Exception as exc:  # report failure without taking down learner
            result = {"checkpoint": str(checkpoint), "error": repr(exc)}
        try:
            result_queue.put(result, timeout=1.0)
        except queue.Full:
            pass
