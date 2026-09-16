"""Evaluate a BC actor checkpoint or an SB3 SAC fine-tuning checkpoint."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from .common import EnvConfig, load_actor, new_tensorboard_run, setup
from .randomized_env import make_randomized_env_factory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-type",
        choices=("auto", "bc", "sac"),
        default="auto",
        help=(
            "checkpoint format: BC .pt, SAC fine-tuning .zip; "
            "auto infers the type from the suffix"
        ),
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--seed-set", choices=("unseen", "train"), default="unseen")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--frame-stack", type=int, default=2)
    parser.add_argument("--max-episode-steps", type=int, default=150)
    parser.add_argument("--action-repeat", type=int, default=8)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.fps <= 0.0:
        parser.error("--fps must be positive")
    return args


def infer_checkpoint_type(path: Path, requested_type: str) -> str:
    """Resolve and validate the checkpoint type before creating an environment."""
    if requested_type != "auto":
        checkpoint_type = requested_type
    elif path.suffix.lower() == ".pt":
        checkpoint_type = "bc"
    elif path.suffix.lower() == ".zip":
        checkpoint_type = "sac"
    else:
        raise ValueError(
            "Cannot infer checkpoint type. Use a BC .pt file, an SAC .zip file, "
            "or pass --checkpoint-type explicitly."
        )

    if checkpoint_type == "bc" and path.suffix.lower() != ".pt":
        raise ValueError("BC checkpoints from bc_pretrain.py must use the .pt format")
    if checkpoint_type == "sac" and path.suffix.lower() != ".zip":
        raise ValueError("SAC checkpoints from sac_finetune.py must use the .zip format")
    return checkpoint_type


def build_sac_environment(args: argparse.Namespace, config: EnvConfig):
    """Build the vector environment used for SB3 checkpoint evaluation."""
    return DummyVecEnv(
        [
            make_randomized_env_factory(
                task="pick_place",
                rank=0,
                seed=args.seed_start,
                config=config,
                randomize=False,
                render_mode="human" if args.gui else None,
            )
        ]
    )


def evaluate_bc(
    actor,
    config: EnvConfig,
    args: argparse.Namespace,
    writer: SummaryWriter | None = None,
) -> dict:
    """Evaluate a standalone BC actor in ordinary Gymnasium episodes."""
    success_count = 0
    grasp_count = 0
    lift_count = 0
    approach_timeout_count = 0
    returns = []
 

    for episode_id in range(args.episodes):
        env = config.make_env(
            seed=args.seed_start + episode_id,
            gui=args.gui,
            randomize=False,
        )
        observation, _ = env.reset(seed=args.seed_start + episode_id)
        episode_return = 0.0

        while True:
            frame_start = time.perf_counter()
            with torch.inference_mode():
                tensors, _ = actor.obs_to_tensor(observation)
                action = actor(
                    tensors,
                    deterministic=not args.stochastic,
                ).cpu().numpy()[0]
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            if args.gui:
                remaining = (1.0 / args.fps) - (time.perf_counter() - frame_start)
                if remaining > 0.0:
                    time.sleep(remaining)
            if terminated or truncated:
                success_count += int(info.get("success", False))
                grasp_count += int(info.get("ever_grasped", False))
                lift_count += int(info.get("ever_lifted", False))
                approach_timeout_count += int(
                    info.get("failure_reason", "") == "approach_timeout"
                )
                break

        returns.append(episode_return)
        if writer is not None:
            writer.add_scalar(
                "eval/episode_success", float(info.get("success", False)), episode_id
            )
            writer.add_scalar(
                "eval/episode_grasp", float(info.get("ever_grasped", False)), episode_id
            )
            writer.add_scalar(
                "eval/episode_lift", float(info.get("ever_lifted", False)), episode_id
            )
            writer.add_scalar("eval/episode_return", episode_return, episode_id)
        env.close()

    result = {
        "mode": "bc",
        "episodes": args.episodes,
        "success_rate": success_count / args.episodes,
        "grasp_rate": grasp_count / args.episodes,
        "lift_rate": lift_count / args.episodes,
        "approach_timeout_rate": approach_timeout_count / args.episodes,
        "mean_return": float(np.mean(returns)),
    }
    if writer is not None:
        for name in (
            "success_rate", "grasp_rate", "lift_rate",
            "approach_timeout_rate", "mean_return",
        ):
            writer.add_scalar(f"eval/{name}", result[name], args.episodes)
    return result


def evaluate_sac(
    model,
    env,
    args: argparse.Namespace,
    writer: SummaryWriter | None = None,
) -> dict:
    """Evaluate an SB3 SAC model in a vector environment."""
    success_count = 0
    grasp_count = 0
    lift_count = 0
    approach_timeout_count = 0
    returns = []
    observation = env.reset()

    for _ in range(args.episodes):
        episode_return = 0.0
        while True:
            frame_start = time.perf_counter()
            action, _ = model.predict(
                observation,
                deterministic=not args.stochastic,
            )
            observation, rewards, dones, infos = env.step(action)
            episode_return += float(rewards[0])
            if dones[0]:
                info = infos[0]
                success_count += int(info.get("success", False))
                grasp_count += int(info.get("ever_grasped", False))
                lift_count += int(info.get("ever_lifted", False))
                approach_timeout_count += int(
                    info.get("failure_reason", "") == "approach_timeout"
                )
                returns.append(episode_return)
                if writer is not None:
                    episode_id = len(returns) - 1
                    writer.add_scalar(
                        "eval/episode_success",
                        float(info.get("success", False)),
                        episode_id,
                    )
                    writer.add_scalar(
                        "eval/episode_grasp",
                        float(info.get("ever_grasped", False)),
                        episode_id,
                    )
                    writer.add_scalar(
                        "eval/episode_lift",
                        float(info.get("ever_lifted", False)),
                        episode_id,
                    )
                    writer.add_scalar("eval/episode_return", episode_return, episode_id)
                break
            if args.gui:
                remaining = (1.0 / args.fps) - (time.perf_counter() - frame_start)
                if remaining > 0.0:
                    time.sleep(remaining)

    result = {
        "mode": "sac",
        "episodes": args.episodes,
        "success_rate": success_count / args.episodes,
        "grasp_rate": grasp_count / args.episodes,
        "lift_rate": lift_count / args.episodes,
        "approach_timeout_rate": approach_timeout_count / args.episodes,
        "mean_return": float(np.mean(returns)),
    }
    if writer is not None:
        for name in (
            "success_rate", "grasp_rate", "lift_rate",
            "approach_timeout_rate", "mean_return",
        ):
            writer.add_scalar(f"eval/{name}", result[name], args.episodes)
    return result


def main() -> None:
    # Checkpoint distinction is intentional:
    #
    # * ``bc_pretrain.py`` writes a standalone visual actor as ``*.pt``.
    #   It is evaluated with ``evaluate_bc`` in an ordinary Gymnasium env.
    # * ``sac_finetune.py`` writes an SB3 SAC model as ``*.zip``.
    #   It is evaluated with ``evaluate_sac`` in a DummyVecEnv.
    # * The privileged vector expert is also a ``*.zip`` file, but it is not a
    #   visual checkpoint. SAC.load() validation below rejects its Box(52,)
    #   observation space and reports the correct source of the error.
    args = parse_args()
    if args.seed_set == "train":
        args.seed_start = 0
    print(
        f"Evaluation seed set: {args.seed_set}, "
        f"seeds={args.seed_start}..{args.seed_start + args.episodes - 1}"
    )
    setup(args.seed_start, args.device)
    checkpoint_type = infer_checkpoint_type(args.checkpoint, args.checkpoint_type)
    tensorboard_path = new_tensorboard_run("evaluate")
    writer = SummaryWriter(log_dir=str(tensorboard_path))
    print(f"TensorBoard logs: {tensorboard_path}")

    if checkpoint_type == "bc":
        actor, config, _ = load_actor(args.checkpoint, args.device)
        result = evaluate_bc(actor, config, args, writer)
    else:
        config = EnvConfig(
            image_size=args.image_size,
            frame_stack=args.frame_stack,
            max_episode_steps=args.max_episode_steps,
            action_repeat=args.action_repeat,
        )
        env = build_sac_environment(args, config)
        try:
            model = SAC.load(args.checkpoint, env=env, device=args.device)
            if not hasattr(model.observation_space, "spaces"):
                raise ValueError(
                    "This .zip contains a vector-observation SAC model, such as "
                    "the SAC_VecObs expert. Pass a SAC_BC_PixelObs sac_finetune "
                    "checkpoint instead."
                )
            result = evaluate_sac(model, env, args, writer)
        finally:
            env.close()

    writer.flush()
    writer.close()
    print(result)


if __name__ == "__main__":
    main()
