"""Behavior cloning pretraining for the visual SAC actor."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .common import EnvConfig, load_actor, make_actor, new_tensorboard_run, save_actor, setup
from .dataset import EpisodeDataset, load_episodes, split_episodes
from .visual_augmentation import ConsistentMultiViewAugmentation


def parse_args() -> argparse.Namespace:
    """Parse command-line options for BC pretraining."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-seed-start", type=int, default=10000)
    parser.add_argument(
        "--train-eval-episodes",
        type=int,
        default=5,
        help="small evaluation count used during training epochs",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=5,
        help="run the small business evaluation every N epochs",
    )
    parser.add_argument(
        "--final-eval-progress-every",
        type=int,
        default=10,
        help="print progress after this many final evaluation episodes",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--augmentation-shift", type=int, default=4)
    parser.add_argument("--frame-stack", type=int, default=2)
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if args.eval_episodes <= 0:
        parser.error("--eval-episodes must be positive")
    if args.train_eval_episodes <= 0:
        parser.error("--train-eval-episodes must be positive")
    if args.eval_every <= 0:
        parser.error("--eval-every must be positive")
    if args.final_eval_progress_every <= 0:
        parser.error("--final-eval-progress-every must be positive")
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be between zero and one")
    if args.augmentation_shift < 0:
        parser.error("--augmentation-shift must be non-negative")
    return args


def load_config(data_dir: Path) -> EnvConfig:
    """Load the environment configuration saved with the showcase data."""
    metadata_path = data_dir / "metadata.json"
    if not metadata_path.exists():
        return EnvConfig()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return EnvConfig(**metadata.get("env_config", {}))


def train_epoch(actor, loader, optimizer, device: torch.device, augmentation) -> float:
    """Run one supervised training epoch and return its mean MSE."""
    actor.train()
    total_loss = 0.0
    sample_count = 0
    for image, proprio, target in loader:
        observation = {
            "image": augmentation(image.to(device)),
            "proprio": proprio.to(device),
        }
        prediction = actor(observation, deterministic=True)
        loss = (prediction - target.to(device)).pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += loss.item() * len(image)
        sample_count += len(image)
    return total_loss / max(sample_count, 1)


def validation_loss(actor, loader, device: torch.device, augmentation) -> float:
    """Evaluate mean action MSE without updating the actor."""
    actor.eval()
    total_loss = 0.0
    sample_count = 0
    with torch.inference_mode():
        for image, proprio, target in loader:
            observation = {
                "image": augmentation(image.to(device)),
                "proprio": proprio.to(device),
            }
            prediction = actor(observation, deterministic=True)
            loss = (prediction - target.to(device)).pow(2).mean()
            total_loss += loss.item() * len(image)
            sample_count += len(image)
    return total_loss / max(sample_count, 1)


def evaluate_business_metrics(
    actor,
    config: EnvConfig,
    episodes: int,
    seed: int,
    progress_every: int | None = None,
    label: str = "BC evaluation",
) -> dict[str, float]:
    """Evaluate BC and return business metrics with optional sparse progress."""
    success = grasp = lift = approach_timeout = 0
    started = time.perf_counter()
    if progress_every is not None:
        print(f"Starting {label}: episodes={episodes}, seed_start={seed}", flush=True)
    for episode_id in range(episodes):
        env = config.make_env(seed=seed + episode_id)
        observation, _ = env.reset(seed=seed + episode_id)
        while True:
            with torch.inference_mode():
                tensors, _ = actor.obs_to_tensor(observation)
                action = actor(tensors, deterministic=True).cpu().numpy()[0]
            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                success += int(info.get("success", False))
                grasp += int(info.get("ever_grasped", False))
                lift += int(info.get("ever_lifted", False))
                approach_timeout += int(info.get("failure_reason", "") == "approach_timeout")
                break
        env.close()
        completed = episode_id + 1
        if progress_every is not None and (
            completed % progress_every == 0 or completed == episodes
        ):
            elapsed = max(time.perf_counter() - started, 1e-6)
            print(
                f"{label}: {completed}/{episodes} episodes, "
                f"success={success / completed:.3f}, "
                f"grasp={grasp / completed:.3f}, "
                f"lift={lift / completed:.3f}, elapsed={elapsed:.1f}s",
                flush=True,
            )
    return {
        "success_rate": success / episodes,
        "grasp_rate": grasp / episodes,
        "lift_rate": lift / episodes,
        "approach_timeout_rate": approach_timeout / episodes,
    }


def main() -> None:
    args = parse_args()
    device = setup(args.seed, args.device)
    episodes = load_episodes(args.data, successful_only=True)
    if not episodes:
        raise SystemExit("No successful demonstrations found")

    config = replace(load_config(args.data), frame_stack=args.frame_stack)
    train_episodes, validation_episodes = split_episodes(
        episodes,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    train_set = EpisodeDataset(train_episodes)
    validation_set = EpisodeDataset(validation_episodes)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(
        validation_set,
        batch_size=args.batch_size,
        shuffle=False,
    )

    actor = make_actor(config, device)
    augmentation = ConsistentMultiViewAugmentation(args.augmentation_shift)
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr)
    checkpoint_path = args.out or (args.data.parent / "bc_model.pt")
    tensorboard_path = new_tensorboard_run("bc_pretrain")
    writer = SummaryWriter(log_dir=str(tensorboard_path))
    print(f"TensorBoard logs: {tensorboard_path}")
    best_validation_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(actor, train_loader, optimizer, device, augmentation)
        val_loss = validation_loss(actor, validation_loader, device, augmentation)
        print(
            f"epoch={epoch} train_mse={train_loss:.6f} "
            f"val_mse={val_loss:.6f}"
        )
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            business = evaluate_business_metrics(
                actor,
                config,
                args.train_eval_episodes,
                args.eval_seed_start,
            )
        else:
            business = None
        writer.add_scalar("bc/train_mse", train_loss, epoch)
        writer.add_scalar("bc/validation_mse", val_loss, epoch)
        writer.add_scalar("bc/train_episode_count", len(train_episodes), epoch)
        writer.add_scalar("bc/validation_episode_count", len(validation_episodes), epoch)
        if business is not None:
            writer.add_scalar("eval/train_periodic_success_rate", business["success_rate"], epoch)
            writer.add_scalar("eval/train_periodic_grasp_rate", business["grasp_rate"], epoch)
            writer.add_scalar("eval/train_periodic_lift_rate", business["lift_rate"], epoch)
            writer.add_scalar(
                "eval/train_periodic_approach_timeout_rate",
                business["approach_timeout_rate"],
                epoch,
            )
        writer.flush()
        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            save_actor(
                checkpoint_path,
                actor,
                config,
                stage="bc_pretrain",
                episodes=len(episodes),
                val_mse=val_loss,
                train_episode_count=len(train_episodes),
                validation_episode_count=len(validation_episodes),
                eval_seed_start=args.eval_seed_start,
                augmentation_shift=args.augmentation_shift,
                frame_stack=config.frame_stack,
            )

    print(f"Best BC checkpoint: {checkpoint_path}")
    # Full unseen-seed evaluation happens once after training, rather than at
    # every epoch. This keeps the training loop fast while retaining the
    # required 100-episode final business measurement.
    final_actor, _, _ = load_actor(checkpoint_path, device=device)
    final_metrics = evaluate_business_metrics(
        final_actor,
        config,
        args.eval_episodes,
        args.eval_seed_start,
        progress_every=args.final_eval_progress_every,
        label="final unseen BC evaluation",
    )
    for name, value in final_metrics.items():
        writer.add_scalar(f"eval/final_unseen_{name}", value, args.epochs)
    writer.flush()
    print(f"Final unseen-seed evaluation ({args.eval_episodes} episodes): {final_metrics}")
    writer.close()


if __name__ == "__main__":
    main()
