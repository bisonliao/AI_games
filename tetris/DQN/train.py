from __future__ import annotations

import argparse

from .config import TrainConfig
from .trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Dueling Double DQN Tetris agent")
    parser.add_argument("--config", default="configs/ddqn_default.toml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--total-transitions", type=int, default=None)
    parser.add_argument("--num-actors", type=int, default=None)
    parser.add_argument("--envs-per-actor", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--eval-max-steps", type=int, default=None)
    parser.add_argument("--log-root", default=None)
    parser.add_argument("--checkpoint-root", default=None)
    args = parser.parse_args()
    config = TrainConfig.from_toml(args.config)
    for key in ("checkpoint_every", "eval_every", "eval_episodes", "eval_max_steps", "log_root", "checkpoint_root"):
        value = getattr(args, key)
        if value is not None:
            setattr(config, key, value)
    log_dir = train(config, device=args.device, total_transitions=args.total_transitions, num_actors=args.num_actors, envs_per_actor=args.envs_per_actor)
    print(f"TensorBoard log directory: {log_dir}")


if __name__ == "__main__":
    main()
