from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

from .runtime import TrainConfig, evaluate_policy, load_checkpoint, run_training


def _add_config_argument(
    parser: argparse.ArgumentParser,
    defaults: TrainConfig,
    flag: str,
    field_name: str,
    *,
    value_type: Callable[[str], Any] | None = None,
    help_text: str | None = None,
) -> None:
    """Expose a TrainConfig field without declaring its default twice."""
    default = getattr(defaults, field_name)
    if value_type is None:
        if default is None:
            raise ValueError(f"value_type is required for optional field {field_name!r}")
        value_type = type(default)
    parser.add_argument(
        flag,
        dest=field_name,
        type=value_type,
        default=default,
        metavar=flag.removeprefix("--").replace("-", "_").upper(),
        help=help_text or f"Override TrainConfig.{field_name}",
    )


def build_train_parser(task: str) -> argparse.ArgumentParser:
    """Expose selected CLI overrides; TrainConfig owns every default value."""
    defaults = TrainConfig(task=task)
    parser = argparse.ArgumentParser(
        description=f"Train the BasicMath {task} DDDQN policy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    def expose(
        flag: str,
        field_name: str,
        *,
        value_type: Callable[[str], Any] | None = None,
        help_text: str | None = None,
    ) -> None:
        _add_config_argument(
            parser,
            defaults,
            flag,
            field_name,
            value_type=value_type,
            help_text=help_text,
        )

    expose("--total-steps", "total_transitions")
    if task == "low":
        expose(
            "--distance-reward-scale",
            "low_distance_reward_scale",
            help_text="Reward for reducing the total cyclic digit distance by one",
        )
    expose("--actors", "num_actors")
    expose("--seed", "seed")
    expose("--replay-capacity", "replay_capacity")
    expose("--replay-warmup", "replay_warmup")
    expose("--batch-size", "batch_size")
    expose("--learning-rate", "learning_rate")
    expose("--updates-per-transition", "updates_per_transition")
    if task == "high":
        expose("--epsilon-start", "epsilon_start")
        expose("--epsilon-end", "epsilon_end")
        expose(
            "--epsilon-decay-steps",
            "epsilon_decay_transitions",
            value_type=int,
            help_text="Defaults to --total-steps when unset",
        )
    expose("--checkpoint-interval", "checkpoint_interval")
    expose("--eval-interval", "evaluation_interval")
    expose("--eval-episodes", "evaluation_episodes")
    expose("--report-interval", "report_interval")
    expose("--runs-dir", "runs_dir")
    expose("--checkpoints-dir", "checkpoints_dir")
    return parser


def train_from_args(task: str) -> None:
    parser = build_train_parser(task)
    args = parser.parse_args()
    config = TrainConfig(task=task, **vars(args))
    final_checkpoint = run_training(config)
    print(f"Final checkpoint: {final_checkpoint}")


def eval_from_args(task: str) -> None:
    parser = argparse.ArgumentParser(
        description=f"Evaluate the BasicMath {task} DDDQN policy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10_001)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help="Maximum policy decisions per second; 0 means unthrottled",
    )
    args = parser.parse_args()

    model, checkpoint = load_checkpoint(args.checkpoint)
    checkpoint_task = checkpoint.get("task")
    if checkpoint_task != task:
        raise ValueError(f"Checkpoint task is {checkpoint_task!r}, expected {task!r}")
    expected_spec = TrainConfig(task=task).network_spec
    if model.spec != expected_spec:
        if task == "low" and model.spec.macro_dim == 21:
            raise ValueError(
                "Legacy low-level checkpoints use a 21-dimensional conditioning "
                "input; the current RAM-conditioned policy requires 45 dimensions "
                "and must be retrained"
            )
        raise ValueError(
            f"Checkpoint network {model.spec} does not match expected {expected_spec}"
        )
    saved_config = checkpoint.get("config", {})
    config = TrainConfig(**saved_config) if saved_config else TrainConfig(task=task)
    metrics = evaluate_policy(
        model,
        config,
        episodes=args.episodes,
        seed=args.seed,
        gui=args.gui,
        fps=args.fps,
    )
    print(f"checkpoint={args.checkpoint}")
    print(f"global_transitions={checkpoint.get('global_transitions', 'unknown')}")
    for name, value in metrics.items():
        print(f"{name}={value:.6f}")
