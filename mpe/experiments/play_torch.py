"""Evaluate or render a checkpoint produced by ``train_torch.py``.

Examples:

  # Headless deterministic evaluation (the default)
  python -m experiments.play_torch --checkpoint ./chkpt/maddpg/legacy/official/simple_spread

  # Open the MPE viewer and play at roughly 10 environment steps per second
  python -m experiments.play_torch --checkpoint ./chkpt/maddpg/legacy/official/simple_spread --render
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

from experiments.train_torch import (
    ALGORITHM_NAME,
    CHECKPOINT_VERSION,
    SUPPORTED_CHECKPOINT_VERSIONS,
    _seed_everything,
    evaluate_checkpoint,
)
from maddpg.common.tf_util_torch import get_device, load_state, resolve_state_path


_REQUIRED_METADATA = (
    "algorithm",
    "env_backend",
    "scenario",
    "policy_mode",
    "target_init",
    "max_episode_len",
    "num_adversaries",
    "good_policy",
    "adv_policy",
    "num_units",
    "lr",
    "gamma",
    "batch_size",
    "action_specs",
)
_REQUIRED_V3_METADATA = tuple(
    key for key in _REQUIRED_METADATA if key != "algorithm"
)
_REQUIRED_V2_METADATA = (
    "env_backend",
    "scenario",
    "policy_mode",
    "target_init",
    "action_specs",
)

_STRICT_SUCCESS_METRIC = "episode_success"
_LANDMARK_CENTER_SUCCESS_METRIC = "landmark_center_success"


def _infer_num_units(checkpoint):
    try:
        weight = checkpoint["trainers"][0]["p_net"]["fc1.weight"]
        return int(weight.shape[0])
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise ValueError(
            "version 2 checkpoint does not contain an inferable actor network"
        ) from exc


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        "Play or evaluate a PyTorch MADDPG checkpoint",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="checkpoint file or directory; a directory selects the greatest step",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="number of evaluation episodes (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10000,
        help="base environment/evaluation seed (default: 10000)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="open the MPE GUI; omitted by default for headless evaluation",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="render speed in environment steps/second; 0 disables throttling",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        default=False,
        help="sample policy actions instead of deterministic argmax/mean actions",
    )
    parser.add_argument(
        "--report-json",
        default="",
        help="optional path for a JSON evaluation report",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="force CPU inference",
    )
    return parser.parse_args(argv)


def _evaluation_args_from_checkpoint(checkpoint, options):
    if not isinstance(checkpoint, dict):
        raise ValueError("unsupported checkpoint: expected a dictionary")
    version = checkpoint.get("checkpoint_version")
    if version not in SUPPORTED_CHECKPOINT_VERSIONS:
        raise ValueError(
            "unsupported checkpoint version {}; supported versions are {}".format(
                version, SUPPORTED_CHECKPOINT_VERSIONS
            )
        )
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("unsupported checkpoint: metadata must be a dictionary")
    required_metadata = (
        _REQUIRED_V2_METADATA
        if version == 2
        else _REQUIRED_V3_METADATA
        if version == 3
        else _REQUIRED_METADATA
    )
    missing = [key for key in required_metadata if key not in metadata]
    if missing:
        raise ValueError(
            "unsupported checkpoint: missing metadata fields {}".format(
                missing
            )
        )
    if version == CHECKPOINT_VERSION and metadata["algorithm"] != ALGORITHM_NAME:
        raise ValueError(
            "unsupported checkpoint algorithm {!r}; expected {!r}".format(
                metadata["algorithm"], ALGORITHM_NAME
            )
        )
    num_units = (
        int(metadata["num_units"])
        if "num_units" in metadata
        else _infer_num_units(checkpoint)
    )

    return SimpleNamespace(
        scenario=metadata["scenario"],
        env_backend=metadata["env_backend"],
        policy_mode=metadata["policy_mode"],
        target_init=metadata["target_init"],
        max_episode_len=int(metadata.get("max_episode_len", 25)),
        num_adversaries=int(metadata.get("num_adversaries", 0)),
        good_policy=metadata.get("good_policy", "maddpg"),
        adv_policy=metadata.get("adv_policy", "maddpg"),
        num_units=num_units,
        lr=float(metadata.get("lr", 1e-2)),
        gamma=float(metadata.get("gamma", 0.95)),
        batch_size=int(metadata.get("batch_size", 1024)),
        checkpoint_eval_episodes=int(options.episodes),
        checkpoint_eval_seed=int(options.seed),
        display=bool(options.render),
    )


def _print_episode(result, total_episodes):
    agent_rewards = ", ".join(
        "{}({})={:.6f}".format(name, role, value)
        for name, role, value in zip(
            result["agent_names"],
            result["agent_roles"],
            result["agent_episode_rewards"],
        )
    )
    task_metrics = ""
    if result["task_metrics"]:
        task_metrics = ", task: " + ", ".join(
            "{}={:.6f}".format(name, value)
            for name, value in sorted(result["task_metrics"].items())
        )
    print(
        "[Play {}/{}] seed: {}, reward: {:.6f}, length: {}, "
        "agent rewards: [{}]{}".format(
            result["episode_index"],
            total_episodes,
            result["seed"],
            result["episode_reward"],
            result["episode_length"],
            agent_rewards,
            task_metrics,
        )
    )


def _add_success_summary(evaluation, metric_name, field_prefix):
    success_rate = evaluation["task_metrics"].get(metric_name)
    if success_rate is None:
        return None
    success_count = int(
        round(float(success_rate) * evaluation["evaluation_episodes"])
    )
    evaluation[field_prefix + "_count"] = success_count
    evaluation[field_prefix + "_rate"] = float(success_rate)
    return success_count, float(success_rate)


def play(options):
    if options.episodes <= 0:
        raise ValueError("--episodes must be greater than 0")
    if options.fps < 0:
        raise ValueError("--fps must be greater than or equal to 0")

    device = get_device(use_cuda=not options.no_cuda)
    checkpoint_input = os.path.expanduser(options.checkpoint)
    checkpoint = load_state(checkpoint_input, map_location=device)
    evaluation_args = _evaluation_args_from_checkpoint(checkpoint, options)
    _seed_everything(options.seed)

    checkpoint_path = Path(resolve_state_path(checkpoint_input)).resolve()
    metadata = checkpoint["metadata"]
    print("Checkpoint: {}".format(checkpoint_path))
    if checkpoint["checkpoint_version"] == 2:
        print(
            "[Compatibility] version 2 checkpoint: inferred network width; "
            "using the historical default evaluation settings"
        )
    print(
        "Environment: {} / {}, policy: {}, device: {}, GUI: {}".format(
            metadata["env_backend"],
            metadata["scenario"],
            metadata["policy_mode"],
            device,
            "on" if options.render else "off",
        )
    )

    render_delay = (
        1.0 / options.fps if options.render and options.fps > 0 else 0.0
    )
    evaluation = evaluate_checkpoint(
        checkpoint,
        evaluation_args,
        device,
        render=options.render,
        deterministic=not options.stochastic,
        render_delay=render_delay,
        episode_callback=lambda result: _print_episode(
            result, options.episodes
        ),
    )
    evaluation["checkpoint"] = os.fspath(checkpoint_path)
    evaluation["render"] = bool(options.render)
    evaluation["metadata"] = metadata
    strict_success = _add_success_summary(
        evaluation, _STRICT_SUCCESS_METRIC, "task_success"
    )
    landmark_center_success = _add_success_summary(
        evaluation,
        _LANDMARK_CENTER_SUCCESS_METRIC,
        "task_landmark_center_success",
    )

    print(
        "[Summary] episodes: {}, mean reward: {:.6f}, std: {:.6f}, "
        "mean length: {:.2f}".format(
            evaluation["evaluation_episodes"],
            evaluation["episode_reward_mean"],
            evaluation["episode_reward_std"],
            evaluation["episode_length_mean"],
        )
    )
    print(
        "[Summary] mean agent rewards: [{}]".format(
            ", ".join(
                "{}({})={:.6f}".format(name, role, value)
                for name, role, value in zip(
                    evaluation["agent_names"],
                    evaluation["agent_roles"],
                    evaluation["agent_episode_reward_mean"],
                )
            )
        )
    )
    if evaluation["task_metrics"]:
        print(
            "[Summary] task metrics: {}".format(
                ", ".join(
                    "{}={:.6f}".format(name, value)
                    for name, value in sorted(
                        evaluation["task_metrics"].items()
                    )
                )
            )
        )
    if strict_success is not None:
        print(
            "[Summary] strict full-coverage success (d < 0.10): "
            "{}/{} ({:.6f})".format(
                strict_success[0],
                evaluation["evaluation_episodes"],
                strict_success[1],
            )
        )
    if landmark_center_success is not None:
        print(
            "[Summary] landmark-center success (d < 0.15): "
            "{}/{} ({:.6f})".format(
                landmark_center_success[0],
                evaluation["evaluation_episodes"],
                landmark_center_success[1],
            )
        )

    if options.report_json:
        report_path = Path(options.report_json).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as stream:
            json.dump(evaluation, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print("Report: {}".format(report_path))

    return evaluation


def main(argv=None):
    options = parse_args(argv)
    try:
        play(options)
    except KeyboardInterrupt:
        print("\nPlay interrupted; environment closed.")


if __name__ == "__main__":
    main()
