"""Generate synchronized visual demonstrations from the vector SAC expert."""
from __future__ import annotations

import argparse
from pathlib import Path

from .common import DEFAULT_EXPERT, EnvConfig, new_run, setup
from .dataset import save_episode, write_metadata
from .expert import VectorExpert
from .randomized_env import RandomizedPixelTaskEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert", type=Path, default=DEFAULT_EXPERT)
    parser.add_argument("--vecnormalize", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--frame-stack", type=int, default=2)
    parser.add_argument("--max-episode-steps", type=int, default=150)
    parser.add_argument("--action-repeat", type=int, default=8)
    parser.add_argument("--camera-scale", type=float, default=1.0)
    parser.add_argument("--train-seed-start", type=int, default=0)
    parser.add_argument("--camera-jitter", type=float, default=0.00)
    parser.add_argument("--object-position-jitter", type=float, default=0.04)
    parser.add_argument("--goal-position-jitter", type=float, default=0.04)
    parser.add_argument("--initial-joint-jitter", type=float, default=0.00) #必须为0，否则会导致反关节而不能成功
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    return args


def main() -> None:
    args = parse_args()
    setup(args.seed)
    config = EnvConfig(
        image_size=args.image_size,
        frame_stack=args.frame_stack,
        max_episode_steps=args.max_episode_steps,
        action_repeat=args.action_repeat,
        camera_scale=args.camera_scale,
    )
    root = new_run("expert_showcase", args.out)
    teacher = VectorExpert(args.expert, args.vecnormalize)
    successes = 0

    for episode_id in range(args.episodes):
        episode_seed = args.train_seed_start + episode_id
        env = RandomizedPixelTaskEnv(
            task=config.task,
            image_size=config.image_size,
            frame_stack=config.frame_stack,
            camera_scale=config.camera_scale,
            max_episode_steps=config.max_episode_steps,
            action_repeat=config.action_repeat,
            seed=episode_seed,
            camera_jitter=args.camera_jitter,
            object_position_jitter=args.object_position_jitter,
            goal_position_jitter=args.goal_position_jitter,
            initial_joint_jitter=args.initial_joint_jitter,
            randomize=True,
        )
        observation, _ = env.reset(seed=episode_seed)
        images, proprio, actions, rewards = [], [], [], []
        terminated, truncated, stages = [], [], []
        episode_return = 0.0

        while True:
            action = teacher.action(env)
            images.append(observation["image"].copy())
            proprio.append(observation["proprio"].copy())
            actions.append(action.copy())
            observation, reward, is_terminal, is_truncated, info = env.step(action)
            rewards.append(reward)
            terminated.append(is_terminal)
            truncated.append(is_truncated)
            stages.append(info.get("stage_index", -1))
            episode_return += reward
            if is_terminal or is_truncated:
                break

        success = bool(info.get("success", False))
        successes += int(success)
        save_episode(
            root / f"episode_{episode_id:06d}.npz",
            image=images,
            proprio=proprio,
            action=actions,
            reward=rewards,
            terminated=terminated,
            truncated=truncated,
            success=success,
            episode_return=episode_return,
            stage=stages,
            failure_reason=info.get("failure_reason", ""),
            episode_metadata=env.randomization_metadata,
        )
        env.close()

    write_metadata(
        root,
        stage="expert_showcase",
        expert=str(args.expert.resolve()),
        vecnormalize=str(teacher.normalizer_path),
        env_config=config.__dict__,
        seed=args.seed,
        train_seed_start=args.train_seed_start,
        unseen_eval_seed_start=10000,
        unseen_eval_episodes=100,
        camera_jitter=args.camera_jitter,
        object_position_jitter=args.object_position_jitter,
        goal_position_jitter=args.goal_position_jitter,
        initial_joint_jitter=args.initial_joint_jitter,
        episodes=args.episodes,
        successes=successes,
    )
    if successes == 0:
        raise SystemExit("No successful demonstrations; verify checkpoint and seeds.")
    print(f"Saved {args.episodes} episodes ({successes} successful) to {root}")


if __name__ == "__main__":
    main()
