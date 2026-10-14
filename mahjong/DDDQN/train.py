import argparse
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from MahjongEnv.config import EnvConfig

from .agent import DDDQNAgent
from .config import TrainConfig
from .evaluate import evaluate
from .metrics import EpisodeMetrics, MetricsLogger, resource_values
from .replay import ReplayBuffer
from .subproc_vec_env import SubprocessVectorEnv


def resolve_device(requested: str) -> torch.device:
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')
    if requested == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        print('WARNING: CUDA is unavailable; falling back to CPU', flush=True)
        return torch.device('cpu')
    return torch.device(requested)


def _root_path(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else Path(__file__).resolve().parent.parent / value


def save_checkpoint(path: Path, state: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        torch.save(state, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def metric_key(metrics: dict[str, float]) -> tuple[float, float, float]:
    return (metrics.get('eval/self_draw_rate', 0.0), metrics.get('eval/return', 0.0), -metrics.get('eval/length', 0.0))


def train(config: TrainConfig, resume: str | None = None, run_id: str | None = None):
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    if device.type == 'cpu':
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    agent = DDDQNAgent(config, device)
    replay = ReplayBuffer(config.replay_capacity, config.obs_dim, config.action_dim, config.seed)
    env_config = EnvConfig(
        observation_mode='vector',
        render_mode=None,
        opponent_action_delay=config.opponent_action_delay,
    )
    envs = SubprocessVectorEnv(config.num_envs, env_config, config.seed)
    log_root = _root_path(config.log_dir)
    logger = MetricsLogger(str(log_root), run_id=run_id)
    # 使用 logger 最终确定的 run_id，自动生成和手动指定的运行都独立保存。
    checkpoint_root = _root_path(config.checkpoint_dir) / logger.run_id
    episode_metrics = EpisodeMetrics(config.num_envs)
    rng = np.random.default_rng(config.seed)
    env_steps = 0
    best_metric = None
    last_log_time = time.monotonic()
    last_log_steps = 0
    next_log = config.log_interval_steps
    next_checkpoint = config.checkpoint_interval_steps
    train_cursor = config.learning_starts
    last_checkpoint_step = 0
    last_train_stats = {}

    try:
        if resume:
            checkpoint = torch.load(resume, map_location=device, weights_only=False)
            env_steps = agent.load_checkpoint(checkpoint)
            # 在原目录续训时沿用历史最佳；另开 run 时从本次评测重新排名，
            # 避免新目录因未超过旧 run 的最佳成绩而一直没有 best.pt。
            if Path(resume).resolve().parent == checkpoint_root.resolve():
                best_metric = checkpoint.get('best_metric')
            if checkpoint.get('numpy_rng_state') is not None:
                rng.bit_generator.state = checkpoint['numpy_rng_state']
            train_cursor = max(
                config.learning_starts,
                env_steps + config.train_frequency * config.num_envs,
            )
            print(f'Resumed {resume} at env_steps={env_steps}', flush=True)
        obs, _ = envs.reset()
        while env_steps < config.total_env_steps:
            epsilon = agent.epsilon(env_steps)
            actions = agent.select_actions(obs['vector'], obs['action_mask'], epsilon, rng)
            next_obs, rewards, terminated, truncated, infos = envs.step(actions)
            replay.add_batch(
                obs['vector'],
                actions,
                rewards,
                next_obs['vector'],
                terminated,
                truncated,
                obs['action_mask'],
                next_obs['action_mask'],
            )
            episode_metrics.update(rewards, terminated, truncated, infos)
            obs = next_obs
            env_steps += config.num_envs

            if env_steps >= config.learning_starts and len(replay) >= config.batch_size:
                while train_cursor <= env_steps:
                    for _ in range(config.gradient_steps):
                        last_train_stats = agent.train_batch(replay.sample(config.batch_size))
                    # SB3's train_freq is measured in vector steps. Each
                    # vector step advances num_envs transitions, so match its
                    # update ratio: train once per train_frequency*num_envs
                    # environment transitions.
                    train_cursor += config.train_frequency * config.num_envs

            if env_steps >= next_log:
                now = time.monotonic()
                elapsed = max(1e-6, now - last_log_time)
                values = episode_metrics.values()
                values.update({f'train/{key}': value for key, value in last_train_stats.items()})
                values.update({
                    'train/epsilon': epsilon,
                    'train/learning_rate': config.learning_rate,
                    'train/replay_size': len(replay),
                    'train/updates': agent.learner_updates,
                    'train/env_steps': env_steps,
                    'train/steps_per_second': (env_steps - last_log_steps) / elapsed,
                })
                values.update(resource_values())
                logger.log(env_steps, values)
                last_log_time = now
                last_log_steps = env_steps
                while next_log <= env_steps:
                    next_log += config.log_interval_steps

            if env_steps >= next_checkpoint:
                eval_metrics = evaluate(agent, config)
                logger.log(env_steps, eval_metrics)
                current_metric = metric_key(eval_metrics)
                is_best = best_metric is None or current_metric > tuple(best_metric)
                if is_best:
                    best_metric = current_metric
                checkpoint_state = agent.checkpoint_state(env_steps, best_metric, config, eval_metrics)
                checkpoint_state['run_id'] = logger.run_id
                checkpoint_state['numpy_rng_state'] = rng.bit_generator.state
                save_checkpoint(checkpoint_root / 'last.pt', checkpoint_state)
                if is_best:
                    save_checkpoint(checkpoint_root / 'best.pt', checkpoint_state)
                last_checkpoint_step = env_steps
                while next_checkpoint <= env_steps:
                    next_checkpoint += config.checkpoint_interval_steps

        if env_steps > 0 and last_checkpoint_step != env_steps:
            eval_metrics = evaluate(agent, config)
            logger.log(env_steps, eval_metrics)
            current_metric = metric_key(eval_metrics)
            is_best = best_metric is None or current_metric > tuple(best_metric)
            if is_best:
                best_metric = current_metric
            checkpoint_state = agent.checkpoint_state(env_steps, best_metric, config, eval_metrics)
            checkpoint_state['run_id'] = logger.run_id
            checkpoint_state['numpy_rng_state'] = rng.bit_generator.state
            save_checkpoint(checkpoint_root / 'last.pt', checkpoint_state)
            if is_best:
                save_checkpoint(checkpoint_root / 'best.pt', checkpoint_state)
    finally:
        envs.close()
        logger.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Train masked Dueling Double DQN on Hongzhong Mahjong')
    defaults = TrainConfig()
    parser.add_argument('--total-env-steps', type=int, default=defaults.total_env_steps)
    parser.add_argument('--num-envs', type=int, default=defaults.num_envs)
    parser.add_argument('--eval-num-envs', type=int, default=defaults.eval_num_envs)
    parser.add_argument('--seed', type=int, default=defaults.seed)
    parser.add_argument('--batch-size', type=int, default=defaults.batch_size)
    parser.add_argument('--replay-capacity', type=int, default=defaults.replay_capacity)
    parser.add_argument('--learning-rate', type=float, default=defaults.learning_rate)
    parser.add_argument('--gamma', type=float, default=defaults.gamma)
    parser.add_argument('--learning-starts', type=int, default=defaults.learning_starts)
    parser.add_argument('--train-frequency', type=int, default=defaults.train_frequency)
    parser.add_argument('--gradient-steps', type=int, default=defaults.gradient_steps)
    parser.add_argument('--target-update-interval', type=int, default=defaults.target_update_interval)
    parser.add_argument('--hidden-dim', type=int, default=defaults.hidden_dim)
    parser.add_argument('--epsilon-start', type=float, default=defaults.epsilon_start)
    parser.add_argument('--epsilon-final', type=float, default=defaults.epsilon_final)
    parser.add_argument(
        '--epsilon-decay-steps',
        type=int,
        default=defaults.epsilon_decay_steps,
    )
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=defaults.max_grad_norm,
    )
    parser.add_argument('--log-interval-steps', type=int, default=defaults.log_interval_steps)
    parser.add_argument('--checkpoint-interval-steps', type=int, default=defaults.checkpoint_interval_steps)
    parser.add_argument('--eval-episodes', type=int, default=defaults.eval_episodes)
    parser.add_argument(
        '--opponent-action-delay',
        type=float,
        default=defaults.opponent_action_delay,
    )
    parser.add_argument('--checkpoint-dir', default=defaults.checkpoint_dir)
    parser.add_argument('--log-dir', default=defaults.log_dir)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default=defaults.device)
    parser.add_argument('--resume')
    parser.add_argument(
        '--run-id',
        help='Subdirectory for logs and checkpoints; defaults to a timestamp_pid ID',
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = TrainConfig(**{key: getattr(args, key) for key in (
        'total_env_steps', 'num_envs', 'eval_num_envs', 'seed', 'batch_size', 'replay_capacity',
        'learning_rate', 'gamma', 'learning_starts', 'train_frequency', 'gradient_steps',
        'target_update_interval', 'hidden_dim', 'epsilon_start', 'epsilon_final',
        'epsilon_decay_steps', 'max_grad_norm', 'log_interval_steps',
        'checkpoint_interval_steps', 'eval_episodes', 'checkpoint_dir', 'log_dir',
        'device', 'opponent_action_delay')})
    train(config, args.resume, args.run_id)


if __name__ == '__main__':
    main()
