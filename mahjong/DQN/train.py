import argparse
from pathlib import Path
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from .algorithm import MaskedDoubleDQN
from .policy import MaskedDuelingPolicy
from .callbacks import ActionStatsCallback, make_eval_callback
from .config import DQNConfig
from .envs import make_vec_env

def train(cfg: DQNConfig, run_id='default'):
    log_path = Path(cfg.log_dir) / f'sb3_{run_id}'
    # 与 TensorBoard 运行目录使用同一标识，避免不同 run 覆盖模型。
    checkpoint_path = Path(cfg.checkpoint_dir) / f'sb3_{run_id}'
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    env = make_vec_env(cfg.num_envs, cfg.seed)
    eval_env = make_vec_env(cfg.eval_num_envs, cfg.seed + 100_000)
    model = MaskedDoubleDQN(
        MaskedDuelingPolicy,
        env,
        learning_rate=cfg.learning_rate,
        buffer_size=cfg.buffer_size,
        batch_size=cfg.batch_size,
        learning_starts=cfg.learning_starts,
        train_freq=cfg.train_freq,
        gradient_steps=cfg.gradient_steps,
        target_update_interval=cfg.target_update_interval,
        gamma=cfg.gamma,
        exploration_fraction=cfg.exploration_fraction,
        exploration_initial_eps=cfg.exploration_initial_eps,
        exploration_final_eps=cfg.exploration_final_eps,
        max_grad_norm=cfg.max_grad_norm,
        tensorboard_log=str(log_path),
        verbose=1,
        seed=cfg.seed,
        device=cfg.device,
        policy_kwargs={'hidden_dim': cfg.hidden_dim},
    )
    callback_freq = max(1, cfg.eval_freq // cfg.num_envs)
    # SB3's DQN ``log_interval`` is measured in rollout iterations. One
    # iteration advances train_freq * num_envs environment steps.
    log_interval = max(1, cfg.log_interval_steps // (cfg.train_freq * cfg.num_envs))
    callbacks = CallbackList([
        ActionStatsCallback(),
        CheckpointCallback(
            save_freq=callback_freq,
            save_path=str(checkpoint_path),
            name_prefix='sb3_checkpoint',
        ),
        make_eval_callback(
            eval_env,
            callback_freq,
            cfg.eval_episodes,
            str(checkpoint_path / 'best'),
            str(log_path / 'eval'),
        ),
    ])
    try:
        model.learn(
            total_timesteps=cfg.total_timesteps,
            callback=callbacks,
            tb_log_name=run_id,
            log_interval=log_interval,
            progress_bar=False,
        )
        model.save(str(checkpoint_path / 'sb3_last'))
    finally:
        env.close()
        eval_env.close()

def main():
    parser = argparse.ArgumentParser(description='Train masked SB3 Double-DQN on Hongzhong Mahjong')
    defaults = DQNConfig()
    integer_options = (
        'total_timesteps', 'num_envs', 'eval_num_envs', 'seed', 'batch_size',
        'buffer_size', 'learning_starts', 'train_freq', 'gradient_steps',
        'target_update_interval', 'hidden_dim', 'eval_freq', 'eval_episodes',
        'log_interval_steps',
    )
    for name in integer_options:
        parser.add_argument(f'--{name.replace("_", "-")}', type=int, default=getattr(defaults, name))
    parser.add_argument('--learning-rate', type=float, default=defaults.learning_rate)
    parser.add_argument('--exploration-fraction', type=float, default=defaults.exploration_fraction)
    parser.add_argument('--exploration-initial-eps', type=float, default=defaults.exploration_initial_eps)
    parser.add_argument('--exploration-final-eps', type=float, default=defaults.exploration_final_eps)
    parser.add_argument('--device', default=defaults.device)
    parser.add_argument('--run-id', default='run')
    parser.add_argument('--log-dir', default=defaults.log_dir)
    parser.add_argument('--checkpoint-dir', default=defaults.checkpoint_dir)
    args = parser.parse_args()
    values = {name: getattr(args, name) for name in DQNConfig.__dataclass_fields__ if hasattr(args, name)}
    train(DQNConfig(**values), args.run_id)
if __name__=='__main__': main()
