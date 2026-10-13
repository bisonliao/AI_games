from dataclasses import dataclass

@dataclass
class DQNConfig:
    total_timesteps: int = 5_000_000
    num_envs: int = 8
    eval_num_envs: int = 4
    seed: int = 1
    gamma: float = 0.99
    learning_rate: float = 1e-4
    buffer_size: int = 200_000
    batch_size: int = 256
    learning_starts: int = 20_000
    train_freq: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 10_000
    hidden_dim: int = 512
    exploration_fraction: float = 0.4
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    max_grad_norm: float = 10.0
    log_interval_steps: int = 10_000
    eval_freq: int = 100_000
    eval_episodes: int = 500
    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'tb_logs'
    device: str = 'auto'
