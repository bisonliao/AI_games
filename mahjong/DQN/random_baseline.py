"""Evaluate a uniformly random legal-action policy."""

import argparse

import numpy as np

from MahjongEnv.config import EnvConfig
from MahjongEnv.env import MahjongEnv


def run_random_baseline(episodes: int = 100, seed: int = 1) -> dict[str, float]:
    """Play ``episodes`` games with uniformly sampled legal actions."""
    if episodes <= 0:
        raise ValueError('episodes must be positive')

    env = MahjongEnv(
        EnvConfig(
            observation_mode='vector',
            render_mode=None,
            opponent_action_delay=0.0,
        ),
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    wins = draws = losses = 0
    return_sum = 0.0
    length_sum = 0

    try:
        completed = 0
        while completed < episodes:
            observation, info = env.reset()
            episode_return = 0.0
            episode_length = 0

            # A deal can terminate immediately because of a heaven win.
            if info.get('terminal_reason') is not None:
                reason = info['terminal_reason']
            else:
                while True:
                    legal_actions = np.flatnonzero(observation['action_mask'])
                    if legal_actions.size == 0:
                        raise RuntimeError('environment returned an empty action mask')
                    action = int(rng.choice(legal_actions))
                    observation, reward, terminated, truncated, info = env.step(action)
                    episode_return += float(reward)
                    episode_length += 1
                    if terminated or truncated:
                        reason = info.get('terminal_reason')
                        break

            completed += 1
            return_sum += episode_return
            length_sum += episode_length
            if reason == 'win':
                wins += 1
            elif reason == 'draw':
                draws += 1
            elif reason == 'loss':
                losses += 1
            else:
                raise RuntimeError(f'unknown terminal reason: {reason!r}')
    finally:
        env.close()

    total = float(episodes)
    return {
        'episodes': total,
        'win_rate': wins / total,
        'self_draw_rate': wins / total,
        'draw_rate': draws / total,
        'loss_rate': losses / total,
        'mean_return': return_sum / total,
        'mean_length': length_sum / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate a random legal-action Mahjong policy')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    metrics = run_random_baseline(args.episodes, args.seed)
    print(f"episodes={int(metrics['episodes'])}")
    print(f"win_rate={metrics['win_rate']:.4f} ({metrics['win_rate']:.2%})")
    print(f"self_draw_rate={metrics['self_draw_rate']:.4f} ({metrics['self_draw_rate']:.2%})")
    print(f"draw_rate={metrics['draw_rate']:.4f} ({metrics['draw_rate']:.2%})")
    print(f"loss_rate={metrics['loss_rate']:.4f} ({metrics['loss_rate']:.2%})")
    print(f"mean_return={metrics['mean_return']:.4f}")
    print(f"mean_length={metrics['mean_length']:.4f}")


if __name__ == '__main__':
    main()
