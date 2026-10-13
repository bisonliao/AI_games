import argparse
import json

import numpy as np

from MahjongEnv.config import EnvConfig
from MahjongEnv.env import ACTION_NAMES, ACTION_OFFSETS
from MahjongEnv.tiles import NUM_TILES, RED, tile_name

from .agent import DDDQNAgent
from .config import TrainConfig
from .subproc_vec_env import SubprocessVectorEnv


def format_hand(vector, drawn_tile=None):
    """Decode the observed counts, separating exactly one newly drawn tile."""
    counts = np.rint(np.asarray(vector[:NUM_TILES]) * 4).astype(int)
    hand = np.repeat(np.arange(NUM_TILES), counts).tolist()
    if drawn_tile is not None:
        hand.remove(int(drawn_tile))
    hand.sort(key=lambda tile: (tile != RED, tile))
    text = ' '.join(tile_name(tile) for tile in hand) or '（空）'
    if drawn_tile is not None:
        text += f'  | 摸到：{tile_name(drawn_tile)}'
    return text


def format_action(action):
    if action == ACTION_OFFSETS['pass']:
        return '过'
    labels = ('打出', '碰', '明杠', '暗杠', '加杠')
    for name, label in zip(ACTION_NAMES, labels):
        tile = int(action) - ACTION_OFFSETS[name]
        if 0 <= tile < NUM_TILES:
            return f'{label} {tile_name(tile)}'
    raise ValueError(f'unknown action: {action}')


def print_decision(vector, info, action, env_index, episode, step):
    phase = '响应弃牌' if info['phase'] == 'claim' else '出牌'
    print(f'\n[环境 {env_index + 1} · 第 {episode} 局 · 第 {step} 步 · {phase}]')
    if info['phase'] == 'claim':
        print(f"牌友 {info['claim_from']} 打出：{tile_name(info['last_discard'])}")
    print(f"手牌：{format_hand(vector, info.get('drawn_tile'))}")
    print(f'模型动作：{format_action(action)}', flush=True)


def evaluate(
    agent: DDDQNAgent,
    config: TrainConfig,
    episodes: int | None = None,
    num_envs: int | None = None,
    seed_offset: int = 100_000,
    verbose: bool = False,
) -> dict[str, float]:
    episodes = int(episodes or config.eval_episodes)
    num_envs = int(num_envs or config.eval_num_envs)
    env_config = EnvConfig(observation_mode='vector', render_mode=None, opponent_action_delay=0.0)
    envs = SubprocessVectorEnv(num_envs, env_config, config.seed + seed_offset)
    returns = np.zeros(num_envs, dtype=np.float64)
    lengths = np.zeros(num_envs, dtype=np.int64)
    total = 0
    return_sum = 0.0
    length_sum = 0
    wins = draws = losses = 0
    episode_numbers = np.ones(num_envs, dtype=np.int64)
    try:
        obs, observation_infos = envs.reset()
        rng = np.random.default_rng(config.seed + seed_offset)
        while total < episodes:
            actions = agent.select_actions(obs['vector'], obs['action_mask'], epsilon=0.0, rng=rng)
            if verbose:
                for index, action in enumerate(actions):
                    print_decision(
                        obs['vector'][index], observation_infos[index], action,
                        index, episode_numbers[index], lengths[index] + 1,
                    )
            next_obs, rewards, terminated, truncated, infos = envs.step(actions)
            returns += rewards
            lengths += 1
            for index in range(num_envs):
                if bool(terminated[index]) or bool(truncated[index]):
                    reason = infos[index].get('terminal_reason')
                    if verbose:
                        result = {'win': '自摸成功', 'loss': '牌友胡牌', 'draw': '流局'}.get(reason, reason)
                        final_obs = infos[index]['final_observation']
                        print(
                            f'\n[环境 {index + 1} · 第 {episode_numbers[index]} 局结束] '
                            f'{result}，累计奖励：{returns[index]:+.4f}'
                        )
                        print(
                            '终局手牌：' + format_hand(
                                final_obs['vector'], infos[index].get('drawn_tile')
                            ), flush=True,
                        )
                    total += 1
                    return_sum += returns[index]
                    length_sum += lengths[index]
                    if reason == 'win':
                        wins += 1
                    elif reason == 'draw':
                        draws += 1
                    elif reason == 'loss':
                        losses += 1
                    returns[index] = 0.0
                    lengths[index] = 0
                    episode_numbers[index] += 1
                    if total >= episodes:
                        break
            obs = next_obs
            # After autoreset, next_obs belongs to the new episode. The
            # top-level info still describes the finished episode.
            observation_infos = [info.get('reset_info', info) for info in infos]
    finally:
        envs.close()
    denominator = max(1, total)
    return {
        'eval/self_draw_rate': wins / denominator,
        'eval/win_rate': wins / denominator,
        'eval/draw_rate': draws / denominator,
        'eval/loss_rate': losses / denominator,
        'eval/return': return_sum / denominator,
        'eval/length': length_sum / denominator,
        'eval/episodes': float(total),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description='Evaluate a DDDQN checkpoint')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--episodes', type=int)
    parser.add_argument('--num-envs', type=int)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--quiet', action='store_true', help='只输出评测汇总，不打印逐步手牌与动作')
    args = parser.parse_args(argv)
    if args.device == 'cuda' and not __import__('torch').cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')
    import torch
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = TrainConfig.from_dict(checkpoint['config'])
    agent = DDDQNAgent(config, device)
    agent.load_checkpoint(checkpoint)
    # A single environment makes the printed game sequence easy to follow.
    num_envs = args.num_envs if args.num_envs is not None else (None if args.quiet else 1)
    metrics = evaluate(
        agent, config, episodes=args.episodes, num_envs=num_envs,
        verbose=not args.quiet,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
