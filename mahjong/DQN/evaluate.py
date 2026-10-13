import argparse
import numpy as np
from .algorithm import MaskedDoubleDQN
from .envs import make_vec_env

def evaluate(model, env, episodes=500):
    obs = env.reset()
    completed = 0
    wins = draws = losses = invalid = 0
    return_sum = length_sum = 0.0
    returns = np.zeros(env.num_envs, dtype=np.float64)
    lengths = np.zeros(env.num_envs, dtype=np.int64)
    while completed < episodes:
        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(actions)
        invalid += sum(bool(info.get('invalid_action')) for info in infos)
        for i, done in enumerate(dones):
            returns[i] += rewards[i]
            lengths[i] += 1
            if done:
                completed += 1
                reason = infos[i].get('terminal_reason')
                wins += reason == 'win'
                draws += reason == 'draw'
                losses += reason == 'loss'
                return_sum += returns[i]
                length_sum += lengths[i]
                returns[i] = 0.0
                lengths[i] = 0
    total = max(completed, 1)
    return {
        'episodes': completed,
        'self_draw_rate': wins / total,
        'draw_rate': draws / total,
        'loss_rate': losses / total,
        'return': return_sum / total,
        'length': length_sum / total,
        'invalid_action_count': invalid,
    }

def main():
    p = argparse.ArgumentParser(); p.add_argument('--model', required=True); p.add_argument('--episodes', type=int, default=500); p.add_argument('--num-envs', type=int, default=4); p.add_argument('--device', default='auto'); a=p.parse_args()
    env=make_vec_env(a.num_envs, 100001); model=MaskedDoubleDQN.load(a.model, env=env, device=a.device); print(evaluate(model, env, a.episodes)); env.close()
if __name__ == '__main__': main()
