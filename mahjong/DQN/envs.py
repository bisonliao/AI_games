from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from MahjongEnv.config import EnvConfig
from MahjongEnv.env import MahjongEnv


class PlayableMahjongEnv(MahjongEnv):
    """MahjongEnv wrapper that never exposes an immediately terminal reset."""

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        while info.get('terminal_reason') is not None:
            observation, info = super().reset(options=options)
        return observation, info

def make_env(seed: int, rank: int = 0):
    def factory():
        return PlayableMahjongEnv(
            EnvConfig(observation_mode='vector', render_mode=None, opponent_action_delay=0.0),
            seed=int(seed) + int(rank),
        )
    return factory

def make_vec_env(num_envs: int, seed: int = 1, start_method: str = 'spawn'):
    factories = [make_env(seed, rank) for rank in range(int(num_envs))]
    if num_envs == 1:
        return DummyVecEnv(factories)
    return SubprocVecEnv(factories, start_method=start_method)
