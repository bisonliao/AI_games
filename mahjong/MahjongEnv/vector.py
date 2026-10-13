from functools import partial

from .env import MahjongEnv, make_env
from .config import EnvConfig

def make_env_fn(config=None, seed=None, rank=0):
    # module-level callable + pickleable arguments for SubprocVecEnv
    return partial(
        make_env,
        config=config or EnvConfig(),
        seed=seed,
        rank=rank,
    )

def make_env_fns(n, config=None, seed=None):
    return [make_env_fn(config, seed, i) for i in range(n)]

def rollout(envs, policy, steps):
    obs = [e.reset()[0] if hasattr(e, 'reset') else e for e in envs]
    out = []
    for _ in range(steps):
        acts = [int(policy(observation, env)) for observation, env in zip(obs, envs)]
        for index, env in enumerate(envs):
            next_obs, reward, terminated, truncated, info = env.step(acts[index])
            out.append(
                (
                    obs[index],
                    acts[index],
                    reward,
                    next_obs,
                    terminated,
                    truncated,
                    info,
                )
            )
            obs[index] = (
                env.reset()[0] if terminated or truncated else next_obs
            )
    return out
