"""Environment adapters for the faithful PyTorch MADDPG port."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np

from maddpg.common.distributions_torch import ActionSpec


SCENARIO_TO_PETTINGZOO = {
    "simple": "simple_v3",
    "simple_spread": "simple_spread_v3",
    "simple_tag": "simple_tag_v3",
    "simple_adversary": "simple_adversary_v3",
    "simple_push": "simple_push_v3",
    "simple_reference": "simple_reference_v3",
    "simple_crypto": "simple_crypto_v3",
    "simple_speaker_listener": "simple_speaker_listener_v4",
    "simple_world_comm": "simple_world_comm_v3",
}


def _legacy_gym_reraise(prefix=None, suffix=None):
    """Compatibility implementation removed from Gym after old MPE shipped."""

    _, exception, traceback = sys.exc_info()
    if exception is None:
        raise RuntimeError("reraise() must be called while handling an exception")
    message_parts = []
    if prefix:
        message_parts.append(str(prefix))
    message_parts.append(str(exception))
    if suffix:
        message_parts.append(str(suffix))
    message = " ".join(message_parts)
    try:
        compatible_exception = type(exception)(message)
    except TypeError:
        compatible_exception = RuntimeError(message)
    raise compatible_exception.with_traceback(traceback) from exception


def _ensure_legacy_rendering_compatibility():
    """Provide the one Gym 0.10 helper imported by archived MPE rendering."""

    import gym.utils

    if not hasattr(gym.utils, "reraise"):
        gym.utils.reraise = _legacy_gym_reraise


class LegacyMPEAdapter:
    """Expose OpenAI's list-based MultiAgentEnv through a ParallelEnv-like API."""

    def __init__(self, env, max_cycles: int):
        self.env = env
        self.max_cycles = int(max_cycles)
        self.possible_agents = ["agent_{}".format(i) for i in range(env.n)]
        self.agents = list(self.possible_agents)
        self._steps = 0

    @property
    def unwrapped(self):
        return self.env

    def observation_space(self, agent):
        return self.env.observation_space[self.possible_agents.index(agent)]

    def action_space(self, agent):
        return self.env.action_space[self.possible_agents.index(agent)]

    def reset(self, seed=None, options=None):
        del options
        if seed is not None:
            # The archived MPE uses NumPy's module-level RandomState.
            np.random.seed(seed)
        observations = self.env.reset()
        self.agents = list(self.possible_agents)
        self._steps = 0
        return dict(zip(self.possible_agents, observations)), {
            agent: {} for agent in self.possible_agents
        }

    def step(self, actions):
        action_n = [actions[agent] for agent in self.possible_agents]
        observations, rewards, dones, raw_info = self.env.step(action_n)
        self._steps += 1
        truncated = self._steps >= self.max_cycles
        infos = raw_info.get("n", []) if isinstance(raw_info, dict) else []
        return (
            dict(zip(self.possible_agents, observations)),
            dict(zip(self.possible_agents, rewards)),
            dict(zip(self.possible_agents, (bool(done) for done in dones))),
            {agent: truncated for agent in self.possible_agents},
            {
                agent: (infos[i] if i < len(infos) else {})
                for i, agent in enumerate(self.possible_agents)
            },
        )

    def render(self):
        _ensure_legacy_rendering_compatibility()
        return self.env.render()

    def close(self):
        close = getattr(self.env, "close", None)
        if close is not None:
            close()


class PettingZooActionAdapter:
    """Translate official soft-action ordering to PettingZoo continuous MPE."""

    def __init__(self, env, policy_mode: str):
        self.env = env
        self.policy_mode = policy_mode
        self.possible_agents = list(env.possible_agents)

    @property
    def agents(self):
        return self.env.agents

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def observation_space(self, agent):
        return self.env.observation_space(agent)

    def action_space(self, agent):
        return self.env.action_space(agent)

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def _translate_action(self, agent_index: int, action):
        if self.policy_mode != "official":
            return action

        translated = np.asarray(action, dtype=np.float32).copy()
        world_agent = self.unwrapped.world.agents[agent_index]
        if world_agent.movable:
            # Archived MPE's vector path computes [a1-a2, a3-a4], whereas
            # PettingZoo computes [a2-a1, a4-a3]. Preserve archived semantics.
            translated[[1, 2]] = translated[[2, 1]]
            translated[[3, 4]] = translated[[4, 3]]
        return translated

    def step(self, actions):
        translated = {
            agent: self._translate_action(i, actions[agent])
            for i, agent in enumerate(self.possible_agents)
            if agent in actions
        }
        return self.env.step(translated)

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


def infer_action_specs(env, agent_list: Sequence[str], policy_mode: str):
    if policy_mode == "gaussian":
        return [ActionSpec(mode="gaussian") for _ in agent_list]

    raw_env = env.unwrapped
    world = raw_env.world
    if len(world.agents) != len(agent_list):
        raise ValueError("world agent order does not match environment agent order")

    specs = []
    for world_agent in world.agents:
        branches = []
        if world_agent.movable:
            branches.append(world.dim_p * 2 + 1)
        if not world_agent.silent:
            branches.append(world.dim_c)
        if not branches or any(size <= 0 for size in branches):
            raise ValueError("agent has no valid categorical action branch")
        specs.append(ActionSpec(mode="official", branch_sizes=tuple(branches)))
    return specs


def _make_legacy_env(scenario_name, max_cycles, benchmark):
    # Commit 6ed7cac imports the removed ``gym.spaces.prng`` singleton.
    # Recreate only that narrow API instead of downgrading the process to Gym 0.10.
    try:
        import gym.spaces

        if not hasattr(gym.spaces, "prng"):
            gym.spaces.prng = SimpleNamespace(np_random=np.random)
    except ImportError as exc:
        raise ImportError("legacy MPE requires the gym compatibility package") from exc
    try:
        from multiagent.environment import MultiAgentEnv
        import multiagent.scenarios as scenarios
    except ImportError as exc:
        raise ImportError(
            "legacy MPE is required for --env-backend legacy; install "
            "requirements-torch.txt (pinned to OpenAI MPE commit 6ed7cac)"
        ) from exc

    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    if world.dim_c == 0 and all(agent.silent for agent in world.agents):
        # Gym 0.26 rejects the archived code's unused Discrete(0). A one-slot
        # silent channel is behaviorally inert and keeps the old environment loadable.
        world.dim_c = 1
    info_callback = scenario.benchmark_data if benchmark else None
    env = MultiAgentEnv(
        world,
        scenario.reset_world,
        scenario.reward,
        scenario.observation,
        info_callback,
    )
    return LegacyMPEAdapter(env, max_cycles=max_cycles)


def _make_pettingzoo_env(
    scenario_name, max_cycles, benchmark, display, policy_mode
):
    del benchmark
    env_name = SCENARIO_TO_PETTINGZOO.get(scenario_name, scenario_name)
    try:
        env_module = importlib.import_module("pettingzoo.mpe.{}".format(env_name))
    except ImportError as exc:
        raise ImportError("install the pinned pettingzoo[mpe] dependency") from exc

    # Official mode deliberately uses the continuous API as a transport for
    # differentiable categorical vectors, matching archived MPE's vector path.
    env = env_module.parallel_env(
        max_cycles=max_cycles,
        continuous_actions=True,
        render_mode="human" if display else None,
    )
    return PettingZooActionAdapter(env, policy_mode=policy_mode)


def make_env(
    scenario_name,
    env_backend,
    max_cycles,
    benchmark=False,
    display=False,
    policy_mode="official",
):
    if env_backend == "legacy":
        if policy_mode == "gaussian":
            raise ValueError("gaussian policy mode is only supported by PettingZoo")
        return _make_legacy_env(scenario_name, max_cycles, benchmark)
    if env_backend == "pettingzoo":
        return _make_pettingzoo_env(
            scenario_name, max_cycles, benchmark, display, policy_mode
        )
    raise ValueError("unknown environment backend: {}".format(env_backend))
