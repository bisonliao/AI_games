"""Direct adapter for the archived (legacy) OpenAI MPE environment.

QMIX chooses categorical action indices.  The official MADDPG/MPE execution
path, however, feeds one-hot vectors to ``MultiAgentEnv``.  This module keeps
that vector path and never imports PettingZoo.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
import sys
from types import SimpleNamespace
from typing import Sequence

import numpy as np


def _legacy_gym_reraise(prefix=None, suffix=None):
    """Compatibility implementation removed after the Gym version MPE used."""

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


def _ensure_legacy_rendering_compatibility() -> None:
    """Provide the single old Gym helper imported by archived MPE rendering."""

    import gym.utils

    if not hasattr(gym.utils, "reraise"):
        gym.utils.reraise = _legacy_gym_reraise


@dataclass(frozen=True)
class ActionSpec:
    """The categorical branches making up one legacy MPE action."""

    branch_sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.branch_sizes or any(size <= 0 for size in self.branch_sizes):
            raise ValueError("every agent needs at least one non-empty action branch")

    @property
    def n_actions(self) -> int:
        return int(prod(self.branch_sizes))

    @property
    def vector_size(self) -> int:
        return int(sum(self.branch_sizes))

    def to_vector(self, action_index: int) -> np.ndarray:
        """Encode a (possibly composite) action as concatenated hard one-hots."""

        action_index = int(action_index)
        if action_index < 0 or action_index >= self.n_actions:
            raise ValueError(
                f"action index {action_index} is outside [0, {self.n_actions})"
            )
        branch_indices = np.unravel_index(action_index, self.branch_sizes)
        action = np.zeros(self.vector_size, dtype=np.float32)
        offset = 0
        for size, branch_index in zip(self.branch_sizes, branch_indices):
            action[offset + int(branch_index)] = 1.0
            offset += size
        return action


class LegacyMPEEnv:
    """Small list-based wrapper around OpenAI's archived ``MultiAgentEnv``."""

    backend = "legacy"
    policy_mode = "official"

    def __init__(self, scenario_name: str) -> None:
        # Commit 6ed7cac imports the Gym 0.10-era ``gym.spaces.prng`` object.
        # Recreate only that removed object, matching the existing project.
        try:
            import gym.spaces

            if not hasattr(gym.spaces, "prng"):
                gym.spaces.prng = SimpleNamespace(np_random=np.random)
        except ImportError as exc:
            raise ImportError("legacy MPE requires the gym package") from exc

        try:
            from multiagent.environment import MultiAgentEnv
            import multiagent.scenarios as scenarios
        except ImportError as exc:
            raise ImportError(
                "legacy MPE is missing; install the repository requirements, "
                "including the pinned OpenAI MPE commit 6ed7cac"
            ) from exc

        scenario = scenarios.load(scenario_name + ".py").Scenario()
        world = scenario.make_world()
        if world.dim_c == 0 and all(agent.silent for agent in world.agents):
            # Gym 0.26 rejects the archived code's unused Discrete(0).
            world.dim_c = 1
        self._env = MultiAgentEnv(
            world,
            scenario.reset_world,
            scenario.reward,
            scenario.observation,
        )
        if self._env.discrete_action_input:
            raise ValueError("official mode requires legacy MPE's vector action path")

        self.scenario_name = scenario_name
        self.n_agents = int(self._env.n)
        self.observation_dims = tuple(
            int(space.shape[0]) for space in self._env.observation_space
        )
        self.action_specs = tuple(self._infer_action_specs())
        self.shared_reward = bool(self._env.shared_reward)

    @property
    def world(self):
        return self._env.world

    @property
    def state_dim(self) -> int:
        """Central state is the stable concatenation of all local observations."""

        return int(sum(self.observation_dims))

    def _infer_action_specs(self) -> Sequence[ActionSpec]:
        specs = []
        for agent in self.world.agents:
            branches = []
            if agent.movable:
                branches.append(self.world.dim_p * 2 + 1)
            if not agent.silent:
                branches.append(self.world.dim_c)
            specs.append(ActionSpec(tuple(int(size) for size in branches)))
        return specs

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            # Archived MPE scenarios use NumPy's module-level RandomState.
            np.random.seed(seed)
        observations = self._env.reset()
        return self._stack_observations(observations)

    def step(
        self, action_indices: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, object]:
        if len(action_indices) != self.n_agents:
            raise ValueError(
                f"expected {self.n_agents} actions, got {len(action_indices)}"
            )
        action_vectors = [
            spec.to_vector(index)
            for spec, index in zip(self.action_specs, action_indices)
        ]
        observations, rewards, dones, info = self._env.step(action_vectors)
        return (
            self._stack_observations(observations),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.bool_),
            info,
        )

    def state(self, observations: np.ndarray) -> np.ndarray:
        observations = np.asarray(observations, dtype=np.float32)
        if observations.shape != (self.n_agents, self.observation_dims[0]):
            raise ValueError(
                "state construction requires homogeneous observations with shape "
                f"({self.n_agents}, {self.observation_dims[0]}), got "
                f"{observations.shape}"
            )
        return observations.reshape(-1).copy()

    def render(self, mode: str = "human"):
        """Render through the archived MPE viewer without changing its API."""

        _ensure_legacy_rendering_compatibility()
        return self._env.render(mode=mode)

    def _stack_observations(self, observations: Sequence[np.ndarray]) -> np.ndarray:
        arrays = [np.asarray(obs, dtype=np.float32) for obs in observations]
        if len(set(self.observation_dims)) != 1:
            raise ValueError(
                "this parameter-sharing QMIX implementation requires equal "
                "observation dimensions for all agents"
            )
        return np.stack(arrays, axis=0)

    def close(self) -> None:
        close = getattr(self._env, "close", None)
        if close is not None:
            close()
