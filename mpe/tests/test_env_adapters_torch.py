import unittest
from types import SimpleNamespace

import numpy as np

from experiments.train_torch import make_env
from maddpg.common.env_adapters_torch import (
    _ensure_legacy_rendering_compatibility,
)
from maddpg.common.env_adapters_torch import infer_action_specs


def _args(backend, scenario):
    del scenario
    return SimpleNamespace(
        env_backend=backend,
        policy_mode="official",
        max_episode_len=25,
        display=False,
    )


class EnvironmentAdapterTest(unittest.TestCase):
    def _make_or_skip(self, scenario, backend):
        try:
            return make_env(scenario, _args(backend, scenario))
        except ImportError as exc:
            if backend == "legacy":
                self.skipTest(str(exc))
            raise

    def test_legacy_rendering_restores_removed_gym_reraise(self):
        import gym.utils

        original = getattr(gym.utils, "reraise", None)
        if hasattr(gym.utils, "reraise"):
            del gym.utils.reraise
        try:
            _ensure_legacy_rendering_compatibility()
            self.assertTrue(callable(gym.utils.reraise))
            try:
                raise ImportError("base import failure")
            except ImportError:
                try:
                    gym.utils.reraise(prefix="prefix", suffix="suffix")
                except ImportError as exc:
                    self.assertIn("prefix", str(exc))
                    self.assertIn("base import failure", str(exc))
                    self.assertIn("suffix", str(exc))
                else:
                    self.fail("reraise did not propagate the active exception")
        finally:
            if original is None:
                if hasattr(gym.utils, "reraise"):
                    del gym.utils.reraise
            else:
                gym.utils.reraise = original

    def test_simple_and_simple_spread_are_distinct_for_both_backends(self):
        for backend in ("legacy", "pettingzoo"):
            simple = self._make_or_skip("simple", backend)
            spread = self._make_or_skip("simple_spread", backend)
            try:
                simple_obs, _ = simple.reset(seed=7)
                spread_obs, _ = spread.reset(seed=7)
                self.assertEqual(len(simple_obs), 1)
                self.assertEqual(next(iter(simple_obs.values())).shape, (4,))
                self.assertEqual(len(spread_obs), 3)
                self.assertTrue(all(obs.shape == (18,) for obs in spread_obs.values()))
            finally:
                simple.close()
                spread.close()

    def test_communication_action_specs_have_separate_branches(self):
        for backend in ("legacy", "pettingzoo"):
            env = self._make_or_skip("simple_reference", backend)
            try:
                observations, _ = env.reset(seed=3)
                specs = infer_action_specs(env, list(observations), "official")
                self.assertEqual([spec.branch_sizes for spec in specs], [(5, 10), (5, 10)])
            finally:
                env.close()

    def test_pettingzoo_action_adapter_matches_legacy_vector_direction(self):
        legacy = self._make_or_skip("simple", "legacy")
        modern = self._make_or_skip("simple", "pettingzoo")
        try:
            legacy.reset(seed=11)
            modern.reset(seed=11)
            for env in (legacy, modern):
                world = env.unwrapped.world
                world.agents[0].state.p_pos = np.zeros(2)
                world.agents[0].state.p_vel = np.zeros(2)
                world.landmarks[0].state.p_pos = np.array([0.5, 0.0])
                world.landmarks[0].state.p_vel = np.zeros(2)
            # Archived vector semantics: branch index 1 accelerates in +x.
            action = np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            legacy.step({"agent_0": action})
            modern.step({"agent_0": action})
            # PettingZoo changed the order of position/velocity integration, so
            # exact positions after one step are intentionally not compared.
            # The adapter must nevertheless produce the same physical force
            # and therefore the same movement direction and velocity.
            np.testing.assert_allclose(
                legacy.unwrapped.world.agents[0].action.u,
                modern.unwrapped.world.agents[0].action.u,
                rtol=1e-6,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                legacy.unwrapped.world.agents[0].state.p_vel,
                modern.unwrapped.world.agents[0].state.p_vel,
                rtol=1e-6,
                atol=1e-6,
            )
            self.assertGreater(legacy.unwrapped.world.agents[0].state.p_vel[0], 0.0)
            self.assertGreater(modern.unwrapped.world.agents[0].state.p_vel[0], 0.0)
        finally:
            legacy.close()
            modern.close()


if __name__ == "__main__":
    unittest.main()
