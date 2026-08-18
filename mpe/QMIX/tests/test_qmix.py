from __future__ import annotations

import unittest
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

from QMIX.env import (
    ActionSpec,
    LegacyMPEEnv,
    _ensure_legacy_rendering_compatibility,
)
from QMIX.learner import LearnerConfig, QMIXLearner
from QMIX.networks import QMixer
from QMIX.replay_buffer import EpisodeBuilder, EpisodeReplayBuffer
from QMIX.train import (
    _checkpoint_payload,
    _load_checkpoint,
    _metadata,
    _rollout_episode,
    _state_dir,
)


class ActionSpecTest(unittest.TestCase):
    def test_composite_action_is_concatenated_one_hot(self):
        spec = ActionSpec((3, 2))
        np.testing.assert_array_equal(
            spec.to_vector(3),
            np.asarray([0, 1, 0, 0, 1], dtype=np.float32),
        )


class MixerTest(unittest.TestCase):
    def test_mixer_is_monotonic_in_agent_q_values(self):
        torch.manual_seed(1)
        mixer = QMixer(n_agents=3, state_dim=7)
        agent_qs = torch.randn(2, 4, 3, requires_grad=True)
        states = torch.randn(2, 4, 7)
        mixer(agent_qs, states).sum().backward()
        self.assertTrue(torch.all(agent_qs.grad >= -1e-7).item())


class LearnerTest(unittest.TestCase):
    def test_padded_episode_batch_trains(self):
        n_agents, obs_dim, state_dim, n_actions, horizon = 3, 4, 12, 5, 5
        replay = EpisodeReplayBuffer(capacity=4)
        for episode_length in (3, 5):
            initial_obs = np.zeros((n_agents, obs_dim), dtype=np.float32)
            builder = EpisodeBuilder(
                initial_obs, initial_obs.reshape(-1), horizon
            )
            for timestep in range(episode_length):
                next_obs = np.full_like(initial_obs, timestep + 1)
                builder.add(
                    np.asarray([0, 1, 2]),
                    reward=-1.0,
                    terminated=timestep == episode_length - 1,
                    next_observations=next_obs,
                    next_state=next_obs.reshape(-1),
                )
            replay.add(builder.finish())

        device = torch.device("cpu")
        learner = QMIXLearner(
            LearnerConfig(
                n_agents=n_agents,
                obs_dim=obs_dim,
                state_dim=state_dim,
                n_actions=n_actions,
                grad_norm_clip=0.05,
            ),
            device,
        )
        metrics = learner.train(replay.sample(2, device))
        self.assertTrue(np.isfinite(metrics["loss"]))
        self.assertGreaterEqual(
            metrics["pre_clip_grad_norm"], metrics["post_clip_grad_norm"]
        )
        self.assertLessEqual(metrics["post_clip_grad_norm"], 0.05001)
        self.assertEqual(learner.train_updates, 1)

        guarded_learner = QMIXLearner(
            LearnerConfig(
                n_agents=n_agents,
                obs_dim=obs_dim,
                state_dim=state_dim,
                n_actions=n_actions,
                max_abs_q=1e-30,
            ),
            device,
        )
        with self.assertRaisesRegex(FloatingPointError, "divergence guard"):
            guarded_learner.train(replay.sample(2, device))


class LegacyEnvironmentTest(unittest.TestCase):
    def test_rendering_restores_removed_gym_reraise(self):
        import gym.utils

        original = getattr(gym.utils, "reraise", None)
        if hasattr(gym.utils, "reraise"):
            del gym.utils.reraise
        try:
            _ensure_legacy_rendering_compatibility()
            self.assertTrue(callable(gym.utils.reraise))
        finally:
            if hasattr(gym.utils, "reraise"):
                del gym.utils.reraise
            if original is not None:
                gym.utils.reraise = original

    def test_simple_spread_shapes_and_shared_reward(self):
        env = LegacyMPEEnv("simple_spread")
        try:
            observations = env.reset(seed=7)
            self.assertEqual(observations.shape, (3, 18))
            self.assertEqual([spec.n_actions for spec in env.action_specs], [5, 5, 5])
            next_observations, rewards, dones, _ = env.step([0, 1, 2])
            self.assertEqual(next_observations.shape, observations.shape)
            self.assertTrue(np.allclose(rewards, rewards[0]))
            self.assertFalse(np.any(dones))
        finally:
            env.close()

    def test_checkpoint_path_identifies_qmix(self):
        self.assertEqual(
            _state_dir("shared-checkpoint-root", "simple_spread"),
            Path("shared-checkpoint-root/qmix/legacy/official/simple_spread"),
        )

    def test_rollout_scales_reward_and_terminates_finite_horizon(self):
        env = LegacyMPEEnv("simple_spread")
        try:
            learner = QMIXLearner(
                LearnerConfig(
                    n_agents=env.n_agents,
                    obs_dim=env.observation_dims[0],
                    state_dim=env.state_dim,
                    n_actions=env.action_specs[0].n_actions,
                ),
                torch.device("cpu"),
            )
            args = SimpleNamespace(
                max_episode_len=1,
                epsilon_start=1.0,
                epsilon_finish=0.05,
                epsilon_anneal_steps=50_000,
                scenario="simple_spread",
                reward_scale=1.0 / env.n_agents,
                bootstrap_time_limit=False,
            )
            episode, metrics, _ = _rollout_episode(
                env,
                learner,
                args,
                env_steps=0,
                seed=11,
                deterministic=True,
            )
            self.assertEqual(episode.terminated[0, 0], 1.0)
            self.assertAlmostEqual(
                episode.rewards[0, 0],
                metrics["team_episode_reward"] / env.n_agents,
                places=5,
            )
            self.assertAlmostEqual(
                metrics["scaled_team_episode_reward"],
                metrics["team_episode_reward"] / env.n_agents,
            )

            args.bootstrap_time_limit = True
            bootstrapped_episode, _, _ = _rollout_episode(
                env,
                learner,
                args,
                env_steps=0,
                seed=11,
                deterministic=True,
            )
            self.assertEqual(bootstrapped_episode.terminated[0, 0], 0.0)
        finally:
            env.close()

    def test_v1_checkpoint_can_be_loaded_with_legacy_options(self):
        env = LegacyMPEEnv("simple_spread")
        try:
            config = LearnerConfig(
                n_agents=env.n_agents,
                obs_dim=env.observation_dims[0],
                state_dim=env.state_dim,
                n_actions=env.action_specs[0].n_actions,
                gamma=0.99,
                lr=5e-4,
                td_loss="mse",
                max_abs_q=0.0,
            )
            learner = QMIXLearner(config, torch.device("cpu"))
            args = SimpleNamespace(
                scenario="simple_spread",
                max_episode_len=25,
                reward_scale=1.0,
                bootstrap_time_limit=True,
            )
            payload = _checkpoint_payload(
                learner,
                _metadata(args, env, config),
                env_steps=25,
                completed_episodes=1,
                last_target_update_episode=0,
            )
            payload["checkpoint_version"] = 1
            payload["metadata"].pop("reward_scale")
            payload["metadata"].pop("bootstrap_time_limit")
            for key in ("td_loss", "huber_delta", "max_abs_q"):
                payload["metadata"]["learner_config"].pop(key)

            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "state_steps_25.pt"
                torch.save(payload, path)
                restored = QMIXLearner(config, torch.device("cpu"))
                progress = _load_checkpoint(
                    path,
                    restored,
                    _metadata(args, env, config),
                    torch.device("cpu"),
                )
                self.assertEqual(progress, (25, 1, 0))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
