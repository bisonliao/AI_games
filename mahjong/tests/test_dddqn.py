import io
import unittest

import numpy as np
import torch

from MahjongEnv.config import EnvConfig
from DDDQN.network import DuelingQNetwork, masked_argmax
from DDDQN.replay import ReplayBuffer
from DDDQN.subproc_vec_env import SubprocessVectorEnv
from DDDQN.agent import DDDQNAgent
from DDDQN.config import TrainConfig
from DDDQN.replay import ReplayBatch


class DDDQNTests(unittest.TestCase):
    def check_checkpoint_rng_restore(self, device):
        original_rng = torch.get_rng_state()
        original_cuda_rng = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        try:
            config = TrainConfig(obs_dim=6, action_dim=5, hidden_dim=16)
            agent = DDDQNAgent(config, device)
            checkpoint = agent.checkpoint_state(123, None, config, {})
            expected_rng = checkpoint['torch_rng_state'].clone()
            stream = io.BytesIO()
            torch.save(checkpoint, stream)
            stream.seek(0)
            loaded = torch.load(stream, map_location=device, weights_only=False)
            self.assertEqual(loaded['torch_rng_state'].device.type, device.type)
            torch.rand(10)
            self.assertEqual(agent.load_checkpoint(loaded), 123)
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
            if device.type == 'cuda':
                for actual, expected in zip(
                    torch.cuda.get_rng_state_all(), checkpoint['cuda_rng_state']
                ):
                    self.assertTrue(torch.equal(actual, expected))
        finally:
            torch.set_rng_state(original_rng)
            if original_cuda_rng is not None:
                torch.cuda.set_rng_state_all(original_cuda_rng)

    def test_cpu_checkpoint_restores_rng(self):
        self.check_checkpoint_rng_restore(torch.device('cpu'))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_cuda_mapped_checkpoint_restores_cpu_rng(self):
        self.check_checkpoint_rng_restore(torch.device('cuda'))

    def test_dueling_network_masks_invalid_actions(self):
        network = DuelingQNetwork(obs_dim=6, action_dim=5, hidden_dim=16)
        obs = torch.zeros((2, 6))
        mask = torch.tensor([[1, 0, 1, 0, 0], [0, 1, 0, 1, 0]], dtype=torch.bool)
        q_values = network(obs, mask)
        self.assertEqual(tuple(q_values.shape), (2, 5))
        self.assertTrue(torch.isneginf(q_values[0, 1]))
        self.assertTrue(mask[1, masked_argmax(q_values, mask)[1]])

    def test_raw_q_uses_legal_actions_for_dueling_baseline(self):
        network = DuelingQNetwork(obs_dim=1, action_dim=3, hidden_dim=1)
        with torch.no_grad():
            for parameter in network.trunk.parameters():
                parameter.zero_()
            network.value.weight.zero_()
            network.value.bias.zero_()
            network.advantage.weight.zero_()
            network.advantage.bias.copy_(torch.tensor([0.0, 10.0, 20.0]))
        obs = torch.zeros((1, 1))
        mask = torch.tensor([[1, 0, 1]], dtype=torch.bool)
        raw = network.raw(obs, mask)
        self.assertTrue(torch.allclose(raw, torch.tensor([[-10.0, 0.0, 10.0]])))
        self.assertTrue(torch.isneginf(network(obs, mask)[0, 1]))

    def test_replay_buffer_wraps_without_disk_files(self):
        buffer = ReplayBuffer(capacity=3, obs_dim=4, action_dim=5, seed=1)
        obs = np.zeros((4, 4), np.float32)
        next_obs = np.ones((4, 4), np.float32)
        actions = np.arange(4, dtype=np.int16)
        rewards = np.ones(4, np.float32)
        done = np.zeros(4, bool)
        masks = np.ones((4, 5), np.uint8)
        buffer.add_batch(obs, actions, rewards, next_obs, done, done, masks, masks)
        self.assertEqual(len(buffer), 3)
        self.assertEqual(buffer.sample(2).obs.shape, (2, 4))

    def test_spawn_vector_env_shapes_and_autoreset(self):
        envs = SubprocessVectorEnv(2, EnvConfig(observation_mode='vector', render_mode=None), seed=7)
        try:
            obs, _ = envs.reset()
            self.assertEqual(obs['vector'].shape, (2, 306))
            self.assertEqual(obs['action_mask'].shape, (2, 141))
            actions = np.asarray([np.flatnonzero(mask)[0] for mask in obs['action_mask']], dtype=np.int64)
            next_obs, rewards, terminated, truncated, infos = envs.step(actions)
            self.assertEqual(next_obs['vector'].shape, (2, 306))
            self.assertEqual(rewards.shape, (2,))
            self.assertEqual(len(infos), 2)
        finally:
            envs.close()

    def test_autoreset_never_exposes_terminal_zero_mask(self):
        envs = SubprocessVectorEnv(8, EnvConfig(observation_mode='vector', render_mode=None), seed=1)
        try:
            obs, _ = envs.reset()
            rng = np.random.default_rng(3)
            for _ in range(300):
                self.assertTrue(np.all(obs['action_mask'].sum(axis=1) > 0))
                actions = np.asarray([rng.choice(np.flatnonzero(mask)) for mask in obs['action_mask']])
                obs, _, _, _, _ = envs.step(actions)
        finally:
            envs.close()

    def test_train_batch_handles_empty_terminal_next_mask(self):
        config = TrainConfig(obs_dim=6, action_dim=5, hidden_dim=16, batch_size=4)
        agent = DDDQNAgent(config, torch.device('cpu'))
        batch = ReplayBatch(
            obs=np.zeros((4, 6), dtype=np.float32),
            actions=np.asarray([0, 1, 2, 3], dtype=np.int16),
            rewards=np.ones(4, dtype=np.float32),
            next_obs=np.zeros((4, 6), dtype=np.float32),
            terminated=np.ones(4, dtype=np.bool_),
            truncated=np.zeros(4, dtype=np.bool_),
            action_mask=np.ones((4, 5), dtype=np.uint8),
            next_action_mask=np.zeros((4, 5), dtype=np.uint8),
        )
        stats = agent.train_batch(batch)
        self.assertTrue(all(np.isfinite(value) for value in stats.values()))

    def test_select_actions_never_returns_invalid_for_nonempty_masks(self):
        config = TrainConfig(obs_dim=6, action_dim=5, hidden_dim=16)
        agent = DDDQNAgent(config, torch.device('cpu'))
        obs = np.zeros((2, 6), dtype=np.float32)
        masks = np.asarray([[0, 1, 0, 0, 0], [0, 0, 1, 0, 1]], dtype=np.uint8)
        actions = agent.select_actions(obs, masks, epsilon=0.0, rng=np.random.default_rng(1))
        self.assertTrue(np.all(masks[np.arange(2), actions] == 1))


if __name__ == '__main__':
    unittest.main()
