import unittest
import numpy as np
from DQN.envs import make_vec_env
from DQN.algorithm import MaskedDoubleDQN
from DQN.policy import MaskedDuelingPolicy

class SB3DQNTests(unittest.TestCase):
    def test_masked_smoke_train(self):
        env = make_vec_env(1, 17)
        try:
            model = MaskedDoubleDQN(MaskedDuelingPolicy, env, buffer_size=64, batch_size=4, learning_starts=4, train_freq=1, gradient_steps=1, device='cpu', policy_kwargs={'hidden_dim': 32}, verbose=0)
            obs = env.reset()
            for _ in range(8):
                action, _ = model.predict(obs, deterministic=False)
                self.assertTrue(np.all(obs['action_mask'][np.arange(1), np.asarray(action).reshape(-1)] == 1))
                obs, _, _, _ = env.step(action)
            model.learn(20)
        finally:
            env.close()

    def test_vector_reset_always_has_legal_action(self):
        env = make_vec_env(2, 1)
        try:
            observation = env.reset()
            self.assertTrue(np.all(observation['action_mask'].sum(axis=1) > 0))
        finally:
            env.close()

if __name__ == '__main__': unittest.main()
