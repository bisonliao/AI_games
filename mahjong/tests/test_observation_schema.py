import unittest

import numpy as np

from MahjongEnv.config import EnvConfig
from MahjongEnv.env import MahjongEnv
from MahjongEnv.state import Meld


class ObservationSchemaTests(unittest.TestCase):
    def test_action_mask_observation_flag_is_respected(self):
        env = MahjongEnv(
            EnvConfig(
                observation_mode='vector',
                include_action_mask_in_obs=False,
            ),
            seed=1,
        )
        observation, _ = env.reset()
        self.assertNotIn('action_mask', observation)
        self.assertIn('vector', observation)

    def test_episode_id_increments_across_resets(self):
        env = MahjongEnv(seed=1)
        _, first_info = env.reset()
        _, second_info = env.reset()
        self.assertEqual(second_info['episode_id'], first_info['episode_id'] + 1)

    def test_public_history_has_fixed_positions_when_melds_exist(self):
        env = MahjongEnv(seed=1)
        env.reset()
        env.state.players[0].discards = [0]
        env.state.players[0].melds = [Meld('peng', (1, 1, 1))]
        env.state.players[1].discards = [9]
        env.state.players[1].melds = [Meld('concealed_kong', (10, 10, 10, 10))]
        vector = env._vector()
        self.assertEqual(vector.shape, (306,))
        # The discard block is always [player0, player1, player2, player3].
        discards = vector[56:168].reshape(4, 28)
        self.assertEqual(discards[0, 0], 0.25)
        self.assertEqual(discards[1, 9], 0.25)
        # The opponent's concealed kong is hidden from public meld tiles.
        public_melds = vector[168:280].reshape(4, 28)
        self.assertEqual(public_melds[0, 1], 0.75)
        self.assertEqual(public_melds[1, 10], 0.0)


if __name__ == '__main__':
    unittest.main()
