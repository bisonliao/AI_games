import unittest

import numpy as np

from MahjongEnv.bots import choose_discard, discard_score
from MahjongEnv.config import EnvConfig
from MahjongEnv.env import MahjongEnv
from MahjongEnv.state import GameState
from MahjongEnv.tiles import RED


class BotDiscardTests(unittest.TestCase):
    def test_red_center_is_never_discarded_when_natural_tiles_exist(self):
        hand = [RED, 0, 1, 2, 9, 10, 11, 18, 19, 20, 25, 26, 4, 5]
        for seed in range(20):
            self.assertNotEqual(choose_discard(hand, np.random.default_rng(seed), randomness=1.0), RED)

    def test_isolated_tile_scores_below_connected_tiles(self):
        hand = [0, 1, 2, 3, 4, 5, 9, 10, 11, 18, 19, 20, 25, RED]
        self.assertLess(discard_score(hand, 25), discard_score(hand, 1))

    def test_opponent_self_draw_is_checked_before_discard(self):
        env = MahjongEnv(EnvConfig(observation_mode='vector'), seed=1)
        env.state = GameState(wall=[0])
        env.state.players[1].hand = [0, 1, 2, 9, 10, 11, 18, 19, 20, 21, 22, 23, 0]
        env._advance_others([1], animate=False)
        self.assertEqual(env.state.terminal_reason, 'loss')
        self.assertEqual(env.state.players[1].discards, [])


if __name__ == '__main__':
    unittest.main()
