import unittest

from MahjongEnv.config import EnvConfig
from MahjongEnv.reward import shaped_reward
from MahjongEnv.state import PlayerState


class RewardBoundaryTests(unittest.TestCase):
    def test_only_explicit_discard_delta_gets_shaping(self):
        before = PlayerState(hand=[0] * 13)
        after = PlayerState(hand=[0] * 12 + [1])
        cfg = EnvConfig()
        self.assertEqual(shaped_reward(before, after, cfg=cfg), 0.0)
        self.assertEqual(shaped_reward(before, after, cfg=cfg, shaping_delta=4), 2.0)

    def test_terminal_reward_is_independent_of_shaping(self):
        cfg = EnvConfig()
        self.assertEqual(shaped_reward(None, None, cfg=cfg, terminal=True, outcome='win'), 10.0)


if __name__ == '__main__':
    unittest.main()
