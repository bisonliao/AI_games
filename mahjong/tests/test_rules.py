import unittest

import numpy as np

from MahjongEnv.rules import isolated_count, is_win, potential, ukeire
from MahjongEnv.tiles import RED


class WinRuleTests(unittest.TestCase):
    def test_red_center_can_complete_sequence_before_lowest_natural_tile(self):
        # 234万, 345万, 789万, red+89条, and 55条.
        hand = [1, 2, 2, 3, 3, 4, 6, 7, 8, 22, 22, 25, 26, RED]
        self.assertTrue(is_win(hand))

    def test_red_center_can_still_complete_a_pair(self):
        hand = [0, 0, 1, 2, 9, 10, 11, 18, 19, 20, 21, 22, 23, RED]
        self.assertTrue(is_win(hand))

    def test_two_red_centers_can_complete_two_separate_sequences(self):
        # 456万, red+12饼, red+89饼, 234条, and 55条.
        hand = [RED, RED, 3, 4, 5, 9, 10, 16, 17, 19, 20, 21, 22, 22]
        self.assertTrue(is_win(hand))


class ResidualPotentialTests(unittest.TestCase):
    def test_residual_tile_weights_and_non_overlapping_groups(self):
        cases = [
            ([7, 8], 1.5),  # 89万边张：两张各 0.75。
            ([6, 8], 1.5),  # 79万嵌张：两张各 0.75。
            ([0, 3], 2.0),  # 14万：两个孤子。
            ([8, 9], 2.0),  # 9万1饼不能跨花色组成搭子。
            ([0, 2, 4], 2.5),  # 135万：嵌张加孤子，不能重复使用 3万。
            ([0, 2, 2, 4], 2.0),  # 33万作将、1万5万作孤子优于两个嵌张。
            ([0, 1, 2], 0.0),  # 完整顺子不计残子。
            ([0, 0], 0.0),  # 唯一的一对作为将牌。
            ([0, 0, 7, 7], 0.5),  # 除去将牌，另一整对计 0.5。
            ([RED], 0.25),
            ([6, 8, RED], 0.0),  # 红中作 8万，组成 789万。
            ([0, 1, 2, RED], 0.25),
        ]
        for hand, expected in cases:
            with self.subTest(hand=hand):
                self.assertEqual(isolated_count(hand), expected)
                self.assertEqual(isolated_count(list(reversed(hand))), expected)

    def test_tatsu_types_in_all_suits(self):
        for base in (0, 9, 18):
            for low in range(8):
                expected = 1.5 if low in (0, 7) else 1.0
                with self.subTest(base=base, low=low, gap=1):
                    self.assertEqual(
                        isolated_count([base + low, base + low + 1]), expected
                    )
            for low in range(7):
                with self.subTest(base=base, low=low, gap=2):
                    self.assertEqual(
                        isolated_count([base + low, base + low + 2]), 1.5
                    )
        # 重叠搭子取更优拆法：124万取24嵌张(1.5)+1孤子(1)。
        self.assertEqual(isolated_count([0, 1, 3]), 2.5)
        # 134万取34两面(1)+1孤子(1)，不取13嵌张。
        self.assertEqual(isolated_count([0, 2, 3]), 2.0)

    def test_user_hand_prefers_preserving_89_wan_over_isolated_tile(self):
        hand = [1, 3, 3, 7, 10, 14, 14, 15, 16, 20, 21, 22, 22, 8]
        discard_8_wan = hand.copy()
        discard_8_wan.remove(7)
        discard_2_bing = hand.copy()
        discard_2_bing.remove(10)
        self.assertEqual(isolated_count(discard_8_wan), 5.0)
        self.assertEqual(isolated_count(discard_2_bing), 4.5)
        self.assertEqual(ukeire(discard_8_wan), 0)
        self.assertEqual(ukeire(discard_2_bing), 0)
        self.assertEqual(
            potential(discard_2_bing) - potential(discard_8_wan), 0.5
        )

    def test_potential_does_not_reward_completed_melds(self):
        hand = [0, 1, 2, 9, 10, 11, 18, 19, 20, 4, 4, 5, 6]
        expected = -isolated_count(hand) + 0.5 * ukeire(hand)
        self.assertAlmostEqual(potential(hand), expected)

    def test_potential_ukeire_weight_is_configurable(self):
        hand = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4]
        self.assertAlmostEqual(
            potential(hand, ukeire_weight=0.5),
            -isolated_count(hand) + 0.5 * ukeire(hand),
        )

    def test_ukeire_counts_only_live_tiles(self):
        # 111/222/333/444/55 waiting on the fourth 5.
        hand = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4]
        self.assertEqual(ukeire(hand), 14)
        visible = np.zeros(28, dtype=np.int8)
        visible[4] = 2
        self.assertEqual(ukeire(hand, visible_counts=visible), 12)

if __name__ == '__main__':
    unittest.main()
