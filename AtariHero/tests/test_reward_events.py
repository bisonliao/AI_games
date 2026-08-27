from __future__ import annotations

from curri_DQN.reward import HeroRewardEventParser


def test_documented_event_values() -> None:
    parser = HeroRewardEventParser()
    assert parser.observe(75).walls_destroyed == 1
    assert parser.observe(50).creatures_killed == 1
    event = parser.observe(1000)
    assert event.miner_rescued == 1
    assert event.dynamite_bonus_sticks == 0


def test_rescue_reward_includes_remaining_dynamite_without_kills() -> None:
    parser = HeroRewardEventParser()
    event = parser.observe(1150)
    assert event.miner_rescued == 1
    assert event.dynamite_bonus_sticks == 3
    assert event.creatures_killed == 0
    assert parser.observe(50).dynamite_bonus_sticks == 1
    assert parser.observe(50).creatures_killed == 0


def test_level_reset_restores_fifty_point_kill_interpretation() -> None:
    parser = HeroRewardEventParser()
    parser.observe(1000)
    parser.reset()
    assert parser.observe(50).creatures_killed == 1


def test_unknown_reward_is_diagnostic_only() -> None:
    parser = HeroRewardEventParser()
    event = parser.observe(40)
    assert event.walls_destroyed == 0
    assert event.creatures_killed == 0
    assert event.miner_rescued == 0
    assert event.unmapped_reward == 40


def test_duplicate_miner_reward_is_not_paid_twice() -> None:
    parser = HeroRewardEventParser()
    assert parser.observe(1000).miner_rescued == 1
    assert parser.observe(1000).miner_rescued == 0
