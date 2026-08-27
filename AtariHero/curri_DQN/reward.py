"""H.E.R.O.-specific reward-event decoding from ALE frame rewards."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HeroRewardEvents:
    walls_destroyed: int = 0
    creatures_killed: int = 0
    miner_rescued: int = 0
    dynamite_bonus_sticks: int = 0
    unmapped_reward: float = 0.0

    def __add__(self, other: "HeroRewardEvents") -> "HeroRewardEvents":
        return HeroRewardEvents(
            walls_destroyed=self.walls_destroyed + other.walls_destroyed,
            creatures_killed=self.creatures_killed + other.creatures_killed,
            miner_rescued=self.miner_rescued + other.miner_rescued,
            dynamite_bonus_sticks=(
                self.dynamite_bonus_sticks + other.dynamite_bonus_sticks
            ),
            unmapped_reward=self.unmapped_reward + other.unmapped_reward,
        )


class HeroRewardEventParser:
    """Decode documented H.E.R.O. score events without image processing.

    The game awards 1000 points for rescuing a miner and 50 points for every
    dynamite stick left at that rescue.  Once the 1000-point event has been
    observed, subsequent 50-point events in that Level are therefore bonus
    dynamite points rather than creature kills.
    """

    def __init__(self) -> None:
        self.miner_reward_seen = False

    def reset(self) -> None:
        self.miner_reward_seen = False

    def mark_miner_rescued(self) -> HeroRewardEvents:
        if self.miner_reward_seen:
            return HeroRewardEvents()
        self.miner_reward_seen = True
        return HeroRewardEvents(miner_rescued=1)

    def observe(self, raw_reward: float) -> HeroRewardEvents:
        """Decode one ALE frame's raw score delta.

        Unknown values are deliberately retained as diagnostics and never
        turned into learner reward.  A rescue value may be 1000 + 50*n.
        """
        value = int(round(float(raw_reward)))
        if value <= 0:
            return HeroRewardEvents()
        if value == 75:
            return HeroRewardEvents(walls_destroyed=1)
        if value == 50:
            if self.miner_reward_seen:
                return HeroRewardEvents(dynamite_bonus_sticks=1)
            return HeroRewardEvents(creatures_killed=1)
        if value >= 1000:
            remainder = value - 1000
            bonus, extra = divmod(remainder, 50)
            event = self.mark_miner_rescued()
            if extra:
                event = event + HeroRewardEvents(unmapped_reward=float(extra))
            if bonus:
                event = event + HeroRewardEvents(
                    dynamite_bonus_sticks=bonus
                )
            return event
        return HeroRewardEvents(unmapped_reward=float(raw_reward))
