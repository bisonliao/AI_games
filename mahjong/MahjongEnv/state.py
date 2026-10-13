from dataclasses import dataclass, field
import numpy as np
from .tiles import NUM_TILES, counts

@dataclass
class Meld:
    kind: str
    tiles: tuple[int,...]
    from_player: int | None = None

@dataclass
class PlayerState:
    hand: list[int] = field(default_factory=list)
    melds: list[Meld] = field(default_factory=list)
    discards: list[int] = field(default_factory=list)

@dataclass
class GameState:
    players: list[PlayerState] = field(default_factory=lambda:[PlayerState() for _ in range(4)])
    wall: list[int] = field(default_factory=list)
    current_player: int = 0
    last_discard: int | None = None
    phase: str = 'draw'
    step: int = 0
    episode_id: int = 0
    terminal_reason: str | None = None
    last_action: int = -1
    robbed_kong: bool = False
    claim_from: int | None = None
    must_discard: bool = False
    drawn_tile: int | None = None
    start_player: int = 0
    pending_players: list[int] = field(default_factory=list)
    heaven_win: bool = False

    def discard_counts(self):
        return np.stack([counts(p.discards) for p in self.players])

    def public_meld_counts(self, viewer=0):
        """Fixed [player, tile] counts without leaking opponents' concealed kongs."""
        result = np.zeros((4, NUM_TILES), dtype=np.int8)
        for player_index, player in enumerate(self.players):
            for meld in player.melds:
                if player_index == viewer or meld.kind != 'concealed_kong':
                    result[player_index] += counts(meld.tiles)
        return result

    def meld_kind_counts(self):
        kinds = ('peng', 'ming_kong', 'concealed_kong', 'added_kong')
        result = np.zeros((4, len(kinds)), dtype=np.int8)
        for player_index, player in enumerate(self.players):
            for meld in player.melds:
                if meld.kind in kinds:
                    result[player_index, kinds.index(meld.kind)] += 1
        return result

    def visible_counts(self, viewer=0):
        """Backward-compatible fixed aggregate: discards plus visible melds per player."""
        return (self.discard_counts() + self.public_meld_counts(viewer)).reshape(-1).tolist()
