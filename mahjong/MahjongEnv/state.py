"""麻将牌局状态的数据结构。

这些 dataclass 只保存牌局事实，不负责推进回合或判断胡牌。环境中的观测
编码会从这里提取手牌、弃牌和公开副露等信息。
"""

from dataclasses import dataclass, field
import numpy as np
from .tiles import NUM_TILES, counts

@dataclass
class Meld:
    """一组副露。

    ``kind`` 当前使用 ``peng``、``ming_kong``、``concealed_kong`` 和
    ``added_kong``；``tiles`` 保存组成副露的牌；``from_player`` 记录吃牌
    来源（本项目不吃，因此通常只对碰/明杠有意义）。
    """
    kind: str
    tiles: tuple[int,...]
    from_player: int | None = None

@dataclass
class PlayerState:
    """单个玩家在一局中的牌面状态。"""
    hand: list[int] = field(default_factory=list)
    melds: list[Meld] = field(default_factory=list)
    discards: list[int] = field(default_factory=list)

@dataclass
class GameState:
    """完整的一局牌局状态。

    玩家编号固定为 0--3，其中 0 是训练 agent。``phase`` 主要取
    ``draw``、``play``、``claim``、``terminal``；列表字段使用 ``default_factory``
    创建，避免不同牌局或不同玩家共享同一个可变列表。
    """
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
        """返回形状为 ``(4, 28)`` 的各家弃牌计数矩阵。"""
        return np.stack([counts(p.discards) for p in self.players])

    def public_meld_counts(self, viewer=0):
        """返回形状为 ``(4, 28)`` 的公开副露计数矩阵。

        观察者本人可以看到自己的暗杠；对于其他玩家，暗杠只保留其发生过
        的类型信息，不泄露具体牌种。这与向量观测的固定布局保持一致。
        """
        result = np.zeros((4, NUM_TILES), dtype=np.int8)
        for player_index, player in enumerate(self.players):
            for meld in player.melds:
                if player_index == viewer or meld.kind != 'concealed_kong':
                    result[player_index] += counts(meld.tiles)
        return result

    def meld_kind_counts(self):
        """统计每位玩家四类副露的数量，返回形状 ``(4, 4)`` 的矩阵。"""
        kinds = ('peng', 'ming_kong', 'concealed_kong', 'added_kong')
        result = np.zeros((4, len(kinds)), dtype=np.int8)
        for player_index, player in enumerate(self.players):
            for meld in player.melds:
                if meld.kind in kinds:
                    result[player_index, kinds.index(meld.kind)] += 1
        return result

    def visible_counts(self, viewer=0):
        """返回各玩家弃牌与公开副露的合计计数（扁平列表）。

        这是旧接口保留的聚合形式；需要按牌种扣除可见死牌时，环境通常使用
        ``discard_counts()`` 与 ``public_meld_counts()`` 的总和。
        """
        return (self.discard_counts() + self.public_meld_counts(viewer)).reshape(-1).tolist()
