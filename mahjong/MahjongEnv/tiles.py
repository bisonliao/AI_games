"""牌面编码及牌堆工具。

项目只使用三种数牌（万、饼、条）和红中，共 28 种牌。普通牌按花色连续
编码，便于用整数索引计数向量；每种牌在牌堆中各有四张。
"""

import numpy as np

# 红中的固定编码。普通牌占用 0--26，红中放在最后一个位置。
RED = 27
# 牌种数量，而不是牌堆中的牌张总数；完整牌堆共有 28 * 4 = 112 张。
NUM_TILES = 28
# 编码顺序为：1--9 万、1--9 饼、1--9 条、红中。
TILE_NAMES = [f'{n}{s}' for s in ('万','饼','条') for n in range(1,10)] + ['红中']
# 每个牌种复制四次，dtype 使用小整数以节省存储。
DECK = np.repeat(np.arange(NUM_TILES, dtype=np.int8), 4)

def tile_name(tile: int) -> str:
    """将牌编码转换为显示名称。"""
    return TILE_NAMES[int(tile)]


def is_suited(tile: int) -> bool:
    """判断是否为万、饼、条中的普通数牌；红中不属于数牌。"""
    return 0 <= tile < 27


def suit(tile: int) -> int:
    """返回花色编号：0=万，1=饼，2=条。

    调用方应先确认 ``tile`` 是普通数牌；红中没有花色。
    """
    return tile // 9


def rank(tile: int) -> int:
    """返回牌在同花色中的零基序号，即 1 的序号为 0、9 的序号为 8。"""
    return tile % 9


def make_deck(rng: np.random.Generator) -> list[int]:
    """复制并洗混一副完整牌，返回 Python 整数列表。

    使用传入的独立随机数生成器，避免依赖全局随机状态，便于并行环境复现。
    """
    deck = DECK.copy()
    rng.shuffle(deck)
    return [int(tile) for tile in deck]


def counts(tiles) -> np.ndarray:
    """统计牌列表中每种牌的数量，返回长度为 28 的向量。"""
    c = np.zeros(NUM_TILES, dtype=np.int8)
    for t in tiles:
        c[int(t)] += 1
    return c


def can_sequence(a,b,c):
    """判断三张牌是否构成同花色、连续三张的顺子。

    红中不在这里作为万能牌处理；带红中的组合由规则模块单独判断。
    """
    return (
        is_suited(a)
        and suit(a) == suit(b) == suit(c)
        and rank(b) == rank(a) + 1
        and rank(c) == rank(a) + 2
    )
