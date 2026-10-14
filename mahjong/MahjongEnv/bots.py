import numpy as np

from .tiles import RED, counts, is_suited, rank, suit


def discard_score(hand, tile):
    """估计 ``tile`` 对当前手牌的保留价值。

    这是牌友机器人使用的启发式评分，不是完整的麻将牌效搜索：

    * 分数越高，表示这张牌越值得保留；
    * 分数越低，表示这张牌越像孤张，越适合作为弃牌；
    * 对子/刻子保护优先级最高；
    * 能和同花色邻牌组成顺子的牌也会获得加分。

    因此 ``choose_discard()`` 会寻找分数最低的牌，而不是最高的牌。
    """
    # 将手牌转换成长度为 28 的计数向量，c[x] 表示牌 x 在手里有几张。
    c = counts(hand)

    # 每种牌的第一张不额外加分，后续重复牌每张加 6 分：
    #   1 张：0 分（普通单张）
    #   2 张：6 分（对子，应尽量保留）
    #   3 张：12 分（刻子，应更优先保留）
    #   4 张：18 分（杠子候选）
    # max(0, ...) 防止单张牌得到负分。
    score = 6 * max(0, int(c[tile]) - 1)

    # 红中不是普通数牌，没有花色和序号，不能参与下面的顺子邻牌计算。
    # 正常情况下 choose_discard() 会直接排除红中；这里保留提前返回，
    # 让 discard_score() 单独被调用时也不会对红中做非法的花色运算。
    if not is_suited(tile):
        return score

    # rank() 是零基序号：1万为 0，2万为 1，……，9万为 8。
    tile_rank = rank(tile)

    # 检查目标牌前后两格以内的同花色牌：
    #   -1 / +1：直接相邻，最容易组成两面或嵌入顺子，加 3 分；
    #   -2 / +2：隔一张牌的嵌张关系，仍有价值，但加 1 分。
    # 例如目标为 3万：2万、4万各加 3 分，1万、5万各加 1 分。
    # 邻牌必须满足 0 <= neighbor < 27，避免越界到红中；还必须同花色，
    # 防止 9万和1饼被错误地当成连续牌。
    for delta, weight in ((-2, 1), (-1, 3), (1, 3), (2, 1)):
        neighbor = tile + delta
        if 0 <= neighbor < 27 and suit(neighbor) == suit(tile) and c[neighbor]:
            score += weight

    # 2--8 的中张牌额外加 1 分。中张向两边都有较多组成顺子的可能，
    # 相比 1、9 这类边张更有保留价值。这里 rank 为 0--8，所以条件
    # 0 < rank < 8 正好对应牌面 2--8。
    if 0 < tile_rank < 8:
        score += 1

    # 调用方会选择最低分牌弃掉；返回的只是相对启发式分数，不是概率，
    # 也不代表严格的最优牌效。
    return score


def choose_discard(hand, rng, randomness=0.35):
    """按保留价值选择弃牌；有普通牌时不弃红中。

    randomness（预期取值 0--1）控制是否扩大随机选择范围，而不是控制
    是否随机打破最低分平局。即使设为 0，最低分牌有多种时仍会随机选。
    """
    candidates = sorted({int(tile) for tile in hand if int(tile) != RED})
    if not candidates:
        return int(rng.choice(hand))
    scores = {tile: discard_score(hand, tile) for tile in candidates}
    best = min(scores.values())
    # 候选池包含最低分牌，以及比最低分高至多 1 分的牌。
    # 例如 A、B、C、D 四种牌的分数为 2、2、3、8，则 best=2，
    # pool=[A, B, C]；C 的保留价值略高，弃掉它是允许的轻微偏离。
    pool = [tile for tile in candidates if scores[tile] <= best + 1]
    # 以 randomness 的概率，从整个候选池中等概率选择一个牌种。
    # 默认 0.35 表示约 35% 的决策使用这个较宽的范围；此时仍可能选中
    # 最低分牌，所以不等于有 35% 的概率一定弃掉较高分牌。
    # 上例中，这个分支会在 A、B、C 中随机选择。
    if randomness > 0 and rng.random() < randomness:
        return int(rng.choice(pool))
    # 其余时候严格选择最低分，只在同分牌种之间随机打破平局。
    # 上例中，这个分支只会选择 A 或 B，不会选择 C。
    # randomness=0 时始终走此分支；=1 时始终走上面的宽范围分支。
    # 如果 pool 中全部都是最低分牌，两条分支的选择分布就完全相同。
    return int(rng.choice([tile for tile in pool if scores[tile] == best]))


def choose_action(env, randomness=0.35):
    legal = env.available_actions()
    if not legal:
        return 140
    disc = [a for a in legal if a < 28]
    if disc:
        tile = choose_discard(env.state.players[0].hand, env.rng, randomness)
        if tile in disc:
            return tile
        return int(env.rng.choice(disc))
    return int(env.rng.choice(legal))
