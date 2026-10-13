import numpy as np
from functools import lru_cache

from .tiles import RED, counts, is_suited, rank, suit


def _meldable(c, groups):
    if groups == 0:
        return all(x == 0 for x in c)
    i = next((i for i, x in enumerate(c) if x), None)
    if i is None:
        return False
    if c[i] >= 3:
        c[i] -= 3
        if _meldable(c, groups - 1):
            return True
        c[i] += 3
    if (
        is_suited(i)
        and rank(i) <= 6
        and c[i + 1]
        and c[i + 2]
        and suit(i) == suit(i + 1) == suit(i + 2)
    ):
        c[i] -= 1
        c[i + 1] -= 1
        c[i + 2] -= 1
        if _meldable(c, groups - 1):
            return True
        c[i] += 1
        c[i + 1] += 1
        c[i + 2] += 1
    return False


def is_win(tiles, meld_count=0):
    return _is_win_cached(tuple(sorted(tiles)), int(meld_count))


@lru_cache(maxsize=200_000)
def _is_win_cached(tiles, meld_count=0):
    """若干成句加一对将，支持 0 到 4 组；红中可替代任意牌。

    ``tiles`` 只包含手牌，已碰/杠的牌通过 ``meld_count`` 传入。
    因此成句数由剩余手牌张数推导，而不是固定写死为四组。
    """
    c = counts(tiles)
    wild = int(c[RED])
    c[RED] = 0
    # Concealed/open meld tiles are stored separately and are excluded from hand.
    if len(tiles) < 2 or (len(tiles) - 2) % 3:
        return False
    need_groups = (len(tiles) - 2) // 3
    if (
        not 0 <= need_groups <= 4
        or meld_count < 0
        or need_groups + meld_count > 4
    ):
        return False
    for pair in range(27):
        if c[pair] >= 2:
            use = 2
        elif c[pair] == 1 and wild:
            use = 1
        elif wild >= 2:
            use = 0
        else:
            use = -1
        if use < 0:
            continue
        c[pair] -= max(0, use)
        w = wild - (2 - use)
        if w >= 0:
            # Try all distributions of wildcards among the remaining melds.
            if w == 0:
                complete = _meldable(c, need_groups)
            else:
                complete = _complete_with_wild(c, w, need_groups)
            if complete:
                return True
        c[pair] += max(0, use)
    return False

def _complete_with_wild(c, wild, groups):
    if groups == 0:
        return sum(c) == 0 and wild == 0
    i = next((i for i, x in enumerate(c) if x), None)
    if i is None:
        return wild == 3 * groups
    # triplet with 0-2 wilds
    take = min(3, c[i])
    for natural in range(take, -1, -1):
        missing = 3 - natural
        if missing <= wild and c[i] >= natural:
            c[i] -= natural
            if _complete_with_wild(c, wild - missing, groups - 1):
                return True
            c[i] += natural
    # The first natural tile may be the middle or end of a sequence when a
    # red center fills an earlier tile (for example red + 8条 + 9条).
    for start in range(max(0, i - 2), min(26, i) + 1):
        if (
            not is_suited(start)
            or rank(start) > 6
            or not start <= i <= start + 2
        ):
            continue
        required = [start, start + 1, start + 2]
        for mask in range(8):
            used = []
            missing = 0
            valid = True
            for j, tile in enumerate(required):
                if (mask >> j) & 1:
                    missing += 1
                elif c[tile]:
                    c[tile] -= 1
                    used.append(tile)
                else:
                    valid = False
                    break
            if (
                valid
                and missing <= wild
                and _complete_with_wild(c, wild - missing, groups - 1)
            ):
                return True
            for tile in used:
                c[tile] += 1
    return False


@lru_cache(maxsize=200_000)
def _residual_score(counts_key, wild, pair_used):
    """Minimum residual score after extracting melds and one pair.

    成句和一对将不计残子分。剩余天然对子整对计 PAIR_WEIGHT=0.5；
    两面搭子每张计 OPEN_TATSU_TILE_WEIGHT=0.5（整组 1.0）；
    边张和嵌张每张计 CLOSED_TATSU_TILE_WEIGHT=0.75（整组 1.5）；
    其他孤子每张计 1.0。红中参与成句/将牌拆解，
    未使用的红中每张计 RED_RESIDUAL_WEIGHT=0.25。

    尝试各个互不重叠的拆法，取最低残子分。每个分支先消耗选中的牌，
    再递归处理剩余牌，因此一张牌不会重复计入对子、搭子或成句。
    """
    c = list(counts_key)
    first = next((i for i, amount in enumerate(c) if amount), None)
    if first is None:
        if wild == 0:
            return 0.0
        options = [wild * RED_RESIDUAL_WEIGHT]
        if not pair_used and wild >= 2:
            options.append(_residual_score(tuple(c), wild - 2, True))
        if wild >= 3:
            options.append(_residual_score(tuple(c), wild - 3, pair_used))
        return min(options)

    best = float('inf')

    # Leave this tile for the residual hand.
    c[first] -= 1
    best = min(best, 1.0 + _residual_score(tuple(c), wild, pair_used))
    c[first] += 1
    if c[first] >= 2:
        # A residual natural pair costs 0.5 for BOTH tiles, not per tile.
        c[first] -= 2
        best = min(
            best,
            PAIR_WEIGHT + _residual_score(tuple(c), wild, pair_used),
        )
        c[first] += 2

    # 残余搭子分三类（同花色）：
    # 两面：23、34、45、56、67、78，每张 0.5，整组 1.0。
    # 边张：12 只能进 3、89 只能进 7，每张 0.75，整组 1.5。
    # 嵌张：如 79 只能进 8，每张 0.75，整组 1.5。
    # first 是剩余最小牌，只需找后面的 +1/+2，且不能跨花色。
    # 例如 1万3万5万只能拆出一个搭子和一个孤子，不能重复使用 3万。
    if is_suited(first):
        for delta in (1, 2):
            neighbor = first + delta
            if (
                is_suited(neighbor)
                and suit(neighbor) == suit(first)
                and c[neighbor] > 0
            ):
                # rank() 从 0 开始，低位牌为 1 或 8 对应 rank 0 或 7。
                is_edge = delta == 1 and rank(first) in (0, 7)
                is_gap = delta == 2
                tile_weight = (
                    CLOSED_TATSU_TILE_WEIGHT
                    if is_edge or is_gap
                    else OPEN_TATSU_TILE_WEIGHT
                )
                c[first] -= 1
                c[neighbor] -= 1
                best = min(
                    best,
                    2 * tile_weight
                    + _residual_score(tuple(c), wild, pair_used),
                )
                c[first] += 1
                c[neighbor] += 1

    # Use it in a natural/wildcard triplet.
    natural = min(3, c[first])
    for take in range(1, natural + 1):
        missing = 3 - take
        if missing <= wild:
            c[first] -= take
            best = min(
                best,
                _residual_score(tuple(c), wild - missing, pair_used),
            )
            c[first] += take

    # Use it in a pair (the eyes), with zero or one wildcard.
    if not pair_used:
        for take in (2, 1):
            missing = 2 - take
            if c[first] >= take and missing <= wild:
                c[first] -= take
                best = min(
                    best,
                    _residual_score(tuple(c), wild - missing, True),
                )
                c[first] += take

    # Use it in any suited sequence containing this tile, filling gaps with
    # red centers. The first natural tile need not be the sequence start.
    if is_suited(first):
        for start in range(max(0, first - 2), min(26, first) + 1):
            if (
                not is_suited(start)
                or rank(start) > 6
                or not start <= first <= start + 2
            ):
                continue
            required = (start, start + 1, start + 2)
            used = []
            missing = 0
            valid = True
            for tile in required:
                if c[tile]:
                    c[tile] -= 1
                    used.append(tile)
                else:
                    missing += 1
            if missing <= wild:
                best = min(best, _residual_score(tuple(c), wild - missing, pair_used))
            for tile in used:
                c[tile] += 1

    return best


PAIR_WEIGHT = 0.5
OPEN_TATSU_TILE_WEIGHT = 0.5
CLOSED_TATSU_TILE_WEIGHT = 0.75
RED_RESIDUAL_WEIGHT = 0.25
UKEIRE_WEIGHT = 0.5


def isolated_count(tiles):
    """Return the residual-tile score after extracting melds and one pair."""
    c = counts(tiles)
    wild = int(c[RED])
    c[RED] = 0
    return _residual_score(tuple(int(value) for value in c), wild, False)


def _ukeire_for_hand(tiles, meld_count, visible_counts):
    """Return the number of remaining tiles that complete a winning hand."""
    c = counts(tiles)
    visible = (
        counts([])
        if visible_counts is None
        else np.asarray(visible_counts, dtype=int)
    )
    if visible.shape != c.shape:
        raise ValueError('visible_counts must have shape (28,)')
    ukeire = 0
    # A winning draw must complete a pair/triplet or a suited sequence.  Its
    # value is therefore either already present, within two ranks of a suited
    # tile, or a red center.
    candidates = set()
    for tile, amount in enumerate(c):
        if amount <= 0:
            continue
        candidates.add(tile)
        if is_suited(tile):
            for delta in (-2, -1, 1, 2):
                candidate = tile + delta
                if 0 <= candidate < 27 and suit(candidate) == suit(tile):
                    candidates.add(candidate)
    # A drawn red center is itself a legal wildcard draw even when the hand
    # currently contains no red center.
    candidates.add(RED)
    if c[RED]:
        candidates.update(range(len(c)))
    for tile in candidates:
        if is_win(list(tiles) + [tile], meld_count):
            ukeire += max(
                0,
                4 - int(visible[tile]) - int(c[tile]),
            )
    return ukeire


def ukeire(tiles, meld_count=0, visible_counts=None):
    """Count live winning draws for an already-selected residual hand.

    The caller is responsible for choosing and removing the discard.  This
    function intentionally does not search over discards: reward shaping is
    evaluated immediately after the agent completes a discard action.
    """
    # Ukeire is defined here only for a normal 13-tile pre-draw hand
    # (accounting for tiles already placed in melds). Other intermediate or
    # partial hands are not considered listening states.
    if len(tiles) + 3 * int(meld_count) != 13:
        return 0
    return _ukeire_for_hand(list(tiles), meld_count, visible_counts)


@lru_cache(maxsize=100_000)
def _potential_cached(tiles, meld_count, visible_counts):
    return float(
        -isolated_count(tiles)
        + UKEIRE_WEIGHT * ukeire(tiles, meld_count, visible_counts)
    )


def potential(tiles, meld_count=0, visible_counts=None, ukeire_weight=UKEIRE_WEIGHT):
    """Residual-hand potential used by reward shaping.

    Completed melds have no direct value here: the objective is to reduce
    isolated tiles and increase the number of live winning draws.
    """
    tile_key = tuple(sorted(int(tile) for tile in tiles))
    visible_key = (
        None
        if visible_counts is None
        else tuple(int(value) for value in visible_counts)
    )
    if float(ukeire_weight) == UKEIRE_WEIGHT:
        return _potential_cached(tile_key, int(meld_count), visible_key)
    return float(
        -isolated_count(tile_key)
        + float(ukeire_weight) * ukeire(tile_key, meld_count, visible_key)
    )

def concealed_kongs(hand):
    c = counts(hand)
    return [tile for tile in range(28) if c[tile] >= 4]
