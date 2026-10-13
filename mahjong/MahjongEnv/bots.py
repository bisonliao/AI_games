import numpy as np

from .tiles import RED, counts, is_suited, rank, suit


def discard_score(hand, tile):
    """Estimate how useful one tile is to the current hand.

    Pairs/triplets are strongest, followed by adjacent tiles that can form a
    sequence. Lower scores therefore identify isolated, unformed tiles.
    """
    c = counts(hand)
    score = 6 * max(0, int(c[tile]) - 1)
    if not is_suited(tile):
        return score
    tile_rank = rank(tile)
    for delta, weight in ((-2, 1), (-1, 3), (1, 3), (2, 1)):
        neighbor = tile + delta
        if 0 <= neighbor < 27 and suit(neighbor) == suit(tile) and c[neighbor]:
            score += weight
    if 0 < tile_rank < 8:
        score += 1
    return score


def choose_discard(hand, rng, randomness=0.35):
    """Choose a discard while never discarding a red center."""
    candidates = sorted({int(tile) for tile in hand if int(tile) != RED})
    if not candidates:
        return int(rng.choice(hand))
    scores = {tile: discard_score(hand, tile) for tile in candidates}
    best = min(scores.values())
    pool = [tile for tile in candidates if scores[tile] <= best + 1]
    if randomness > 0 and rng.random() < randomness:
        return int(rng.choice(pool))
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
