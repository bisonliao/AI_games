import numpy as np

RED = 27
NUM_TILES = 28
TILE_NAMES = [f'{n}{s}' for s in ('万','饼','条') for n in range(1,10)] + ['红中']
DECK = np.repeat(np.arange(NUM_TILES, dtype=np.int8), 4)

def tile_name(tile: int) -> str:
    return TILE_NAMES[int(tile)]


def is_suited(tile: int) -> bool:
    return 0 <= tile < 27


def suit(tile: int) -> int:
    return tile // 9


def rank(tile: int) -> int:
    return tile % 9

def make_deck(rng: np.random.Generator) -> list[int]:
    deck = DECK.copy()
    rng.shuffle(deck)
    return [int(tile) for tile in deck]

def counts(tiles) -> np.ndarray:
    c = np.zeros(NUM_TILES, dtype=np.int8)
    for t in tiles: c[int(t)] += 1
    return c

def can_sequence(a,b,c):
    return (
        is_suited(a)
        and suit(a) == suit(b) == suit(c)
        and rank(b) == rank(a) + 1
        and rank(c) == rank(a) + 2
    )
