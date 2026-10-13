import os
from functools import lru_cache

import numpy as np

from .tiles import tile_name


HAND_X = 20
HAND_TILE_WIDTH = 58
HAND_TILE_GAP = 4
HAND_TILE_Y_FROM_BOTTOM = 150
HAND_TILE_HEIGHT = 68
DISCARD_KEYS = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'Q', 'E', 'R', 'T')


def _hand_sort_key(tile):
    """Visual suit order: red center, wan, bing, tiao."""
    if tile == 27:
        return (0, 0)
    if 0 <= tile < 9:
        return (1, tile)
    if 9 <= tile < 18:
        return (2, tile)
    return (3, tile)


def display_hand(hand, drawn_tile=None):
    """Return the visual hand order, with a newly drawn tile separated right."""
    ordered = sorted(hand, key=_hand_sort_key)
    if drawn_tile is not None and drawn_tile in ordered:
        ordered.remove(drawn_tile)
        ordered.append(drawn_tile)
    return ordered


def _action_ids(actions):
    if isinstance(actions, set):
        actions = sorted(actions)
    values = np.asarray(actions)
    if values.shape == (141,) and np.all((values == 0) | (values == 1)):
        return [int(x) for x in np.flatnonzero(values)]
    return [int(x) for x in values.reshape(-1)]


@lru_cache(maxsize=8)
def _font(size):
    import pygame

    configured = os.environ.get('MAHJONG_FONT')
    if configured and os.path.exists(configured):
        return pygame.font.Font(configured, size)
    for path in (
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'fonts', 'NotoSansCJK-Regular.ttc'),
        '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    ):
        if os.path.exists(path):
            return pygame.font.Font(path, size)
    for name in ('Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei', 'Microsoft YaHei'):
        path = pygame.font.match_font(name)
        if path:
            return pygame.font.Font(path, size)
    return pygame.font.SysFont(None, size)


@lru_cache(maxsize=8)
def _latin_font(size):
    import pygame

    return pygame.font.SysFont('DejaVu Sans', size)


def _render_text(text, font, color):
    """Render mixed Latin/CJK text using the appropriate glyph coverage."""
    import pygame

    size = max(12, font.get_height() - 2)
    latin = _latin_font(size)
    parts = []
    start = 0
    while start < len(text):
        is_latin = ord(text[start]) < 128
        end = start + 1
        while end < len(text) and (ord(text[end]) < 128) == is_latin:
            end += 1
        renderer = latin if is_latin else font
        parts.append(renderer.render(text[start:end], True, color))
        start = end
    width = sum(part.get_width() for part in parts)
    height = max((part.get_height() for part in parts), default=font.get_height())
    result = pygame.Surface((max(1, width), height), pygame.SRCALPHA)
    x = 0
    for part in parts:
        result.blit(part, (x, (height - part.get_height()) // 2))
        x += part.get_width()
    return result


def _tile_fill(tile):
    if 0 <= tile < 9:       # 万
        return (246, 198, 198)
    if 9 <= tile < 18:      # 饼
        return (224, 246, 224)
    return (242, 240, 220)  # 条、红中


def _action_label(action, self_draw=False, heaven_win=False):
    if action == 140:
        return '过 [Space]'
    group, tile = divmod(action, 28)
    name = {1: '碰', 2: '杠', 3: '暗杠', 4: '加杠'}[group]
    key = {1: 'P', 2: 'M', 3: 'C', 4: 'A'}[group]
    return f'{name} [{key}]  {tile_name(tile)}'


def action_button_rects(legal_actions, size=(1200, 900), self_draw=False):
    w, _ = size
    actions = sorted(a for a in set(_action_ids(legal_actions)) if a >= 28)
    bx, by = w - 235, 125
    result = []
    for action in actions:
        result.append((action, (bx, by, 215, 38)))
        by += 45
    return result


def _draw_tiles(surface, tiles, x, y, width, height, font, fill=(242, 240, 220), highlight_index=None):
    import pygame

    for index, tile in enumerate(tiles):
        rect = pygame.Rect(x + index * (width + 3), y, width, height)
        tile_fill = _tile_fill(tile) if fill is None else fill
        pygame.draw.rect(surface, tile_fill, rect, border_radius=2)
        border = (220, 35, 35) if index == highlight_index else (35, 35, 30)
        border_width = 4 if index == highlight_index else 1
        pygame.draw.rect(surface, border, rect, border_width, border_radius=2)
        label_color = (210, 25, 25) if tile == 27 else (25, 25, 25)
        label = _render_text(tile_name(tile), font, label_color)
        surface.blit(label, label.get_rect(center=rect.center))


def render_state(state, size=(1200, 900), mode=None, legal_actions=None):
    import pygame

    w, h = size
    img = np.full((h, w, 3), (25, 105, 65), dtype=np.uint8)
    legal = set(_action_ids(legal_actions if legal_actions is not None else []))
    if not pygame.display.get_init():
        pygame.display.init()
    if not pygame.font.get_init():
        pygame.font.init()
    screen = pygame.display.get_surface()
    if mode == 'human' and (screen is None or screen.get_size() != (w, h)):
        screen = pygame.display.set_mode((w, h))

    surface = pygame.Surface((w, h))
    surface.fill((25, 105, 65))
    title_font = _font(28)
    text_font = _font(22)
    small_font = _font(18)
    turn_text = '轮到你' if state.current_player == 0 or state.phase == 'claim' else f'牌友 {state.current_player} 出牌'
    surface.blit(_render_text(f'红中麻将 - {turn_text}', title_font, (245, 245, 230)), (20, 15))
    phase_label = '响应' if state.phase == 'claim' else '出牌'
    surface.blit(
        _render_text(
            f'牌墙: {len(state.wall)}    阶段: {phase_label}',
            text_font,
            (235, 235, 220),
        ),
        (20, 50),
    )

    # Show every opponent's discards and melds as small tile graphics.
    for row, player in enumerate(state.players[1:]):
        y = 105 + row * 185
        meld_text = ' '.join(f'{meld.kind} {tile_name(meld.tiles[0])}' for meld in player.melds)
        label = f'牌友 {row + 1}  弃牌' + (f'  副露: {meld_text}' if meld_text else '')
        surface.blit(_render_text(label, text_font, (235, 235, 220)), (20, y))
        for line, start in enumerate(range(0, len(player.discards), 14)):
            _draw_tiles(
                surface,
                player.discards[start:start + 14],
                20,
                y + 28 + line * 34,
                34,
                30,
                small_font,
                (215, 225, 205),
            )

    hand_y = h - HAND_TILE_Y_FROM_BOTTOM
    pygame.draw.line(surface, (65, 145, 95), (20, hand_y - 105), (w - 20, hand_y - 105), 2)
    hand = display_hand(state.players[0].hand, state.drawn_tile)
    highlight = len(hand) - 1 if state.drawn_tile is not None else None
    _draw_tiles(
        surface,
        hand,
        HAND_X,
        hand_y,
        HAND_TILE_WIDTH,
        HAND_TILE_HEIGHT,
        text_font,
        fill=None,
        highlight_index=highlight,
    )
    for index, key in enumerate(DISCARD_KEYS[:len(hand)]):
        x = HAND_X + index * (HAND_TILE_WIDTH + HAND_TILE_GAP)
        surface.blit(
            _render_text(key, small_font, (250, 230, 100)),
            (x + HAND_TILE_WIDTH // 2 - 5, hand_y + HAND_TILE_HEIGHT + 4),
        )
    # Keep our discard history visible separately from the current hand.
    own_discards = state.players[0].discards
    surface.blit(_render_text('我方弃牌', small_font, (240, 235, 200)), (20, hand_y - 68))
    if own_discards:
        discard_width = max(18, min(30, (w - 135) // len(own_discards)))
        _draw_tiles(surface, own_discards, 100, hand_y - 72, discard_width, 28, small_font, fill=(215, 225, 205))
    own_melds = '副露: ' + ' '.join(f'{m.kind} {tile_name(m.tiles[0])}' for m in state.players[0].melds)
    surface.blit(_render_text(own_melds, small_font, (240, 235, 200)), (20, hand_y - 35))

    buttons = action_button_rects(legal, size) if state.current_player == 0 or state.phase == 'claim' else []
    if buttons:
        panel = pygame.Rect(w - 250, 105, 235, 45 + 45 * len(buttons))
        pygame.draw.rect(surface, (17, 75, 49), panel, border_radius=5)
        surface.blit(_render_text('可选操作', text_font, (250, 235, 180)), (w - 235, 112))
        for action, (bx, by, width, height) in buttons:
            rect = pygame.Rect(bx, by, width, height)
            pygame.draw.rect(surface, (190, 145, 55), rect, border_radius=4)
            surface.blit(
                _render_text(
                    _action_label(
                        action,
                        self_draw=(
                            state.phase == 'play'
                            and state.drawn_tile is not None
                        ),
                        heaven_win=state.heaven_win,
                    ),
                    text_font,
                    (20, 20, 20),
                ),
                (bx + 8, by + 7),
            )
    if state.terminal_reason == 'win':
        self_draw = pygame.Rect(w - 250, 70, 235, 30)
        pygame.draw.rect(surface, (255, 205, 205), self_draw, border_radius=4)
        pygame.draw.rect(surface, (210, 25, 25), self_draw, 3, border_radius=4)
        label = '天胡' if state.heaven_win else '自摸'
        surface.blit(_render_text(label, text_font, (190, 20, 20)), (w - 235, 73))

    if mode == 'human':
        screen.blit(surface, (0, 0))
        pygame.display.flip()
    return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2)).copy()
