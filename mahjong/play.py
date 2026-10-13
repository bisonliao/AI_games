"""Human expert data collection entry point."""

import argparse

import numpy as np

from MahjongEnv.config import EnvConfig
from MahjongEnv.dataset import EpisodeRecorder
from MahjongEnv.env import MahjongEnv
from MahjongEnv.renderer import (
    DISCARD_KEYS,
    HAND_TILE_GAP,
    HAND_TILE_HEIGHT,
    HAND_TILE_WIDTH,
    HAND_TILE_Y_FROM_BOTTOM,
    HAND_X,
    _font,
    _render_text,
    action_button_rects,
    display_hand,
)


def run(args):
    cfg = EnvConfig(
        observation_mode='both' if not args.headless else 'vector',
        opponent_action_delay=getattr(args, 'opponent_delay', 1.5),
    )
    env = MahjongEnv(cfg, render_mode=None if args.headless else 'human', seed=args.seed)
    try:
        completed = 0
        while args.episodes <= 0 or completed < args.episodes:
            rec = EpisodeRecorder()
            obs, reset_info = env.reset()
            if reset_info.get('terminal_reason') is not None:
                # A non-human starter can win on the initial 14 tiles.
                if not args.headless:
                    _wait_for_next_game(env, reset_info)
                completed += 1
                continue
            done = False
            while not done:
                legal = np.flatnonzero(obs['action_mask'])
                action = int(env.rng.choice(legal)) if args.headless else _human_action(env, legal)
                next_obs, reward, terminated, truncated, info = env.step(action)
                rec.add(obs, action, reward, next_obs, terminated, truncated, info)
                obs = next_obs
                done = terminated or truncated
            if args.output:
                rec.save(args.output)
            if not args.headless:
                _wait_for_next_game(env, info)
            completed += 1
    finally:
        env.close()


def _human_action(env, legal):
    import pygame

    legal = set(int(action) for action in legal)
    tile_w, tile_gap = HAND_TILE_WIDTH, HAND_TILE_GAP
    tile_y = env.config.image_size[1] - HAND_TILE_Y_FROM_BOTTOM
    key_events = {
        pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2, pygame.K_4: 3,
        pygame.K_5: 4, pygame.K_6: 5, pygame.K_7: 6, pygame.K_8: 7,
        pygame.K_9: 8, pygame.K_0: 9, pygame.K_q: 10, pygame.K_e: 11,
        pygame.K_r: 12, pygame.K_t: 13,
    }
    env.render()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and 140 in legal:
                    return 140
                shortcuts = {
                    pygame.K_p: 1,  # peng
                    pygame.K_m: 2,  # ming kong
                    pygame.K_c: 3,  # concealed kong
                    pygame.K_a: 4,  # added kong
                }
                if event.key in shortcuts:
                    group = shortcuts[event.key]
                    choices = sorted(a for a in legal if group * 28 <= a < (group + 1) * 28)
                    if choices:
                        return choices[0]
                if event.key in key_events and env.state.phase == 'play':
                    index = key_events[event.key]
                    hand = display_hand(env.state.players[0].hand, env.state.drawn_tile)
                    if index < len(hand) and hand[index] in legal:
                        return int(hand[index])
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                x, y = event.pos
                if tile_y <= y <= tile_y + HAND_TILE_HEIGHT:
                    index = (x - HAND_X) // (tile_w + tile_gap)
                    if index >= 0 and (x - HAND_X) % (tile_w + tile_gap) < tile_w:
                        hand = display_hand(env.state.players[0].hand, env.state.drawn_tile)
                        if index < len(hand) and hand[index] in legal:
                            return int(hand[index])
                for action, (bx, by, width, height) in action_button_rects(legal, env.config.image_size):
                    if bx <= x <= bx + width and by <= y <= by + height:
                        return action
        pygame.time.wait(20)


def _wait_for_next_game(env, info):
    """Show the terminal result and require explicit confirmation to reset."""
    import pygame

    env.render()
    screen = pygame.display.get_surface()
    if screen is None:
        return
    width, height = screen.get_size()
    reason = info.get('terminal_reason')
    if reason == 'win':
        title = '我方自摸胜利'
    elif reason == 'loss':
        title = '牌友胡牌，本局结束'
    elif reason == 'draw':
        title = '流局'
    else:
        title = '本局结束'
    if info.get('heaven_win') and reason == 'loss':
        title = '牌友天胡，本局结束'
    overlay = pygame.Surface((width, height), pygame.SRCALPHA)
    overlay.fill((8, 35, 25, 170))
    panel = pygame.Rect(width // 2 - 250, height // 2 - 90, 500, 180)
    pygame.draw.rect(overlay, (20, 75, 50, 245), panel, border_radius=8)
    title_surface = _render_text(title, _font(34), (255, 245, 215))
    hint_surface = _render_text('按 Enter 或 Space 进入下一局', _font(23), (245, 225, 160))
    overlay.blit(title_surface, title_surface.get_rect(center=(width // 2, height // 2 - 28)))
    overlay.blit(hint_surface, hint_surface.get_rect(center=(width // 2, height // 2 + 38)))
    screen.blit(overlay, (0, 0))
    pygame.display.flip()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                pygame.event.clear()
                return
        pygame.time.wait(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=0, help='局数；0 表示持续运行，直到关闭窗口或 Ctrl-C')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--output', default='expert_data', help='每局数据保存目录')
    parser.add_argument('--opponent-delay', type=float, default=1.5, help='每位牌友出牌后的显示秒数')
    parser.add_argument('--headless', action='store_true')
    run(parser.parse_args())
