import numpy as np
import time
import gymnasium as gym
from gymnasium import spaces
from .config import EnvConfig
from .tiles import NUM_TILES, RED, make_deck, counts
from .state import GameState, PlayerState, Meld
from .rules import is_win, concealed_kongs, potential
from .reward import shaped_reward
from .bots import choose_discard

ACTION_NAMES = ('discard','peng','ming_kong','concealed_kong','added_kong')
ACTION_OFFSETS = {n:i*28 for i,n in enumerate(ACTION_NAMES)}
ACTION_OFFSETS.update({'pass':140})
N_ACTIONS = 141

class MahjongEnv(gym.Env):
    metadata={'render_modes':['human','rgb_array'], 'render_fps':8}
    def __init__(self, config=None, render_mode=None, seed=None):
        super().__init__()
        self.config = config or EnvConfig()
        self.render_mode = (
            render_mode if render_mode is not None else self.config.render_mode
        )
        self.observation_mode = self.config.observation_mode
        self._seed = seed if seed is not None else self.config.seed
        self.rng = np.random.default_rng(self._seed)
        self.state = None
        self._renderer = None
        self._episode_id = 0
        self.last_discard_potential = None
        # hand, own melds, per-player discards, public meld tiles, meld kinds,
        # scalar state, phase flags, and two shaping features.
        self.vector_dim = 28 + 28 + 4 * 28 + 4 * 28 + 4 * 4 + 4 + 4 + 2
        vd = spaces.Box(
            -np.inf,
            np.inf,
            (self.vector_dim,),
            dtype=np.float32,
        )
        mask = spaces.MultiBinary(N_ACTIONS)
        image_space = spaces.Box(
            0,
            255,
            (self.config.image_size[1], self.config.image_size[0], 3),
            dtype=np.uint8,
        )
        observation_fields = {}
        if self.observation_mode in ('vector', 'both'):
            observation_fields['vector'] = vd
        if self.observation_mode in ('rgb', 'both'):
            observation_fields['image'] = image_space
        if self.config.include_action_mask_in_obs:
            observation_fields['action_mask'] = mask
        self.observation_space = spaces.Dict(observation_fields)
        self.action_space = spaces.Discrete(N_ACTIONS)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # A reset without an explicit seed starts a new random deal from the
        # existing RNG stream.  Re-seeding with ``self._seed`` on every reset
        # would replay the same deal forever in direct/single-env use.
        if seed is not None or self.state is None:
            self.rng = np.random.default_rng(
                seed if seed is not None else self._seed
            )
        self._episode_id += 1
        self.state = GameState(wall=make_deck(self.rng))
        self.state.episode_id = self._episode_id
        self.last_discard_potential = None
        self.state.start_player = int(self.rng.integers(0, 4))
        for player in self.state.players:
            player.hand.clear()
            player.melds.clear()
            player.discards.clear()
        for _ in range(13):
            for player in self.state.players:
                player.hand.append(self.state.wall.pop())
        starter = self.state.start_player
        if starter == 0:
            self._draw(0)
        else:
            self.state.players[starter].hand.append(self.state.wall.pop())
            self.state.players[starter].hand.sort()
        self.state.current_player = starter
        self.state.phase = 'play'
        if is_win(self.state.players[starter].hand, len(self.state.players[starter].melds)):
            self.state.heaven_win = True
            self.state.terminal_reason = 'win' if starter == 0 else 'loss'
            self.state.phase = 'terminal'
            return self._obs(), self._info()
        if starter != 0:
            starter_hand = self.state.players[starter].hand
            tile = int(self.rng.choice(starter_hand))
            starter_hand.remove(tile)
            self.state.players[starter].discards.append(tile)
            self.state.last_discard = tile
            self.state.claim_from = starter
            order = []
            current = (starter + 1) % 4
            while current != 0:
                order.append(current)
                current = (current + 1) % 4
            self.state.pending_players = order
            if self._claim_available():
                self.state.phase = 'claim'
            else:
                self._advance_others(order, animate=False)
        return self._obs(), self._info()

    def _legal_mask(self):
        mask = np.zeros(N_ACTIONS, dtype=np.int8)
        player = self.state.players[0]
        hand = player.hand
        if self.state.terminal_reason is not None:
            return mask
        if self.state.phase == 'claim':
            t = self.state.last_discard
            if t is not None:
                if hand.count(t) >= 2:
                    mask[ACTION_OFFSETS['peng'] + t] = 1
                if hand.count(t) >= 3:
                    mask[ACTION_OFFSETS['ming_kong'] + t] = 1
            mask[ACTION_OFFSETS['pass']] = 1
            return mask
        for tile in set(hand):
            mask[ACTION_OFFSETS['discard'] + tile] = 1
        if self.state.must_discard:
            return mask
        for tile in concealed_kongs(hand):
            mask[ACTION_OFFSETS['concealed_kong'] + tile] = 1
        for meld in player.melds:
            if meld.kind == 'peng' and meld.tiles and meld.tiles[0] in hand:
                mask[ACTION_OFFSETS['added_kong'] + meld.tiles[0]] = 1
        return mask

    def _vector(self):
        state = self.state
        player = state.players[0]
        vector = []
        vector.extend((counts(player.hand) / 4).astype(np.float32))
        own_meld_tiles = []
        for meld in player.melds:
            own_meld_tiles.extend(meld.tiles)
        vector.extend((counts(own_meld_tiles) / 4).astype(np.float32))
        vector.extend(
            (state.discard_counts().reshape(-1) / 4).astype(np.float32)
        )
        vector.extend(
            (state.public_meld_counts().reshape(-1) / 4).astype(np.float32)
        )
        vector.extend(
            (state.meld_kind_counts().reshape(-1) / 4).astype(np.float32)
        )
        vector.extend(
            [
                len(state.wall) / 112,
                state.current_player / 3,
                state.step / self.config.max_steps,
                float(
                    state.last_discard if state.last_discard is not None else -1
                ) / 27,
            ]
        )
        vector.extend(
            [
                float(state.phase == 'play'),
                float(state.phase == 'claim'),
                float(state.robbed_kong),
                float(is_win(player.hand, len(player.melds))),
            ]
        )
        vector.extend(
            [
                potential(
                    player.hand,
                    len(player.melds),
                    self._visible_tile_counts(),
                    self.config.ukeire_weight,
                ) / 10,
                len(player.discards) / 30,
            ]
        )
        return np.asarray(vector, dtype=np.float32)

    def _visible_tile_counts(self):
        """Tiles unavailable based on the player's visible information."""
        return (
            self.state.discard_counts().sum(axis=0)
            + self.state.public_meld_counts(viewer=0).sum(axis=0)
        ).astype(np.int16)

    def _obs(self):
        out = {}
        if self.config.include_action_mask_in_obs:
            out['action_mask'] = self._legal_mask()
        if self.observation_mode in ('vector', 'both'):
            out['vector'] = self._vector()
        if self.observation_mode in ('rgb', 'both'):
            out['image'] = self.render()
        return out

    def _info(self):
        return {
            'action_mask': self._legal_mask().copy(),
            'episode_id': self.state.episode_id,
            'step_index': self.state.step,
            'terminal_reason': self.state.terminal_reason,
            'win': self.state.terminal_reason == 'win',
            'draw': self.state.terminal_reason == 'draw',
            'robbed_kong': self.state.robbed_kong,
            'start_player': self.state.start_player,
            'heaven_win': self.state.heaven_win,
            # Display metadata only; these fields do not change policy inputs.
            'drawn_tile': self.state.drawn_tile,
            'phase': self.state.phase,
            'last_discard': self.state.last_discard,
            'claim_from': self.state.claim_from,
        }

    def step(self, action):
        action = int(action)
        before = PlayerState(
            list(self.state.players[0].hand),
            list(self.state.players[0].melds),
            list(self.state.players[0].discards),
        )
        if not 0 <= action < N_ACTIONS:
            raise ValueError(f'action must be in [0, {N_ACTIONS}), got {action}')
        if not self._legal_mask()[action]:
            info = {**self._info(), 'invalid_action': True}
            # A terminal state has no legal actions.  Treat an accidental
            # step on it as terminal instead of returning a non-advancing
            # transition forever (which would train Q toward -0.2).
            if self.state.terminal_reason is not None:
                outcome = self.state.terminal_reason
                reward = {'win': self.config.win_reward,
                          'loss': self.config.loss_reward,
                          'draw': self.config.draw_reward}.get(outcome, 0.0)
                return self._obs(), float(reward), True, False, info
            return self._obs(), -0.2, False, False, info
        typ = next(
            (
                name
                for name, offset in ACTION_OFFSETS.items()
                if action == offset
                or (name in ACTION_NAMES and offset <= action < offset + 28)
            ),
            'invalid',
        )
        tile = action - ACTION_OFFSETS.get(typ, 0) if typ in ACTION_NAMES else -1
        p = self.state.players[0]
        event = None
        terminated = False
        outcome = None
        shaping_delta = None
        if typ=='discard':
            p.hand.remove(tile)
            p.discards.append(tile)
            self.state.last_discard = tile
            self.state.drawn_tile = None
            self.state.must_discard = False
            event = 'discard'
            # Evaluate exactly at the completed-discard boundary.  The hand
            # now has its residual size (normally 13), before opponents act
            # and before the next draw changes the visible state.
            new_potential = potential(
                p.hand,
                len(p.melds),
                self._visible_tile_counts(),
                self.config.ukeire_weight,
            )
            if self.last_discard_potential is not None:
                shaping_delta = new_potential - self.last_discard_potential
            self.last_discard_potential = new_potential
            self._advance_others()
        elif typ=='concealed_kong':
            p.hand = [x for x in p.hand if x != tile]
            p.melds.append(Meld('concealed_kong', (tile,) * 4))
            event = 'kong'
            self._draw(0)
            self._check_self_draw()
        elif typ=='peng':
            p.hand.remove(tile)
            p.hand.remove(tile)
            p.melds.append(Meld('peng', (tile,) * 3, self.state.claim_from))
            event = 'peng'
            self.state.drawn_tile = None
            self.state.phase = 'play'
            self.state.current_player = 0
            self.state.must_discard = True
        elif typ=='ming_kong':
            for _ in range(3): p.hand.remove(tile)
            p.melds.append(Meld('ming_kong', (tile,) * 4, self.state.claim_from))
            event = 'kong'
            self._draw(0)
            self.state.phase = 'play'
            self.state.current_player = 0
            self._check_self_draw()
        elif typ=='added_kong':
            p.hand.remove(tile)
            for meld in p.melds:
                if meld.kind == 'peng' and meld.tiles[0] == tile:
                    meld.kind = 'added_kong'
                    meld.tiles = (tile,) * 4
                    break
            event = 'kong'
            self._draw(0)
            self._check_self_draw()
        elif typ=='pass':
            self._advance_others(self.state.pending_players)
        if self.state.terminal_reason is not None and not terminated:
            terminated = True
            outcome = self.state.terminal_reason
        self.state.last_action = action
        self.state.step += 1
        if not terminated and not self.state.wall:
            terminated = True
            outcome = 'draw'
            self.state.terminal_reason = 'draw'
        if self.state.step >= self.config.max_steps and not terminated:
            terminated = True
            outcome = 'draw'
            self.state.terminal_reason = 'draw'
        reward = shaped_reward(
            before,
            self.state.players[0],
            event,
            terminated,
            outcome,
            self.config,
            shaping_delta=shaping_delta,
        )
        return self._obs(), reward, terminated, False, self._info()

    def _draw(self, i):
        if self.state.wall:
            tile = self.state.wall.pop()
            self.state.players[i].hand.append(tile); self.state.players[i].hand.sort()
            if i == 0: self.state.drawn_tile = tile

    def _check_self_draw(self):
        if (
            self.state.terminal_reason is None
            and is_win(self.state.players[0].hand, len(self.state.players[0].melds))
        ):
            self.state.terminal_reason = 'win'
            self.state.phase = 'terminal'

    def _advance_others(self, players=None, animate=True):
        if players is None:
            players = [1, 2, 3]
        self.state.pending_players = list(players)
        while self.state.pending_players:
            i = self.state.pending_players.pop(0)
            self._draw(i)
            hand = self.state.players[i].hand
            if is_win(hand, len(self.state.players[i].melds)):
                self.state.terminal_reason = 'loss'
                return
            tile = choose_discard(
                hand,
                self.rng,
                self.config.robot_randomness,
            )
            hand.remove(tile)
            self.state.players[i].discards.append(tile)
            self.state.last_discard = tile
            self.state.current_player = i
            self.state.claim_from = i
            if animate:
                self._pause_after_opponent_action()
            if self._claim_available():
                self.state.phase = 'claim'
                return
        self._draw(0)
        self.state.current_player = 0
        self.state.phase = 'play'
        self.state.claim_from = None
        self._check_self_draw()

    def _claim_available(self):
        t = self.state.last_discard
        if t is None:
            return False
        p = self.state.players[0]
        return p.hand.count(t) >= 2

    def _pause_after_opponent_action(self):
        delay = float(self.config.opponent_action_delay)
        if self.render_mode != 'human' or delay <= 0:
            return
        self.render()
        try:
            import pygame
            deadline = time.monotonic() + delay
            while True:
                # Drain stale keyboard input while the table is showing the
                # opponent's move. A repeated Space must not auto-pass later.
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise SystemExit
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                pygame.time.wait(min(50, max(1, int(remaining * 1000))))
        except ImportError:
            time.sleep(delay)

    def render(self):
        from .renderer import render_state
        return render_state(self.state,self.config.image_size,self.render_mode,self._legal_mask())
    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def available_actions(self):
        return np.flatnonzero(self._legal_mask()).tolist()

def make_env(config=None, seed=None, rank=0):
    cfg = config or EnvConfig()
    base = seed if seed is not None else cfg.seed
    env_seed = None if base is None else int(base) + int(rank)
    return MahjongEnv(cfg, seed=env_seed)
