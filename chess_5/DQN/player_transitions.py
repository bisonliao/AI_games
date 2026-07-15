from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

try:
    from .agent import encode_boards
    from .returns import NStepAccumulator, Transition, shaped_reward
except ImportError:
    from agent import encode_boards
    from returns import NStepAccumulator, Transition, shaped_reward


class PlayerTransitionCollector:
    """Build same-player decision-interval transitions for one board color.

    One instance owns only one player's pending states and n-step accumulators.
    A dual-sided rollout composes two independent instances (black and white)
    instead of mixing their state machines in one set of branches.
    """

    def __init__(
        self,
        *,
        player: int,
        num_envs: int,
        board_size: int,
        n_step: int,
        gamma: float,
        reward_shaping_scale: float = 0.0,
    ) -> None:
        if player not in (1, -1):
            raise ValueError("player must be 1 (black) or -1 (white)")
        self.player = int(player)
        self.num_envs = int(num_envs)
        self.board_size = int(board_size)
        self.action_dim = self.board_size * self.board_size
        self.gamma = float(gamma)
        self.reward_shaping_scale = float(reward_shaping_scale)
        self.pending_states: List[np.ndarray | None] = [None] * self.num_envs
        self.pending_actions: List[int | None] = [None] * self.num_envs
        self.n_step = NStepAccumulator(self.num_envs, n_step, gamma)

    def record_actions(
        self,
        boards: np.ndarray,
        env_indices: Sequence[int],
        actions: Sequence[int],
    ) -> None:
        """Record this player's pre-action states for the selected env slots."""
        indices = np.asarray(env_indices, dtype=np.int64)
        action_array = np.asarray(actions, dtype=np.int64)
        board_array = np.asarray(boards, dtype=np.int8)
        if board_array.ndim == 2:
            board_array = board_array[None, ...]
        if len(indices) != len(action_array) or len(indices) != len(board_array):
            raise ValueError("boards, env_indices, and actions must have equal batch size")
        states = encode_boards(
            board_array, np.full(len(indices), self.player, dtype=np.int8)
        )
        for row, env_index_value in enumerate(indices):
            env_index = int(env_index_value)
            if self.pending_states[env_index] is not None:
                raise RuntimeError(
                    f"Player {self.player} already has a pending transition "
                    f"for env {env_index}"
                )
            self.pending_states[env_index] = states[row]
            self.pending_actions[env_index] = int(action_array[row])

    def finish_step(
        self,
        *,
        acted_players: np.ndarray,
        next_obs: Dict[str, np.ndarray],
        dones: np.ndarray,
        rewards: np.ndarray,
    ) -> List[Transition]:
        """Close transitions after an opponent reply or either terminal move."""
        acted = np.asarray(acted_players).reshape(-1)
        done_array = np.asarray(dones, dtype=np.bool_).reshape(-1)
        reward_array = np.asarray(rewards, dtype=np.float32).reshape(-1)
        if len(acted) != self.num_envs or len(done_array) != self.num_envs:
            raise ValueError("finish_step arrays must contain one item per environment")
        if len(reward_array) != self.num_envs:
            raise ValueError("rewards must contain one item per environment")

        emitted: List[Transition] = []
        for env_index in range(self.num_envs):
            state = self.pending_states[env_index]
            if state is None:
                continue
            done = bool(done_array[env_index])
            # A non-terminal move by this same player only opens the interval.
            # Its transition closes after the opponent has replied.
            if not done and int(acted[env_index]) == self.player:
                continue

            if done:
                next_state = np.zeros(
                    (3, self.board_size, self.board_size), dtype=np.float32
                )
                next_mask = np.zeros(self.action_dim, dtype=np.bool_)
            else:
                next_player = int(next_obs["current_player"][env_index].item())
                if next_player != self.player:
                    raise RuntimeError(
                        f"Expected player {self.player} after opponent reply in env "
                        f"{env_index}, got {next_player}"
                    )
                next_state = encode_boards(
                    next_obs["board"][env_index],
                    next_obs["current_player"][env_index],
                )[0]
                next_mask = next_obs["action_mask"][env_index].astype(np.bool_)

            reward = shaped_reward(
                float(reward_array[env_index]), state, next_state, done,
                self.gamma, self.reward_shaping_scale,
            )
            action = self.pending_actions[env_index]
            if action is None:
                raise RuntimeError("pending state is missing its action")
            emitted.extend(self.n_step.add(env_index, Transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                next_mask=next_mask,
                done=done,
            )))
            self.pending_states[env_index] = None
            self.pending_actions[env_index] = None
        return emitted

    def pending_count(self) -> int:
        return sum(state is not None for state in self.pending_states)
