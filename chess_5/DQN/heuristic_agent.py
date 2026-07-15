from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np


class HeuristicAgent:
    """Seeded tactical Gomoku policy with a shallow opponent-response search."""

    def __init__(self, seed: int = 0, max_candidates: int = 12) -> None:
        self.rng = np.random.default_rng(seed)
        self.max_candidates = max(1, int(max_candidates))

    def select_actions(
        self,
        boards: np.ndarray,
        current_players: np.ndarray,
        action_masks: np.ndarray,
        epsilon: float = 0.0,
    ) -> np.ndarray:
        del epsilon
        boards = np.asarray(boards, dtype=np.int8)
        if boards.ndim == 2:
            boards = boards[None, ...]
        players = np.asarray(current_players).reshape(-1).astype(np.int8)
        masks = np.asarray(action_masks).reshape(boards.shape[0], -1).astype(bool)
        if players.size != boards.shape[0] or masks.shape[1] != boards.shape[1] * boards.shape[2]:
            raise ValueError("HeuristicAgent inputs have incompatible batch shapes")

        actions = np.empty(boards.shape[0], dtype=np.int64)
        for index in range(boards.shape[0]):
            actions[index] = self._select_one(boards[index], int(players[index]), masks[index])
        return actions

    def _select_one(self, board: np.ndarray, player: int, mask: np.ndarray) -> int:
        size = board.shape[0]
        if board.shape != (size, size):
            raise ValueError("Gomoku board must be square")
        legal = np.flatnonzero(mask & (board.reshape(-1) == 0))
        if legal.size == 0:
            raise RuntimeError("No legal actions available")
        if not np.any(board):
            center = (size // 2) * size + size // 2
            if center in legal:
                return int(center)

        winning = self._winning_moves(board, player, legal)
        if winning:
            return self._choose_tied(winning)

        opponent = -player
        opponent_wins = self._winning_moves(board, opponent, legal)
        if opponent_wins:
            return self._best_from_candidates(board, player, opponent_wins, shallow_search=False)

        own_forks = self._fork_moves(board, player, legal)
        if own_forks:
            return self._best_from_candidates(board, player, own_forks, shallow_search=False)

        opponent_forks = self._fork_moves(board, opponent, legal)
        if opponent_forks:
            return self._best_from_candidates(board, player, opponent_forks, shallow_search=False)

        candidates = self._shortlist(board, player, legal)
        return self._best_from_candidates(board, player, candidates, shallow_search=True)

    def _best_from_candidates(
        self,
        board: np.ndarray,
        player: int,
        candidates: Sequence[int],
        *,
        shallow_search: bool,
    ) -> int:
        scores: List[Tuple[float, int]] = []
        for action in candidates:
            next_board = board.copy()
            row, col = divmod(int(action), board.shape[0])
            if next_board[row, col] != 0:
                continue
            next_board[row, col] = player
            score = self._position_score(next_board, player)
            own_threats = len(self._winning_moves(next_board, player))
            opponent_threats = len(self._winning_moves(next_board, -player))
            score += own_threats * 8_000.0 - opponent_threats * 20_000.0
            if shallow_search and opponent_threats == 0:
                score = min(score, self._worst_opponent_reply(next_board, player))
            scores.append((score, int(action)))
        if not scores:
            raise RuntimeError("Heuristic candidate set contained no legal action")
        best_score = max(score for score, _ in scores)
        tied = [action for score, action in scores if np.isclose(score, best_score)]
        return self._choose_tied(tied)

    def _worst_opponent_reply(self, board: np.ndarray, player: int) -> float:
        legal = np.flatnonzero(board.reshape(-1) == 0)
        if legal.size == 0:
            return self._position_score(board, player)
        opponent = -player
        replies = self._winning_moves(board, opponent, legal)
        if replies:
            return -1_000_000_000.0
        reply_candidates = self._shortlist(board, opponent, legal)
        worst = float("inf")
        for action in reply_candidates:
            reply_board = board.copy()
            row, col = divmod(int(action), board.shape[0])
            reply_board[row, col] = opponent
            own_threats = len(self._winning_moves(reply_board, player))
            opponent_threats = len(self._winning_moves(reply_board, opponent))
            score = self._position_score(reply_board, player)
            score += own_threats * 5_000.0 - opponent_threats * 12_000.0
            worst = min(worst, score)
        return worst

    def _shortlist(self, board: np.ndarray, player: int, legal: np.ndarray) -> List[int]:
        if legal.size <= self.max_candidates:
            return [int(action) for action in legal]
        scored: List[Tuple[float, int]] = []
        size = board.shape[0]
        center = (size - 1) / 2.0
        occupied = np.argwhere(board != 0)
        for action in legal:
            row, col = divmod(int(action), size)
            candidate = board.copy()
            candidate[row, col] = player
            score = self._position_score(candidate, player)
            score -= 0.15 * ((row - center) ** 2 + (col - center) ** 2)
            if occupied.size:
                distance = np.max(np.abs(occupied - np.asarray([row, col])), axis=1).min()
                score -= 0.5 * float(distance)
            scored.append((score, int(action)))
        scored.sort(key=lambda item: item[0], reverse=True)
        cutoff = scored[min(self.max_candidates, len(scored)) - 1][0]
        return [action for score, action in scored if score >= cutoff][: self.max_candidates]

    def _fork_moves(
        self,
        board: np.ndarray,
        player: int,
        legal: Iterable[int],
    ) -> List[int]:
        forks: List[int] = []
        size = board.shape[0]
        for action in legal:
            row, col = divmod(int(action), size)
            candidate = board.copy()
            candidate[row, col] = player
            if self._has_five_from(candidate, row, col, player):
                continue
            if len(self._winning_moves(candidate, player)) >= 2:
                forks.append(int(action))
        return forks

    def _winning_moves(
        self,
        board: np.ndarray,
        player: int,
        legal: Iterable[int] | None = None,
    ) -> List[int]:
        if legal is None:
            legal = np.flatnonzero(board.reshape(-1) == 0)
        wins: List[int] = []
        size = board.shape[0]
        for action in legal:
            row, col = divmod(int(action), size)
            if board[row, col] != 0:
                continue
            board[row, col] = player
            won = self._has_five_from(board, row, col, player)
            board[row, col] = 0
            if won:
                wins.append(int(action))
        return wins

    @staticmethod
    def _has_five_from(board: np.ndarray, row: int, col: int, player: int) -> bool:
        size = board.shape[0]
        for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
            count = 1
            for sign in (-1, 1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < size and 0 <= c < size and board[r, c] == player:
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= 5:
                return True
        return False

    def _position_score(self, board: np.ndarray, player: int) -> float:
        own = self._pattern_score(board, player)
        opponent = self._pattern_score(board, -player)
        return own - 1.12 * opponent

    def _pattern_score(self, board: np.ndarray, player: int) -> float:
        weights = (0.0, 1.0, 8.0, 60.0, 600.0, 1_000_000.0)
        score = 0.0
        for line in self._lines(board):
            for start in range(max(0, len(line) - 4)):
                window = line[start : start + 5]
                if np.any(window == -player):
                    continue
                score += weights[int(np.count_nonzero(window == player))]
            index = 0
            while index < len(line):
                if line[index] != player:
                    index += 1
                    continue
                end = index
                while end + 1 < len(line) and line[end + 1] == player:
                    end += 1
                length = min(5, end - index + 1)
                open_ends = int(index > 0 and line[index - 1] == 0) + int(
                    end + 1 < len(line) and line[end + 1] == 0
                )
                if open_ends:
                    score += weights[length] * (0.35 if open_ends == 1 else 0.8)
                index = end + 1
        return score

    @staticmethod
    def _lines(board: np.ndarray) -> Iterable[np.ndarray]:
        size = board.shape[0]
        for index in range(size):
            yield board[index, :]
            yield board[:, index]
        for offset in range(-(size - 5), size - 4):
            yield np.diagonal(board, offset=offset)
            yield np.diagonal(np.fliplr(board), offset=offset)

    def _choose_tied(self, actions: Sequence[int]) -> int:
        return int(self.rng.choice(np.asarray(actions, dtype=np.int64)))
