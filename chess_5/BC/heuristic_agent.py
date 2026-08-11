"""BC 数据标注与评测共用的冻结启发式五子棋专家。

专家不参与学习。它以固定优先级处理中心开局、一步必胜、必须封堵、制造双杀、
阻止双杀，最后才对普通候选执行棋形评分和一层最坏回复搜索。该排序是监督标签
语义的一部分；修改决策结果时必须同步升级 oracle.py 中的专家版本。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ExpertDecision:
    """专家输出：有序候选动作、是否必须 greedy，以及可审计的决策原因。"""
    actions: tuple[int, ...]
    tactical: bool
    reason: str


class HeuristicAgent:
    """带战术规则、棋形估值和浅搜索的确定性候选排序专家。"""

    def __init__(self, seed: int = 0, max_candidates: int = 12) -> None:
        self.rng = np.random.default_rng(seed)
        self.max_candidates = max(1, int(max_candidates))
        self._position_cache: dict[tuple[bytes, int], float] = {}

    def select_actions(self, boards: np.ndarray, current_players: np.ndarray,
                       action_masks: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
        """批量兼容接口：对每个棋盘返回专家 top-1 动作。"""
        del epsilon
        boards = np.asarray(boards, dtype=np.int8)
        if boards.ndim == 2:
            boards = boards[None, ...]
        players = np.asarray(current_players).reshape(-1).astype(np.int8)
        masks = np.asarray(action_masks).reshape(boards.shape[0], -1).astype(bool)
        if players.size != boards.shape[0] or masks.shape[1] != boards.shape[1] * boards.shape[2]:
            raise ValueError("HeuristicAgent inputs have incompatible batch shapes")
        return np.asarray([self.ranked_decision(board, int(player), mask, top_k=1).actions[0]
                           for board, player, mask in zip(boards, players, masks)], dtype=np.int64)

    def ranked_decision(self, board: np.ndarray, player: int, mask: np.ndarray,
                        top_k: int = 4) -> ExpertDecision:
        """按固定战术优先级返回最多 top_k 个候选动作。

        win/block 只返回唯一明确动作并标记 tactical，生成器遇到该标记会停止随机
        采样；fork 和普通局面允许多个合理候选，供 top-4 软标签使用。
        """
        self._position_cache.clear()
        size = board.shape[0]
        if board.shape != (size, size):
            raise ValueError("Gomoku board must be square")
        legal = np.flatnonzero(mask & (board.reshape(-1) == 0))
        if legal.size == 0:
            raise RuntimeError("No legal actions available")
        # 规则优先级不可随意调整，它决定数据集中 reason 和 top-1 标签的含义。
        if not np.any(board):
            center = (size // 2) * size + size // 2
            if center in legal:
                return ExpertDecision((int(center),), False, "opening_center")
        winning = self._winning_moves(board, player, legal)
        if winning:
            return ExpertDecision(tuple(self._rank_candidates(
                board, player, winning, shallow_search=False, limit=1)), True, "win")
        opponent = -player
        opponent_wins = self._winning_moves(board, opponent, legal)
        if opponent_wins:
            return ExpertDecision(tuple(self._rank_candidates(
                board, player, opponent_wins, shallow_search=False, limit=1)), True, "block")
        own_forks = self._fork_moves(board, player, legal)
        if own_forks:
            return ExpertDecision(tuple(self._rank_candidates(
                board, player, own_forks, shallow_search=False, limit=top_k)), False, "own_fork")
        opponent_forks = self._fork_moves(board, opponent, legal)
        if opponent_forks:
            return ExpertDecision(tuple(self._rank_candidates(
                board, player, opponent_forks, shallow_search=False, limit=top_k)), False, "block_fork")
        return ExpertDecision(tuple(self._rank_candidates(
            board, player, self._shortlist(board, player, legal),
            shallow_search=True, limit=top_k)), False, "normal")

    def _rank_candidates(self, board: np.ndarray, player: int, candidates: Sequence[int],
                         *, shallow_search: bool, limit: int) -> List[int]:
        """综合静态棋形、双方一步威胁和可选浅搜索，对候选稳定排序。"""
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
            # 已经暴露对手一步胜时无需继续浅搜索；普通候选按最坏回复重新估值。
            if shallow_search and opponent_threats == 0:
                score = min(score, self._worst_opponent_reply(next_board, player))
            scores.append((score, int(action)))
        if not scores:
            raise RuntimeError("Heuristic candidate set contained no legal action")
        # 同分时按动作下标稳定排序，保证跨进程、跨轮专家标签一致。
        scores.sort(key=lambda item: (-item[0], item[1]))
        return [action for _, action in scores[:max(1, int(limit))]]

    def _worst_opponent_reply(self, board: np.ndarray, player: int) -> float:
        """执行一层 minimax：返回对手最佳回复后当前方能保证的最差估值。"""
        legal = np.flatnonzero(board.reshape(-1) == 0)
        if legal.size == 0:
            return self._position_score(board, player)
        opponent = -player
        if self._winning_moves(board, opponent, legal):
            return -1_000_000_000.0
        worst = float("inf")
        for action in self._shortlist(board, opponent, legal):
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
        """从所有合法动作中保留高分、靠近中心和已有棋形的有限候选。"""
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
        return [action for score, action in scored if score >= cutoff][:self.max_candidates]

    def _fork_moves(self, board: np.ndarray, player: int, legal: Iterable[int]) -> List[int]:
        """查找落子后同时产生至少两个一步胜点的双杀动作。"""
        forks: List[int] = []
        size = board.shape[0]
        for action in legal:
            row, col = divmod(int(action), size)
            candidate = board.copy()
            candidate[row, col] = player
            if not self._has_five_from(candidate, row, col, player) and \
                    len(self._winning_moves(candidate, player)) >= 2:
                forks.append(int(action))
        return forks

    def _winning_moves(self, board: np.ndarray, player: int,
                       legal: Iterable[int] | None = None) -> List[int]:
        """通过原地试落并回滚，枚举一步形成五连的动作。"""
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
        """从新落点向横、竖、两条对角线检查连续五子。"""
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
        """当前方棋形分减去 1.12 倍对手棋形分，并缓存本次决策中的重复局面。"""
        key = (board.tobytes(), int(player))
        cached = self._position_cache.get(key)
        if cached is None:
            cached = self._pattern_score(board, player) - 1.12 * self._pattern_score(board, -player)
            self._position_cache[key] = cached
        return cached

    def _pattern_score(self, board: np.ndarray, player: int) -> float:
        """累加棋盘所有有效行上的五格窗口与连续棋形贡献。"""
        score = 0.0
        for line in self._lines(board):
            for contribution in _line_pattern_contributions(line.tobytes(), int(player)):
                score += contribution
        return score

    @staticmethod
    def _lines(board: np.ndarray) -> Iterable[np.ndarray]:
        """按冻结顺序遍历横线、竖线和长度至少为 5 的两向对角线。"""
        flat = np.asarray(board).reshape(-1)
        for indices in _line_indices(board.shape[0]):
            yield flat[indices]

    def _choose_tied(self, actions: Sequence[int]) -> int:
        return int(self.rng.choice(np.asarray(actions, dtype=np.int64)))


@lru_cache(maxsize=5)
def _line_indices(size: int) -> tuple[np.ndarray, ...]:
    """缓存 heuristic-v1 使用的精确行顺序；顺序会影响浮点累加和同分结果。"""
    marker = np.arange(size * size, dtype=np.int16).reshape(size, size)
    lines: list[np.ndarray] = []
    for index in range(size):
        lines.append(marker[index, :].copy())
        lines.append(marker[:, index].copy())
    for offset in range(-(size - 5), size - 4):
        lines.append(np.diagonal(marker, offset=offset).copy())
        lines.append(np.diagonal(np.fliplr(marker), offset=offset).copy())
    return tuple(lines)


@lru_cache(maxsize=500_000)
def _line_pattern_contributions(raw: bytes, player: int) -> tuple[float, ...]:
    """缓存一条线的有序棋形贡献，不改变原浮点累加顺序。"""
    line = np.frombuffer(raw, dtype=np.int8)
    weights = (0.0, 1.0, 8.0, 60.0, 600.0, 1_000_000.0)
    contributions: list[float] = []
    # 第一部分：统计不含对手棋子的每个连续五格窗口。
    for start in range(max(0, len(line) - 4)):
        window = line[start:start + 5]
        if not np.any(window == -player):
            contributions.append(weights[int(np.count_nonzero(window == player))])
    # 第二部分：连续己方棋子按长度和开口数加分，双开口价值更高。
    index = 0
    while index < len(line):
        if line[index] != player:
            index += 1
            continue
        end = index
        while end + 1 < len(line) and line[end + 1] == player:
            end += 1
        length = min(5, end - index + 1)
        open_ends = int(index > 0 and line[index - 1] == 0) + \
            int(end + 1 < len(line) and line[end + 1] == 0)
        if open_ends:
            contributions.append(weights[length] * (0.35 if open_ends == 1 else 0.8))
        index = end + 1
    return tuple(contributions)
