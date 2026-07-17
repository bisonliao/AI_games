"""Persistent symmetry-aware cache of ranked expert decisions."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from .heuristic_agent import ExpertDecision, HeuristicAgent
from .sampling import rank_softmax_action
from .symmetry import canonicalize, inverse_action


class ExpertCache:
    FORMAT_VERSION = 2

    def __init__(self, path: Path, board_size: int, max_candidates: int, seed: int,
                 top_k: int = 4, temperature: float = 1.0,
                 stochastic_moves: int = 6) -> None:
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self.size = int(board_size); self.top_k = max(1, int(top_k))
        self.temperature = float(temperature); self.stochastic_moves = int(stochastic_moves)
        self.agent = HeuristicAgent(seed=seed, max_candidates=max_candidates)
        self.rng = np.random.default_rng(seed + 991)
        self.db = sqlite3.connect(self.path)
        self.db.execute("CREATE TABLE IF NOT EXISTS metadata (name TEXT PRIMARY KEY, value TEXT NOT NULL)")
        expected = {"format_version": str(self.FORMAT_VERSION), "board_size": str(board_size),
                    "max_candidates": str(max_candidates), "seed": str(seed),
                    "top_k": str(self.top_k), "temperature": str(self.temperature),
                    "stochastic_moves": str(self.stochastic_moves)}
        actual = dict(self.db.execute("SELECT name, value FROM metadata"))
        if actual and actual != expected:
            self.db.close(); raise ValueError(f"expert cache configuration mismatch: {self.path}")
        self.db.execute("CREATE TABLE IF NOT EXISTS decisions ("
                        "key BLOB PRIMARY KEY, tactical INTEGER NOT NULL, reason TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS candidates ("
                        "key BLOB NOT NULL, rank INTEGER NOT NULL, action INTEGER NOT NULL, "
                        "PRIMARY KEY (key, rank))")
        self.db.executemany("INSERT OR REPLACE INTO metadata VALUES (?, ?)", expected.items())
        self.db.commit(); self.hits = self.misses = self.expert_queries = 0

    def decision(self, board: np.ndarray, player: int) -> ExpertDecision:
        key, canonical, transform = canonicalize(np.asarray(board, dtype=np.int8), player)
        row = self.db.execute("SELECT tactical, reason FROM decisions WHERE key = ?", (key,)).fetchone()
        candidates = self.db.execute(
            "SELECT action FROM candidates WHERE key = ? ORDER BY rank", (key,)).fetchall()
        if row is None or not candidates:
            mask = canonical.reshape(-1) == 0
            canonical_decision = self.agent.ranked_decision(canonical, player, mask, self.top_k)
            self.db.execute("INSERT OR REPLACE INTO decisions VALUES (?, ?, ?)",
                            (key, int(canonical_decision.tactical), canonical_decision.reason))
            self.db.executemany("INSERT OR REPLACE INTO candidates VALUES (?, ?, ?)",
                                [(key, rank, action)
                                 for rank, action in enumerate(canonical_decision.actions)])
            actions = canonical_decision.actions; tactical = canonical_decision.tactical
            reason = canonical_decision.reason; self.misses += 1; self.expert_queries += 1
            if self.misses % 100 == 0:
                self.db.commit()
        else:
            actions = tuple(int(item[0]) for item in candidates)
            tactical = bool(row[0]); reason = str(row[1]); self.hits += 1
        transformed = tuple(inverse_action(action, self.size, transform) for action in actions)
        return ExpertDecision(transformed, tactical, reason)

    def label(self, board: np.ndarray, player: int) -> int:
        decision = self.decision(board, player)
        move_index = int(np.count_nonzero(np.asarray(board) == player))
        return rank_softmax_action(decision.actions, move_index, self.rng, top_k=self.top_k,
                                   temperature=self.temperature,
                                   stochastic_moves=self.stochastic_moves,
                                   force_greedy=decision.tactical)

    def close(self) -> None:
        self.db.commit(); self.db.close()

    def __enter__(self) -> "ExpertCache":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
