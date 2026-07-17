"""Persistent, symmetry-aware cache for expensive heuristic labels."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from DQN.heuristic_agent import HeuristicAgent
from .symmetry import canonicalize, inverse_action


class ExpertCache:
    def __init__(self, path: Path, board_size: int, max_candidates: int, seed: int,
                 labels_per_state: int = 4) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.size = int(board_size)
        self.labels_per_state = max(1, int(labels_per_state))
        self.agent = HeuristicAgent(seed=seed, max_candidates=max_candidates)
        self.rng = np.random.default_rng(seed + 991)
        self.db = sqlite3.connect(self.path)
        self.db.execute("CREATE TABLE IF NOT EXISTS labels (key BLOB NOT NULL, slot INTEGER NOT NULL, "
                        "action INTEGER NOT NULL, PRIMARY KEY (key, slot))")
        self.db.execute("CREATE TABLE IF NOT EXISTS metadata (name TEXT PRIMARY KEY, value TEXT NOT NULL)")
        expected = {"board_size": str(board_size), "max_candidates": str(max_candidates),
                    "seed": str(seed), "labels_per_state": str(self.labels_per_state)}
        actual = dict(self.db.execute("SELECT name, value FROM metadata"))
        if actual and actual != expected:
            self.db.close()
            raise ValueError(f"expert cache configuration mismatch: {self.path}")
        self.db.executemany("INSERT OR REPLACE INTO metadata VALUES (?, ?)", expected.items())
        self.db.commit()
        self.hits = 0
        self.misses = 0
        self.expert_queries = 0

    def label(self, board: np.ndarray, player: int) -> int:
        key, canonical, transform = canonicalize(np.asarray(board, dtype=np.int8), player)
        rows = self.db.execute("SELECT action FROM labels WHERE key = ? ORDER BY slot", (key,)).fetchall()
        if not rows:
            mask = canonical.reshape(-1) == 0
            canonical_actions = [int(self.agent.select_actions(canonical, [player], mask)[0])
                                 for _ in range(self.labels_per_state)]
            self.db.executemany("INSERT OR IGNORE INTO labels VALUES (?, ?, ?)",
                                [(key, slot, action) for slot, action in enumerate(canonical_actions)])
            self.misses += 1
            self.expert_queries += self.labels_per_state
            if self.misses % 100 == 0:
                self.db.commit()
        else:
            canonical_actions = [int(row[0]) for row in rows]; self.hits += 1
        canonical_action = int(self.rng.choice(canonical_actions))
        return inverse_action(canonical_action, self.size, transform)

    def close(self) -> None:
        self.db.commit(); self.db.close()

    def __enter__(self) -> "ExpertCache":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
