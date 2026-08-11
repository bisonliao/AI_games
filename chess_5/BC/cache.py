"""可持久化、对称感知的专家候选动作缓存。

同一棋盘的 8 种旋转/镜像先归一化为一个 canonical key，专家只计算一次。
缓存保存确定性的候选排名和战术原因；随机采样发生在读取之后，因此缓存只提升
标注速度，不改变 heuristic-v1 对同一状态的答案。
"""

from __future__ import annotations

import sqlite3
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .heuristic_agent import ExpertDecision, HeuristicAgent
from .sampling import rank_softmax_action
from .symmetry import canonicalize, inverse_action


class ExpertCache:
    """两级专家缓存：进程内 LRU -> 本 worker SQLite -> 上一轮共享只读 SQLite。"""
    FORMAT_VERSION = 3

    def __init__(self, path: Path, board_size: int, max_candidates: int, seed: int,
                 top_k: int = 4, temperature: float = 1.5,
                 stochastic_moves: int = 6, shared_path: Path | None = None,
                 memory_size: int = 4096) -> None:
        """打开 worker 缓存，并校验所有影响专家答案的配置完全一致。"""
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self.size = int(board_size); self.top_k = max(1, int(top_k))
        self.temperature = float(temperature); self.stochastic_moves = int(stochastic_moves)
        self.agent = HeuristicAgent(seed=seed, max_candidates=max_candidates)
        self.rng = np.random.default_rng(seed + 991)
        self.memory_size = max(0, int(memory_size))
        self.memory: OrderedDict[bytes, tuple[tuple[int, ...], bool, str]] = OrderedDict()
        self.db = sqlite3.connect(self.path)
        self.shared_db = None
        # shared cache 只读打开；命中结果会回填当前 worker 数据库，便于本轮最终合并。
        if shared_path is not None and Path(shared_path).is_file():
            uri = f"file:{Path(shared_path).resolve()}?mode=ro"
            self.shared_db = sqlite3.connect(uri, uri=True)
        self.db.execute("CREATE TABLE IF NOT EXISTS metadata (name TEXT PRIMARY KEY, value TEXT NOT NULL)")
        expected = {"format_version": str(self.FORMAT_VERSION), "board_size": str(board_size),
                    "max_candidates": str(max_candidates),
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
        """返回原棋盘坐标系下的专家排名，并维护命中/查询统计。"""
        key, canonical, transform = canonicalize(np.asarray(board, dtype=np.int8), player)
        # canonicalize 返回原棋盘到规范棋盘的 transform；缓存动作存规范坐标，
        # 返回调用方前必须 inverse_action 还原。
        cached = self.memory.get(key)
        if cached is not None:
            self.memory.move_to_end(key)
            actions, tactical, reason = cached
            self.hits += 1
            transformed = tuple(inverse_action(action, self.size, transform) for action in actions)
            return ExpertDecision(transformed, tactical, reason)
        row, candidates = self._database_decision(self.db, key)
        # 当前 worker 未命中时先查历史共享库，最后才真正调用启发式专家。
        if (row is None or not candidates) and self.shared_db is not None:
            row, candidates = self._database_decision(self.shared_db, key)
            if row is not None and candidates:
                self.db.execute("INSERT OR IGNORE INTO decisions VALUES (?, ?, ?)",
                                (key, int(row[0]), str(row[1])))
                self.db.executemany("INSERT OR IGNORE INTO candidates VALUES (?, ?, ?)",
                                    [(key, rank, int(item[0]))
                                     for rank, item in enumerate(candidates)])
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
        # 热点状态进入有界 LRU，减少 SQLite 往返；满时淘汰最久未使用项。
        if self.memory_size:
            self.memory[key] = (tuple(actions), bool(tactical), str(reason))
            self.memory.move_to_end(key)
            while len(self.memory) > self.memory_size:
                self.memory.popitem(last=False)
        transformed = tuple(inverse_action(action, self.size, transform) for action in actions)
        return ExpertDecision(transformed, tactical, reason)

    @staticmethod
    def _database_decision(connection: sqlite3.Connection, key: bytes):
        """从 SQLite 同时读取决策元信息和按 rank 排序的候选动作。"""
        row = connection.execute(
            "SELECT tactical, reason FROM decisions WHERE key = ?", (key,)
        ).fetchone()
        candidates = connection.execute(
            "SELECT action FROM candidates WHERE key = ? ORDER BY rank", (key,)
        ).fetchall()
        return row, candidates

    def label(self, board: np.ndarray, player: int) -> int:
        """兼容旧调用：从缓存排名执行受控采样，战术决策保持 greedy。"""
        decision = self.decision(board, player)
        move_index = int(np.count_nonzero(np.asarray(board) == player))
        return rank_softmax_action(decision.actions, move_index, self.rng, top_k=self.top_k,
                                   temperature=self.temperature,
                                   stochastic_moves=self.stochastic_moves,
                                   force_greedy=decision.tactical)

    def close(self) -> None:
        """提交当前 worker 新增答案并关闭本地/共享连接。"""
        self.db.commit(); self.db.close()
        if self.shared_db is not None:
            self.shared_db.close()

    def __enter__(self) -> "ExpertCache":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def merge_caches(target: Path, sources: list[Path]) -> None:
    """将配置兼容的 worker 缓存合并为下一轮可复用的共享只读缓存。"""
    sources = [Path(path) for path in sources if Path(path).is_file()]
    if not sources:
        return
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    first = sqlite3.connect(sources[0])
    metadata = dict(first.execute("SELECT name, value FROM metadata"))
    first.close()
    output = sqlite3.connect(target)
    output.execute("CREATE TABLE IF NOT EXISTS metadata (name TEXT PRIMARY KEY, value TEXT NOT NULL)")
    output.execute("CREATE TABLE IF NOT EXISTS decisions ("
                   "key BLOB PRIMARY KEY, tactical INTEGER NOT NULL, reason TEXT NOT NULL)")
    output.execute("CREATE TABLE IF NOT EXISTS candidates ("
                   "key BLOB NOT NULL, rank INTEGER NOT NULL, action INTEGER NOT NULL, "
                   "PRIMARY KEY (key, rank))")
    existing = dict(output.execute("SELECT name, value FROM metadata"))
    if existing and existing != metadata:
        output.close()
        raise ValueError(f"cache metadata mismatch: {target}")
    output.executemany("INSERT OR REPLACE INTO metadata VALUES (?, ?)", metadata.items())
    for source in sources:
        connection = sqlite3.connect(source)
        if dict(connection.execute("SELECT name, value FROM metadata")) != metadata:
            connection.close(); output.close()
            raise ValueError(f"cache metadata mismatch: {source}")
        output.executemany("INSERT OR IGNORE INTO decisions VALUES (?, ?, ?)",
                           connection.execute("SELECT key, tactical, reason FROM decisions"))
        output.executemany("INSERT OR IGNORE INTO candidates VALUES (?, ?, ?)",
                           connection.execute("SELECT key, rank, action FROM candidates"))
        connection.close()
    output.commit(); output.close()
