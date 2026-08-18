"""Interval-aggregated TensorBoard logging for QMIX."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter


@dataclass
class _Mean:
    total: float = 0.0
    count: int = 0

    def add(self, value: float) -> None:
        self.total += float(value)
        self.count += 1

    @property
    def value(self) -> float:
        return self.total / self.count


def make_run_dir(runs_dir: str, scenario: str, exp_name: str | None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = ""
    if exp_name:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", exp_name).strip("_")
        suffix = f"_{safe_name}" if safe_name else ""
    name = (
        f"qmix_legacy_official_{scenario}{suffix}_"
        f"{timestamp}_pid{os.getpid()}"
    )
    return Path(runs_dir) / name


class TensorBoardLogger:
    def __init__(
        self,
        interval: int,
        initial_step: int,
        log_dir: Path | None,
        config: dict | None = None,
        writer=None,
    ) -> None:
        if interval < 0:
            raise ValueError("TensorBoard interval must be non-negative")
        self.interval = int(interval)
        self.last_step = int(initial_step)
        self.writer = writer
        if self.writer is None:
            self.writer = (
                SummaryWriter(log_dir=str(log_dir))
                if log_dir is not None and interval > 0
                else None
            )
        self._means: dict[str, _Mean] = defaultdict(_Mean)
        self._latest: dict[str, float] = {}
        self._success_total = 0.0
        self._success_count = 0
        self._recent_successes: deque[float] = deque(maxlen=100)
        if self.writer is not None and config is not None:
            self.writer.add_text(
                "config/json",
                "```json\n" + json.dumps(config, indent=2, sort_keys=True) + "\n```",
                initial_step,
            )

    @property
    def enabled(self) -> bool:
        return self.writer is not None and self.interval > 0

    def mean(self, tag: str, value: float) -> None:
        if self.enabled:
            self._means[tag].add(value)

    def latest(self, tag: str, value: float) -> None:
        if self.enabled:
            self._latest[tag] = float(value)

    def record_task_metrics(self, metrics: dict[str, float]) -> None:
        """Record scenario metrics with the same success tags as MADDPG."""

        if not self.enabled:
            return
        for name, value in metrics.items():
            self._means[f"task/{name}"].add(value)
        if "episode_success" in metrics:
            success = float(metrics["episode_success"])
            self._success_total += success
            self._success_count += 1
            self._recent_successes.append(success)

    def immediate(self, metrics: dict[str, float], step: int) -> None:
        if not self.enabled:
            return
        for tag, value in metrics.items():
            self.writer.add_scalar(tag, float(value), int(step))
        self.writer.flush()

    def maybe_flush(self, step: int, force: bool = False) -> bool:
        if not self.enabled or (not self._means and not self._latest):
            return False
        step = int(step)
        if not force and step - self.last_step < self.interval:
            return False
        for tag, accumulator in self._means.items():
            self.writer.add_scalar(tag, accumulator.value, step)
        if self._success_count:
            self.writer.add_scalar(
                "task/success_rate",
                self._success_total / self._success_count,
                step,
            )
            self.writer.add_scalar(
                "task/success_rate_roll100",
                sum(self._recent_successes) / len(self._recent_successes),
                step,
            )
        for tag, value in self._latest.items():
            self.writer.add_scalar(tag, value, step)
        self._means.clear()
        self._latest.clear()
        self.last_step = step
        return True

    def close(self, step: int) -> None:
        if self.writer is None:
            return
        self.maybe_flush(step, force=True)
        self.writer.flush()
        self.writer.close()
