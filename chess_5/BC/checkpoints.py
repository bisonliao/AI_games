"""按 BC pipeline 的 run/round 目录结构发现和解析 checkpoint。"""

from __future__ import annotations

import re
from pathlib import Path


def validate_name(value: str, label: str) -> str:
    """只接受单个路径分量，防止 run/stage 参数跳出 checkpoint 根目录。"""
    if not value or value in (".", "..") or Path(value).name != value:
        raise ValueError(f"invalid {label}: {value!r}")
    return value


def checkpoint_sort_key(path: Path) -> tuple[int, str, str]:
    """优先按 round 目录开头的数字排序，再用名称稳定打破平局。"""
    match = re.match(r"(\d+)", path.parent.name)
    return (int(match.group(1)) if match else -1, path.parent.name, path.name)


def checkpoints_for_run(root: Path, run_name: str, kind: str = "best") -> list[Path]:
    """返回一个 run 下按轮次排序的 best、latest 或全部 checkpoint。"""
    run_dir = Path(root).expanduser().resolve() / validate_name(run_name, "run name")
    if not run_dir.is_dir():
        raise FileNotFoundError(f"BC run directory does not exist: {run_dir}")
    pattern = "*.pt" if kind == "all" else f"{kind}.pt"
    checkpoints = sorted(run_dir.rglob(pattern), key=checkpoint_sort_key)
    if not checkpoints:
        raise FileNotFoundError(f"No {pattern} checkpoints found in {run_dir}")
    return checkpoints


def resolve_checkpoint(root: Path, *, direct: Path | None = None,
                       run_name: str | None = None, stage: str | None = None,
                       checkpoint_name: str = "best.pt") -> Path:
    """解析直接路径或 run/stage；未指定 stage 时选择数值最大的已有轮次。"""
    if direct is not None:
        candidate = direct.expanduser().resolve()
    else:
        if run_name is None:
            raise ValueError("either a direct checkpoint or run name is required")
        run_dir = Path(root).expanduser().resolve() / validate_name(run_name, "run name")
        checkpoint_name = validate_name(checkpoint_name, "checkpoint name")
        if stage is not None:
            candidate = run_dir / validate_name(stage, "stage") / checkpoint_name
        else:
            matches = sorted(run_dir.rglob(checkpoint_name), key=checkpoint_sort_key)
            if not matches:
                raise FileNotFoundError(f"No {checkpoint_name} checkpoints found in {run_dir}")
            candidate = matches[-1]
    if not candidate.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {candidate}")
    return candidate
