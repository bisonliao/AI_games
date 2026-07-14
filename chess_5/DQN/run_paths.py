from __future__ import annotations

import re
from pathlib import Path


RUN_NAME_PATTERN = re.compile(r"^[\w.-]+$", re.UNICODE)


def validate_run_name(run_name: str) -> str:
    value = run_name.strip()
    if not value:
        raise ValueError("Training name cannot be empty.")
    if value in (".", "..") or not RUN_NAME_PATTERN.fullmatch(value):
        raise ValueError(
            "Training name may only contain letters, numbers, underscores, "
            "hyphens, and dots, and cannot be '.' or '..'."
        )
    return value


def named_directory(base_directory: Path, run_name: str) -> Path:
    return base_directory.expanduser() / validate_run_name(run_name)


def checkpoint_filename(run_name: str, index: int) -> str:
    return f"{validate_run_name(run_name)}_{int(index):06d}.pt"
