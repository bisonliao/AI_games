from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from HeroEnv.curriculum import (
    DRAFT_MANIFEST,
    DepthTracker,
    RECORDING_MODE,
    episode_budget,
    freeze_draft,
    manifest_digest,
    validate_frozen_manifest,
)


def test_manual_depth_starts_at_minus_one_and_only_d_adds() -> None:
    tracker = DepthTracker()
    assert tracker.global_depth == -1

    first = tracker.manual_add(1, 1, 73)
    assert (first.global_depth, first.local_band, first.milestone_id) == (
        0,
        0,
        "GD000-L01-R01-B00",
    )
    second = tracker.manual_add(1, 2, 60)
    assert (second.global_depth, second.local_band) == (1, 0)

    assert tracker.undo_last() == second
    assert tracker.global_depth == 0
    assert tracker.undo_last() == first
    assert tracker.global_depth == -1

    tracker.manual_add(1, 2, 60)
    tracker.reset()  # Entering the next Level starts a new curriculum episode.
    assert tracker.global_depth == -1
    assert tracker.manual_add(2, 1, 73).global_depth == 0


def _checkpoint_entry(identifier: str, milestone: str, filename: str) -> dict:
    return {
        "checkpoint_id": identifier,
        "milestone_id": milestone,
        "checkpoint": filename,
        "checkpoint_sha256": hashlib.sha256(b"state").hexdigest(),
        "metadata": filename.replace(".chkpt", ".json"),
        "screenshot": filename.replace(".chkpt", ".png"),
        "accepted": True,
        "health": {"lives": 3, "power_ratio": 0.8},
        "validation": {
            "accepted": True,
            "reasons": [],
            "noop_survival_frames": 120,
            "responsive_actions": ["LEFT"],
            "training_smoke_seeds": 8,
        },
        "demo": {
            "session_id": "test",
            "episode": 1,
            "remaining_raw_frames": 400,
            "budget_decisions": 200,
        },
    }


def test_freeze_uses_max_depth_minus_checkpoint_depth(tmp_path: Path) -> None:
    milestones = [
        {
            "milestone_id": "GD000-L01-R01-B00",
            "kind": "depth",
            "global_depth": 0,
            "level": 1,
            "room": 1,
            "local_band": 0,
        },
        {
            "milestone_id": "GD001-L01-R02-B00",
            "kind": "depth",
            "global_depth": 1,
            "level": 1,
            "room": 2,
            "local_band": 0,
        },
        {
            "milestone_id": "GD000-L02-R01-B00",
            "kind": "depth",
            "global_depth": 0,
            "level": 2,
            "room": 1,
            "local_band": 0,
        },
        {
            "milestone_id": "GD002-L02-R03-B00",
            "kind": "depth",
            "global_depth": 2,
            "level": 2,
            "room": 3,
            "local_band": 0,
        },
    ]
    checkpoints = []
    for index, milestone in enumerate(milestones, start=1):
        filename = f"state-{index}.chkpt"
        (tmp_path / filename).write_bytes(b"state")
        (tmp_path / filename.replace(".chkpt", ".json")).write_text(
            json.dumps(
                {"checkpoint": {"sha256": hashlib.sha256(b"state").hexdigest()}}
            ),
            encoding="utf-8",
        )
        (tmp_path / filename.replace(".chkpt", ".png")).write_bytes(b"png")
        checkpoints.append(
            _checkpoint_entry(
                f"{milestone['milestone_id']}-V01",
                milestone["milestone_id"],
                filename,
            )
        )
    draft = {
        "format_version": 2,
        "status": "draft",
        "recording_mode": RECORDING_MODE,
        "rom_md5": "test",
        "milestones": milestones,
        "checkpoints": checkpoints,
        "rejections": [],
    }
    (tmp_path / DRAFT_MANIFEST).write_text(json.dumps(draft), encoding="utf-8")

    version_path = freeze_draft(tmp_path, max_level=2)
    assert version_path.name == "curriculum-v0001.json"
    frozen = json.loads((tmp_path / "curriculum.json").read_text(encoding="utf-8"))
    validate_frozen_manifest(frozen)
    assert frozen["manifest_sha256"] == manifest_digest(frozen)
    task_stages = {task["task_id"]: task["stage"] for task in frozen["tasks"]}
    assert task_stages["GD001-L01-R02-B00"] == 1
    assert task_stages["GD000-L01-R01-B00"] == 2
    assert task_stages["GD002-L02-R03-B00"] == 1
    assert task_stages["GD000-L02-R01-B00"] == 3
    assert frozen["max_depth_by_level"] == {"1": 1, "2": 2}


def test_freeze_rejects_empty_manual_recording(tmp_path: Path) -> None:
    draft = {
        "format_version": 2,
        "status": "draft",
        "recording_mode": RECORDING_MODE,
        "rom_md5": "test",
        "milestones": [],
        "checkpoints": [],
        "rejections": [],
    }
    (tmp_path / DRAFT_MANIFEST).write_text(json.dumps(draft), encoding="utf-8")
    with pytest.raises(ValueError, match="no manual-D checkpoints"):
        freeze_draft(tmp_path, max_level=1)


def test_demo_budget_defaults() -> None:
    assert episode_budget(400, action_repeat=4) == 200
    assert episode_budget(1, action_repeat=4) == 100
    assert episode_budget(100_000, action_repeat=4) == 5_000
