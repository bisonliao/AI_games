"""Curriculum depth tracking, draft manifests, and checkpoint metadata.

The teacher records candidate states while a human completes a level.  Only
states from a successful demonstration are committed to the draft manifest;
the draft is frozen into an immutable training manifest explicitly.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from .game_progress import LEVEL_ROOM_COUNTS
except ImportError:  # Support teacher.py when executed as a script.
    from game_progress import LEVEL_ROOM_COUNTS


RAM_PLAYER_X = 27
RAM_PLAYER_Y = 31
RAM_POWER = 43

DRAFT_MANIFEST = "curriculum.draft.json"
FROZEN_MANIFEST = "curriculum.json"
MANIFEST_FORMAT_VERSION = 2
RECORDING_MODE = "manual_depth_key_v1"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def manifest_digest(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("manifest_sha256", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class DepthEvent:
    global_depth: int
    level: int
    room: int
    local_band: int
    player_y: int
    manual: bool = True

    @property
    def milestone_id(self) -> str:
        return milestone_id(
            self.global_depth, self.level, self.room, self.local_band, "depth"
        )


class DepthTracker:
    """Manual depth counter: only the D key can create an event."""

    def __init__(self) -> None:
        self.global_depth = -1
        self.room_bands: dict[tuple[int, int], int] = {}
        self.events: list[DepthEvent] = []

    @property
    def local_band(self) -> int:
        return self.events[-1].local_band if self.events else -1

    def reset(self) -> None:
        self.global_depth = -1
        self.room_bands.clear()
        self.events.clear()

    def manual_add(self, level: int, room: int, player_y: int) -> DepthEvent:
        key = (level, room)
        local_band = self.room_bands.get(key, -1) + 1
        self.room_bands[key] = local_band
        self.global_depth += 1
        event = DepthEvent(
            global_depth=self.global_depth,
            level=level,
            room=room,
            local_band=local_band,
            player_y=player_y,
        )
        self.events.append(event)
        return event

    def undo_last(self) -> DepthEvent | None:
        if not self.events:
            return None
        event = self.events.pop()
        key = (event.level, event.room)
        if event.local_band == 0:
            self.room_bands.pop(key, None)
        else:
            self.room_bands[key] = event.local_band - 1
        self.global_depth -= 1
        return event


def milestone_id(
    global_depth: int,
    level: int,
    room: int,
    local_band: int,
    kind: str,
) -> str:
    suffix = "E" if kind == "entry" else f"B{local_band:02d}"
    return f"GD{global_depth:03d}-L{level:02d}-R{room:02d}-{suffix}"


def checkpoint_id(milestone: str, variant: int) -> str:
    return f"{milestone}-V{variant:02d}"


@dataclass
class CandidateState:
    milestone_id: str
    kind: str
    global_depth: int
    level: int
    room: int
    local_band: int
    session_id: str
    episode: int
    capture_frame: int
    lives: int
    power_raw: int
    player_x: int
    player_y: int
    state_bytes: bytes
    observation: np.ndarray
    manual: bool = False


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    reasons: tuple[str, ...]
    noop_survival_frames: int
    responsive_actions: tuple[str, ...]
    training_smoke_seeds: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateRequest:
    milestone_id: str
    kind: str
    global_depth: int
    level: int
    room: int
    local_band: int
    manual: bool = False


def episode_budget(
    demo_remaining_raw_frames: int,
    *,
    action_repeat: int = 4,
    multiplier: float = 2.0,
    minimum: int = 100,
    maximum: int = 5_000,
) -> int:
    if demo_remaining_raw_frames < 1:
        raise ValueError("demo_remaining_raw_frames must be positive")
    value = math.ceil(multiplier * demo_remaining_raw_frames / action_repeat)
    return min(maximum, max(minimum, value))


def empty_draft(*, rom_md5: str | None) -> dict[str, Any]:
    return {
        "format_version": MANIFEST_FORMAT_VERSION,
        "status": "draft",
        "recording_mode": RECORDING_MODE,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "rom_md5": rom_md5,
        "milestones": [],
        "checkpoints": [],
        "rejections": [],
    }


def load_draft(checkpoint_dir: Path, *, rom_md5: str | None) -> dict[str, Any]:
    path = checkpoint_dir / DRAFT_MANIFEST
    if not path.exists():
        return empty_draft(rom_md5=rom_md5)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("format_version", 0)) != MANIFEST_FORMAT_VERSION:
        raise ValueError(f"unsupported draft manifest: {path}")
    if payload.get("status") != "draft":
        raise ValueError(f"manifest is not a draft: {path}")
    if payload.get("recording_mode") != RECORDING_MODE:
        raise ValueError(
            "draft was not created by the manual-D recorder; archive or clear "
            "the checkpoint directory before recording"
        )
    if payload.get("rom_md5") != rom_md5:
        raise ValueError("draft curriculum ROM differs from the active ROM")
    return payload


def upsert_milestone(draft: dict[str, Any], request: CandidateRequest) -> None:
    if any(item["milestone_id"] == request.milestone_id for item in draft["milestones"]):
        return
    draft["milestones"].append(
        {
            "milestone_id": request.milestone_id,
            "kind": request.kind,
            "global_depth": request.global_depth,
            "level": request.level,
            "room": request.room,
            "local_band": request.local_band,
            "manual": request.manual,
        }
    )


def accepted_for_milestone(
    draft: dict[str, Any], milestone: str
) -> list[dict[str, Any]]:
    return [
        item
        for item in draft["checkpoints"]
        if item["milestone_id"] == milestone and item.get("accepted", True)
    ]


def record_rejection(
    draft: dict[str, Any], candidate: CandidateState, validation: ValidationResult
) -> None:
    draft["rejections"].append(
        {
            "milestone_id": candidate.milestone_id,
            "session_id": candidate.session_id,
            "episode": candidate.episode,
            "capture_frame": candidate.capture_frame,
            "reasons": list(validation.reasons),
            "recorded_at": now_iso(),
        }
    )
    draft["updated_at"] = now_iso()


def checkpoint_quality(entry: dict[str, Any]) -> tuple[float, int, int, int]:
    health = entry["health"]
    validation = entry["validation"]
    return (
        float(health["power_ratio"]),
        int(health["lives"]),
        int(validation["noop_survival_frames"]),
        -int(entry["demo"]["remaining_raw_frames"]),
    )


def select_active_variants(
    entries: Iterable[dict[str, Any]], *, limit: int = 3
) -> list[dict[str, Any]]:
    by_episode: dict[tuple[str, int], dict[str, Any]] = {}
    for entry in entries:
        episode = (
            str(entry["demo"]["session_id"]),
            int(entry["demo"]["episode"]),
        )
        existing = by_episode.get(episode)
        if existing is None or checkpoint_quality(entry) > checkpoint_quality(existing):
            by_episode[episode] = entry
    return sorted(by_episode.values(), key=checkpoint_quality, reverse=True)[:limit]


def validate_frozen_manifest(payload: dict[str, Any]) -> None:
    if int(payload.get("format_version", 0)) != MANIFEST_FORMAT_VERSION:
        raise ValueError("curriculum manifest must use format version 2")
    if payload.get("status") != "frozen":
        raise ValueError("curriculum manifest is not frozen")
    if payload.get("recording_mode") != RECORDING_MODE:
        raise ValueError("curriculum was not recorded with the manual-D workflow")
    expected = payload.get("manifest_sha256")
    if not expected or expected != manifest_digest(payload):
        raise ValueError("curriculum manifest hash mismatch")
    if not payload.get("stages") or not payload.get("tasks"):
        raise ValueError("curriculum manifest has no stages/tasks")
    max_level = int(payload.get("max_level", 0))
    if not 1 <= max_level <= len(LEVEL_ROOM_COUNTS):
        raise ValueError("invalid max_level in curriculum manifest")
    max_depth_by_level = {
        int(level): int(depth)
        for level, depth in payload.get("max_depth_by_level", {}).items()
    }
    if set(max_depth_by_level) != {
        int(task["level"]) for task in payload["tasks"]
    }:
        raise ValueError("max_depth_by_level does not cover every task Level")
    task_ids: set[str] = set()
    checkpoint_ids: set[str] = set()
    task_stage: dict[str, int] = {}
    for task in payload["tasks"]:
        identifier = str(task["task_id"])
        if identifier in task_ids:
            raise ValueError(f"duplicate curriculum task_id: {identifier}")
        task_ids.add(identifier)
        stage = int(task["stage"])
        level = int(task["level"])
        room = int(task["room"])
        if not 1 <= level <= len(LEVEL_ROOM_COUNTS):
            raise ValueError(f"invalid Level for task {identifier}")
        total_rooms = LEVEL_ROOM_COUNTS[level - 1]
        if not 1 <= room <= total_rooms:
            raise ValueError(f"invalid room for task {identifier}")
        depth = int(task["depth"])
        if depth < 0:
            raise ValueError(f"negative checkpoint depth for task {identifier}")
        expected_stage = max_depth_by_level[level] - depth + 1
        if stage != expected_stage or stage < 1:
            raise ValueError(f"invalid reverse-depth Stage for task {identifier}")
        target = task.get("target", {})
        if target.get("type") != "rescue_miner" or int(target.get("level", 0)) != int(
            task["level"]
        ):
            raise ValueError(f"invalid miner-rescue target for task {identifier}")
        if not task.get("variants"):
            raise ValueError(f"task has no checkpoint variants: {identifier}")
        task_stage[identifier] = stage
        for variant in task["variants"]:
            checkpoint = str(variant["checkpoint_id"])
            if checkpoint in checkpoint_ids:
                raise ValueError(f"duplicate checkpoint_id: {checkpoint}")
            checkpoint_ids.add(checkpoint)
            if int(variant.get("demo", {}).get("budget_decisions", 0)) < 1:
                raise ValueError(f"invalid episode budget for {checkpoint}")

    listed_tasks: set[str] = set()
    for stage_data in payload["stages"]:
        stage = int(stage_data["stage"])
        for identifier in stage_data.get("task_ids", []):
            identifier = str(identifier)
            if identifier not in task_stage or task_stage[identifier] != stage:
                raise ValueError(f"stage references an invalid task: {identifier}")
            if identifier in listed_tasks:
                raise ValueError(f"task appears in multiple stage lists: {identifier}")
            listed_tasks.add(identifier)
    if listed_tasks != task_ids:
        raise ValueError("stage task lists do not cover every curriculum task")
    if int(payload.get("checkpoint_count", -1)) != len(checkpoint_ids):
        raise ValueError("checkpoint_count does not match curriculum variants")
    if int(payload.get("task_count", -1)) != len(task_ids):
        raise ValueError("task_count does not match curriculum tasks")


def freeze_draft(
    checkpoint_dir: Path,
    *,
    max_level: int,
    variants_per_milestone: int = 3,
) -> Path:
    draft_path = checkpoint_dir / DRAFT_MANIFEST
    if not draft_path.exists():
        raise ValueError(f"draft curriculum does not exist: {draft_path}")
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    if int(draft.get("format_version", 0)) != MANIFEST_FORMAT_VERSION:
        raise ValueError("draft curriculum must use format version 2")
    if draft.get("recording_mode") != RECORDING_MODE:
        raise ValueError("only manual-D curriculum drafts can be frozen")

    milestones = {item["milestone_id"]: item for item in draft["milestones"]}
    selected_by_milestone: dict[str, list[dict[str, Any]]] = {}
    missing_milestones = []
    for identifier in sorted(milestones):
        selected = select_active_variants(
            accepted_for_milestone(draft, identifier), limit=variants_per_milestone
        )
        if not selected:
            missing_milestones.append(identifier)
        else:
            for entry in selected:
                for field in ("checkpoint", "metadata", "screenshot"):
                    artifact = checkpoint_dir / entry[field]
                    if not artifact.is_file():
                        raise ValueError(
                            f"curriculum artifact is missing for {entry['checkpoint_id']}: "
                            f"{artifact}"
                        )
                metadata_path = checkpoint_dir / entry["metadata"]
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                expected_sha = entry.get("checkpoint_sha256") or metadata.get(
                    "checkpoint", {}
                ).get("sha256")
                actual_sha = hashlib.sha256(
                    (checkpoint_dir / entry["checkpoint"]).read_bytes()
                ).hexdigest()
                if not expected_sha or expected_sha != actual_sha:
                    raise ValueError(
                        f"checkpoint hash mismatch for {entry['checkpoint_id']}"
                    )
            selected_by_milestone[identifier] = selected

    if not milestones:
        raise ValueError("curriculum draft contains no manual-D checkpoints")
    if missing_milestones:
        raise ValueError(
            "curriculum checkpoints were rejected or are missing: "
            + ",".join(missing_milestones)
        )

    max_depth_by_level: dict[int, int] = {}
    for milestone in milestones.values():
        level = int(milestone["level"])
        max_depth_by_level[level] = max(
            max_depth_by_level.get(level, -1), int(milestone["global_depth"])
        )
    tasks = []
    for identifier, milestone in sorted(
        milestones.items(), key=lambda pair: (int(pair[1]["global_depth"]), pair[0]), reverse=True
    ):
        selected = selected_by_milestone[identifier]
        level = int(milestone["level"])
        room = int(milestone["room"])
        total_rooms = LEVEL_ROOM_COUNTS[level - 1]
        rooms_after = total_rooms - int(milestone["room"])
        depth = int(milestone["global_depth"])
        stage = max_depth_by_level[level] - depth + 1
        variants = []
        for entry in selected:
            variants.append(
                {
                    "checkpoint_id": entry["checkpoint_id"],
                    "checkpoint": entry["checkpoint"],
                    "checkpoint_sha256": entry.get("checkpoint_sha256"),
                    "metadata": entry["metadata"],
                    "screenshot": entry["screenshot"],
                    "health": entry["health"],
                    "validation": entry["validation"],
                    "demo": entry["demo"],
                }
            )
        tasks.append(
            {
                "task_id": identifier,
                "stage": stage,
                "depth": depth,
                "global_depth": int(milestone["global_depth"]),
                "level": int(milestone["level"]),
                "room": int(milestone["room"]),
                "rooms_after": rooms_after,
                "local_band": int(milestone["local_band"]),
                "kind": milestone["kind"],
                "target": {"type": "rescue_miner", "level": int(milestone["level"])},
                "variants": variants,
            }
        )

    stages = []
    for stage in sorted({int(task["stage"]) for task in tasks}):
        stage_tasks = [task for task in tasks if int(task["stage"]) == stage]
        stages.append(
            {
                "stage": stage,
                "task_ids": [task["task_id"] for task in stage_tasks],
            }
        )

    existing_versions = []
    for path in checkpoint_dir.glob("curriculum-v*.json"):
        try:
            existing_versions.append(int(path.stem.split("-v", 1)[1]))
        except (IndexError, ValueError):
            continue
    version = max(existing_versions, default=0) + 1
    frozen = {
        "format_version": MANIFEST_FORMAT_VERSION,
        "status": "frozen",
        "recording_mode": RECORDING_MODE,
        "version": version,
        "frozen_at": now_iso(),
        "rom_md5": draft.get("rom_md5"),
        "capture_config": draft.get("capture_config"),
        "capture_config_history": draft.get("capture_config_history", []),
        "stage_semantics": (
            "stage = max_depth_of_checkpoint_level - checkpoint_depth + 1"
        ),
        "max_level": max_level,
        "max_depth_by_level": {
            str(level): depth for level, depth in sorted(max_depth_by_level.items())
        },
        "checkpoint_count": sum(len(task["variants"]) for task in tasks),
        "task_count": len(tasks),
        "stages": stages,
        "tasks": tasks,
    }
    frozen["manifest_sha256"] = manifest_digest(frozen)
    validate_frozen_manifest(frozen)
    version_path = checkpoint_dir / f"curriculum-v{version:04d}.json"
    write_json_atomic(version_path, frozen)
    write_json_atomic(checkpoint_dir / FROZEN_MANIFEST, frozen)
    return version_path
