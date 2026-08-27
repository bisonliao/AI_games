#!/usr/bin/env python3
"""Play H.E.R.O. through ALE and capture curriculum checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import ale_py
import gymnasium as gym
import numpy as np

try:
    from .game_progress import GameProgress, decode_game_progress
    from .curriculum import (
        DRAFT_MANIFEST,
        CandidateRequest,
        CandidateState,
        DepthEvent,
        DepthTracker,
        RAM_PLAYER_X,
        RAM_PLAYER_Y,
        RAM_POWER,
        ValidationResult,
        checkpoint_id,
        episode_budget,
        freeze_draft,
        load_draft,
        now_iso,
        record_rejection,
        select_active_variants,
        upsert_milestone,
        validate_frozen_manifest,
        write_json_atomic as write_curriculum_json_atomic,
    )
except ImportError:  # Support running this file directly.
    from game_progress import GameProgress, decode_game_progress
    from curriculum import (  # type: ignore[no-redef]
        DRAFT_MANIFEST,
        CandidateRequest,
        CandidateState,
        DepthEvent,
        DepthTracker,
        RAM_PLAYER_X,
        RAM_PLAYER_Y,
        RAM_POWER,
        ValidationResult,
        checkpoint_id,
        episode_budget,
        freeze_draft,
        load_draft,
        now_iso,
        record_rejection,
        select_active_variants,
        upsert_milestone,
        validate_frozen_manifest,
        write_json_atomic as write_curriculum_json_atomic,
    )

try:
    import pygame
except ImportError as exc:  # pragma: no cover - only reached with a missing optional dependency
    raise SystemExit(
        "pygame is required. Install it with: python -m pip install pygame"
    ) from exc


ENV_ID = "ALE/Hero-v5"
DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
CURRICULUM_MANIFEST = "curriculum.json"
STATUS_BAR_HEIGHT = 30
TEACHER_MAX_LEVEL = 2


@dataclass
class SessionStats:
    episode: int = 1
    episode_steps: int = 0
    episode_return: float = 0.0
    checkpoints: int = 0
    last_action: str = "NOOP"

    def reset_episode(self) -> None:
        self.episode += 1
        self.episode_steps = 0
        self.episode_return = 0.0
        self.last_action = "NOOP"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play H.E.R.O. and record validated vertical-depth curriculum states."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help=f"checkpoint output directory (default: {DEFAULT_CHECKPOINT_DIR})",
    )
    parser.add_argument("--seed", type=int, default=0, help="ALE random seed")
    parser.add_argument(
        "--mode",
        type=int,
        default=0,
        choices=range(5),
        metavar="{0,1,2,3,4}",
        help="H.E.R.O. game mode (default: 0)",
    )
    parser.add_argument(
        "--frameskip",
        type=int,
        default=1,
        help="ALE frames per input step (default: 1 for precise manual control)",
    )
    parser.add_argument(
        "--sticky-action-probability",
        type=float,
        default=0.0,
        help="probability of repeating the previous action (default: 0.0)",
    )
    parser.add_argument(
        "--fps", type=int, default=60, help="input steps rendered per second (default: 60)"
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=3,
        help="initial integer window scale (default: 3)",
    )
    start_group = parser.add_mutually_exclusive_group()
    start_group.add_argument(
        "--load-checkpoint",
        type=Path,
        help="restore a specific .chkpt file before play",
    )
    start_group.add_argument(
        "--start-stage",
        type=int,
        help="restore the highest-life checkpoint from a curriculum stage",
    )
    parser.add_argument(
        "--freeze-curriculum",
        action="store_true",
        help="validate manual-D checkpoints and write a frozen v2 manifest",
    )
    parser.add_argument("--minimum-lives", type=int, default=2)
    parser.add_argument("--minimum-power-ratio", type=float, default=0.60)
    parser.add_argument("--noop-validation-frames", type=int, default=120)
    parser.add_argument("--action-validation-frames", type=int, default=60)
    parser.add_argument("--training-smoke-seeds", type=int, default=8)
    parser.add_argument("--training-action-repeat", type=int, default=4)
    parser.add_argument("--max-variants-per-milestone", type=int, default=3)
    parser.add_argument(
        "--coverage-interval",
        type=float,
        default=10.0,
        help="seconds between coverage reports; 0 disables periodic reports",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.frameskip < 1:
        parser.error("--frameskip must be at least 1")
    if args.fps < 1:
        parser.error("--fps must be at least 1")
    if args.scale < 1:
        parser.error("--scale must be at least 1")
    if not 0.0 <= args.sticky_action_probability <= 1.0:
        parser.error("--sticky-action-probability must be between 0 and 1")
    if args.start_stage is not None and args.start_stage < 1:
        parser.error("--start-stage must be at least 1")
    if args.coverage_interval < 0:
        parser.error("--coverage-interval cannot be negative")
    if args.frameskip != 1 and not args.freeze_curriculum:
        parser.error("depth curriculum recording requires --frameskip 1")
    if args.minimum_lives < 1:
        parser.error("--minimum-lives must be positive")
    if not 0 < args.minimum_power_ratio <= 1:
        parser.error("--minimum-power-ratio must be in (0, 1]")
    if args.noop_validation_frames < 1 or args.action_validation_frames < 1:
        parser.error("validation frame counts must be positive")
    if args.training_smoke_seeds < 1 or args.training_action_repeat < 1:
        parser.error("training smoke seeds/action repeat must be positive")
    if args.max_variants_per_milestone < 1:
        parser.error("--max-variants-per-milestone must be positive")

    return args


def make_env(args: argparse.Namespace) -> gym.Env:
    gym.register_envs(ale_py)
    return gym.make(
        ENV_ID,
        mode=args.mode,
        difficulty=0,
        obs_type="rgb",
        frameskip=args.frameskip,
        repeat_action_probability=args.sticky_action_probability,
        render_mode="rgb_array",
    )


def action_from_keyboard(
    keys: Sequence[bool], action_indices: dict[str, int]
) -> tuple[int, str]:
    up = bool(keys[pygame.K_UP])
    down = bool(keys[pygame.K_DOWN])
    left = bool(keys[pygame.K_LEFT])
    right = bool(keys[pygame.K_RIGHT])
    fire = bool(
        keys[pygame.K_SPACE]
        or keys[pygame.K_LCTRL]
        or keys[pygame.K_RCTRL]
    )

    vertical = "UP" if up and not down else "DOWN" if down and not up else ""
    horizontal = (
        "RIGHT" if right and not left else "LEFT" if left and not right else ""
    )
    action_name = vertical + horizontal
    if fire:
        action_name += "FIRE"
    if not action_name:
        action_name = "NOOP"

    return action_indices[action_name], action_name


def current_game_progress(env: gym.Env) -> GameProgress | None:
    return decode_game_progress(env.unwrapped.ale.getRAM())


def rom_metadata() -> dict[str, str | None]:
    rom_path = ale_py.roms.get_rom_path("hero")
    if rom_path is None:
        return {"path": None, "md5": None}

    digest = hashlib.md5(usedforsecurity=False)
    with rom_path.open("rb") as rom_file:
        for block in iter(lambda: rom_file.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(rom_path), "md5": digest.hexdigest()}


def checkpoint_for_stage(checkpoint_dir: Path, stage: int) -> Path:
    manifest_path = checkpoint_dir / CURRICULUM_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(
            f"curriculum manifest not found: {manifest_path}; run --freeze-curriculum"
        ) from exc

    validate_frozen_manifest(manifest)
    variants = [
        variant
        for task in manifest.get("tasks", [])
        if int(task.get("stage", -1)) == stage
        for variant in task.get("variants", [])
    ]
    if not variants:
        raise ValueError(f"curriculum stage {stage} does not exist in {manifest_path}")
    selected = max(
        variants,
        key=lambda item: (
            float(item.get("health", {}).get("power_ratio", 0.0)),
            int(item.get("health", {}).get("lives", 0)),
            item.get("checkpoint", ""),
        ),
    )
    return checkpoint_dir / selected["checkpoint"]


def restore_teacher_checkpoint(
    env: gym.Env, state_path: Path
) -> tuple[np.ndarray, dict[str, int], SessionStats]:
    if not state_path.is_file():
        raise ValueError(f"checkpoint does not exist: {state_path}")

    state = ale_py.ALEState(state_path.read_bytes())
    env.unwrapped.ale.restoreSystemState(state)
    observation = env.unwrapped.ale.getScreenRGB()
    info = {
        "lives": int(env.unwrapped.ale.lives()),
        "episode_frame_number": int(state.getEpisodeFrameNumber()),
        "frame_number": int(state.getFrameNumber()),
    }
    stats = SessionStats()

    metadata_path = state_path.with_suffix(".json")
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            demo = metadata["demo"]
            stats.episode = int(demo.get("episode", stats.episode))
            stats.episode_steps = int(demo.get("capture_frame", 0))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid checkpoint metadata: {metadata_path}") from exc

    print(f"Loaded checkpoint: {state_path}", flush=True)
    return observation, info, stats


def request_for_depth_event(event: DepthEvent) -> CandidateRequest:
    return CandidateRequest(
        milestone_id=event.milestone_id,
        kind="depth",
        global_depth=event.global_depth,
        level=event.level,
        room=event.room,
        local_band=event.local_band,
        manual=event.manual,
    )


def capture_candidate_state(
    env: gym.Env,
    observation: np.ndarray,
    request: CandidateRequest,
    *,
    session_id: str,
    episode: int,
) -> CandidateState:
    ram = env.unwrapped.ale.getRAM()
    state = env.unwrapped.ale.cloneSystemState()
    return CandidateState(
        milestone_id=request.milestone_id,
        kind=request.kind,
        global_depth=request.global_depth,
        level=request.level,
        room=request.room,
        local_band=request.local_band,
        session_id=session_id,
        episode=episode,
        capture_frame=int(state.getEpisodeFrameNumber()),
        lives=int(env.unwrapped.ale.lives()),
        power_raw=int(ram[RAM_POWER]),
        player_x=int(ram[RAM_PLAYER_X]),
        player_y=int(ram[RAM_PLAYER_Y]),
        state_bytes=state.serialize(),
        observation=np.array(observation, copy=True),
        manual=request.manual,
    )


def _restore_validation_state(
    env: gym.Env, state_bytes: bytes, *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    env.reset(seed=seed)
    env.unwrapped.ale.restoreSystemState(ale_py.ALEState(state_bytes))
    return env.unwrapped.ale.getScreenRGB(), env.unwrapped.ale.getRAM()


def _validation_rollout(
    env: gym.Env,
    state_bytes: bytes,
    *,
    seed: int,
    action: int,
    frames: int,
    expected_level: int,
    expected_room: int,
    expected_lives: int,
) -> tuple[bool, int, int, np.ndarray, str | None]:
    observation, _ = _restore_validation_state(env, state_bytes, seed=seed)
    for frame in range(1, frames + 1):
        observation, _, terminated, truncated, _ = env.step(action)
        progress = current_game_progress(env)
        lives = int(env.unwrapped.ale.lives())
        if lives < expected_lives:
            return False, frame, lives, observation, "life_lost"
        if terminated or truncated:
            return False, frame, lives, observation, "terminated"
        if (
            progress is None
            or progress.level != expected_level
            or progress.room != expected_room
        ):
            return False, frame, lives, observation, "progress_changed"
    ram = env.unwrapped.ale.getRAM()
    return (
        True,
        frames,
        int(env.unwrapped.ale.lives()),
        observation,
        None,
    )


def validate_candidate_state(
    env: gym.Env,
    candidate: CandidateState,
    args: argparse.Namespace,
) -> ValidationResult:
    action_meanings = env.unwrapped.get_action_meanings()
    indices = {name: index for index, name in enumerate(action_meanings)}
    reasons: list[str] = []

    noop_ok, noop_frames, _, _, noop_error = _validation_rollout(
        env,
        candidate.state_bytes,
        seed=args.seed + 700_000,
        action=indices["NOOP"],
        frames=args.noop_validation_frames,
        expected_level=candidate.level,
        expected_room=candidate.room,
        expected_lives=candidate.lives,
    )
    if not noop_ok:
        reasons.append(f"noop_{noop_error}_at_{noop_frames}")

    _, baseline_ram = _restore_validation_state(
        env, candidate.state_bytes, seed=args.seed + 710_000
    )
    baseline_ok, _, _, baseline_observation, _ = _validation_rollout(
        env,
        candidate.state_bytes,
        seed=args.seed + 710_000,
        action=indices["NOOP"],
        frames=args.action_validation_frames,
        expected_level=candidate.level,
        expected_room=candidate.room,
        expected_lives=candidate.lives,
    )
    baseline_ram = env.unwrapped.ale.getRAM().copy()
    responsive = []
    if baseline_ok:
        for name in ("LEFT", "RIGHT", "UP", "DOWN", "FIRE"):
            ok, _, _, branch_observation, _ = _validation_rollout(
                env,
                candidate.state_bytes,
                seed=args.seed + 710_000,
                action=indices[name],
                frames=args.action_validation_frames,
                expected_level=candidate.level,
                expected_room=candidate.room,
                expected_lives=candidate.lives,
            )
            if not ok:
                continue
            branch_ram = env.unwrapped.ale.getRAM()
            displacement = abs(int(branch_ram[RAM_PLAYER_X]) - int(baseline_ram[RAM_PLAYER_X]))
            displacement += abs(int(branch_ram[RAM_PLAYER_Y]) - int(baseline_ram[RAM_PLAYER_Y]))
            image_delta = float(
                np.mean(
                    np.abs(
                        branch_observation.astype(np.int16)
                        - baseline_observation.astype(np.int16)
                    )
                )
            )
            if displacement >= 4 or image_delta >= 1.0:
                responsive.append(name)
    if not responsive:
        reasons.append("actions_not_responsive")

    smoke_passed = 0
    for smoke_seed in range(args.training_smoke_seeds):
        ok, frame, _, _, error = _validation_rollout(
            env,
            candidate.state_bytes,
            seed=args.seed + 720_000 + smoke_seed,
            action=indices["NOOP"],
            frames=args.noop_validation_frames,
            expected_level=candidate.level,
            expected_room=candidate.room,
            expected_lives=candidate.lives,
        )
        if not ok:
            reasons.append(f"training_smoke_{error}_seed_{smoke_seed}_at_{frame}")
            break
        smoke_passed += 1

    return ValidationResult(
        accepted=not reasons,
        reasons=tuple(reasons),
        noop_survival_frames=noop_frames,
        responsive_actions=tuple(responsive),
        training_smoke_seeds=smoke_passed,
    )


def _next_variant(draft: dict[str, Any], milestone: str, checkpoint_dir: Path) -> int:
    used = {
        int(str(entry["checkpoint_id"]).rsplit("-V", 1)[1])
        for entry in draft["checkpoints"]
        if entry["milestone_id"] == milestone and "-V" in entry["checkpoint_id"]
    }
    variant = 1
    while variant in used or (checkpoint_dir / f"hero-{milestone}-V{variant:02d}.chkpt").exists():
        variant += 1
    return variant


def _save_candidate_image(observation: np.ndarray, path: Path) -> None:
    surface = pygame.surfarray.make_surface(np.transpose(observation, (1, 0, 2)))
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    pygame.image.save(surface, temporary)
    temporary.replace(path)


def _write_state_atomic(path: Path, state_bytes: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(state_bytes)
    temporary.replace(path)


def quarantine_candidate(
    checkpoint_dir: Path,
    candidate: CandidateState,
    validation: ValidationResult,
) -> None:
    quarantine = checkpoint_dir / "quarantine"
    quarantine.mkdir(parents=True, exist_ok=True)
    stem = (
        f"rejected-{candidate.milestone_id}-{candidate.session_id}"
        f"-ep{candidate.episode:03d}-frame{candidate.capture_frame:07d}"
    )
    state_path = quarantine / f"{stem}.chkpt"
    _write_state_atomic(state_path, candidate.state_bytes)
    write_curriculum_json_atomic(
        quarantine / f"{stem}.json",
        {
            "format_version": 2,
            "status": "rejected",
            "milestone_id": candidate.milestone_id,
            "session_id": candidate.session_id,
            "episode": candidate.episode,
            "capture_frame": candidate.capture_frame,
            "validation": validation.as_dict(),
        },
    )


def quarantine_superseded_variants(
    draft: dict[str, Any],
    checkpoint_dir: Path,
    milestone: str,
    *,
    limit: int,
) -> None:
    entries = [
        entry
        for entry in draft["checkpoints"]
        if entry["milestone_id"] == milestone and entry.get("accepted", True)
    ]
    active_ids = {
        entry["checkpoint_id"]
        for entry in select_active_variants(entries, limit=limit)
    }
    destination = checkpoint_dir / "quarantine" / "superseded"
    for entry in entries:
        if entry["checkpoint_id"] in active_ids:
            continue
        destination.mkdir(parents=True, exist_ok=True)
        for field in ("checkpoint", "metadata", "screenshot"):
            source = checkpoint_dir / entry[field]
            target = destination / source.name
            if source.exists():
                source.replace(target)
            entry[field] = str(target.relative_to(checkpoint_dir))
        entry["accepted"] = False
        entry["status"] = "superseded"
        entry["superseded_at"] = now_iso()


def commit_level_candidates(
    *,
    checkpoint_dir: Path,
    candidates: list[CandidateState],
    observed_requests: list[CandidateRequest],
    rescued_level: int,
    rescue_frame: int,
    full_power: int,
    args: argparse.Namespace,
    validation_env: gym.Env,
    rom: dict[str, str | None],
) -> tuple[int, int]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    draft = load_draft(
        checkpoint_dir,
        rom_md5=rom.get("md5"),
    )
    capture_config = {
        "recording_trigger": "D_key_only",
        "minimum_lives": args.minimum_lives,
        "minimum_power_ratio": args.minimum_power_ratio,
        "noop_validation_frames": args.noop_validation_frames,
        "action_validation_frames": args.action_validation_frames,
        "training_smoke_seeds": args.training_smoke_seeds,
        "training_action_repeat": args.training_action_repeat,
    }
    previous_capture_config = draft.get("capture_config")
    if previous_capture_config not in (None, capture_config):
        history = draft.setdefault("capture_config_history", [])
        if previous_capture_config not in history:
            history.append(previous_capture_config)
    draft["capture_config"] = capture_config
    for request in observed_requests:
        if request.level == rescued_level:
            upsert_milestone(draft, request)

    accepted = 0
    rejected = 0
    for candidate in candidates:
        if candidate.level != rescued_level:
            continue
        power_ratio = candidate.power_raw / max(1, full_power)
        validation = validate_candidate_state(validation_env, candidate, args)
        reasons = list(validation.reasons)
        if candidate.lives < args.minimum_lives:
            reasons.append(f"lives_below_{args.minimum_lives}")
        if power_ratio < args.minimum_power_ratio:
            reasons.append(f"power_ratio_below_{args.minimum_power_ratio:.2f}")
        if reasons:
            validation = ValidationResult(
                accepted=False,
                reasons=tuple(dict.fromkeys(reasons)),
                noop_survival_frames=validation.noop_survival_frames,
                responsive_actions=validation.responsive_actions,
                training_smoke_seeds=validation.training_smoke_seeds,
            )
            record_rejection(draft, candidate, validation)
            quarantine_candidate(checkpoint_dir, candidate, validation)
            rejected += 1
            continue

        remaining = max(1, rescue_frame - candidate.capture_frame)
        budget = episode_budget(
            remaining, action_repeat=args.training_action_repeat
        )
        variant = _next_variant(draft, candidate.milestone_id, checkpoint_dir)
        identifier = checkpoint_id(candidate.milestone_id, variant)
        stem = f"hero-{identifier}"
        state_path = checkpoint_dir / f"{stem}.chkpt"
        image_path = checkpoint_dir / f"{stem}.png"
        metadata_path = checkpoint_dir / f"{stem}.json"
        _write_state_atomic(state_path, candidate.state_bytes)
        _save_candidate_image(candidate.observation, image_path)
        checkpoint_sha256 = hashlib.sha256(candidate.state_bytes).hexdigest()
        health = {
            "lives": candidate.lives,
            "power_raw": candidate.power_raw,
            "full_power_raw": full_power,
            "power_ratio": power_ratio,
            "player_x": candidate.player_x,
            "player_y": candidate.player_y,
        }
        demo = {
            "session_id": candidate.session_id,
            "episode": candidate.episode,
            "capture_frame": candidate.capture_frame,
            "rescue_frame": rescue_frame,
            "remaining_raw_frames": remaining,
            "budget_decisions": budget,
            "budget_multiplier": 2.0,
            "training_action_repeat": args.training_action_repeat,
        }
        metadata = {
            "format_version": 2,
            "created_at": now_iso(),
            "checkpoint_id": identifier,
            "milestone_id": candidate.milestone_id,
            "checkpoint": {
                "file": state_path.name,
                "sha256": checkpoint_sha256,
            },
            "rom": rom,
            "curriculum": {
                "global_depth": candidate.global_depth,
                "level": candidate.level,
                "room": candidate.room,
                "local_band": candidate.local_band,
                "kind": candidate.kind,
                "manual": candidate.manual,
                "target": {"type": "rescue_miner", "level": candidate.level},
            },
            "health": health,
            "validation": validation.as_dict(),
            "demo": demo,
            "screenshot": image_path.name,
        }
        write_curriculum_json_atomic(metadata_path, metadata)
        draft["checkpoints"].append(
            {
                "checkpoint_id": identifier,
                "milestone_id": candidate.milestone_id,
                "checkpoint": state_path.name,
                "checkpoint_sha256": checkpoint_sha256,
                "metadata": metadata_path.name,
                "screenshot": image_path.name,
                "accepted": True,
                "health": health,
                "validation": validation.as_dict(),
                "demo": demo,
            }
        )
        accepted += 1

    for milestone in {candidate.milestone_id for candidate in candidates}:
        quarantine_superseded_variants(
            draft,
            checkpoint_dir,
            milestone,
            limit=args.max_variants_per_milestone,
        )
    draft["updated_at"] = now_iso()
    write_curriculum_json_atomic(checkpoint_dir / DRAFT_MANIFEST, draft)
    return accepted, rejected


def progress_status(
    progress: GameProgress | None,
    *,
    global_depth: int,
    buffered: int,
) -> str:
    if progress is None:
        return f"LEVEL -- | ROOM --/-- | DEPTH {global_depth:03d}"
    return (
        f"LEVEL {progress.level:02d} | ROOM {progress.room:02d}/{progress.total_rooms:02d}"
        f" | DEPTH {global_depth:03d} | BUFFERED {buffered:02d}"
    )


def draw_frame(
    screen: pygame.Surface,
    observation: np.ndarray,
    progress: GameProgress | None,
    status_font: pygame.font.Font,
    *,
    global_depth: int,
    buffered: int,
) -> None:
    screen_width, screen_height = screen.get_size()
    frame_height, frame_width = observation.shape[:2]
    game_area_height = max(1, screen_height - STATUS_BAR_HEIGHT)
    scale = min(screen_width / frame_width, game_area_height / frame_height)
    draw_width = max(1, round(frame_width * scale))
    draw_height = max(1, round(frame_height * scale))
    offset_x = (screen_width - draw_width) // 2
    offset_y = STATUS_BAR_HEIGHT + (game_area_height - draw_height) // 2

    frame_surface = pygame.surfarray.make_surface(np.transpose(observation, (1, 0, 2)))
    scaled_frame = pygame.transform.scale(frame_surface, (draw_width, draw_height))
    screen.fill((8, 8, 8))
    pygame.draw.rect(screen, (28, 30, 33), (0, 0, screen_width, STATUS_BAR_HEIGHT))
    status_surface = status_font.render(
        progress_status(
            progress,
            global_depth=global_depth,
            buffered=buffered,
        ),
        True,
        (236, 226, 87),
    )
    screen.blit(status_surface, (8, (STATUS_BAR_HEIGHT - status_surface.get_height()) // 2))
    screen.blit(scaled_frame, (offset_x, offset_y))
    pygame.display.flip()


def update_caption(
    stats: SessionStats,
    info: dict,
    progress: GameProgress | None,
    paused: bool,
    *,
    global_depth: int,
    buffered: int,
) -> None:
    lives = int(info.get("lives", 0))
    status = "paused" if paused else stats.last_action
    caption = (
        f"H.E.R.O. Teacher | score {stats.episode_return:g} | lives {lives}"
        f" | {status} | checkpoints {stats.checkpoints}"
    )
    if progress is not None:
        caption += (
            f" | level {progress.level} room {progress.room}/{progress.total_rooms}"
            f" | depth {global_depth} | buffered {buffered}"
        )
    pygame.display.set_caption(caption)


def run(args: argparse.Namespace) -> int:
    if args.freeze_curriculum:
        path = freeze_draft(
            args.checkpoint_dir,
            max_level=TEACHER_MAX_LEVEL,
            variants_per_milestone=args.max_variants_per_milestone,
        )
        print(f"Frozen curriculum: {path}", flush=True)
        return 0

    # Fail before opening the game if this directory contains data generated
    # by any older automatic/entry recorder.
    if (args.checkpoint_dir / DRAFT_MANIFEST).exists():
        load_draft(
            args.checkpoint_dir,
            rom_md5=rom_metadata().get("md5"),
        )

    env = make_env(args)
    observation, info = env.reset(seed=args.seed)
    start_path = args.load_checkpoint
    if args.start_stage is not None:
        start_path = checkpoint_for_stage(args.checkpoint_dir, args.start_stage)
    loaded_stats: SessionStats | None = None
    if start_path is not None:
        observation, info, loaded_stats = restore_teacher_checkpoint(env, start_path)

    action_meanings = env.unwrapped.get_action_meanings()
    action_indices = {name: index for index, name in enumerate(action_meanings)}
    required_actions = {
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE",
    }
    missing_actions = required_actions.difference(action_indices)
    if missing_actions:
        env.close()
        raise RuntimeError(f"H.E.R.O. action set is missing: {sorted(missing_actions)}")

    pygame.display.init()
    pygame.font.init()
    frame_height, frame_width = observation.shape[:2]
    screen = pygame.display.set_mode(
        (frame_width * args.scale, frame_height * args.scale + STATUS_BAR_HEIGHT),
        pygame.RESIZABLE,
    )
    status_font = pygame.font.Font(None, 18)
    clock = pygame.time.Clock()
    stats = loaded_stats if loaded_stats is not None else SessionStats()
    stats.checkpoints = len(list(args.checkpoint_dir.glob("*.chkpt")))
    session_id = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    tracker = DepthTracker()
    progress = current_game_progress(env)
    observed: dict[int, dict[str, CandidateRequest]] = {}
    candidates: dict[int, list[CandidateState]] = {}
    full_power: dict[int, int] = {}
    validation_env = gym.make(
        ENV_ID,
        mode=args.mode,
        difficulty=0,
        obs_type="rgb",
        frameskip=1,
        repeat_action_probability=0.25,
    )
    rom = rom_metadata()
    last_coverage_report = time.monotonic()
    print(
        "Manual recorder: depth starts at -1; D increments depth and captures "
        "the exact checkpoint; U undoes the last D; R/F2 restarts. "
        "Captured points commit only after miner rescue.",
        flush=True,
    )
    paused = False
    running = True
    rendered_steps = 0

    def clear_ephemeral(*, reset_depth: bool) -> None:
        observed.clear()
        candidates.clear()
        full_power.clear()
        if reset_depth:
            tracker.reset()

    try:
        while running:
            reset_requested = False
            manual_depth_requested = False
            undo_depth_requested = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_d:
                        manual_depth_requested = True
                    elif event.key == pygame.K_u:
                        undo_depth_requested = True
                    elif event.key in (pygame.K_r, pygame.K_F2):
                        reset_requested = True
                    elif event.key == pygame.K_p:
                        paused = not paused

            if not running:
                break

            if reset_requested:
                if start_path is not None:
                    checkpoint_count = stats.checkpoints
                    observation, info, stats = restore_teacher_checkpoint(env, start_path)
                    stats.checkpoints = checkpoint_count
                else:
                    observation, info = env.reset()
                    stats.reset_episode()
                progress = current_game_progress(env)
                clear_ephemeral(reset_depth=True)
                print("Recording session depth reset to -1", flush=True)
                last_coverage_report = time.monotonic()

            if undo_depth_requested:
                undone = tracker.undo_last()
                if undone is not None:
                    observed.get(undone.level, {}).pop(undone.milestone_id, None)
                    candidates[undone.level] = [
                        item
                        for item in candidates.get(undone.level, [])
                        if item.milestone_id != undone.milestone_id
                    ]
                    print(
                        f"Undid {undone.milestone_id}; depth={tracker.global_depth}",
                        flush=True,
                    )

            if (
                manual_depth_requested
                and progress is not None
                and progress.level <= TEACHER_MAX_LEVEL
            ):
                ram = env.unwrapped.ale.getRAM()
                depth_event = tracker.manual_add(
                    progress.level, progress.room, int(ram[RAM_PLAYER_Y])
                )
                request = request_for_depth_event(depth_event)
                candidate = capture_candidate_state(
                    env,
                    observation,
                    request,
                    session_id=session_id,
                    episode=stats.episode,
                )
                observed.setdefault(progress.level, {})[
                    request.milestone_id
                ] = request
                candidates.setdefault(progress.level, []).append(candidate)
                print(
                    f"Captured {request.milestone_id} at frame "
                    f"{candidate.capture_frame}; awaiting miner rescue",
                    flush=True,
                )

            if not paused and not reset_requested:
                previous_progress = progress
                previous_lives = int(env.unwrapped.ale.lives())
                action, action_name = action_from_keyboard(
                    pygame.key.get_pressed(), action_indices
                )
                observation, reward, terminated, truncated, info = env.step(action)
                stats.episode_steps += 1
                stats.episode_return += float(reward)
                stats.last_action = action_name
                rendered_steps += 1
                progress = current_game_progress(env)

                ram = env.unwrapped.ale.getRAM()
                current_lives = int(env.unwrapped.ale.lives())
                if progress is not None and progress.level <= TEACHER_MAX_LEVEL:
                    full_power[progress.level] = max(
                        full_power.get(progress.level, 0), int(ram[RAM_POWER])
                    )

                life_lost = current_lives < previous_lives
                if life_lost:
                    level = previous_progress.level if previous_progress is not None else None
                    if level is not None:
                        observed.pop(level, None)
                        candidates.pop(level, None)
                        full_power.pop(level, None)
                    tracker.reset()
                    stats.reset_episode()
                    print(
                        f"Life lost: discarded the previous curriculum episode; "
                        f"Level {level} retry depth reset to -1",
                        flush=True,
                    )

                rescued_level = None
                if (
                    previous_progress is not None
                    and progress is not None
                    and progress.level > previous_progress.level
                ):
                    rescued_level = previous_progress.level

                if rescued_level is not None:
                    rescue_frame = int(env.unwrapped.ale.getEpisodeFrameNumber())
                    accepted, rejected = commit_level_candidates(
                        checkpoint_dir=args.checkpoint_dir,
                        candidates=candidates.get(rescued_level, []),
                        observed_requests=list(observed.get(rescued_level, {}).values()),
                        rescued_level=rescued_level,
                        rescue_frame=rescue_frame,
                        full_power=full_power.get(rescued_level, 1),
                        args=args,
                        validation_env=validation_env,
                        rom=rom,
                    )
                    stats.checkpoints += accepted
                    print(
                        f"Miner rescued in Level {rescued_level}: committed {accepted}, "
                        f"quarantined {rejected}; "
                        f"draft={args.checkpoint_dir / DRAFT_MANIFEST}",
                        flush=True,
                    )
                    observed.pop(rescued_level, None)
                    candidates.pop(rescued_level, None)
                    full_power.pop(rescued_level, None)

                    # Every Level is an independent curriculum episode.  The
                    # newly entered Level must begin at -1, so its first D
                    # capture is depth 0 rather than continuing the previous
                    # Level's counter.
                    if rescued_level < TEACHER_MAX_LEVEL:
                        tracker.reset()
                        print(
                            f"Level {rescued_level + 1} curriculum episode: "
                            "depth reset to -1",
                            flush=True,
                        )

                if rescued_level == TEACHER_MAX_LEVEL:
                    checkpoint_count = stats.checkpoints
                    observation, info = env.reset()
                    stats.reset_episode()
                    stats.checkpoints = checkpoint_count
                    progress = current_game_progress(env)
                    clear_ephemeral(reset_depth=True)
                    print(
                        f"Level {TEACHER_MAX_LEVEL} complete: Level 3 is outside "
                        "the curriculum; started a new recording game",
                        flush=True,
                    )
                    last_coverage_report = time.monotonic()
                elif terminated or truncated:
                    if start_path is not None:
                        checkpoint_count = stats.checkpoints
                        observation, info, stats = restore_teacher_checkpoint(
                            env, start_path
                        )
                        stats.checkpoints = checkpoint_count
                    else:
                        observation, info = env.reset()
                        stats.reset_episode()
                    progress = current_game_progress(env)
                    clear_ephemeral(reset_depth=True)
                    print("Game reset: uncommitted candidates discarded", flush=True)
                    last_coverage_report = time.monotonic()

            now = time.monotonic()
            if (
                args.coverage_interval > 0
                and now - last_coverage_report >= args.coverage_interval
            ):
                print(
                    progress_status(
                        progress,
                        global_depth=tracker.global_depth,
                        buffered=sum(len(items) for items in candidates.values()),
                    ),
                    flush=True,
                )
                last_coverage_report = now

            draw_frame(
                screen,
                observation,
                progress,
                status_font,
                global_depth=tracker.global_depth,
                buffered=sum(len(items) for items in candidates.values()),
            )
            update_caption(
                stats,
                info,
                progress,
                paused,
                global_depth=tracker.global_depth,
                buffered=sum(len(items) for items in candidates.values()),
            )
            clock.tick(args.fps)

            if args.max_frames and rendered_steps >= args.max_frames:
                running = False
    finally:
        env.close()
        validation_env.close()
        pygame.quit()

    return 0


def main() -> int:
    try:
        return run(parse_args())
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"teacher.py: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
