"""Gymnasium environment for H.E.R.O. levels 1 and 2."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ale_py
import gymnasium as gym
import numpy as np
from PIL import Image

from .game_progress import (
    MAX_LEVEL,
    ROM_MD5,
    GameProgress,
    decode_game_progress,
    decode_level,
)
from .curriculum import MANIFEST_FORMAT_VERSION, validate_frozen_manifest


ALE_ENV_ID = "ALE/Hero-v5"
DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
CURRICULUM_MANIFEST = "curriculum.json"
MINER_RESCUE_REWARD = 100.0


@dataclass(frozen=True)
class CurriculumCheckpoint:
    checkpoint_id: str
    task_id: str
    path: Path
    screenshot_path: Path
    stage: int
    global_depth: int
    level: int
    room: int
    local_band: int
    lives: int | None
    power_ratio: float | None
    budget_decisions: int


@dataclass(frozen=True)
class CurriculumTask:
    task_id: str
    stage: int
    global_depth: int
    level: int
    room: int
    local_band: int
    checkpoints: tuple[CurriculumCheckpoint, ...]


def _load_curriculum_pools(
    checkpoint_dir: Path, max_level: int
) -> dict[int, tuple[CurriculumTask, ...]]:
    manifest_path = checkpoint_dir / CURRICULUM_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(
            f"curriculum manifest not found: {manifest_path}; "
            "record depth checkpoints and run teacher.py --freeze-curriculum"
        ) from exc

    if int(manifest.get("format_version", 0)) != MANIFEST_FORMAT_VERSION:
        raise ValueError(
            f"unsupported curriculum manifest: {manifest_path}; "
            "record and freeze a format-v2 depth curriculum"
        )
    validate_frozen_manifest(manifest)
    if manifest.get("rom_md5") != ROM_MD5:
        raise ValueError(
            f"curriculum ROM MD5 differs from the supported H.E.R.O. ROM: {manifest_path}"
        )

    pools: dict[int, list[CurriculumTask]] = {}
    checkpoint_ids: set[str] = set()
    for task_data in manifest.get("tasks", []):
        level = int(task_data["level"])
        if not 1 <= level <= max_level:
            continue
        stage = int(task_data["stage"])
        task_id = str(task_data["task_id"])
        checkpoints: list[CurriculumCheckpoint] = []
        for entry in task_data.get("variants", []):
            state_path = checkpoint_dir / str(entry["checkpoint"])
            screenshot_path = checkpoint_dir / str(entry["screenshot"])
            if not state_path.is_file():
                raise ValueError(f"curriculum checkpoint not found: {state_path}")
            if not screenshot_path.is_file():
                raise ValueError(
                    f"curriculum checkpoint screenshot not found: {screenshot_path}"
                )
            expected_sha = entry.get("checkpoint_sha256")
            actual_sha = hashlib.sha256(state_path.read_bytes()).hexdigest()
            if not expected_sha or str(expected_sha) != actual_sha:
                raise ValueError(f"curriculum checkpoint hash mismatch: {state_path}")
            identifier = str(entry["checkpoint_id"])
            if identifier in checkpoint_ids:
                raise ValueError(f"duplicate checkpoint_id in manifest: {identifier}")
            checkpoint_ids.add(identifier)
            health = entry.get("health", {})
            demo = entry.get("demo", {})
            checkpoints.append(
                CurriculumCheckpoint(
                    checkpoint_id=identifier,
                    task_id=task_id,
                    path=state_path,
                    screenshot_path=screenshot_path,
                    stage=stage,
                    global_depth=int(task_data["global_depth"]),
                    level=level,
                    room=int(task_data["room"]),
                    local_band=int(task_data["local_band"]),
                    lives=(
                        int(health["lives"])
                        if health.get("lives") is not None
                        else None
                    ),
                    power_ratio=(
                        float(health["power_ratio"])
                        if health.get("power_ratio") is not None
                        else None
                    ),
                    budget_decisions=int(demo["budget_decisions"]),
                )
            )
        if checkpoints:
            pools.setdefault(stage, []).append(
                CurriculumTask(
                    task_id=task_id,
                    stage=stage,
                    global_depth=int(task_data["global_depth"]),
                    level=level,
                    room=int(task_data["room"]),
                    local_band=int(task_data["local_band"]),
                    checkpoints=tuple(checkpoints),
                )
            )

    if not pools:
        raise ValueError(f"no checkpoints for levels 1-{max_level} in {manifest_path}")
    return {
        stage: tuple(sorted(tasks, key=lambda task: task.task_id))
        for stage, tasks in pools.items()
    }


class HeroLevelRangeEnv(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Reset from depth tasks and terminate when their miner is rescued.

    Every reset mode fails immediately on a lost life. The successful terminal
    observation remains in the reset Level, so a next-Level frame is never
    exposed. Full-game starts use the same Level 2 cap.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        max_level: int = 2,
        checkpoint_dir: Path | str | None = None,
        curriculum_stage: int | None = None,
        checkpoint_reset_probability: float = 1.0,
        include_easier_stages: bool = True,
    ) -> None:
        gym.utils.RecordConstructorArgs.__init__(
            self,
            max_level=max_level,
            checkpoint_dir=checkpoint_dir,
            curriculum_stage=curriculum_stage,
            checkpoint_reset_probability=checkpoint_reset_probability,
            include_easier_stages=include_easier_stages,
        )
        gym.Wrapper.__init__(self, env)
        if not 1 <= max_level <= MAX_LEVEL:
            raise ValueError(f"max_level must be between 1 and {MAX_LEVEL}")
        if not 0.0 <= checkpoint_reset_probability <= 1.0:
            raise ValueError("checkpoint_reset_probability must be between 0 and 1")
        if self.observation_space.shape != (210, 160, 3):
            raise ValueError(
                "HeroLevelRangeEnv requires raw RGB ALE observations with shape "
                "(210, 160, 3); apply preprocessing outside this wrapper"
            )

        self.max_level = max_level
        self.checkpoint_reset_probability = checkpoint_reset_probability
        self.include_easier_stages = include_easier_stages
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self._curriculum_pools = (
            _load_curriculum_pools(self.checkpoint_dir, max_level)
            if self.checkpoint_dir is not None
            else {}
        )
        if self.checkpoint_dir is not None:
            manifest = json.loads(
                (self.checkpoint_dir / CURRICULUM_MANIFEST).read_text(encoding="utf-8")
            )
            self._curriculum_identity = {
                "format_version": int(manifest["format_version"]),
                "version": int(manifest["version"]),
                "manifest_sha256": str(manifest["manifest_sha256"]),
            }
        else:
            self._curriculum_identity = None
        self._curriculum_stage: int | None = None
        self.set_curriculum_stage(curriculum_stage)

        self._reset_source = "game_start"
        self._reset_checkpoint: CurriculumCheckpoint | None = None
        self._last_in_range_observation: np.ndarray | None = None
        self._last_progress: GameProgress | None = None
        self._reset_level: int | None = None
        self._reset_lives: int | None = None
        self._last_lives: int | None = None

    @property
    def available_curriculum_stages(self) -> tuple[int, ...]:
        return tuple(sorted(self._curriculum_pools))

    @property
    def curriculum_stage(self) -> int | None:
        return self._curriculum_stage

    @property
    def curriculum_identity(self) -> dict[str, Any] | None:
        return copy.deepcopy(self._curriculum_identity)

    def set_curriculum_stage(self, stage: int | None) -> None:
        if stage is not None and stage not in self._curriculum_pools:
            available = ", ".join(map(str, sorted(self._curriculum_pools))) or "none"
            raise ValueError(
                f"curriculum stage {stage} is unavailable; available stages: {available}"
            )
        self._curriculum_stage = stage

    def reload_curriculum(self) -> None:
        if self.checkpoint_dir is None:
            return
        current_stage = self._curriculum_stage
        self._curriculum_pools = _load_curriculum_pools(
            self.checkpoint_dir, self.max_level
        )
        manifest = json.loads(
            (self.checkpoint_dir / CURRICULUM_MANIFEST).read_text(encoding="utf-8")
        )
        self._curriculum_identity = {
            "format_version": int(manifest["format_version"]),
            "version": int(manifest["version"]),
            "manifest_sha256": str(manifest["manifest_sha256"]),
        }
        self.set_curriculum_stage(current_stage)

    def _sample_checkpoint(self, stage: int) -> CurriculumCheckpoint:
        if self.include_easier_stages:
            eligible_stages = [
                candidate for candidate in self.available_curriculum_stages
                if candidate <= stage
            ]
        else:
            eligible_stages = [stage]
        selected_stage = eligible_stages[
            int(self.np_random.integers(len(eligible_stages)))
        ]
        tasks = self._curriculum_pools[selected_stage]
        task = tasks[int(self.np_random.integers(len(tasks)))]
        return task.checkpoints[int(self.np_random.integers(len(task.checkpoints)))]

    def checkpoint_ids_for_stage(self, stage: int) -> tuple[str, ...]:
        tasks = self._curriculum_pools.get(stage, ())
        return tuple(
            checkpoint.checkpoint_id
            for task in tasks
            for checkpoint in task.checkpoints
        )

    def checkpoint_ids_for_level_start(self, level: int) -> tuple[str, ...]:
        return tuple(
            checkpoint.checkpoint_id
            for tasks in self._curriculum_pools.values()
            for task in tasks
            if task.level == level and task.room == 1
            for checkpoint in task.checkpoints
        )

    def _checkpoint_by_id(self, checkpoint_id: str) -> CurriculumCheckpoint:
        for tasks in self._curriculum_pools.values():
            for task in tasks:
                for checkpoint in task.checkpoints:
                    if checkpoint.checkpoint_id == checkpoint_id:
                        return checkpoint
        raise ValueError(f"curriculum checkpoint_id is unavailable: {checkpoint_id}")

    def _restore_checkpoint(
        self, checkpoint: CurriculumCheckpoint, info: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        state = ale_py.ALEState(checkpoint.path.read_bytes())
        ale = self.env.unwrapped.ale
        ale.restoreSystemState(state)

        # ALEState does not contain the framebuffer. Use the RGB frame captured
        # atomically with this state; the first step advances the restored ALE
        # state normally and returns a freshly rendered successor frame.
        with Image.open(checkpoint.screenshot_path) as image:
            observation = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        if observation.shape != self.observation_space.shape:
            raise ValueError(
                f"checkpoint screenshot has unexpected shape: "
                f"{checkpoint.screenshot_path}: {observation.shape}"
            )

        progress = decode_game_progress(ale.getRAM())
        if progress is None or progress.level > self.max_level:
            raise ValueError(
                f"checkpoint is outside levels 1-{self.max_level}: {checkpoint.path}"
            )
        if (progress.level, progress.room) != (checkpoint.level, checkpoint.room):
            raise ValueError(
                f"checkpoint metadata does not match ALE state: {checkpoint.path}"
            )

        info.update(
            {
                "lives": int(self.env.unwrapped.ale.lives()),
                "episode_frame_number": int(state.getEpisodeFrameNumber()),
                "frame_number": int(state.getFrameNumber()),
            }
        )
        return observation, info

    def _add_info(
        self,
        info: dict[str, Any],
        progress: GameProgress | None,
        *,
        level_cap_reached: bool,
        next_level: int | None = None,
    ) -> dict[str, Any]:
        info["hero_max_level"] = self.max_level
        info["hero_reset_source"] = self._reset_source
        info["hero_level_cap_reached"] = level_cap_reached
        info.setdefault("hero_miner_rescued", level_cap_reached)
        info.setdefault("hero_life_lost", False)
        info.setdefault("hero_time_limit_reached", False)
        info.setdefault("hero_terminal_reason", None)
        info["is_success"] = bool(info["hero_miner_rescued"])
        if progress is not None:
            info["hero_level"] = progress.level
            info["hero_room"] = progress.room
            info["hero_total_rooms"] = progress.total_rooms
            info["hero_rooms_after"] = progress.rooms_after
            info["hero_lesson_id"] = progress.lesson_id
        if next_level is not None:
            info["hero_next_level"] = next_level
        if self._reset_checkpoint is not None:
            info["hero_curriculum_stage"] = self._reset_checkpoint.stage
            info["hero_checkpoint"] = self._reset_checkpoint.path.name
            info["hero_checkpoint_id"] = self._reset_checkpoint.checkpoint_id
            info["hero_task_id"] = self._reset_checkpoint.task_id
            info["hero_global_depth"] = self._reset_checkpoint.global_depth
            info["hero_local_band"] = self._reset_checkpoint.local_band
            info["hero_budget_decisions"] = self._reset_checkpoint.budget_decisions
            assert self._curriculum_identity is not None
            info["hero_curriculum_version"] = self._curriculum_identity["version"]
            info["hero_curriculum_sha256"] = self._curriculum_identity[
                "manifest_sha256"
            ]
        return info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        custom_options = dict(options or {})
        force_game_start = bool(custom_options.pop("force_game_start", False))
        stage = custom_options.pop("curriculum_stage", self._curriculum_stage)
        checkpoint_id = custom_options.pop("checkpoint_id", None)
        observation, info = self.env.reset(
            seed=seed, options=custom_options or None
        )

        self._reset_source = "game_start"
        self._reset_checkpoint = None
        use_checkpoint = checkpoint_id is not None or (
            not force_game_start
            and stage is not None
            and self._curriculum_pools
            and self.np_random.random() < self.checkpoint_reset_probability
        )
        if use_checkpoint:
            if checkpoint_id is not None:
                self._reset_checkpoint = self._checkpoint_by_id(str(checkpoint_id))
                if stage is not None and self._reset_checkpoint.stage != stage:
                    raise ValueError(
                        f"checkpoint {checkpoint_id} belongs to stage "
                        f"{self._reset_checkpoint.stage}, not {stage}"
                    )
            else:
                if stage not in self._curriculum_pools:
                    raise ValueError(f"curriculum stage {stage} is unavailable")
                self._reset_checkpoint = self._sample_checkpoint(stage)
            observation, info = self._restore_checkpoint(self._reset_checkpoint, info)
            self._reset_source = "checkpoint"

        progress = decode_game_progress(self.env.unwrapped.ale.getRAM())
        if progress is None or progress.level > self.max_level:
            raise RuntimeError(
                f"reset produced a state outside levels 1-{self.max_level}"
            )
        self._last_progress = progress
        self._last_in_range_observation = np.array(observation, copy=True)
        self._reset_level = progress.level if self._reset_checkpoint is not None else None
        self._reset_lives = int(self.env.unwrapped.ale.lives())
        self._last_lives = self._reset_lives
        return observation, self._add_info(
            info, progress, level_cap_reached=False
        )

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        ale_reward = float(reward)
        ram = self.env.unwrapped.ale.getRAM()
        level = decode_level(ram)
        progress = decode_game_progress(ram)
        raw_level_cap_reached = level is not None and level > self.max_level
        current_lives = int(self.env.unwrapped.ale.lives())
        life_lost = (
            self._last_lives is not None
            and current_lives < self._last_lives
        )
        level_cap_reached = raw_level_cap_reached and not life_lost
        miner_rescued = (
            self._reset_checkpoint is not None
            and self._reset_level is not None
            and level is not None
            and level > self._reset_level
        )
        self._last_lives = current_lives

        if life_lost:
            # Losing a life is an immediate episode failure for every reset
            # mode, including full-game and after-curriculum episodes.
            terminated = True
            truncated = False
            reward = -1.0
            info["hero_terminal_reason"] = "life-lost"
        elif miner_rescued or level_cap_reached:
            terminated = True
            reward = MINER_RESCUE_REWARD
            info["hero_terminal_reason"] = "miner-rescued"
            if self._last_in_range_observation is not None:
                observation = copy.deepcopy(self._last_in_range_observation)
                progress = self._last_progress
        elif progress is not None:
            self._last_progress = progress
            self._last_in_range_observation = np.array(observation, copy=True)

        info["hero_miner_rescued"] = bool(
            not life_lost and (miner_rescued or level_cap_reached)
        )
        info["hero_life_lost"] = bool(life_lost)
        info["hero_ale_reward"] = ale_reward

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            self._add_info(
                info,
                progress,
                level_cap_reached=level_cap_reached,
                next_level=(
                    level
                    if not life_lost and (miner_rescued or level_cap_reached)
                    else None
                ),
            ),
        )


def make_hero_level_1_to_2_env(
    *,
    training: bool,
    checkpoint_dir: Path | str = DEFAULT_CHECKPOINT_DIR,
    curriculum_stage: int = 1,
    checkpoint_reset_probability: float = 0.9,
    include_easier_stages: bool = True,
    frameskip: int = 4,
    repeat_action_probability: float = 0.25,
    render_mode: str | None = None,
) -> HeroLevelRangeEnv:
    """Create matching Level 1-2 environments that stop before Level 3."""
    gym.register_envs(ale_py)
    base_env = gym.make(
        ALE_ENV_ID,
        mode=0,
        difficulty=0,
        obs_type="rgb",
        frameskip=frameskip,
        repeat_action_probability=repeat_action_probability,
        render_mode=render_mode,
    )
    try:
        return HeroLevelRangeEnv(
            base_env,
            max_level=2,
            checkpoint_dir=checkpoint_dir if training else None,
            curriculum_stage=curriculum_stage if training else None,
            checkpoint_reset_probability=(
                checkpoint_reset_probability if training else 0.0
            ),
            include_easier_stages=include_easier_stages,
        )
    except Exception:
        base_env.close()
        raise
