#!/usr/bin/env python3
"""Export every frozen curriculum reset image as ``<stage>_<idx>.jpg``."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from HeroEnv.hero_env import (  # noqa: E402
    DEFAULT_CHECKPOINT_DIR,
    make_hero_level_1_to_2_env,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help=f"frozen curriculum directory (default: {DEFAULT_CHECKPOINT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "tmp",
        help=f"JPEG output directory (default: {PROJECT_ROOT / 'tmp'})",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in [1, 100]")
    return args


def export_reset_images(args: argparse.Namespace) -> int:
    checkpoint_dir = args.checkpoint_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_name = re.compile(r"\d+_\d+\.jpg")
    for existing in output_dir.glob("*.jpg"):
        if generated_name.fullmatch(existing.name):
            existing.unlink()

    env = make_hero_level_1_to_2_env(
        training=True,
        checkpoint_dir=checkpoint_dir,
        curriculum_stage=1,
        checkpoint_reset_probability=1.0,
        include_easier_stages=False,
        frameskip=1,
        repeat_action_probability=0.25,
    )
    exported = 0
    stages = env.available_curriculum_stages
    try:
        for stage in stages:
            checkpoint_ids = env.checkpoint_ids_for_stage(stage)
            if not checkpoint_ids:
                raise RuntimeError(f"Stage {stage} has no checkpoint variants")
            env.set_curriculum_stage(stage)
            for index, checkpoint_id in enumerate(checkpoint_ids, start=1):
                observation, info = env.reset(
                    seed=args.seed + stage * 10_000 + index,
                    options={
                        "curriculum_stage": stage,
                        "checkpoint_id": checkpoint_id,
                    },
                )
                frame = np.asarray(observation, dtype=np.uint8)
                if frame.shape != (210, 160, 3):
                    raise RuntimeError(
                        f"unexpected reset observation shape for {checkpoint_id}: "
                        f"{frame.shape}"
                    )
                output_path = output_dir / f"{stage}_{index}.jpg"
                Image.fromarray(frame, mode="RGB").save(
                    output_path,
                    format="JPEG",
                    quality=args.jpeg_quality,
                    subsampling=0,
                )
                exported += 1
                print(
                    f"{output_path}: task={info['hero_task_id']} "
                    f"checkpoint={info['hero_checkpoint_id']}",
                    flush=True,
                )
    finally:
        env.close()

    print(
        f"Exported {exported} reset images from "
        f"{len(stages)} Stage(s) to {output_dir}",
        flush=True,
    )
    return exported


def main() -> int:
    export_reset_images(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
