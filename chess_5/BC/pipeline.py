"""Resumable quality-first 9x9 bootstrap and multi-round DAgger pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quality-first 9x9 BC/DAgger training.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--artifact-root", type=Path,
                        default=Path(os.environ.get("ARTIFACT_ROOT", ROOT / "BC")))
    parser.add_argument("--board-size", type=int, default=_env_int("BOARD_SIZE", 9))
    parser.add_argument("--rounds", type=int, default=_env_int("DAGGER_ROUNDS", 6))
    parser.add_argument("--workers", type=int, default=_env_int("GEN_WORKERS", 16))
    parser.add_argument("--eval-workers", type=int, default=_env_int("EVAL_WORKERS", 16))
    parser.add_argument("--train-workers", type=int, default=_env_int("TRAIN_WORKERS", 4))
    parser.add_argument("--device", default=os.environ.get("DEVICE", "auto"))
    parser.add_argument("--epochs", type=int, default=_env_int("EPOCHS", 100))
    parser.add_argument("--batch-size", type=int, default=_env_int("BATCH_SIZE", 256))
    parser.add_argument("--bootstrap-min-games", type=int, default=2_000)
    parser.add_argument("--bootstrap-max-games", type=int, default=8_000)
    parser.add_argument("--bootstrap-target-states", type=int, default=100_000)
    parser.add_argument("--dagger-min-games", type=int, default=1_000)
    parser.add_argument("--dagger-max-games", type=int, default=4_000)
    parser.add_argument("--dagger-target-states", type=int, default=50_000)
    parser.add_argument("--challenge-ood-states", type=int, default=20_000)
    parser.add_argument("--challenge-prefixes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=_env_int("SEED", 0))
    args = parser.parse_args(argv)
    if args.board_size != 9:
        parser.error("the quality-first pipeline is intentionally fixed to 9x9")
    if not 1 <= args.rounds <= 6:
        parser.error("--rounds must be in [1, 6]")
    return args


def _run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("RUN", " ".join(command), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True); log.write(line); log.flush()
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def _complete_dataset(path: Path) -> bool:
    metadata = path / "metadata.json"
    return metadata.is_file() and json.loads(metadata.read_text()).get("status") == "complete"


def _coverage_ok(path: Path) -> None:
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata.get("coverage_stalled"):
        raise RuntimeError(
            f"generation reached max games without its canonical-state target: {path}"
        )


def _atomic_state(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    python = sys.executable
    root = args.artifact_root.resolve()
    data_root = root / "data" / args.run_name
    checkpoint_root = root / "checkpoints" / args.run_name
    run_root = root / "runs" / args.run_name
    evaluation_root = root / "evaluations" / args.run_name
    state_path = root / "pipeline_state" / args.run_name / "state.json"
    challenge_bank = data_root / "challenge_v1.npz"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {
        "run_name": args.run_name, "board_size": 9, "completed_round": -1,
        "consecutive_passes": 0, "status": "running",
    }

    bootstrap = data_root / "round_00_bootstrap"
    if not _complete_dataset(bootstrap):
        _run([
            python, "BC/generate.py", "--output", str(bootstrap), "--mode", "bootstrap",
            "--board-size", "9", "--workers", str(args.workers), "--seed", str(args.seed),
            "--games", str(args.bootstrap_max_games), "--min-games", str(args.bootstrap_min_games),
            "--max-games", str(args.bootstrap_max_games), "--target-new-states",
            str(args.bootstrap_target_states), "--batch-games", "250", "--quality-gate",
            "--target-win-states", "2000", "--target-block-states", "2000",
            "--target-fork-states", "1000",
            "--tb-dir", str(run_root / "round_00_generate"),
        ], run_root / "round_00_generate" / "console.log")
    _coverage_ok(bootstrap)

    if not challenge_bank.is_file():
        _run([
            python, "BC/challenge.py", "build", "--data-dir", str(bootstrap),
            "--output", str(challenge_bank), "--seed", str(args.seed),
            "--ood-states", str(args.challenge_ood_states),
            "--prefix-count", str(args.challenge_prefixes),
        ], run_root / "challenge_build" / "console.log")

    datasets = [bootstrap]
    checkpoints: list[Path] = []
    if state.get("status") == "complete":
        print(f"Pipeline already complete: {state_path}", flush=True)
        return
    consecutive = 0
    for round_index in range(args.rounds + 1):
        label = f"round_{round_index:02d}"
        if round_index:
            data = data_root / f"{label}_dagger"
            previous = checkpoints[-1]
            if not _complete_dataset(data):
                command = [
                    python, "BC/generate.py", "--output", str(data), "--mode", "dagger",
                    "--checkpoint", str(previous), "--board-size", "9",
                    "--workers", str(args.workers), "--seed", str(args.seed + round_index * 1000),
                    "--dagger-round", str(round_index), "--games", str(args.dagger_max_games),
                    "--min-games", str(args.dagger_min_games), "--max-games",
                    str(args.dagger_max_games), "--target-new-states",
                    str(args.dagger_target_states), "--batch-games", "250", "--quality-gate",
                    "--shared-cache", str(datasets[-1] / "cache" / "shared.sqlite3"),
                    "--tb-dir", str(run_root / f"{label}_generate"),
                ]
                for checkpoint in checkpoints:
                    command.extend(("--history-checkpoint", str(checkpoint)))
                for old_data in datasets:
                    command.extend(("--previous-data-dir", str(old_data)))
                _run(command, run_root / f"{label}_generate" / "console.log")
            _coverage_ok(data); datasets.append(data)

        stage_root = checkpoint_root / label
        best = stage_root / "best.pt"
        if not best.is_file():
            command = [
                python, "BC/train.py", "--data-dir", *map(str, datasets),
                "--run-name", label, "--output-dir", str(checkpoint_root),
                "--board-size", "9", "--hidden-channels", "128", "--num-res-blocks", "8",
                "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                "--workers", str(args.train_workers), "--device", args.device,
                "--seed", str(args.seed), "--challenge-bank", str(challenge_bank),
                "--lr", "0.0003" if round_index == 0 else "0.0001",
                "--tb-dir", str(run_root / f"{label}_train"),
            ]
            latest = stage_root / "latest.pt"
            if latest.is_file():
                command.append("--resume")
            elif round_index:
                command.extend(("--init-checkpoint", str(checkpoints[-1])))
            _run(command, run_root / f"{label}_train" / "console.log")
        checkpoints.append(best)

        evaluation = evaluation_root / f"{label}.json"
        if not evaluation.is_file():
            command = [
                python, "BC/challenge.py", "evaluate", "--checkpoint", str(best),
                "--bank", str(challenge_bank), "--board-size", "9",
                "--workers", str(args.eval_workers), "--seed",
                str(args.seed + 20_000 + round_index * 1000), "--output", str(evaluation),
            ]
            if round_index:
                command.extend(("--audit-data", str(datasets[-1])))
            _run(command, run_root / f"{label}_evaluate" / "console.log")
        result = json.loads(evaluation.read_text())
        consecutive = consecutive + 1 if result.get("passed") else 0
        state.update({"completed_round": round_index, "consecutive_passes": consecutive,
                      "last_checkpoint": str(best), "last_evaluation": str(evaluation)})
        _atomic_state(state_path, state)
        if consecutive >= 2:
            state["status"] = "complete"; _atomic_state(state_path, state)
            print(f"SUCCESS: expert-level composite gate passed in {label}", flush=True)
            return
    state["status"] = "max_rounds_reached"; _atomic_state(state_path, state)
    print("Pipeline reached the configured DAgger round limit without two consecutive passes.",
          flush=True)


if __name__ == "__main__":
    main()
