"""可恢复的 9x9 BC/DAgger 训练总编排器。

完整流程为：生成 Bootstrap 数据 -> 冻结 challenge bank -> 训练 round_00 ->
用当前策略收集 DAgger 状态 -> 混合新旧数据 warm-start 下一轮 -> 复合评测。
连续两轮通过复合门槛后提前结束，否则运行到配置的最大轮数。

本文件不直接实现训练算法，而是以子进程串联 generate.py、train.py 和
challenge.py。每个阶段都有独立日志，并用已落盘产物判断能否从中断处恢复。
"""

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
    """读取整数环境变量；命令行参数仍可覆盖这里提供的默认值。"""
    return int(os.environ.get(name, default))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析整条 pipeline 的规模、并行度和停止条件。"""
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
    """运行一个阶段并保存日志；子进程失败时立即中止后续流水线。"""
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
    """只有 metadata 声明的全部 shard 都存在时，数据集才算完成。"""
    metadata = path / "metadata.json"
    return metadata.is_file() and json.loads(metadata.read_text()).get("status") == "complete"


def _coverage_ok(path: Path) -> None:
    """执行数据覆盖质量门，多样性不足或生成停滞时禁止进入训练。"""
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata.get("coverage_stalled"):
        raise RuntimeError(
            f"generation reached max games without its canonical-state target: {path}"
        )


def _atomic_state(path: Path, value: dict) -> None:
    """原子更新 pipeline 状态，避免中断后留下半个 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> None:
    """按轮次执行数据生成、监督训练和固定挑战集评测。"""
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

    # 阶段一：没有神经网络策略时，先用带受控扰动的专家生成初始覆盖。
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

    # 阶段二：训练前冻结挑战集。train.py 会排除其中的 canonical 状态，
    # 防止挑战样本泄漏进训练集导致评测虚高。
    if not challenge_bank.is_file():
        _run([
            python, "BC/challenge.py", "build", "--data-dir", str(bootstrap),
            "--output", str(challenge_bank), "--seed", str(args.seed),
            "--ood-states", str(args.challenge_ood_states),
            "--prefix-count", str(args.challenge_prefixes),
        ], run_root / "challenge_build" / "console.log")

    # datasets 随轮次增长：每轮训练都会读取 Bootstrap 和历轮 DAgger 数据；
    # train.py 的 WeightedRandomSampler 再负责控制新旧数据占比。
    datasets = [bootstrap]
    checkpoints: list[Path] = []
    if state.get("status") == "complete":
        print(f"Pipeline already complete: {state_path}", flush=True)
        return
    consecutive = 0
    for round_index in range(args.rounds + 1):
        label = f"round_{round_index:02d}"
        if round_index:
            # 阶段三：上一轮 best.pt 负责推进棋局，每个访问状态仍由冻结专家标注。
            # 历史 checkpoint 作为部分对局的对手，降低对单一策略版本的过拟合。
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

        # 阶段四：round_00 随机初始化；后续轮次只继承上一轮网络权重，
        # optimizer 和学习率调度器重新初始化，以干净状态适应新增数据。
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

        # 阶段五：best.pt 要经过固定状态一致率、当前 rollout audit 和交换黑白
        # 的前缀对局评测。这里只依据完整复合结果决定是否停止。
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
        # 单轮通过可能来自统计波动，因此要求连续两轮通过；任何失败都会清零。
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
