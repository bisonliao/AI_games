"""Offline masked-cross-entropy training for Gomoku BC."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from BC.dataset import GomokuDataset, discover_shards
from BC.diversity import assess_diversity
from BC.network import GomokuPolicyNet
from BC.oracle import REASON_TO_ID, SOURCE_TO_ID, oracle_identity, validate_oracle_identity


DIVERSITY_KEYS = (
    "canonical_effective_trajectory_ratio",
    "dominant_canonical_trajectory_fraction",
    "canonical_state_unique_ratio",
)


def load_diversity_reports(data_dirs: list[Path]) -> list[dict[str, Any]]:
    """Load the immutable diversity report for every dataset used by training."""
    reports: list[dict[str, Any]] = []
    for root in data_dirs:
        root = root.expanduser().resolve()
        metadata = json.loads((root / "metadata.json").read_text())
        diversity = metadata.get("diversity")
        if diversity is None and (root / "diversity.json").is_file():
            diversity = json.loads((root / "diversity.json").read_text())
        if not isinstance(diversity, dict):
            raise ValueError(f"diversity report is missing from dataset: {root}")
        missing = [key for key in DIVERSITY_KEYS if key not in diversity]
        if missing:
            raise ValueError(f"diversity report missing {missing}: {root}")
        # Recompute with the current standard thresholds so legacy reports also
        # receive consistent Chinese conclusions instead of stale message text.
        quality = assess_diversity(diversity)
        reports.append({"root": str(root), "label": root.name, "diversity": diversity,
                        "quality": quality})
    return reports


def print_diversity_reports(reports: list[dict[str, Any]]) -> None:
    marker = "!!!!!!!!!!!!!!!!"
    print(f"{marker} 数据多样性检查开始 {marker}", flush=True)
    for report in reports:
        values = report["diversity"]
        print(f"{marker} 数据集：{report['label']} {marker}", flush=True)
        print(f"{marker} 有效轨迹比例："
              f"{values['canonical_effective_trajectory_ratio']:.2%}（越高越好，表示真正不同的对局分支） {marker}",
              flush=True)
        print(f"{marker} 最大单一轨迹占比："
              f"{values['dominant_canonical_trajectory_fraction']:.2%}（越低越好，表示是否被一种对局垄断） {marker}",
              flush=True)
        print(f"{marker} 独特状态比例："
              f"{values['canonical_state_unique_ratio']:.2%}（越高越好，表示不同棋盘局面的覆盖程度） {marker}",
              flush=True)
        quality = report.get("quality") or {}
        if quality.get("passed") is False:
            conclusion = "不可接受，禁止训练"
        elif quality.get("warnings"):
            conclusion = "可接受，但多样性仍有警告，需要关注"
        else:
            conclusion = "可接受，多样性良好"
        print(f"{marker} 数据质量结论：{conclusion} {marker}", flush=True)
        if quality.get("warnings"):
            print(f"{marker} 警告："
                  + " | ".join(quality["warnings"]) + f" {marker}", flush=True)
        if quality.get("passed") is False:
            print(f"{marker} 不通过原因："
                  + " | ".join(quality.get("failures", [])) + f" {marker}", flush=True)
    print(f"{marker} 数据多样性检查结束 {marker}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Gomoku behavioral-cloning policy.")
    parser.add_argument("--data-dir", type=Path, nargs="+", required=True,
                        help="Immutable dataset versions; first is the base expert dataset.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "BC" / "checkpoints")
    parser.add_argument("--board-size", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--num-res-blocks", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--aggregate-max-samples", type=int, default=0,
                        help="Per aggregation dataset cap; 0 keeps every sample.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true",
                        help="Resume this run from latest.pt if it exists.")
    parser.add_argument("--init-checkpoint", type=Path, default=None,
                        help="Warm-start model weights while resetting optimizer/scheduler.")
    parser.add_argument("--rank-temperature", type=float, default=1.0)
    parser.add_argument("--challenge-bank", type=Path, default=None,
                        help="Frozen bank whose canonical states are excluded from training.")
    parser.add_argument("--tb-dir", type=Path, default=None,
                        help="Exact TensorBoard directory for this pipeline step.")
    return parser.parse_args()


def _datasets(args: argparse.Namespace, split: str) -> ConcatDataset:
    datasets = []
    excluded_keys: set[bytes] = set()
    if split == "train" and args.challenge_bank is not None:
        with np.load(args.challenge_bank) as challenge:
            excluded_keys = {bytes(key) for key in challenge["canonical_keys"]}
    for index, root in enumerate(args.data_dir):
        metadata = json.loads((root / "metadata.json").read_text())
        if int(metadata["board_size"]) != args.board_size:
            raise ValueError(f"board size mismatch in {root}")
        cap = args.aggregate_max_samples if index and args.aggregate_max_samples > 0 else None
        datasets.append(GomokuDataset(discover_shards([root]), split=split,
                                      val_fraction=args.val_fraction, augment=split == "train",
                                      seed=args.seed, max_samples=cap, rich=True,
                                      excluded_keys=excluded_keys))
    return ConcatDataset(datasets)


def _training_sampler(dataset: ConcatDataset, seed: int) -> WeightedRandomSampler:
    children = list(dataset.datasets)
    count = len(children)
    all_keys = np.concatenate([np.asarray(child.canonical_keys, dtype="S32")
                               for child in children])
    if len(all_keys):
        _, inverse, canonical_counts = np.unique(all_keys, return_inverse=True,
                                                 return_counts=True)
        global_frequency_weights = 1.0 / np.sqrt(np.minimum(canonical_counts[inverse], 64))
    else:
        global_frequency_weights = np.empty(0, dtype=np.float64)
    if count == 1:
        dataset_shares = [1.0]
    elif count == 2:
        dataset_shares = [0.5, 0.5]
    elif count == 3:
        dataset_shares = [0.3, 0.35, 0.35]
    else:
        dataset_shares = [0.3] + [0.2 / max(1, count - 3)] * (count - 3) + [0.25, 0.25]
    weights = []
    phase_shares = np.asarray([0.30, 0.45, 0.25], dtype=np.float64)
    offset = 0
    for share, child in zip(dataset_shares, children):
        if len(child) == 0:
            weights.append(torch.empty(0, dtype=torch.double)); continue
        players = np.asarray(child.players)
        plies = np.asarray(child.plies)
        phases = np.where(plies <= 15, 0, np.where(plies <= 35, 1, 2))
        player_counts = {value: max(1, int(np.count_nonzero(players == value)))
                         for value in (-1, 1)}
        phase_counts = np.maximum(1, np.bincount(phases, minlength=3))
        frequency_values = global_frequency_weights[offset:offset + len(child)]
        offset += len(child)
        values = np.asarray([
            share / len(child) * (0.5 * len(child) / player_counts[int(player)])
            * (phase_shares[int(phase)] * len(child) / phase_counts[int(phase)])
            * float(frequency)
            for player, phase, frequency in zip(players, phases, frequency_values)
        ], dtype=np.float64)
        weights.append(torch.from_numpy(values))
    all_weights = torch.cat(weights)
    generator = torch.Generator().manual_seed(int(seed))
    return WeightedRandomSampler(all_weights, num_samples=len(dataset), replacement=True,
                                 generator=generator)


def topk_imitation_loss(logits: torch.Tensor, targets: torch.Tensor,
                        candidates: torch.Tensor, candidate_counts: torch.Tensor,
                        reasons: torch.Tensor, frequency_weights: torch.Tensor,
                        *, rank_temperature: float = 1.0) -> torch.Tensor:
    hard = nn.functional.cross_entropy(logits, targets, reduction="none")
    width = candidates.shape[1]
    ranks = torch.arange(width, device=logits.device)[None, :]
    valid = ranks < candidate_counts[:, None]
    rank_logits = (-ranks.to(logits.dtype) / float(rank_temperature)).masked_fill(~valid, -1e9)
    rank_probabilities = torch.softmax(rank_logits, dim=1) * valid
    safe_candidates = candidates.clamp_min(0)
    target_distribution = torch.zeros_like(logits)
    target_distribution.scatter_add_(1, safe_candidates, rank_probabilities)
    soft = -(target_distribution * torch.log_softmax(logits, dim=1)).sum(1)
    tactical_weights = torch.ones_like(hard)
    tactical_weights[(reasons == REASON_TO_ID["win"]) |
                     (reasons == REASON_TO_ID["block"])] = 4.0
    tactical_weights[(reasons == REASON_TO_ID["own_fork"]) |
                     (reasons == REASON_TO_ID["block_fork"])] = 2.0
    weights = tactical_weights * frequency_weights
    return (((0.7 * hard + 0.3 * soft) * weights).sum() / weights.sum().clamp_min(1e-9))


def run_epoch(model: nn.Module, loader: DataLoader, device: torch.device,
              optimizer: torch.optim.Optimizer | None, grad_clip: float,
              rank_temperature: float = 1.0) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total_topk = total_legal = total = 0
    tactical_correct = tactical_total = ood_correct = ood_topk = ood_total = 0
    started = time.perf_counter()
    for batch_data in loader:
        states = batch_data["states"].to(device)
        targets = batch_data["targets"].to(device)
        masks = batch_data["masks"].to(device)
        candidates = batch_data["candidates"].to(device)
        candidate_counts = batch_data["candidate_counts"].to(device)
        reasons = batch_data["reasons"].to(device)
        sources = batch_data["sources"].to(device)
        frequency_weights = batch_data["frequency_weights"].to(device)
        with torch.set_grad_enabled(training):
            logits = model(states).masked_fill(~masks, -1e9)
            loss = topk_imitation_loss(
                logits, targets, candidates, candidate_counts, reasons, frequency_weights,
                rank_temperature=rank_temperature,
            )
            if training:
                optimizer.zero_grad(set_to_none=True); loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        predicted = logits.argmax(1)
        batch = len(targets)
        total += batch; total_loss += float(loss.detach()) * batch
        total_correct += int((predicted == targets).sum())
        valid_candidates = candidates >= 0
        total_topk += int(((predicted[:, None] == candidates) & valid_candidates).any(1).sum())
        total_legal += int(masks.gather(1, predicted[:, None]).sum())
        tactical = (reasons == REASON_TO_ID["win"]) | (reasons == REASON_TO_ID["block"])
        tactical_total += int(tactical.sum())
        tactical_correct += int(((predicted == targets) & tactical).sum())
        ood = sources != SOURCE_TO_ID["expert_selfplay"]
        ood_total += int(ood.sum())
        ood_correct += int(((predicted == targets) & ood).sum())
        ood_topk += int((((predicted[:, None] == candidates) & valid_candidates).any(1) & ood).sum())
    seconds = time.perf_counter() - started
    return {"loss": total_loss / max(1, total), "accuracy": total_correct / max(1, total),
            "top4_accuracy": total_topk / max(1, total),
            "tactical_accuracy": tactical_correct / max(1, tactical_total),
            "ood_accuracy": ood_correct / max(1, ood_total),
            "ood_top4_accuracy": ood_topk / max(1, ood_total),
            "legal_rate": total_legal / max(1, total), "samples_per_second": total / max(1e-9, seconds)}


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler: Any, args: argparse.Namespace, epoch: int,
                    metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {"format_version": 2, "board_size": args.board_size,
                  "model_kwargs": {"hidden_channels": args.hidden_channels,
                                   "num_res_blocks": args.num_res_blocks},
                  "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                  "scheduler_state_dict": scheduler.state_dict(), "epoch": epoch, "metrics": metrics,
                  "data_versions": [str(p.resolve()) for p in args.data_dir],
                  "oracle": getattr(args, "oracle", None), "args": vars(args)}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary); os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1")
    torch.manual_seed(args.seed)
    if args.device == "auto" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the quality-first BC pipeline; pass --device cpu explicitly for tests")
    device = torch.device("cuda" if args.device == "auto" else args.device)
    train_data, val_data = _datasets(args, "train"), _datasets(args, "val")
    if len(train_data) == 0 or len(val_data) == 0:
        raise ValueError("train and validation splits must both contain samples")
    diversity_reports = load_diversity_reports(args.data_dir)
    print_diversity_reports(diversity_reports)
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              sampler=_training_sampler(train_data, args.seed),
                              num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == "cuda")
    challenge_loader = None
    if args.challenge_bank is not None:
        challenge_data = GomokuDataset([args.challenge_bank], split="val", val_fraction=1.0,
                                       augment=False, seed=args.seed, rich=True)
        challenge_loader = DataLoader(challenge_data, batch_size=args.batch_size,
                                      shuffle=False, num_workers=args.workers,
                                      pin_memory=device.type == "cuda")
    model = GomokuPolicyNet(hidden_channels=args.hidden_channels,
                            num_res_blocks=args.num_res_blocks).to(device)
    metadata_oracles = []
    for root in args.data_dir:
        metadata = json.loads((root / "metadata.json").read_text())
        if metadata.get("oracle") is not None:
            metadata_oracles.append(metadata["oracle"])
    args.oracle = (metadata_oracles[0] if metadata_oracles else
                   oracle_identity(max_candidates=12, top_k=4))
    for actual in metadata_oracles[1:]:
        validate_oracle_identity(actual, args.oracle)
    if args.init_checkpoint is not None and args.resume:
        raise ValueError("--init-checkpoint and --resume are mutually exclusive")
    if args.init_checkpoint is not None:
        initial = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        if initial.get("model_kwargs") != {"hidden_channels": args.hidden_channels,
                                           "num_res_blocks": args.num_res_blocks}:
            raise ValueError("init checkpoint model architecture is incompatible")
        if initial.get("oracle") is not None:
            validate_oracle_identity(initial.get("oracle"), args.oracle)
        model.load_state_dict(initial["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=max(1, args.patience // 3))
    run_dir = args.output_dir / args.run_name
    latest_path, best_path = run_dir / "latest.pt", run_dir / "best.pt"
    if best_path.exists() and not args.resume:
        raise FileExistsError(f"run already exists: {run_dir}")
    best_rank = (-1, -1.0, -1.0, float("-inf")); stale = 0; start_epoch = 1
    if args.resume and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        expected_data = [str(path.resolve()) for path in args.data_dir]
        if (int(checkpoint["board_size"]) != args.board_size
                or checkpoint.get("model_kwargs") != {"hidden_channels": args.hidden_channels,
                                                       "num_res_blocks": args.num_res_blocks}
                or checkpoint.get("data_versions") != expected_data):
            raise ValueError("latest checkpoint is incompatible with the requested training run")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        if best_path.exists():
            best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
            previous_metrics = best_checkpoint.get("metrics", {})
            best_rank = (
                int(previous_metrics.get("challenge_tactical_accuracy",
                                         previous_metrics.get("val_tactical_accuracy", 0.0)) >= 0.995),
                float(previous_metrics.get("challenge_ood_accuracy",
                                           previous_metrics.get("val_accuracy", 0.0))),
                float(previous_metrics.get("challenge_ood_top4_accuracy",
                                           previous_metrics.get("val_top4_accuracy", 0.0))),
                -float(previous_metrics.get("challenge_loss",
                                            previous_metrics.get("val_loss", float("inf")))),
            )
        print(f"resumed {latest_path} at epoch {start_epoch}", flush=True)

    writer = None
    if args.tb_dir is not None:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(args.tb_dir), purge_step=start_epoch)
        writer.add_text("Pipeline/config", json.dumps({
            "run_name": args.run_name, "board_size": args.board_size,
            "data_versions": [str(path.resolve()) for path in args.data_dir],
            "device": str(device),
        }, ensure_ascii=False, indent=2), start_epoch - 1)
        for index, report in enumerate(diversity_reports):
            values = report["diversity"]
            label = f"dataset_{index}_{report['label']}"
            writer.add_text(f"DataDiversity/{label}/details",
                            json.dumps({"root": report["root"], **values},
                                       ensure_ascii=False, indent=2), 0)
            for key in DIVERSITY_KEYS:
                writer.add_scalar(f"DataDiversity/{label}/{key}", values[key], 0)
            quality = report.get("quality") or {}
            if "passed" in quality:
                writer.add_scalar(f"DataDiversity/{label}/quality_gate_passed",
                                  float(bool(quality["passed"])), 0)
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_metrics = run_epoch(model, train_loader, device, optimizer, args.grad_clip,
                                      args.rank_temperature)
            val_metrics = run_epoch(model, val_loader, device, None, args.grad_clip,
                                    args.rank_temperature)
            challenge_metrics = (run_epoch(model, challenge_loader, device, None,
                                           args.grad_clip, args.rank_temperature)
                                 if challenge_loader is not None else val_metrics)
            scheduler.step(val_metrics["loss"])
            metrics = {**{f"train_{k}": v for k, v in train_metrics.items()},
                       **{f"val_{k}": v for k, v in val_metrics.items()},
                       **{f"challenge_{k}": v for k, v in challenge_metrics.items()},
                       "lr": optimizer.param_groups[0]["lr"]}
            save_checkpoint(latest_path, model, optimizer, scheduler, args, epoch, metrics)
            checkpoint_rank = (int(challenge_metrics["tactical_accuracy"] >= 0.995),
                               challenge_metrics["ood_accuracy"],
                               challenge_metrics["ood_top4_accuracy"],
                               -challenge_metrics["loss"])
            if checkpoint_rank > best_rank:
                best_rank = checkpoint_rank; stale = 0
                save_checkpoint(best_path, model, optimizer, scheduler, args, epoch, metrics)
            else:
                stale += 1
            if writer is not None:
                for name, value in train_metrics.items():
                    writer.add_scalar(f"Train/{name}", value, epoch)
                for name, value in val_metrics.items():
                    writer.add_scalar(f"Validation/{name}", value, epoch)
                for name, value in challenge_metrics.items():
                    writer.add_scalar(f"Challenge/{name}", value, epoch)
                writer.add_scalar("Train/learning_rate", optimizer.param_groups[0]["lr"], epoch)
                writer.flush()
            print(f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
                  f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.3f} "
                  f"legal={val_metrics['legal_rate']:.3f} samples/s={train_metrics['samples_per_second']:.0f}",
                  flush=True)
            if stale >= args.patience:
                print(f"early stopping after {epoch} epochs"); break
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
