"""Offline masked-cross-entropy training for Gomoku BC."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from BC.dataset import GomokuDataset, discover_shards
from BC.network import GomokuPolicyNet


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
    parser.add_argument("--hidden-channels", type=int, default=96)
    parser.add_argument("--num-res-blocks", type=int, default=4)
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
    parser.add_argument("--tb-dir", type=Path, default=None,
                        help="Exact TensorBoard directory for this pipeline step.")
    return parser.parse_args()


def _datasets(args: argparse.Namespace, split: str) -> ConcatDataset:
    datasets = []
    for index, root in enumerate(args.data_dir):
        metadata = json.loads((root / "metadata.json").read_text())
        if int(metadata["board_size"]) != args.board_size:
            raise ValueError(f"board size mismatch in {root}")
        cap = args.aggregate_max_samples if index and args.aggregate_max_samples > 0 else None
        datasets.append(GomokuDataset(discover_shards([root]), split=split,
                                      val_fraction=args.val_fraction, augment=split == "train",
                                      seed=args.seed, max_samples=cap))
    return ConcatDataset(datasets)


def run_epoch(model: nn.Module, loader: DataLoader, device: torch.device,
              optimizer: torch.optim.Optimizer | None, grad_clip: float) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total_legal = total = 0
    started = time.perf_counter()
    for states, targets, masks in loader:
        states, targets, masks = states.to(device), targets.to(device), masks.to(device)
        with torch.set_grad_enabled(training):
            logits = model(states).masked_fill(~masks, -1e9)
            loss = nn.functional.cross_entropy(logits, targets)
            if training:
                optimizer.zero_grad(set_to_none=True); loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        predicted = logits.argmax(1)
        batch = len(targets)
        total += batch; total_loss += float(loss.detach()) * batch
        total_correct += int((predicted == targets).sum())
        total_legal += int(masks.gather(1, predicted[:, None]).sum())
    seconds = time.perf_counter() - started
    return {"loss": total_loss / max(1, total), "accuracy": total_correct / max(1, total),
            "legal_rate": total_legal / max(1, total), "samples_per_second": total / max(1e-9, seconds)}


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler: Any, args: argparse.Namespace, epoch: int,
                    metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {"format_version": 1, "board_size": args.board_size,
                  "model_kwargs": {"hidden_channels": args.hidden_channels,
                                   "num_res_blocks": args.num_res_blocks},
                  "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                  "scheduler_state_dict": scheduler.state_dict(), "epoch": epoch, "metrics": metrics,
                  "data_versions": [str(p.resolve()) for p in args.data_dir], "args": vars(args)}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary); os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    train_data, val_data = _datasets(args, "train"), _datasets(args, "val")
    if len(train_data) == 0 or len(val_data) == 0:
        raise ValueError("train and validation splits must both contain samples")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == "cuda")
    model = GomokuPolicyNet(hidden_channels=args.hidden_channels,
                            num_res_blocks=args.num_res_blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=max(1, args.patience // 3))
    run_dir = args.output_dir / args.run_name
    latest_path, best_path = run_dir / "latest.pt", run_dir / "best.pt"
    if best_path.exists() and not args.resume:
        raise FileExistsError(f"run already exists: {run_dir}")
    best = float("inf"); stale = 0; start_epoch = 1
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
            best = float(best_checkpoint.get("metrics", {}).get("val_loss", float("inf")))
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
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_metrics = run_epoch(model, train_loader, device, optimizer, args.grad_clip)
            val_metrics = run_epoch(model, val_loader, device, None, args.grad_clip)
            scheduler.step(val_metrics["loss"])
            metrics = {**{f"train_{k}": v for k, v in train_metrics.items()},
                       **{f"val_{k}": v for k, v in val_metrics.items()},
                       "lr": optimizer.param_groups[0]["lr"]}
            save_checkpoint(latest_path, model, optimizer, scheduler, args, epoch, metrics)
            if val_metrics["loss"] < best:
                best = val_metrics["loss"]; stale = 0
                save_checkpoint(best_path, model, optimizer, scheduler, args, epoch, metrics)
            else:
                stale += 1
            if writer is not None:
                for name, value in train_metrics.items():
                    writer.add_scalar(f"Train/{name}", value, epoch)
                for name, value in val_metrics.items():
                    writer.add_scalar(f"Validation/{name}", value, epoch)
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
