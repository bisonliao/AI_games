"""Behavior cloning using native Torch data, model, optimizer and checkpoints."""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import time

from .runtime import (CHECKPOINT_DIR, LOG_DIR, DATA_DIR, load_checkpoint, save_checkpoint,
                      to_device, seed_everything, rng_state, restore_rng, frozen_text_eval)
from .data import ExpertDataset, make_manifest
from .evaluate import predict, rollout
import numpy as np
import torch
from torch.utils.data import DataLoader


class StageBatchSampler:
    def __init__(self, dataset, batch_size, batches, generator, sampling="balanced"):
        stages = np.array([entry[2] for entry in dataset.indices])
        counts = np.bincount(stages, minlength=5)
        if sampling not in {"balanced", "random"}:
            raise ValueError(f"Unknown sampling mode: {sampling}")
        if sampling == "balanced" and np.any(counts == 0):
            raise ValueError("All five stages must be present")
        weights = 1.0 / counts[stages] if sampling == "balanced" else np.ones(len(stages))
        self.weights = torch.tensor(weights, dtype=torch.double)
        self.batch_size = batch_size
        self.batches = batches
        self.generator = generator

    def __iter__(self):
        batches_yielded = 0
        while self.batches is None or batches_yielded < self.batches:
            yield torch.multinomial(self.weights, self.batch_size, replacement=True,
                                    generator=self.generator).tolist()
            batches_yielded += 1

    def __len__(self):
        if self.batches is None:
            raise TypeError("Unlimited sampler has no length")
        return self.batches


def discard_nonfinite_update(optimizer, scaler, reason):
    optimizer.zero_grad(set_to_none=True)
    if not scaler.is_enabled():
        raise FloatingPointError(f"Non-finite {reason} in FP32 training")
    parameter = optimizer.param_groups[0]["params"][0]
    scaler.scale(torch.zeros((), device=parameter.device))
    scaler.update(new_scale=scaler.get_scale() * scaler.get_backoff_factor())
    state = scaler.state_dict()
    state["_growth_tracker"] = 0
    scaler.load_state_dict(state)


def prepare_gradients(optimizer, scaler, parameters, max_norm=1.0):
    scaler.unscale_(optimizer)
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        raise RuntimeError("No gradients available for the optimizer update")
    grad_norm = torch.nn.utils.get_total_norm(gradients, error_if_nonfinite=False)
    if not torch.isfinite(grad_norm):
        discard_nonfinite_update(optimizer, scaler, "gradient norm")
        return None
    torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, grad_norm)
    return grad_norm


def resolve_settings(args, stored, parser):
    defaults = dict(batch_size=4, accumulation=8, lr=3e-5, warmup=500,
                    lr_decay=True, decay_start=5000, decay_end=10000,
                    min_lr_ratio=0.1, sampling="balanced")
    saved = stored.get("settings", {})
    for key, default in defaults.items():
        previous = saved.get(key, False if key == "lr_decay" else default)
        provided = getattr(args, key)
        if args.resume and provided is not None and provided != previous:
            parser.error(f"Resume must preserve {key}={previous}; start a new run for ablations")
        setattr(args, key, previous if args.resume else (default if provided is None else provided))
    if min(args.batch_size, args.accumulation, args.warmup) < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("batch size, accumulation, warmup and lr must be positive and finite")
    if not 0 < args.min_lr_ratio <= 1:
        parser.error("--min-lr-ratio must be in (0, 1]")
    if args.lr_decay and not args.warmup <= args.decay_start < args.decay_end:
        parser.error("Decay requires warmup <= decay-start < decay-end")
    return {key: getattr(args, key) for key in defaults}


def learning_rate_factor(step, settings):
    update = step + 1
    warmup_factor = min(1.0, update / settings["warmup"])
    if not settings["lr_decay"] or update <= settings["decay_start"]:
        return warmup_factor
    progress = min(1.0, (update - settings["decay_start"]) /
                   (settings["decay_end"] - settings["decay_start"]))
    return settings["min_lr_ratio"] + (1 - settings["min_lr_ratio"]) * (1 + math.cos(math.pi * progress)) / 2


def validate(model, dataset, metadata, batch_size, max_batches=0):
    model.eval()
    device = next(model.parameters()).device
    total = torch.zeros(4, device=device)
    stage_sum = torch.zeros(5, device=device)
    stage_count = torch.zeros(5, device=device)
    count = 0
    with torch.inference_mode():
        for index, batch in enumerate(DataLoader(dataset, batch_size=batch_size, generator=torch.Generator().manual_seed(0))):
            batch = to_device(batch, device)
            error = (predict(model, batch["observation"], metadata) - batch["action"]).abs().float()
            total += error.sum(0)
            count += len(error)
            for stage in range(5):
                mask = batch["stage"] == stage
                stage_sum[stage] += error[mask].mean(-1).sum()
                stage_count[stage] += mask.sum()
            if max_batches and index + 1 >= max_batches:
                break
    if count == 0:
        raise ValueError("Empty validation dataset")
    result = dict(l1=float(total.mean() / count))
    result.update({f"action_{index}_l1": float(value / count) for index, value in enumerate(total)})
    result.update({f"stage_{stage}_l1": float(stage_sum[stage] / stage_count[stage])
                   for stage in range(5) if stage_count[stage] > 0})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=str(CHECKPOINT_DIR / "base.pt"))
    parser.add_argument("--resume")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--run-id")
    parser.add_argument("--checkpoint-dir", default=str(CHECKPOINT_DIR))
    parser.add_argument("--log-dir", default=str(LOG_DIR))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA FP16 for numerical diagnosis")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--accumulation", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--lr-decay", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable cosine decay after the constant-LR phase (default: enabled for new runs)")
    parser.add_argument("--decay-start", type=int, help="First decay boundary in optimizer updates (default: 5000)")
    parser.add_argument("--decay-end", type=int, help="Update at which minimum LR is reached (default: 10000)")
    parser.add_argument("--min-lr-ratio", type=float, help="Minimum LR / base LR (default: 0.1)")
    parser.add_argument("--sampling", choices=["balanced", "random"],
                        help="Equal stage probability or uniform transition sampling (default: balanced)")
    parser.add_argument("--validation-interval", type=int, default=500)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-max-steps", type=int, default=150)
    parser.add_argument("--validation-batches", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.steps, args.validation_interval, args.save_interval, args.threads) < 1:
        parser.error("steps, intervals and threads must be positive")
    if args.eval_episodes < 0 or args.validation_batches < 0:
        parser.error("evaluation episodes and validation batches cannot be negative")
    torch.set_num_threads(args.threads)
    seed_everything(42)
    # 加载入口：新训练读取 convert.py 由 HF 权重转换并适配后的 base.pt；恢复训练读取 --resume。
    # 实际参数赋值在 runtime.load_checkpoint() 的 model.load_state_dict() 中完成，不在这里访问 HF。
    model, payload = load_checkpoint(args.resume or args.base, args.device)
    metadata = payload["metadata"]
    stored = payload["training"] if args.resume else {}
    settings = resolve_settings(args, stored, parser)
    manifest = make_manifest(args.data_dir, metadata["manifest"]["seed"])
    if manifest != metadata["manifest"]:
        raise ValueError("Dataset differs from the converted checkpoint; convert again")
    # 适配自定义任务：NPZ 转为两帧三视角图像、26 维 proprio 和归一化的 4 维专家动作；见 data.py。
    train_dataset = ExpertDataset(args.data_dir, manifest, "train", metadata["statistics"])
    validation_dataset = ExpertDataset(args.data_dir, manifest, "validation", metadata["statistics"])
    # 适配自定义任务：冻结的 T5 不参与优化；更新视觉/Transformer 预训练参数及新建的 proprio、动作模块。
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: learning_rate_factor(step, settings))
    use_amp = torch.device(args.device).type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    generator = torch.Generator().manual_seed(42)
    step = 0
    skipped_updates = stored.get("skipped_updates", 0)
    consecutive_skips = 0
    best_validation, best_success = float("inf"), -1.0
    if args.resume:
        if stored["amp"] != use_amp:
            raise ValueError("Resume must use the same AMP/device type")
        optimizer.load_state_dict(stored["optimizer"])
        scheduler.load_state_dict(stored["scheduler"])
        scaler.load_state_dict(stored["scaler"])
        generator.set_state(stored["sampler_rng"])
        restore_rng(stored["rng"])
        step = stored["step"]
        best_validation, best_success = stored["best_validation"], stored["best_success"]
    if args.steps <= step:
        parser.error(f"--steps must be greater than resumed step {step}")
    run_id = args.run_id or stored.get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.checkpoint_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    import tensorboard.compat
    from tensorboard.compat import tensorflow_stub
    tensorboard.compat.tf = tensorflow_stub
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(str(Path(args.log_dir) / run_id), purge_step=step if args.resume else None)
    sampler = StageBatchSampler(train_dataset, args.batch_size, None,
                                generator, sampling=args.sampling)
    iterator = iter(DataLoader(train_dataset, batch_sampler=sampler, num_workers=0, generator=torch.Generator().manual_seed(0)))

    def save(name):
        save_checkpoint(output_dir / name, model, metadata, step=step, optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), rng=rng_state(),
                        sampler_rng=generator.get_state(), run_id=run_id, best_validation=best_validation,
                        best_success=best_success, amp=use_amp,
                        settings=settings, skipped_updates=skipped_updates)

    def record_skip(reason):
        nonlocal skipped_updates, consecutive_skips
        skipped_updates += 1
        consecutive_skips += 1
        writer.add_scalar("train/skipped_updates", skipped_updates, step)
        writer.add_scalar("train/amp_scale", scaler.get_scale(), step)
        print(f"Skipped update after step={step}: {reason}; AMP scale={scaler.get_scale():.0f}", flush=True)
        if consecutive_skips >= 10:
            save("latest.pt")
            writer.flush()
            raise FloatingPointError("10 consecutive non-finite updates; saved latest.pt for diagnosis")

    try:
        while step < args.steps:
            model.train()
            frozen_text_eval(model)
            optimizer.zero_grad(set_to_none=True)
            start = time.perf_counter()
            total_loss = 0.0
            finite_loss = True
            for _ in range(args.accumulation):
                batch = to_device(next(iterator), args.device)
                with torch.autocast(device_type=torch.device(args.device).type, dtype=torch.float16, enabled=use_amp):
                    predicted = predict(model, batch["observation"], metadata, train=True)
                    # 适配自定义任务：用当前步专家动作监督 L1 行为克隆；五阶段奖励只用于环境评估，不进入损失。
                    loss = (predicted - batch["action"]).abs().mean()
                if not torch.isfinite(loss):
                    discard_nonfinite_update(optimizer, scaler, "loss")
                    finite_loss = False
                    break
                scaler.scale(loss / args.accumulation).backward()
                total_loss += float(loss.detach()) / args.accumulation
            if not finite_loss:
                record_skip("non-finite loss")
                continue
            parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
            grad_norm = prepare_gradients(optimizer, scaler, parameters)
            if grad_norm is None:
                record_skip("non-finite gradients")
                continue
            if any(parameter.grad is not None for name, parameter in model.named_parameters() if ".hf_model." in name):
                raise RuntimeError("Frozen text encoder received gradients")
            for group in ["block_transformer", "observation_tokenizers", "obs_projections", "heads.action"]:
                if not any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                           for name, parameter in model.named_parameters() if group in name and parameter.requires_grad):
                    raise RuntimeError(f"No nonzero gradients in {group}")
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < previous_scale:
                optimizer.zero_grad(set_to_none=True)
                record_skip("GradScaler skipped an overflowed update")
                continue
            scheduler.step()
            step += 1
            consecutive_skips = 0
            for key, value in dict(l1=total_loss, grad_norm=float(grad_norm), lr=optimizer.param_groups[0]["lr"],
                                   samples_per_second=args.batch_size * args.accumulation / (time.perf_counter() - start)).items():
                writer.add_scalar(f"train/{key}", value, step)
            if use_amp:
                writer.add_scalar("system/cuda_memory_mb", torch.cuda.max_memory_allocated() / 2**20, step)
                writer.add_scalar("train/amp_scale", scaler.get_scale(), step)
                writer.add_scalar("train/skipped_updates", skipped_updates, step)
            print(f"step={step} l1={total_loss:.6f} grad_norm={float(grad_norm):.4f}", flush=True)
            if step % args.validation_interval == 0 or step == args.steps:
                metrics = validate(model, validation_dataset, metadata, args.batch_size, args.validation_batches)
                for key, value in metrics.items():
                    writer.add_scalar(f"validation/{key}", value, step)
                if metrics["l1"] < best_validation:
                    best_validation = metrics["l1"]
                    save("best_validation.pt")
            if step % args.save_interval == 0 or step == args.steps:
                if args.eval_episodes:
                    state = rng_state()
                    result = rollout(model, metadata, args.eval_episodes, max_steps=args.eval_max_steps)
                    restore_rng(state)
                    (output_dir / f"evaluation_{step}.json").write_text(json.dumps(result, indent=2))
                    for key in ["success_rate", "mean_return", "mean_length"]:
                        writer.add_scalar(f"evaluation/{key}", result[key], step)
                    for stage, value in result["stage_reach_rates"].items():
                        writer.add_scalar(f"evaluation/stage_{stage}_reach", value, step)
                    for reason, count in result["failure_reasons"].items():
                        writer.add_scalar(f"evaluation/failure/{reason}", count / args.eval_episodes, step)
                    if result["success_rate"] > best_success:
                        best_success = result["success_rate"]
                        save("best_success.pt")
                save("latest.pt")
                writer.flush()
    except torch.cuda.OutOfMemoryError as error:
        raise RuntimeError("CUDA out of memory: reduce --batch-size and retain effective batch using --accumulation") from error
    finally:
        writer.close()
    print(f"Saved native PyTorch checkpoint: {output_dir / 'latest.pt'}")


if __name__ == "__main__":
    main()
