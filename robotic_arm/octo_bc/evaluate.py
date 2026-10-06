"""Evaluate a native Torch checkpoint in the unchanged staged PixelTaskEnv."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from .runtime import load_checkpoint, to_device, frozen_text_eval, VIEWS
from .data import observation_tensors, denormalize_action
import numpy as np
import torch


def task_batch(metadata, size, device):
    return to_device({key: {name: value.expand(size, *value.shape[1:]) for name, value in values.items()}
                      for key, values in metadata["task"].items()}, device)


def predict(model, observation, metadata, train=False):
    device = next(model.parameters()).device
    task = task_batch(metadata, observation["proprio"].shape[0], device)
    # 适配自定义任务：与转换时一致，为每个相机补零 goal RGB，保持预训练 stem 输入形状。
    for view in VIEWS:
        task[f"image_{view}"] = torch.zeros_like(observation[f"image_{view}"][:, 0])
        task["pad_mask_dict"][f"image_{view}"] = torch.zeros(len(observation["proprio"]), dtype=torch.bool, device=device)
    outputs, _ = model(observation, task, observation["timestep_pad_mask"], train=train, transformer_only=True)
    # 适配自定义任务：只取最新历史时刻的单步 4 维动作；此预测函数同时用于训练和评估。
    return model.module.heads["action"](outputs, train=train)[:, -1, 0]


def rollout(model, metadata, episodes=100, seed=100000, gui=False, max_steps=150):
    from robot_arm.SAC_PixelObs.env import PixelTaskEnv

    if episodes < 1 or max_steps < 1:
        raise ValueError("episodes and max_steps must be positive")
    model.eval()
    frozen_text_eval(model)
    device = next(model.parameters()).device
    env = PixelTaskEnv(task="pick_place", frame_stack=2, render_mode="human" if gui else None,
                       max_episode_steps=max_steps)
    records = []
    try:
        with torch.inference_mode():
            for episode in range(episodes):
                started = time.perf_counter()
                observation, info = env.reset(seed=seed + episode)
                previous_proprio = observation["proprio"].copy()
                reached = {int(info["stage_index"])}
                episode_return = 0.0
                for step in range(max_steps):
                    proprio = np.stack([previous_proprio, observation["proprio"]])
                    batch = observation_tensors(observation["image"], proprio)
                    batch = to_device({key: {name: value.unsqueeze(0) for name, value in values.items()}
                                       if isinstance(values, dict) else values.unsqueeze(0)
                                       for key, values in batch.items()}, device)
                    normalized = predict(model, batch, metadata)[0].float().cpu().numpy()
                    # 适配自定义任务：还原环境动作尺度并裁剪到 [-1,1]，交给保留五阶段逻辑的环境执行。
                    action = np.clip(denormalize_action(normalized, metadata["statistics"]), -1, 1).astype(np.float32)
                    previous_proprio = observation["proprio"].copy()
                    observation, reward, terminated, truncated, info = env.step(action)
                    episode_return += reward
                    reached.add(int(info["stage_index"]))
                    if terminated or truncated:
                        break
                records.append(dict(seed=seed + episode, success=bool(info.get("success", False)),
                                    episode_return=episode_return, length=step + 1,
                                    stages=sorted(reached), failure_reason=info.get("failure_reason", "") or
                                    ("time_limit" if not info.get("success", False) else "")))
                print(f"evaluation episode={episode + 1}/{episodes} success={records[-1]['success']} "
                      f"steps={step + 1} seconds={time.perf_counter() - started:.1f}", flush=True)
    finally:
        env.close()
    return dict(success_rate=float(np.mean([record["success"] for record in records])),
                mean_return=float(np.mean([record["episode_return"] for record in records])),
                mean_length=float(np.mean([record["length"] for record in records])),
                stage_reach_rates={str(stage): float(np.mean([stage in record["stages"] for record in records])) for stage in range(5)},
                failure_reasons=dict(Counter(record["failure_reason"] for record in records if not record["success"])),
                episodes=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--output", default=None)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    model, payload = load_checkpoint(args.checkpoint, args.device)
    result = rollout(model, payload["metadata"], args.episodes, args.seed, args.gui, args.max_steps)
    output = Path(args.output) if args.output else Path(args.checkpoint).with_suffix(".evaluation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
