from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
import time
from typing import Any

import numpy as np

from env.gui_keys import exit_requested
from env.jump_env import JumpEnv, JumpEnvConfig
from td3.agent import BanditTD3


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def resolve_checkpoint(name: str, run_root: str = "runs") -> Path:
    """Resolve a direct checkpoint path or find its name below run_root."""
    requested = Path(name).expanduser()
    direct_candidates = [requested, Path(run_root) / requested]
    if requested.suffix == "":
        direct_candidates.extend(
            [
                requested.with_suffix(".pt"),
                (Path(run_root) / requested).with_suffix(".pt"),
            ]
        )
    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate.resolve()
        nested_checkpoint = candidate / "checkpoint.pt"
        if candidate.is_dir() and nested_checkpoint.is_file():
            return nested_checkpoint.resolve()

    names = {requested.name}
    if requested.suffix == "":
        names.add(f"{requested.name}.pt")
    root = Path(run_root)
    matches = (
        [path for path in root.rglob("*") if path.is_file() and path.name in names]
        if root.is_dir()
        else []
    )
    if root.is_dir():
        for directory in root.rglob("*"):
            if not directory.is_dir():
                continue
            if directory.name == requested.name or directory.name.endswith(
                f"-{requested.name}"
            ):
                nested_checkpoint = directory / "checkpoint.pt"
                if nested_checkpoint.is_file():
                    matches.append(nested_checkpoint)
    if not matches:
        raise FileNotFoundError(
            f"Checkpoint '{name}' does not exist and was not found below '{run_root}'"
        )
    return max(matches, key=lambda path: path.stat().st_mtime).resolve()


def env_config_from_metadata(metadata: dict[str, Any]) -> JumpEnvConfig:
    values = metadata.get("env_config", {})
    if not isinstance(values, dict):
        return JumpEnvConfig()
    # Old checkpoints predate pixel observations and the nonlinear charge curve,
    # so they imply vector mode and the original linear p=1 physics.
    values = dict(values)
    values.setdefault("observation_mode", "vector")
    values.setdefault("charge_exponent", 1.0)
    allowed = {field.name for field in fields(JumpEnvConfig)}
    return JumpEnvConfig(
        **{key: value for key, value in values.items() if key in allowed}
    )


def _wait_for_gui(env: JumpEnv, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and env.is_connected:
        events = env.get_keyboard_events()
        if exit_requested(events):
            return False
        time.sleep(1.0 / 120.0)
    return env.is_connected


def evaluate_gui(
    checkpoint: str,
    *,
    run_root: str = "runs",
    device: str = "cpu",
    episodes: int = 0,
    seed: int = 2_000_000,
    speed: float = 1.0,
    pre_jump_delay: float = 0.7,
    result_delay: float = 1.5,
    show_charge: bool = True,
) -> None:
    checkpoint_path = resolve_checkpoint(checkpoint, run_root)
    agent, metadata = BanditTD3.from_checkpoint(checkpoint_path, device=device)
    agent.actor.eval()
    config = env_config_from_metadata(metadata)
    env = JumpEnv(
        config=config,
        render_mode="human",
        playback_speed=speed,
    )
    successes = 0
    completed = 0
    print(f"加载 checkpoint：{checkpoint_path}")
    print("Esc 或 Ctrl+C 退出 GUI 评估。")
    try:
        while env.is_connected and (episodes <= 0 or completed < episodes):
            observation, info = env.reset(seed=seed + completed)
            action = agent.act(observation[None, :])[0]
            hold_time = (
                (float(np.clip(action[0], -1.0, 1.0)) + 1.0)
                * 0.5
                * config.max_hold_seconds
            )
            print(
                f"第 {completed + 1} 局：距离={info['target_distance']:.3f}m，"
                f"agent蓄力={hold_time:.3f}s"
            )
            if not _wait_for_gui(env, pre_jump_delay):
                break
            if show_charge and not _wait_for_gui(env, hold_time):
                break

            _, reward, _, _, final_info = env.step(action)
            completed += 1
            success = bool(final_info["is_success"])
            successes += int(success)
            result = "成功" if success else "失败"
            env.show_message(
                f"{result}! success={successes}/{completed}",
                color=(0.2, 1.0, 0.2) if success else (1.0, 0.2, 0.2),
                duration=result_delay,
            )
            print(
                f"第 {completed} 局 {result}：reward={reward:.3f}，"
                f"落点误差={final_info['landing_error']:.3f}m，"
                f"累计成功率={successes / completed:.2%}"
            )
            if not _wait_for_gui(env, result_delay):
                break
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
    if completed:
        print(f"共演示 {completed} 局，成功率 {successes / completed:.2%}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="加载 TD3 checkpoint 并在 PyBullet GUI 中演示 agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "checkpoint", help="checkpoint 路径、文件名或时间戳 run 名称"
    )  # 非路径名称会在 run-root 下递归查找
    parser.add_argument(
        "--run-root", default="runs", help="按名称查找 checkpoint 的根目录"
    )  # 多个匹配项中选择最近修改的文件
    parser.add_argument(
        "--device", default="cpu", help="模型加载和推理使用的 PyTorch device"
    )  # CPU 可加载 CUDA 训练得到的 checkpoint
    parser.add_argument(
        "--episodes", type=int, default=0, help="演示局数；0 表示持续到主动退出"
    )  # Esc、Q 或 Ctrl+C 均可退出
    parser.add_argument(
        "--seed", type=int, default=2_000_000, help="第一局的环境随机种子"
    )  # 后续演示局逐一递增
    parser.add_argument(
        "--speed",
        type=_positive_float,
        default=1.0,
        help="物理回放倍速；0.5 表示半速慢放",
    )  # 不改变环境动力学或 agent 动作
    parser.add_argument(
        "--pre-jump-delay", type=float, default=0.7, help="显示新平台后、蓄力前的停留秒数"
    )  # 让观察者先看清本局距离
    parser.add_argument(
        "--result-delay", type=float, default=1.5, help="成功或失败画面的保留秒数"
    )  # 延迟结束后 reset 下一局
    parser.add_argument(
        "--no-show-charge",
        action="store_true",
        help="不按预测蓄力时长等待，直接播放飞行",
    )  # 只省略演示等待，不改变传给环境的 action
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evaluate_gui(
        args.checkpoint,
        run_root=args.run_root,
        device=args.device,
        episodes=args.episodes,
        seed=args.seed,
        speed=args.speed,
        pre_jump_delay=args.pre_jump_delay,
        result_delay=args.result_delay,
        show_charge=not args.no_show_charge,
    )


if __name__ == "__main__":
    main()
