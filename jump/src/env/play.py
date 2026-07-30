from __future__ import annotations

import argparse
import time

from env.gui_keys import SPACE_KEY, exit_requested, was_released, was_triggered
from env.jump_env import JumpEnv, JumpEnvConfig


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _wait_for_gui(env: JumpEnv, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and env.is_connected:
        events = env.get_keyboard_events()
        if exit_requested(events):
            return False
        time.sleep(1.0 / 120.0)
    return env.is_connected


def play(
    *,
    speed: float = 1.0,
    seed: int = 0,
    episodes: int = 0,
    result_delay: float = 1.5,
) -> None:
    """Open the PyBullet GUI and control jumps with the space bar."""
    config = JumpEnvConfig()
    env = JumpEnv(
        config=config,
        render_mode="human",
        playback_speed=speed,
    )
    completed = 0
    held_since: float | None = None
    print("空格键按下开始蓄力，松开后跳跃；Esc 或 Ctrl+C 退出。")
    try:
        _, info = env.reset(seed=seed)
        print(f"第 1 局：平台距离 {info['target_distance']:.3f} m")
        while env.is_connected and (episodes <= 0 or completed < episodes):
            events = env.get_keyboard_events()
            if exit_requested(events):
                break

            if was_triggered(events, SPACE_KEY) and held_since is None:
                held_since = time.monotonic()
                print("蓄力中……")

            if was_released(events, SPACE_KEY) and held_since is not None:
                raw_hold = time.monotonic() - held_since
                held_since = None
                action = config.action_from_hold_time(raw_hold)
                _, reward, _, _, final_info = env.step(action)
                completed += 1
                success = bool(final_info["is_success"])
                shown_hold = float(final_info["hold_time_s"])
                result = "成功" if success else "失败"
                env.show_message(
                    f"{result}! hold={shown_hold:.2f}s",
                    color=(0.2, 1.0, 0.2) if success else (1.0, 0.2, 0.2),
                    duration=result_delay,
                )
                print(
                    f"第 {completed} 局 {result}：按键 {raw_hold:.3f}s，"
                    f"有效蓄力 {shown_hold:.3f}s，reward={reward:.3f}，"
                    f"落点误差={final_info['landing_error']:.3f}m"
                )
                if episodes > 0 and completed >= episodes:
                    _wait_for_gui(env, result_delay)
                    break
                if not _wait_for_gui(env, result_delay):
                    break
                _, info = env.reset(seed=seed + completed)
                print(
                    f"第 {completed + 1} 局："
                    f"平台距离 {info['target_distance']:.3f} m"
                )
            time.sleep(1.0 / 120.0)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用空格键人工操作 PyBullet 跳一跳",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--speed",
        type=_positive_float,
        default=1.0,
        help="物理回放倍速；0.5 表示半速慢放",
    )  # 只影响释放后的飞行动画
    parser.add_argument(
        "--seed", type=int, default=0, help="第一局的平台随机种子"
    )  # 每完成一局自动递增
    parser.add_argument(
        "--episodes", type=int, default=0, help="游戏局数；0 表示持续到主动退出"
    )  # Esc、Q 或 Ctrl+C 均可退出
    parser.add_argument(
        "--result-delay", type=float, default=1.5, help="每局结果画面的保留秒数"
    )  # 延迟结束后自动 reset 下一局
    return parser


def main() -> None:
    args = build_parser().parse_args()
    play(
        speed=args.speed,
        seed=args.seed,
        episodes=args.episodes,
        result_delay=args.result_delay,
    )


if __name__ == "__main__":
    main()
