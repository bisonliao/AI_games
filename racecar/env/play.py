"""用键盘试玩 RaceCarEnv。

在项目根目录运行：

    python -m env.play

PyBullet GUI 窗口获得焦点后，按住左/右方向键控制转向，松开按键后方向
回中；按 ``q`` 退出。油门沿用环境中的固定值。撞墙、到达终点或超时后，
脚本会自动开始下一回合。
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pybullet as p

try:
    from .RaceCarEnv import RaceCarEnv
except ImportError:  # 支持直接执行 ``python env/play.py``
    from RaceCarEnv import RaceCarEnv


def _steering_from_keyboard(physics_client: int) -> tuple[float, bool]:
    """返回当前转向值和是否请求退出。"""
    events = p.getKeyboardEvents(physicsClientId=physics_client)

    # q 的 ASCII key code 在 PyBullet 中是 ord('q')。
    if ord("q") in events:
        q_state = events[ord("q")]
        if q_state & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
            return 0.0, True

    left = events.get(p.B3G_LEFT_ARROW, 0) & p.KEY_IS_DOWN
    right = events.get(p.B3G_RIGHT_ARROW, 0) & p.KEY_IS_DOWN
    if left and not right:
        return -1.0, False
    if right and not left:
        return 1.0, False
    return 0.0, False


def play(fps: int = 20) -> None:
    env = RaceCarEnv(render=True, fps=fps)
    try:
        env.reset()
        print("PyBullet 窗口已启动：按住 ←/→ 转向，按 q 退出。")
        frame_time = 1.0 / fps

        while p.isConnected(env.physicsClient):
            started = time.perf_counter()
            steer, should_quit = _steering_from_keyboard(env.physicsClient)
            if should_quit:
                break

            _, reward, terminated, truncated, info = env.step(
                np.asarray([steer], dtype=np.float32)
            )

            if terminated or truncated:
                reason = info.get("termination_reason", "unknown")
                result = "成功" if info.get("is_success", False) else "结束"
                print(
                    f"回合{result}（{reason}）：steps={info['steps']}, "
                    f"position={np.asarray(info['position']).round(2)}, reward={reward:.3f}"
                )
                env.reset()

            # stepSimulation 本身不会按 fps 阻塞，补足实时显示间隔。
            elapsed = time.perf_counter() - started
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyboard controller for RaceCarEnv")
    parser.add_argument("--fps", type=int, default=20, help="simulation/display frequency")
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    play(args.fps)


if __name__ == "__main__":
    main()
