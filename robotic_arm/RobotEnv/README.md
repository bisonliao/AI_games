# RobotEnv

一个尽量少依赖的 PyBullet 机械臂环境原型，当前提供：

- `PandaTabletopEnv(task="reach")`：末端执行器到达随机目标点；
- `PandaTabletopEnv(task="pick_place")`：夹取台面上的随机方块并放到随机目标区域；
- 4 维连续动作：`[dx, dy, dz, gripper]`，范围为 `[-1, 1]`；
- Gymnasium 风格的 `reset()` / `step()` 接口；
- privileged state observation，另外提供可选 `rgb_array` 渲染；
- `heuristic_action()` 脚本控制器，用于在训练前检查运动和奖励逻辑。

## 运行脚本控制器

在项目 Conda 环境中运行：

```bash
python -m RobotEnv.scripts.run_scripted_demo --task reach --episodes 3
python -m RobotEnv.scripts.run_scripted_demo --task pick_place --episodes 3
```

需要查看 GUI 时加上 `--gui`。无桌面或 CI 环境使用默认的 headless DIRECT 模式即可。

GUI 回放默认按 30 FPS 播放；可以用 `--fps` 调慢或调快，例如：

```bash
python -m RobotEnv.scripts.run_scripted_demo \
    --task pick_place --episodes 1 --gui --fps 5
```

`--fps` 只影响 GUI 回放，不会拖慢无 GUI 的运行。

## 接入 RL

环境没有绑定某一个 RL 库，可以直接交给 Gymnasium 兼容的算法。建议先运行脚本控制器确认接触和夹爪闭合稳定，再接 PPO、SAC 或 TD3，并逐步增加物体位置、尺寸、质量和摩擦随机化。
