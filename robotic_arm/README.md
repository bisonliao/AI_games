# PyBullet 机械臂强化学习实验

本项目用于研究基于 PyBullet 的 Panda 机械臂强化学习环境。机械臂在桌面场景中执行两个任务：

- `reach`：到达随机位置的红色物体；
- `pick_place`：抓取红色物体，并将其放到随机位置的绿色目标区域。

训练采用 PyTorch 和 SAC，并支持多个 PyBullet 环境并行 rollout、checkpoint、评估和 TensorBoard 记录。项目刻意保留两种观测方案，用于比较“精确状态输入”和“视觉输入”对任务学习难度的影响。

## 目录

| 目录 | 作用 |
|---|---|
| [`RobotEnv/`](RobotEnv/) | 基础 PyBullet 场景、Panda IK/夹爪控制、桌面物体和目标区域，以及脚本控制器。它不负责 SAC 训练。 |
| [`SAC_VecObs/`](SAC_VecObs/) | 基于精确状态向量的 SAC 实验。包含 observation、奖励、pick-place 阶段状态机、并行环境、训练、评估和 TensorBoard 逻辑。 |
| [`SAC_PixelObs/`](SAC_PixelObs/) | 基于三个正交 RGB 相机视图和机械臂本体状态的 SAC 实验。物体/目标精确坐标和阶段状态不提供给 policy。 |
| [`SAC_BC_PixelObs/`](SAC_BC_PixelObs/) | 在SAC_PixelObs的基础上，克隆teacher的行为（BC），teacher是SAC_VecObs训练得到的checkpoint。 |

各实验目录下的 `README.md` 记录了对应实现的参数、观测维度、训练命令和指标说明；训练产生的 checkpoint、monitor 和 TensorBoard 日志位于对应目录的 `runs/` 下。

## 实验结论

当前阶段的总体结果如下：

1. **向量化观测**：`SAC_VecObs` 下的 `reach` 和 `pick_place` 都能快速、稳定收敛，成功率可达到 100%。
2. **三个正交 RGB 观测**：`SAC_PixelObs` 下的 `reach` 可以达到 70% 以上成功率；但 `pick_place` 长时间训练仍不能收敛，甚至无法稳定完成第一个 `APPROACH` 阶段。
3. **三个正交 RGB 观测+BC预训练**：pick_place可以达到90%以上的成功率, 需要训练至少8M时间步。



