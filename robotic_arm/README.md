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

各实验目录下的 `README.md` 记录了对应实现的参数、观测维度、训练命令和指标说明；训练产生的 checkpoint、monitor 和 TensorBoard 日志位于对应目录的 `runs/` 下。

## 实验结论

当前阶段的总体结果如下：

1. **向量化观测**：`SAC_VecObs` 下的 `reach` 和 `pick_place` 都能快速、稳定收敛，成功率可达到 100%。
2. **三个正交 RGB 观测**：`SAC_PixelObs` 下的 `reach` 可以达到 70% 以上成功率；但 `pick_place` 长时间训练仍不能收敛，甚至无法稳定完成第一个 `APPROACH` 阶段。

这说明当前 PyBullet 物理环境、动作定义和 pick-place 任务状态机本身是可学习的；主要困难来自从 RGB 图像中恢复三维物体位置、抓取姿态和任务阶段，而不是基础任务逻辑无法完成。

## PixelObs pick-place 失败分析

`SAC_PixelObs` 和 `SAC_VecObs` 的 pick-place 核心奖励公式、阶段判定和成功条件基本一致：`APPROACH` 使用末端到物体距离的差分奖励，后续阶段分别使用抬升、运输、放置等进度奖励，并在成功或阶段违规时终止 episode。

两者的学习问题并不等价：

- `SAC_VecObs` 直接获得末端到物体、物体到目标的相对坐标，以及阶段 one-hot、接触、抓取和抬升状态；
- `SAC_PixelObs` 只能从三个 RGB 视图和本体状态中自行识别这些信息，没有物体/目标坐标，也没有显式阶段标志；
- pick-place 还比 reach 多一个抓爪动作维度，物体会被碰撞和抓取移动，接近物体并不等于满足抓取条件。

当前 PixelObs pick-place 日志中，episode 基本全部以 `approach_timeout` 结束，`grasp_rate=0`、`success_rate=0`。因此不是“已经学会接近，只是回报被最终惩罚遮住”，而是还没有形成稳定的视觉接近策略。

`APPROACH` 的超时失败奖励为一次性的 `-5`，确实会压低并遮蔽距离进度在 episode return 中的表现；但它不是唯一根因。即使去掉该项，当前策略也没有产生足够稳定的接近和抓取样本。更核心的问题是：差分距离奖励只告诉 policy “这一步变近还是变远”，没有直接告诉它三维空间中应向哪个方向运动，而视觉策略还要同时探索末端运动、抓爪闭合和可抓取姿态。训练中探索分布较早变得集中，也进一步降低了发现有效 grasp 行为的概率。

阶段预算是防止策略长期停留刷奖励的安全阀，不能替代视觉定位和探索能力。当前 PixelObs 每阶段使用 100 个 RL step、全局 episode 上限为 150；这个预算差异会改变失败回报，但不足以解释从未进入 GRASP 的现象。

## 当前状态

向量观测版本已经验证了任务定义和训练流程；视觉版本验证了三视图 RGB 观测可以支持 reach，但尚不足以支持当前完整 pick-place 任务。后续若继续研究视觉 pick-place，应优先从观测表达、视觉伺服学习信号和有效抓取样本稀缺性入手，并保持已收敛的 `--task reach` 逻辑不变。

