# SAC_PixelObs

这是基于摄像头观测的 SAC v1 原型。它复用 `SAC_VecObs` 已验证的动作、PyBullet 物理、pick-place 阶段状态机、成功条件和防刷分逻辑，但 policy 不接收物体/目标精确坐标。

## Observation

每个时间点渲染三个同步的正交 RGB 视图：

```text
xy：沿 Z 轴俯视，图像横纵轴对应 X/Y
xz：沿 Y 轴前视，图像横纵轴对应 X/Z
yz：沿 X 轴侧视，图像横纵轴对应 Y/Z
```

默认每张图为 `96×96`。三个视图每个有 RGB 三通道，因此单帧图像是 9 通道；当前默认使用单帧观测，送入 policy 的 `image` shape 为 `(9, 96, 96)`。如需通过运动趋势补充时间信息，可以用 `--frame-stack` 显式增加帧数。

`camera-scale` 表示每张图在相机目标平面覆盖的实际宽度和高度（米），默认
`1.0`。三个相机都以 `[0.52, 0.0, 0.30]` 为中心，因此默认视野覆盖约
`x=0.02…1.02`、`y=-0.50…0.50`、`z=-0.20…0.80`：这包含夹爪控制器的完整
工作范围及随机物体/目标范围。相机通过远距离窄视场透视近似正交投影，因为
PyBullet 的 `computeProjectionMatrix` 本身不是正交矩阵。96×96 下会保留红块和
绿色目标的有效像素；测试会防止相机参数再次把目标压缩到零像素。

`proprio` 是 26 维归一化向量：

```text
维度  0.. 6：Panda 7 个关节角（除以 pi）
维度  7..13：Panda 7 个关节角速度（除以 10 并裁剪）
维度 14..16：末端执行器 XYZ 位置（按控制工作空间归一化到约 [-1, 1]）
维度 17..19：末端执行器 XYZ 线速度（单位尺度为 1 m/s，并裁剪到 [-5, 5]）
维度 20..21：夹爪开口和开口变化
维度 22..25：上一个 action；reach 的第 4 维固定为 0
```

以下字段不会进入 actor observation：物体坐标、目标坐标、阶段 one-hot、`ever_grasped`、`ever_lifted` 和阶段计数器。环境内部仍可使用精确仿真状态计算 reward 和 termination。

这次 observation 已由旧版的 `64×64 + 20 维 proprio` 改为
`96×96 + 26 维 proprio`，网络输入 shape 已变化，因此旧版 checkpoint 不能直接在
新版环境中继续训练或评估；需要启动一次新训练。

## Visual encoder

三个视图和每个历史帧在各自网络内部使用共享 CNN encoder，视觉 feature 与
proprio feature 融合后输入 SAC actor/critic。虽然图像在 observation 中按
`frame × view × RGB` 通道存储，网络实际按 `frame × view × RGB` 独立编码，避免把
三个几何含义不同的视图当成普通自然图像通道。

actor 和 critic 默认使用各自独立的 feature extractor，critic target 也保持独立。
原因是 SB3 在 `share_features_extractor=True` 时会从 critic optimizer 中明确排除
共享 extractor 参数，导致 TD loss 无法训练视觉 encoder；此前策略因此容易依赖
proprio 学习盲扫轨迹。独立后，critic encoder 会直接接受 TD loss 梯度，actor 则
通过已经对图像敏感的 Q 函数学习状态相关动作。若显存确实不足，仍可用
`--share-features-extractor` 恢复旧的低显存模式，但不建议将其用于正式视觉训练。

当前训练和评估都不使用随机平移增强。三个相机视图保持严格的像素几何关系，避免各视图独立平移干扰跨视图定位。

新版视觉 head 使用
`Linear → LayerNorm(elementwise_affine=False) → LeakyReLU`。旧结构中的可学习
LayerNorm bias 曾逐渐把全部视觉 feature 推到负区，之后 ReLU 将其全部截为 0，且
梯度无法恢复。非仿射 LayerNorm 不再具有这种整体负向平移能力；LeakyReLU 在负区
仍保留梯度。训练入口默认并在 `config.json` 中保存 `visual_head_version=2`，也可用
`--visual-head-version` 显式选择；旧 checkpoint 仍按 version 1 结构加载，但必须新开
训练才能获得该修复。

每次常规 checkpoint 都会复用当前 rollout 已经返回的一小批图片执行一次无梯度
视觉探针，不额外渲染或反向传播，并在 checkpoint 的准确 step 上报：

```text
diagnostics/visual_zero_fraction  绝对值不超过 1e-8 的视觉 feature 比例
diagnostics/visual_batch_std      同一视图/feature 跨并行环境的平均标准差
diagnostics/visual_relative_batch_std  batch_std / RMS
diagnostics/visual_rms            视觉 feature 的 RMS 幅度
diagnostics/visual_inactive       是否满足全零、近零或跨样本无变化条件（0/1）
diagnostics/critic_visual_*       critic encoder 对应的上述五项指标
diagnostics/image_action_delta_mean  固定 proprio、循环置换图片后的平均动作变化
diagnostics/image_action_delta_max   上述动作变化的最大值
diagnostics/image_action_insensitive 图片变化引起的平均动作变化是否低于 1e-4（0/1）
```

正常情况下 `visual_inactive` 应始终为 0。`visual_zero_fraction` 对 LeakyReLU 通常也应
接近 0；`visual_batch_std` 用于发现 feature 虽非零但已经不再随输入变化的退化。
当 `n-actors=1` 时跨环境标准差按定义记为 0，判定逻辑不会单独因此报 inactive；
默认的多环境训练才会用该项参与判定。`image_action_insensitive` 比单纯的 feature
非零检查更严格：它直接检验图片变化是否真正传播到 policy 输出；训练早期为 1 并不
异常，但随着策略学习应转为 0，且 `image_action_delta_mean` 应出现数量级增长。

## 运行

短 smoke test：

```bash
python -m SAC_PixelObs.train \
    --task reach \
    --n-actors 1 \
    --image-size 96 \
    --frame-stack 2 \
    --total-timesteps 10000
```

Pick-place 训练：

```bash
python -m SAC_PixelObs.train \
    --task pick_place \
    --n-actors 12 \
    --device cuda \
    --total-timesteps 10000000
```

当前默认 profile 为 `n-actors=12`、`image-size=96`、`frame-stack=1`、`batch-size=128`、`buffer-size=100000`。SB3 的 Dict replay buffer 会分别保存当前和下一帧 observation；仅两份 RGB 数组理论上约占 16.6 GB RAM，尚未包含 PyBullet worker、模型、proprio 和其他进程内存。39 GB 主机需要关注实际内存余量。独立actor/critic encoder也会比旧共享模式增加一份encoder参数、梯度和optimizer state；如果发生内存压力，应优先显式减小`--batch-size`，其次减小`--buffer-size`或`--n-actors`，不建议为了省显存重新共享encoder。

默认 `gradient_steps=-1` 是有意为并行环境设置的。SB3 的 `train_freq=1 step`
表示先让每个并行环境各产生一条 transition；12个环境一次会新增12条数据。若固定
`gradient_steps=1`，则每12条新数据只更新一次，update/data ratio约为`1/12`；此前
视觉训练到约585k transition时实际上只有约48k次参数更新。`-1` 会令SB3按本轮
实际新增transition数执行更新，12环境即做12次，使UTD约为1。TensorBoard新增：

```text
train/n_updates             已完成的梯度更新总数
train/update_to_data_ratio  learning_starts之后累计update/transition比率
```

另外，每 `5000` 个 transition（可通过 `--entropy-log-freq` 调整）会用刚刚由
policy 采样并送入环境的动作上报：

```text
rollout/action_entropy  当前 squashed action 分布的 Monte Carlo 熵估计 -log pi(a|s)
```

该指标包含 tanh 变换的概率密度修正，并对全部动作维度求和，因而是真实动作分布
微分熵的采样估计；连续分布的微分熵允许为负。它与 `train/ent_coef` 不同：后者只是
actor loss 中熵项的自动学习权重 alpha，不能单独用于判断 policy 是否已经接近确定性。
熵越低通常表示动作分布越集中，但应结合任务表现和 `ent_coef` 的反馈方向一起判断。

预期代价是环境步吞吐下降、GPU利用率上升；这是用更多计算换取视觉表示学习。若硬件
无法承受，可以显式使用`--gradient-steps 4`作为折中（12环境时UTD约为1/3），但不
建议重新使用1作为正式视觉训练默认值。

训练中的 rollout action 前向传播由 learner 进程批量执行，而不是由
`SubprocVecEnv` 的 PyBullet worker 执行。只要启动参数使用 `--device cuda`（或
`--device auto` 且 CUDA 可用），actor、critic 的 CNN 都在 GPU 上；worker 进程只
负责环境 stepping 和图像渲染。

## 吞吐耗时统计

训练主进程会累计四个阶段的 wall-clock 时间，并以低频率写入 TensorBoard：

```text
time/env_step       VecEnv 等待返回，包含进程通信、PyBullet stepping 和相机渲染
time/predict        rollout action 的 policy 前向和采样
time/replay_sample  replay buffer 采样及其 CPU→GPU 准备
time/train          SAC 梯度更新（已扣除 replay_sample）
```

默认每 `5000` 个环境步写一次，可通过 `--time-log-freq` 调整。统计只在计时点累加
数值，平时不写 TensorBoard；每个阶段仅增加 `perf_counter` 调用，不执行额外的
GPU 同步，因此不会为了测量明显拖慢训练。四条曲线均位于 TensorBoard 的 `time/`
命名空间下。

评测使用独立的常驻 standby 进程。训练进程在评测间隔只保存一个模型快照并把任务
放入队列，然后立即继续 rollout/update；评测进程用 `SAC.load(..., device="cpu")`
在 CPU 上运行，不会与训练 GPU 争用。结果写入 `eval/results.jsonl`、
`eval/evaluations.npz`，并上报到 TensorBoard 的 `eval/*` 指标；最佳模型仍保存为
`best_model/best_model.zip`。训练结束时会短暂等待 worker 收尾，超时则安全终止。
`--eval-freq` 表示全局 transition 数量，已经包含所有并行环境，因此不会再除以
`n-actors`；默认值 100000 就是大约每 100000 条 transition 触发一次异步评测。
评测队列最多保留一个待处理任务；如果 evaluator 尚在评测旧模型，之后到达的多个
任务只保留最新 checkpoint，避免积压后持续上报过时策略。评测结果仍由 learner
主进程写入已有的 TensorBoard writer，并显式使用被评 checkpoint 的真实 step 作为
横轴，而不是结果返回时 learner 已经运行到的 step。
`reach` 只上报通用的 `eval/success_rate`、`eval/mean_reward` 和
`eval/mean_ep_length`；`eval/lift_rate`、`eval/mean_final_stage` 仅用于
`pick_place`。

训练 episode 指标同样按任务区分：`reach` 的 `task/*` 只包含
`success_rate`、`failure_rate` 和 `truncation_rate`；`grasp_rate`、
`lift_rate`、`final_stage`、`grasp_lost_rate`、`drop_rate` 和
`stage_timeout_rate` 只在 `pick_place` 中上报。

默认使用 `forkserver`，如果运行环境禁止 Unix socket，可显式使用
`--start-method fork`。

评估：

```bash
python -m SAC_PixelObs.evaluate \
    --task pick_place \
    --checkpoint SAC_PixelObs/runs/<run_name>/final_model.zip \
    --episodes 100
```

训练时异步评测默认也使用 100 个 episode（`--eval-episodes 100`），用于显著降低
10 回合评测带来的成功率抽样噪声。每个 checkpoint 使用同一个固定 seed 开始的
确定性 episode 序列，因而不同 checkpoint 的结果可以直接横向比较。CPU evaluator
的单次评测时间会相应增加，但仍与 learner 异步执行。

TensorBoard：

```bash
tensorboard --logdir SAC_PixelObs/runs
```

Reach 重点看 `eval/success_rate`、`eval/mean_reward` 和
`task/success_rate`；pick-place 再重点关注 `task/lift_rate` 和
`task/final_stage`。不要只看 shaping reward。

## 摄像头和训练阶段

第一版使用固定正交相机和简单外观，先验证多视图视觉 reach，再训练 pick-place。之后再逐步加入相机位姿、光照、材质、背景和图像噪声随机化。若要迁移到真实相机，还需要用真实透视相机完成内外参标定、同步、畸变和延迟建模。
