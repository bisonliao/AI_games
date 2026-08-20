# SAC_PixelObs

这是基于摄像头观测的 SAC v1 原型。它复用 `SAC_VecObs` 已验证的动作、PyBullet 物理、pick-place 阶段状态机、成功条件和防刷分逻辑，但 policy 不接收物体/目标精确坐标。

## Observation

每个时间点渲染三个同步的正交 RGB 视图：

```text
xy：沿 Z 轴俯视，图像横纵轴对应 X/Y
xz：沿 Y 轴前视，图像横纵轴对应 X/Z
yz：沿 X 轴侧视，图像横纵轴对应 Y/Z
```

默认每张图为 `96×96`。三个视图每个有 RGB 三通道，因此单帧图像是 9 通道；默认堆叠最近 2 帧，送入 policy 的 `image` shape 为 `(18, 96, 96)`。堆叠用于让 policy 观察物体和夹爪的运动趋势。

`proprio` 是 20 维归一化向量：

```text
Panda 7 个关节角（除以 pi）
Panda 7 个关节角速度（缩放并裁剪）
夹爪开口和开口变化
上一个 action
```

以下字段不会进入 actor observation：物体坐标、目标坐标、阶段 one-hot、`ever_grasped`、`ever_lifted` 和阶段计数器。环境内部仍可使用精确仿真状态计算 reward 和 termination。

## Visual encoder

三个视图和每个历史帧使用共享 CNN encoder，视觉 feature 与 proprio feature 融合后输入 SAC actor/critic。虽然图像在 observation 中按 18 个通道存储，网络实际按 `frame × view × RGB` 独立编码，避免把三个几何含义不同的视图当成普通自然图像通道。训练配置还共享 actor 和 critic 的视觉 encoder（target critic 保持独立），以降低显存和卷积计算量。

训练模式下 encoder 对每个 RGB crop 使用随机平移增强（默认 padding 为 4 pixels），评估模式关闭增强。这是轻量的 DrQ 风格正则化，用于降低 policy 对相机像素精确位置的过拟合；它不会改变 proprioception，也不会改变环境 reward。

## 运行

短 smoke test：

```bash
python -m SAC_PixelObs.train \
    --task reach \
    --n-actors 1 \
    --image-size 64 \
    --frame-stack 2 \
    --total-timesteps 10000
```

Pick-place 训练：

```bash
python -m SAC_PixelObs.train \
    --task pick_place \
    --n-actors 8 \
    --device cuda \
    --total-timesteps 10000000
```

默认 profile 针对 8 GiB GPU 和约 39 GB RAM：`image-size=96`、`frame-stack=2`、`batch-size=128`、`buffer-size=50000`。其中 replay buffer 同时保存当前和下一帧 observation，图像部分约占 15.5 GiB；不要在这台机器上恢复旧的 `128×128、3 帧、100k buffer` 默认组合。若显存仍紧张，可再使用 `--batch-size 64 --image-size 84`；若 RAM 紧张，可使用 `--buffer-size 30000`。

默认使用 `forkserver`，如果运行环境禁止 Unix socket，可显式使用
`--start-method fork`。

评估：

```bash
python -m SAC_PixelObs.evaluate \
    --task pick_place \
    --checkpoint SAC_PixelObs/runs/<run_name>/final_model.zip \
    --episodes 10
```

TensorBoard：

```bash
tensorboard --logdir SAC_PixelObs/runs
```

重点看 `eval/success_rate`、`task/lift_rate` 和 `task/final_stage`，不要只看 shaping reward。

## 摄像头和训练阶段

第一版使用固定正交相机和简单外观，先验证多视图视觉 reach，再训练 pick-place。之后再逐步加入相机位姿、光照、材质、背景和图像噪声随机化。若要迁移到真实相机，还需要用真实透视相机完成内外参标定、同步、畸变和延迟建模。
