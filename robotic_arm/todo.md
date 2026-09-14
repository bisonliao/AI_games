# 视觉 Pick-Place 方案调研与实现路线

## 目标

项目目标是让 PyBullet Panda 机械臂基于三视图 RGB 和 proprioception 稳定完成 `pick_place`。当前状态向量 SAC 已收敛，而视觉 SAC 主要停留在 `APPROACH`，因此优先解决视觉表示、探索和专家数据分布偏移问题。本文档记录候选方法及最终选定的低成本实现路线；当前阶段只规划方案，不修改训练代码。

## 候选方法与论文/代码

### DrQ-v2 / DrQ

- 论文：Yarats 等，《Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning》（2021）。
- 核心：对 replay 中的图像做随机裁剪/平移增强，并结合 off-policy actor-critic、n-step return 和探索噪声衰减。
- 影响力：视觉连续控制的重要基线，作者来自 Facebook AI/NYU 等团队。
- 代码：[facebookresearch/drqv2](https://github.com/facebookresearch/drqv2)，官方 PyTorch 实现，MIT license。
- 适配性：与当前连续动作 SAC、图像 encoder 和 replay buffer 结构接近；可只移植轻量图像增强，避免重写整个算法。
- 成本与风险：中等；增强必须保持多视图几何关系，不能对不同相机独立破坏空间对应。

### RLPD（Reinforcement Learning with Prior Data）

- 论文：Ball 等，《Efficient Online Reinforcement Learning with Offline Data》（2023）。
- 核心：将专家或次优 prior 数据与在线数据共同用于 off-policy RL；论文报告在多类基准上最高约 2.5 倍改进，并强调数据采样、critic 更新归一化和多 critic 对稳定性的作用。
- 影响力：Berkeley/Sergey Levine 团队，论文发表于 PMLR。
- 代码：[ikostrikov/rlpd](https://github.com/ikostrikov/rlpd)，官方实现。
- 适配性：可以直接利用已收敛的 `SAC_VecObs` 专家轨迹，缓解视觉 pick-place 中 GRASP 样本稀缺。
- 成本与风险：中等至较高；完整复现 critic ensemble 会增加显存和代码复杂度，因此本项目第一版只采用“专家/在线 replay 混合”的简化思想。

### AWAC（Advantage Weighted Actor-Critic）

- 论文：[AWAC: Accelerating Online Reinforcement Learning with Offline Datasets](https://arxiv.org/abs/2006.09359)。
- 核心：用 critic advantage 对行为克隆损失加权，从示范数据初始化策略并在线微调。
- 影响力：Berkeley/Levine 团队；在灵巧手、抽屉开启和阀门旋转等机器人任务中验证。
- 代码与项目页：[awacrl.github.io](https://awacrl.github.io/)，另有 [PyTorch 社区实现](https://github.com/hari-sikchi/AWAC)。
- 适配性：适合少量专家轨迹；思想可用于 BC 后的策略更新。
- 成本与风险：中等。第一版不单独实现 AWAC，以免引入第二套 actor 更新逻辑；若简单 BC+SAC 不稳定，再考虑采用 AWAC 的优势加权更新。

### 行为克隆（BC）与 DAgger

- BC 直接拟合观测到动作，工程复杂度最低，可用状态专家自动生成 RGB-动作数据。
- DAgger 通过当前策略访问的新观测调用专家给出纠正动作，缓解 BC 的 covariate shift。
- 本项目在仿真中可以直接查询状态专家，因此若出现明显分布偏移，可实现无人工标注的简化 DAgger。
- 风险：BC/DAgger 优化的是专家动作一致性，不能单独优化最终抓取成功率，仍需在线 RL 微调。

### Transporter Networks

- 论文与代码：[google-research/ravens](https://github.com/google-research/ravens)，CoRL 2020。
- 核心：从图像预测抓取点到放置点的空间对应，专门面向视觉抓取放置。
- 优点：空间定位能力强，且官方环境就是 PyBullet。
- 缺点：动作表示和当前 Panda 末端增量控制、阶段状态机不一致；改造成现有 SAC actor 成本较高。
- 结论：作为后续分层抓取/放置模块备选，暂不作为第一版端到端方案。

### Diffusion Policy / DPPO

- Diffusion Policy 论文：[Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)；官方代码：[real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy)。
- DPPO 项目：[diffusion-il](https://github.com/dhruvsreenivas/diffusion-il)。
- 优点：能表达多模态、长时序动作，在机器人模仿学习中影响力较大。
- 缺点：需要较多高质量轨迹，训练和推理复杂度明显高于 BC/SAC；对当前 toy reach 和首个 pick-place 目标过重。
- 结论：暂不实现。

### RoboBase 与 Octo

- [RoboBase](https://github.com/robobase-org/robobase)：面向多视图视觉、本体状态和向量化机器人环境的综合基线，适合参考 encoder、增强和离线 RL 实现。
- [Octo](https://octo-models.github.io/)：大规模 Open X-Embodiment 预训练通用策略，影响力高但需要动作/观测空间适配和较大计算资源。
- 结论：作为资料和后续对照，不纳入近期实现。

## 最终选定的低成本完整方案

### 阶段 A：自动生成专家数据

1. 使用已收敛的 `SAC_VecObs` 状态策略，在随机物体/目标和随机初始姿态下运行 pick-place。
2. 保存每个时间步的三视图 RGB、proprio、动作、终止原因和成功结果。
3. 优先保留成功轨迹；同时保留少量接近成功的失败轨迹用于后续在线 replay。
4. 阶段标签、物体/目标坐标等仿真真值只用于数据筛选和诊断，不输入最终视觉策略。

### 阶段 B：视觉 BC 预训练

1. 输入保持 `SAC_PixelObs` 的 RGB + proprio 形式，可先使用 `frame-stack=1`，必要时增加短历史帧。
2. 输出保持现有连续动作空间，包括末端位移/姿态控制和夹爪动作。
3. 使用监督动作损失训练视觉 actor，使其先学会 APPROACH、GRASP、LIFT、TRANSPORT、PLACE 的基本动作分布。
4. 在固定评估场景上确认 BC 能进入 GRASP 并产生一定 lift 行为后，才进入在线 RL。

### 阶段 C：BC 初始化的视觉 SAC 微调

1. 用 BC 权重初始化 actor；critic 从 replay 数据重新训练，不直接依赖状态输入。
2. 保留现有奖励、阶段判定、超时和成功条件，保证与当前基线可比较。
3. 引入 DrQ-v2 风格的轻量随机裁剪/平移增强；三视图应作为一个整体处理，避免破坏跨视图几何对应。
4. 在线执行当前视觉策略，收集其实际访问到的新状态，用环境奖励更新 SAC。
5. replay 中固定混合专家轨迹与在线轨迹，例如先采用 25% 专家、75% 在线数据；根据稳定性再调整比例。
6. 优先使用现有双 Q SAC，不实现完整 RLPD ensemble，以控制代码和显存成本。

### 阶段 D：分布偏移处理

BC 后出现专家数据未覆盖的观测是预期现象。在线 SAC 会直接收集并学习这些状态，因此第一版不要求 DAgger。

若策略在 BC 初始化后频繁偏离、误差累积且在线 RL 无法恢复，则加入简化仿真 DAgger：

1. 运行当前视觉策略并记录其访问的观测。
2. 在同一 PyBullet 状态调用状态专家，得到纠正动作。
3. 将“当前策略观测、专家纠正动作”加入 BC 数据集。
4. 短暂重新进行 BC，再继续 SAC 微调。

DAgger 是按需启用的补救步骤，不是第一版必做模块。

## 实现优先级

- **必须实现**：专家数据生成、视觉 BC、BC 初始化的 SAC、专家/在线 replay 混合、轻量 DrQ 图像增强。
- **按需实现**：简化 DAgger、显式阶段 one-hot 辅助输入。
- **暂不实现**：完整 RLPD critic ensemble、AWAC 独立算法、Transporter、Diffusion Policy、DPPO、Octo。

## 验收标准

- BC 阶段：固定场景上能稳定执行接近并产生抓取尝试，不再全部 `approach_timeout`。
- SAC 微调阶段：`grasp_rate`、`lift_rate` 和最终 `success_rate` 持续提升；失败原因从单一 `approach_timeout` 扩展为可诊断的阶段失败分布。
- 最终结果：在至少 3 个随机种子和固定评估场景序列上，pick-place 成功率显著高于当前视觉 SAC 基线，并达到项目设定的可接受成功率。
- 诊断：继续记录视觉特征活跃度和图像置换动作差异，确认策略确实使用图像而非盲目依赖 proprio。

## 参考链接

- [DrQ-v2 官方代码](https://github.com/facebookresearch/drqv2)
- [RLPD 论文与代码](https://proceedings.mlr.press/v202/ball23a/ball23a.pdf)
- [AWAC 项目](https://awacrl.github.io/)
- [Ravens / Transporter Networks](https://github.com/google-research/ravens)
- [Diffusion Policy 论文](https://arxiv.org/abs/2303.04137)
- [RoboBase](https://github.com/robobase-org/robobase)
- [Octo](https://octo-models.github.io/)

## 相机观测方案决策：先验证三正交 RGB，再按判据切换

### 当前决定

第一阶段继续使用当前的三正交 RGB 观测，不立即改动相机设计。这个方案虽然不是现实机器人中最常见的相机接口，但对当前 PyBullet toy task 有明确的几何优势：`xy/xz/yz` 三个投影可以降低单视角下的深度歧义，适合先验证视觉策略是否能够恢复物体、目标和末端之间的三维关系。

三正交相机应被视为当前实验的视觉基线，而不是最终现实机器人相机方案。训练时继续保持三个视图的几何对应关系；图像增强必须对三个视图使用一致的空间变换，不能把不同视图当作互不相关的普通 RGB 图像独立平移。

### 三正交方案的主要风险

- 正交投影与真实相机的透视成像存在 domain gap，训练出的策略未必能直接迁移到真实相机。
- 三张图包含较多重复背景和机械臂信息，会增加渲染、传输和 CNN encoder 的计算量。
- 固定外部视角可能在夹爪接近物体、闭合和放置时出现遮挡。
- 如果三个视图只是沿通道拼接，网络需要自行学习每个视图的坐标语义；应继续使用按视图拆分、再融合的 encoder 结构。

### 三正交方案的继续验证标准

在切换相机设计之前，先完成以下验证：

1. `reach` 能稳定收敛，并且 `image_action_delta_mean` 不接近零，确认策略确实使用图像。
2. BC 预训练后不再几乎全部以 `approach_timeout` 结束，能够稳定进入 `GRASP`。
3. SAC 微调过程中 `grasp_rate`、`lift_rate` 和最终 `success_rate` 持续上升，而不是只提升 shaping reward。
4. 在固定评估场景和至少三个随机种子下，结果具有重复性。
5. 通过遮挡、物体/目标位置变化和相机轻微扰动测试，确认策略不是依赖固定像素模板或单一轨迹。

如果三正交 RGB 在上述条件下能够完成 pick-place，则保留它作为当前项目的主方案，不必为了追求行业常见形式而更换观测接口。

### 失败后的相机切换顺序

如果三正交 RGB 长时间训练后仍然无法稳定进入 `GRASP`，或者策略对遮挡和轻微相机变化非常敏感，按以下顺序切换：

1. **单个固定透视第三人称 RGB 相机**：相机位于桌面前上方，同时看到物体、目标、机械臂和夹爪。它是最简单的行业化对照基线，先用于判断问题是否来自正交投影和多视图融合。
2. **固定透视第三人称 RGB + wrist RGB**：第三人称相机提供全局工作空间定位，腕部相机提供夹爪接触和放置细节。对 pick-place 来说，这是优先级最高的多相机替代方案。
3. **单个或多个 RGB-D 相机**：使用深度和相机内外参生成点云、体素或统一三维特征。只有在 RGB 方案已经证明视觉信息足够、但深度歧义仍限制成功率时再引入。

行业参考中，RoboMimic 的图像任务普遍采用 front/shoulder view 与 wrist view 的组合；SERL 的真实 Franka 示例使用安装在夹爪上的 wrist RealSense；Ravens/Transporter Networks 使用 RGB-D 图像和相机参数，而不是手工构造的三张正交 RGB 图像。参考：[RoboMimic observations](https://robomimic.github.io/docs/tutorials/observations.html)、[RoboMimic study](https://robomimic.github.io/study/)、[SERL real Franka](https://github.com/rail-berkeley/serl/blob/main/docs/real_franka.md)、[Ravens](https://github.com/google-research/ravens)。

### 切换时保持不变的内容

更换相机观测时，优先只替换 observation adapter 和视觉 encoder，保持以下内容不变：

- Panda 动作空间和末端控制语义；
- IK 控制器和 PyBullet 物理场景；
- pick-place 阶段状态机、奖励和成功条件；
- 专家示范生成、BC、Ape-X learner/actor 和 prior/online replay 流程；
- TensorBoard 业务指标和评估 episode 序列。

这样可以把实验差异限制在视觉观测表达本身，避免同时改变任务定义和训练算法。

## 后续独立目标：提升 SAC_VecObs teacher 的随机化泛化能力

### 当前问题记录

当前使用的 vector teacher checkpoint 为：

```text
SAC_VecObs/runs/pick_place_20260818_211439/checkpoints/pick_place_sac_1000000_steps.zip
```

对应的归一化统计文件为：

```text
SAC_VecObs/runs/pick_place_20260818_211439/checkpoints/pick_place_sac_vecnormalize_1000000_steps.pkl
```

该 teacher 在原始 `SAC_VecObs` 环境分布上已经验证过较强能力：之前生成的原始专家示范中，`100/100` 个 episode 成功。

在本轮为了增加视觉 BC 数据覆盖而新增的 `RandomizedPixelTaskEnv` 中，额外加入了：

- 物体 XY 位置 jitter，幅度最多约 `±0.04 m`；
- 目标 XY 位置 jitter，幅度最多约 `±0.04 m`；
- Panda 初始 7 个关节角 jitter，幅度最多约 `±0.05 rad`；
- 三个正交相机共享的中心、角度和尺度 jitter。

在这组随机化环境上，使用同一个 vector teacher 运行 `100` 个 episode，结果为：

```text
success_rate = 58%
成功 episode = 58/100
失败 episode = 42/100
```

失败 teacher 轨迹没有作为 BC 正样本使用。当前失败数据主要来自 teacher 对新增物理状态分布的适应不足，而不是相机输入问题：vector teacher 不读取 RGB，相机 jitter 不会直接影响它的动作。主要未知因素是物体/目标位置变化和初始关节姿态变化的单独影响。

当前应区分两个目标：

1. **SAC_BC_PixelObs 主线**：只需要收集足量、高质量且覆盖范围合理的成功视觉轨迹；teacher 不必在所有随机化状态上都达到完美泛化。
2. **SAC_VecObs 后续研究目标**：提升 vector teacher 对物体、目标和初始机械臂姿态变化的鲁棒性，使它能在更宽的状态分布中稳定生成成功示范。

### 对当前视觉主线的影响

teacher 成功率只有 `58%` 不会阻止 BC 和 SAC 方案继续推进，但会提高示范采集成本，并可能缩小成功轨迹的状态覆盖范围。

如果目标是获得 `300` 条成功示范，按当前约 `58%` 成功率估算，需要运行约 `500–550` 个随机化 episode。采集时必须保留所有 episode 的 metadata，而不是只记录成功数量，以检查成功轨迹是否集中在某个狭窄区域。

成功轨迹至少应记录：

- reset seed；
- 物体位置；
- 目标位置；
- 初始关节角偏移；
- 相机 jitter 参数；
- 最终阶段、成功标记和失败原因。

失败轨迹暂不作为 BC 专家动作标签，但应保留用于分析 teacher 的失败边界，并可在后续 SAC online replay 或 DAgger 研究中使用。

### 后续 teacher 泛化研究路线

该目标独立于当前视觉主线，后续按 curriculum 逐级增加随机化：

1. 在不加入物理 jitter 的原始环境中增加训练 seed，确认 teacher 基础能力和成功率稳定。
2. 只加入较小的物体/目标位置 jitter，例如 `±0.01–0.02 m`。
3. 在物体/目标 jitter 稳定后，再加入较小的初始关节角 jitter，例如 `±0.01–0.02 rad`。
4. 最后逐步扩大到当前目标范围 `±0.04 m` 和 `±0.05 rad`。
5. 每个阶段单独报告原始 seed、训练内 seed 和 unseen seed 的成功率、grasp_rate、lift_rate 及失败原因。

每个 curriculum 阶段的进入条件建议为：

```text
成功率至少 90%–95%
grasp_rate 和 lift_rate 没有明显下降
失败原因不是集中爆发为 approach_timeout
```

teacher 泛化能力提升完成后，可以重新生成更宽状态覆盖的成功视觉示范，作为 SAC_BC_PixelObs 的第二版数据集；当前视觉主线不需要等待该研究完成。
