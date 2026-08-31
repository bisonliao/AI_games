我想基于Tetris和BasicMath，设计一种训练方法/学习方法，让agent最终可以操作原始的动作，成功的完成任务，然后出一篇论文。我怀疑关于动作空间的设计、训练的迭代pipeline这方面还没有类似的论文。

有论文空间，但需要谨慎定位：**“使用宏动作加速 RL”不新，“最终去掉宏动作，只保留原始动作”才可能是关键贡献。**

我建议把研究问题明确成：

> 能否只在训练期间提供任务语义动作，通过逐级动作细化和策略蒸馏，最终得到一个仅使用 primitive actions、但仍能解决长时序稀疏奖励任务的 agent？

**核心方法**

可以称为 Progressive Semantic Action Refinement，工作标题例如：

> From Semantic Decisions to Primitive Control: Progressive Action-Space Refinement for Sparse-Reward Reinforcement Learning

定义一串逐渐细化的动作空间：

```text
A0 semantic -> A1 subtask -> A2 control -> AK primitive
```

每一层动作都有一个闭环 compiler，将它转换为下一层动作序列：

```text
Ck: (state, action_k) -> sequence of actions_(k+1)
```

两个任务可以这样组织：

| 层级 | Tetris | BasicMath |
|---|---|---|
| A0 | `PLACE(rotation, column)` | `WRITE(answer)` |
| A1 | `ROTATE_TO(r), MOVE_TO(x), DROP` | `WRITE_DIGIT(slot,d), SUBMIT` |
| A2 | 单次旋转、单格移动、软降 | 移光标、递增/递减数字、提交 |
| Primitive | 原始按键 | ALE 六动作 |

这不是简单地“把宏动作加入动作空间”。每完成一阶段，就把该层策略蒸馏到下一层，然后移除上一层动作。

**训练 Pipeline**

1. **训练闭环执行器**

执行器接收语义目标，例如 `WRITE("15")` 或 `PLACE(r,x)`，只输出原始动作。使用 goal-conditioned RL、HER 和内部进度奖励训练。它必须观察执行结果并纠错，不能是固定 open-loop action chunk。

2. **训练语义层 Teacher**

高层策略选择完整答案或方块落点，执行器负责真实操作。环境从始至终都实际运行原始帧，宏动作不能暂停游戏时间。

这是一个 SMDP，折扣必须按 option 的真实持续时间使用 `γ^duration`。

3. **收集成功的 primitive trajectories**

Teacher 与执行器组合后产生：

```text
observation history
semantic intent z
primitive action a
value/return
```

这些数据用于训练下一层 student。

4. **蒸馏到更细动作空间**

Student 学习三个目标：

```text
primitive action imitation
semantic intent prediction
teacher value/advantage prediction
```

语义 intent 辅助头很重要。否则 primitive student 只会模仿局部按键，不知道自己为什么向左移动或者为什么正在填写十位。

5. **渐进式控制权交接**

不能只做一次 behavior cloning。应采用类似 DAgger 的 mixed control：

```text
π_mix = β π_teacher_raw + (1-β) π_student
```

开始时 `β≈1`，随后根据 student 成功率降到 `0`。Student 偏离成功轨迹后，Teacher 在这些状态提供恢复动作，解决 imitation learning 的分布偏移。

6. **最终 primitive-only 微调**

最后完全删除 semantic policy、compiler 和 action mask，只保留 recurrent primitive policy，在原始奖励、原始 frameskip 和 sticky action 下继续 RL。

评估时 agent 接口只能看到：

```text
pixels/history -> primitive action
```

**与现有工作的边界**

初步检索已经发现几类必须引用的强相关工作：

| 工作 | 相似点 | 与拟议方法的区别 |
|---|---|---|
| Options / Macro-actions | 时间抽象、缩短 horizon | 通常测试时仍保留 options |
| [Deep RL With Macro-Actions](https://arxiv.org/abs/1606.04615) | Atari 中用宏动作改善稀疏奖励 | 宏动作仍是最终动作空间 |
| [Mix&Match](https://arxiv.org/abs/1806.01780) | action-space curriculum，最终留下复杂 agent | 从 9 个单步动作扩到 756 个单步动作；不是变时长语义动作的逐级编译 |
| [EASpace](https://arxiv.org/abs/2212.03540) | expert policy 变成 macro action | 将宏动作持续加入 primitive action space |
| [Q-chunking](https://arxiv.org/abs/2507.07969) | action chunks 缩短 horizon | 依赖 offline data，测试时仍运行 chunked policy |
| [Iterative Policy-Space Expansion](https://arxiv.org/abs/1912.02532) | 在 Tetris 上逐步解除约束 | 扩展的是线性策略参数空间，动作始终是方块落点，不涉及原始控制动作 |

因此不能宣称“action-space curriculum 从未有人做过”。更可信的新颖点是：

- 非嵌套、变时长的语义动作细化链。
- semantic action 到 primitive action 的闭环 compiler。
- 宏动作只作为训练脚手架，测试时彻底移除。
- 通过 intent distillation 和渐进 handoff 保留高层决策能力。
- 在几何决策 Tetris 和符号决策 BasicMath 上验证同一方法。

**必须做的实验**

基线至少包括：

- Primitive RL from scratch
- Primitive RL + reward shaping
- Hierarchical options，测试时允许宏动作
- Teacher 直接一次性 BC 到 primitive student
- DAgger without progressive refinement
- Mix&Match 风格动作课程
- Action chunking
- 完整方法

主要指标不能只按 agent decision 数计算，因为宏动作会隐藏计算量。必须报告：

- 原始 emulator frames
- 首次获得非零奖励所需 frames
- 最终 primitive-only 得分
- Teacher 到 student 的性能差距
- executor 成功率
- sticky action 鲁棒性
- 未见题目和未见方块序列的泛化
- 多随机种子的置信区间

**最大的论文风险**

最大的审稿意见会是：“语义动作是人工设计的，所以只是 domain engineering。”

应对方式不是立即增加自动 skill discovery，而是证明：

- 同一个训练框架适用于结构完全不同的两个任务。
- 不同粒度的动作链都能工作。
- 最终 agent 不依赖语义接口。
- 收益来自 progressive refinement，而不只是 expert demonstration。
- 在受控环境中系统改变 primitive sequence length、sticky probability 和 reward sparsity，验证尺度规律。

Tetris 和 BasicMath 足以做一个有说服力的初稿；若目标是主会，最好再加入一个“选择棋盘落点后通过光标执行”的任务，例如 Othello、Video Checkers，或者一个可程序化生成的 cursor-control benchmark。

这条研究线最有价值的表述不是“重新设计动作空间”，而是：

> **把好的动作空间当成训练期的特权信息，然后学习如何在没有它的情况下行动。**

下一步应先形成一份 2 至 3 页 research specification：正式定义 action refinement chain、compiler、handoff 算法、primitive-only evaluation contract，以及与 Mix&Match/Q-chunking 的差异。先把论文主张钉死，再开始实现。
