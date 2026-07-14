先给结论：

- 当前代码严格来说是 one-step TD / TD(0)，不是 TD(1)：一次 transition 从“黑棋决策”跨过白棋回应，到下一个黑棋决策。
- TD(λ)可能有帮助，但不建议直接实现经典 eligibility trace。对当前 replay-buffer + off-policy DQN，更适合先做 n-step return，收益相近、实现简单且更稳定。
- 标准 HER 对五子棋帮助通常不大。五子棋没有天然可替换的目标，强行把某个棋形或落点当作 hindsight goal，容易优化出与“赢棋”不一致的行为。
- 最值得优先做的是：n-step return、对局结果回填、优先经验回放、改进 self-play 对手池，以及调整更新比例/采样结构。

## 当前稀疏奖励的核心问题

当前黑棋 transition 大致是：

```text
黑棋状态 s_t
  -> 黑棋动作
  -> 白棋回应
  -> 下一个黑棋状态 s_{t+1}
```

非终局 reward 全是 0，终局才有 `+1/-1/0`。因此终局信号需要通过多次 bootstrapping 才能传播到开局：

```text
终局前一步 -> 终局前两步 -> ... -> 开局
```

同时还有几个附加问题：

- replay 中绝大部分样本 reward=0。
- uniform sampling 很少连续采到同一盘棋的相邻状态。
- self-play 的白棋不断变化，环境对黑棋而言是非平稳的。
- 只训练黑棋，数据分布可能被少数历史白棋策略主导。
- 当前 UTD=0.25 提高了吞吐，但奖励传播速度也会比原先每步更新一次更慢。

## 推荐优先级

| 方案 | 预期帮助 | 复杂度 | 主要风险 | 建议 |
|---|---:|---:|---|---|
| n-step return | 高 | 中 | bootstrap 边界、终局处理 | 第一优先级 |
| 对局结果回填 | 中到高 | 低到中 | 信号偏置、折扣选择 | 很适合先实验 |
| Prioritized Replay | 中到高 | 中 | TD-error 偏差、极端样本垄断 | 第二优先级 |
| 改进 self-play 对手池 | 高 | 中 | 训练非平稳、算力增加 | 非常重要 |
| Double/Dueling DQN | 已有 | — | — | 保留 |
| TD(λ) | 中 | 高 | off-policy 不稳定、实现复杂 | n-step 后再考虑 |
| HER | 低 | 高 | 目标定义不自然 | 暂不建议 |
| 奖励塑形 | 中到高 | 低到中 | 学会刷分而不是获胜 | 谨慎使用 |
| Distributional DQN | 中 | 中到高 | 网络和 loss 改动较大 | 后续优化 |
| 数据增强/棋盘对称 | 高 | 低 | 动作映射错误 | 强烈推荐 |
| 增加状态历史 | 低 | 中 | 五子棋本身近似 Markov | 通常没必要 |

# 1. n-step return

这是最适合当前项目的改造。

现在的 target 是：

```text
y_t = r_t + γ max Q_target(s_{t+1}, a)
```

改成 n-step：

```text
y_t =
    r_t
    + γ r_{t+1}
    + ...
    + γ^(n-1) r_{t+n-1}
    + γ^n max Q_target(s_{t+n}, a)
```

如果中途终局，则不 bootstrap。

对于五子棋，推荐先测试：

```text
n = 3、5、8
```

5×5 棋盘上一局黑棋决策次数通常不算很长，`n=5` 是比较合理的起点。终局奖励可以一次向前传播 5 个黑棋决策，而不是只传播一步。

### 如何适配当前多 actor 架构

每个 actor 为每个环境维护一个长度为 `n` 的 deque：

```text
raw transition
  -> per-env n-step accumulator
  -> 聚合后的 n-step transition
  -> learner queue
  -> replay buffer
```

需要额外保存：

- 累积折扣 reward；
- 实际 bootstrap 步数；
- `next_state`；
- `next_mask`；
- `done`；
- 最好保存 `discount = γ^k`，而不是假定所有样本都是固定 n 步。

终局时必须 flush deque，把不足 n 步的尾部样本全部写入 replay。

### 收益

- 明显加快胜负信号向开局传播。
- 继续兼容 replay buffer。
- 继续兼容 Double DQN、target network 和 prioritized replay。
- 不要求连续采样同一局的 transition。

### 风险

- n 太大会增加策略滞后和估计方差。
- self-play 是非平稳的，多步 return 使用旧 actor 策略生成的数据，off-policy 程度更高。
- 终局 flush 很容易出现少样本、重复样本或错误 bootstrap。
- n-step transition 数仍应等于原始黑棋决策数，不能只保留完整 n 步窗口。

复杂度：中等。

# 2. TD(λ)

TD(λ)本质上是不同长度 n-step return 的加权组合：

```text
G^λ = (1-λ) Σ λ^(n-1) G^(n)
```

它确实能更快传播终局奖励。推荐范围通常可以从：

```text
λ = 0.8～0.95
```

开始实验。

但在当前 DQN 中，直接引入经典 TD(λ)有几个问题。

### 经典 eligibility trace 不适合当前 replay

传统 TD(λ)通常按时间顺序在线更新 eligibility trace，而当前训练是：

- 多 actor 异步采样；
- replay 随机打乱；
- off-policy；
- actor 使用稍旧的策略；
- learner 使用 epsilon-greedy 以外的数据分布。

随机 replay 会破坏 trace 的连续性。若保留 replay，就不能简单地给网络参数维护传统 eligibility trace。

### 可行实现

更可行的是在 actor 端计算 truncated λ-return：

- 保留一段长度 `L` 的轨迹；
- 计算 1 到 L 步 return；
- 用 λ 加权；
- 将最终 target 或轨迹片段送入 replay。

但是如果直接把计算好的 target 存入 replay，target 很快会过时。更稳妥的方式是 replay 保存序列，训练时根据当前 target network 重新计算 λ-return。

这会要求：

- replay 从单 transition 改成 sequence replay；
- 正确处理 episode 边界；
- 按序列 batch 训练；
- 处理不同长度和 padding；
- 处理 off-policy correction。

如果希望理论上更可靠，还会涉及 Retrace、Tree-backup 或 Watkins’s Q(λ)。

### 我的判断

TD(λ)可能有帮助，但性价比低于 n-step：

- n-step 能解决大部分奖励传播问题；
- 更容易与现有 DQN/replay 结合；
- 更容易定位 bug；
- 更容易进行对照实验。

建议先实现 n-step。只有在 n-step 已经验证有效，但传播仍不够快时，再考虑 sequence replay + Retrace(λ)。

复杂度：高。  
风险：中到高。

# 3. HER

标准 HER 适合明确的 goal-conditioned 任务，例如：

```text
状态 + 目标位置 -> 动作
```

失败轨迹可以把实际达到的位置重新标记为目标，从而得到成功奖励。

五子棋的目标固定是“最终获胜”，没有自然的可替换 goal。失败的一盘棋不能简单地改写成：

```text
目标 = 最终形成的某个棋形
```

因为形成该棋形未必对赢棋有价值，甚至可能是败因。

### 如果一定要使用 HER

需要先把网络改成 goal-conditioned Q-network，例如目标可以定义为：

- 在指定位置形成五连；
- 完成某个长度和方向的连线；
- 阻止对手在某个位置形成威胁；
- 达到指定局面特征。

然后从轨迹中重新标记已实现的局部目标。

问题是这些目标不是原始胜负目标，必须设计目标采样、目标编码和奖励函数。最终很可能是在训练一个辅助任务系统，而不是传统 HER。

### 更合适的替代方案

与其做 HER，不如加入棋形辅助任务：

- 预测当前局面胜率；
- 预测最终赢家；
- 预测距离终局还有多少步；
- 预测某个动作是否形成活三、活四、冲四；
- 预测对方下一步致胜点；
- 预测合法动作的战术等级。

共享卷积 backbone，主头仍然输出 Q 值，辅助头提供更密集的监督。

这种方式比把局部棋形伪装成 HER goal 更自然。

复杂度：HER 高。  
预期收益：低或不确定。  
建议：暂不采用。

# 4. 对局结果回填

这是一个非常直接的方案：一盘结束后，将最终结果回填给这一局所有黑棋 transition。

例如黑棋最终获胜：

```text
r_t = γ^(T-t) × 1
```

失败：

```text
r_t = γ^(T-t) × -1
```

和棋为 0，或者设置一个很小的负值鼓励更积极结束对局。

这不完全等同于标准 DQN 的即时奖励定义，更接近 Monte Carlo return 或 reward redistribution。

### 优点

- 终局奖励一次传播到整盘棋。
- 实现比 TD(λ)简单。
- 适合每个 actor 已经天然维护独立 episode 状态的架构。

### 风险

- 每个早期动作都获得同方向奖励，无法区分好棋和“虽然很差但最后仍赢了”的棋。
- 如果同时保留 bootstrap，必须避免重复计算未来奖励。
- 对手较弱时，很多错误动作也会收到正奖励。
- 方差比 one-step TD 大。

### 推荐方式

不要直接把最终结果作为每一步的即时 reward 再继续正常 bootstrap。更清晰的是将其作为 Monte Carlo 样本或混合 target：

```text
target = (1-α) × n-step target + α × discounted_game_result
```

先从较小的：

```text
α = 0.1～0.3
```

开始。

复杂度：低到中。  
风险：中等。

# 5. Prioritized Experience Replay

终局附近以及预测错误大的 transition 通常有较大 TD error。PER 会更频繁地采样这些样本：

```text
priority_i = (|δ_i| + ε)^α
```

训练时通过 importance sampling 修正：

```text
weight_i = (N × P(i))^(-β)
```

建议初始参数：

```text
α = 0.6
β = 0.4，并逐步增长到 1.0
priority_epsilon = 1e-6
```

新 transition 以当前最大 priority 插入，确保至少被采样一次。

### 对稀疏奖励的帮助

终局样本初始 TD error 较大，会被更频繁地训练。之后高 TD error 会逐步向前传播，PER 再继续优先采样这些状态。

PER 与 n-step 通常组合使用效果更好。

### 风险

- self-play 的非平稳性会导致旧数据保持很高 TD error，并长期占据采样。
- 输棋终局可能垄断 replay，造成 Q 值过度悲观。
- 需要 importance weights，否则产生采样偏差。
- priority 更新与环形 buffer 索引管理较容易出错。

可以考虑对 priority 设置上限，或混入一定比例 uniform sampling。

复杂度：中等。

# 6. 棋盘对称数据增强

五子棋具备旋转和镜像对称性。一个 transition 可以生成最多 8 种等价形式：

- 旋转 0°、90°、180°、270°；
- 每种再做镜像。

必须同时变换：

- state；
- next_state；
- action；
- next action mask。

### 推荐方式

训练 batch 采样后，随机选择一种对称变换，而不是把每条数据复制 8 份写入 replay。

这样：

- replay 容量不膨胀；
- 每次训练都可能看到不同变换；
- 提升样本利用率；
- 降低网络对固定方向的偏置。

这是低复杂度、高性价比改进，甚至可以排在 PER 前面。

主要风险是动作索引和 mask 变换错误。需要为 8 种变换写严格的双向映射测试。

复杂度：低。  
风险：低。

# 7. Self-play 对手池优化

稀疏奖励之外，当前项目的另一个重要问题是白棋对手分布。

随机从最近 checkpoint 中选一个白棋，只能提供有限多样性。建议维护分层对手池：

- 随机 agent；
- 当前策略的近期 checkpoint；
- 较旧 checkpoint；
- 对黑棋胜率较高的 hard opponent；
- 少量规则型战术 agent。

采样比例示例：

```text
10% 随机 agent
50% 最近 checkpoint
20% 历史均匀采样
20% hard opponent
```

### 为什么有帮助

- 随机白棋适合早期探索和学习基本连线。
- 较强白棋提供有效失败奖励。
- 历史白棋防止策略只适应最近一个版本。
- hard opponent 能集中暴露黑棋弱点。

### 风险

- 白棋太强时，黑棋几乎全输，奖励仍然稀疏且单一。
- 白棋太弱时，黑棋从错误动作中也能获胜。
- 对手选择变化会增加训练非平稳性。
- 需要记录每个 checkpoint 的胜率或 Elo。

复杂度：中等。  
预期收益：高。

# 8. 奖励塑形

可以加入局部棋形奖励，例如：

- 形成活四；
- 阻挡对方活四；
- 形成双威胁；
- 阻止对方立即获胜；
- 非法动作惩罚。

但普通 reward shaping 风险很高。agent 可能反复追求局部棋形分数，而不是获胜。

更安全的是 potential-based shaping：

```text
r' = r + γΦ(s') - Φ(s)
```

理论上在满足条件时不会改变最优策略。

Potential 可以来自：

- 己方威胁数减去对方威胁数；
- 可立即获胜动作数量；
- 活四、冲四、活三的加权差值。

### 风险

- 五子棋棋形规则细节很多。
- potential 设计错误会引入策略偏差。
- 双方威胁并不是简单线性可加。
- 规则计算会增加 actor CPU 开销。

复杂度：中等。  
风险：高于 n-step 和数据增强。

# 9. 增加辅助价值头

当前 dueling Q 网络只通过 TD target 学习。可以增加一个 outcome head：

```text
V_outcome(s) -> 黑棋最终胜/负/和
```

每局结束后，该局所有黑棋状态都有监督标签：

```text
win  =  1
draw =  0
loss = -1
```

总 loss：

```text
L = L_DQN + c × L_outcome
```

这相当于用整局结果给 backbone 提供密集监督，但不会直接改变 DQN reward 定义。

还可以做三分类：

```text
P(win), P(draw), P(loss)
```

### 优点

- 每盘棋所有状态都能参与 outcome loss。
- 比直接 reward shaping 更不容易改变原始目标。
- 推理时仍然只使用 Q head。
- 可以监控 value calibration。

### 风险

- 早期状态的最终结果受探索和白棋版本影响，不完全由局面决定。
- outcome loss 过大可能压制 Q-learning。
- actor 或 replay 需要保存最终 outcome，或者终局后回填标签。

复杂度：中等。  
风险：中等。

# 10. Distributional DQN

可以将标量 Q 改为收益分布，例如 C51。对于胜、负、和三种主要结果，分布式价值表示可能比单一期望值更适合：

```text
P(return = -1)
P(return = 0)
P(return = +1)
```

不过折扣后的多步 return 会产生中间 support 值，并不只是三个类别。

优势：

- 更好地区分高风险/高收益动作；
- 常与 n-step、PER 组合；
- 在稀疏离散回报任务上可能更稳定。

风险：

- 输出和 loss 改动较大；
- action mask、Double DQN action selection 和 projection 都需要正确实现；
- 调试成本明显高于 n-step。

复杂度：中到高。

# 建议的实施路线

我建议按以下顺序，每一步独立做对照实验：

1. 棋盘 8 对称随机数据增强。
2. 3-step、5-step、8-step return，优先测试 5-step。
3. Prioritized Replay，并保留一部分 uniform sampling。
4. 改进 opponent pool，至少混合随机、近期、历史 checkpoint。
5. 加 outcome auxiliary head，作为密集的最终结果监督。
6. 再评估 Distributional DQN。
7. 只有上述方法仍不够，再考虑 sequence replay + Retrace(λ)。

暂不建议：

- 直接实现经典 TD(λ) eligibility trace；
- 标准 HER；
- 大量手工棋形即时奖励；
- 一开始就同时叠加 n-step、PER、TD(λ)、distributional DQN，届时很难判断训练异常来自哪里。

## 最推荐的第一版组合

如果只做一轮中等规模改造，我会选择：

```text
5-step return
+ 随机棋盘对称增强
+ Prioritized Replay
+ 分层 self-play 对手池
```

其中最先落地的应是：

```text
5-step return + 棋盘对称增强
```

这两项对现有架构侵入适中、理论依据清晰，也最容易通过单元测试验证正确性。TD(λ)可以视作 n-step 验证成功后的进阶选项；HER 则不适合作为这个项目的优先方向。