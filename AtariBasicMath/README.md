# Atari BasicMath：从宏动作到原始动作的三阶段 RL 方案

## 1. 目标与核心问题

Atari BasicMath 的原始动作空间只有六个动作：

```text
NOOP / FIRE / UP / RIGHT / LEFT / DOWN
```

这些动作粒度过细。Agent 不仅要读题和计算，还要通过一长串摇杆动作填写答案。只有最终提交答案时，环境才返回正确或错误的二元反馈。这使任务同时包含：

- 高层决策：这道题的答案是什么；
- 低层控制：如何用原始动作把答案填入游戏并完成提交。

本方案将两者拆开，使用两个模型：

```text
高层模型：游戏画面 -> 宏动作
低层模型：当前/目标画面 + 目标答案 + 当前答案/光标 -> 原始动作
```

训练仍分高层、低层和组合评测三个阶段。本方案暂不考虑把两个模型蒸馏成单一模型，也不设置后续第四阶段。

所有环境默认使用统一的随机化 episode：每次 reset 重新启动原生游戏，自动正确完成前三道预热题，并在每道预热题提交前由环境 RNG 随机等待 `0..59` 个 ALE 帧；随后将第 4 道题作为当前唯一 episode。宏动作或低层原始动作完成后，本次 episode 结束并丢弃剩余题目，下一次 reset 再重新随机化。这样可以避开正误反馈和第 10 题终止边界，同时保留 ROM 的真实题目随机性。

### 核心经验与当前结果

当前实验表明，训练效果主要来自以下三个设计决定：

1. **统一 episode 定义并修复环境随机性。** 高层、低层和组合评测都从新游戏开始，经过三道自动正确完成的预热题后使用第 4 题。预热题提交前的随机帧延迟会驱动 ROM 内部的题目生成器，使低层和高层看到完整的低难度加法题分布，而不是每个 actor 都重复同一题目背景。这样也避开了错误答案反馈和原生第 10 题终止状态对训练的干扰。
2. **分层动作设计。** 高层只输出 `ANSWER(value)` 宏动作，直接学习“这道题答案是多少”；低层接收高层给出的目标答案和当前执行状态，只学习如何用 `NOOP/FIRE/UP/RIGHT/LEFT/DOWN` 完成填写。高层可以快速收敛，并为低层提供清晰的中间目标。
3. **距离变化稠密奖励。** 低层根据当前数字到目标数字的最短拨动距离变化获得逐步奖励，拨近目标得到正反馈，拨远目标得到负反馈，最终正确提交仍获得成功奖励。它为原始动作提供了可传播的中间信用，显著缓解了只依赖最终提交反馈的稀疏奖励问题。

在当前一组训练实验中，低层训练约 8M transitions 后，rollout 和独立评测成功率均稳定在 90% 以上；高低层组合评测的 `hierarchy_success_rate` 达到约 94%，`timeout_rate` 约为 6%。这些数字是当前配置和随机种子下的实验结果，不代表环境的理论上限，但说明统一环境、分层控制和稠密奖励三者能够共同支持稳定收敛。

## 2. 总体结构

```text
                     ┌────────────────┐
当前游戏画面 ───────>│    高层模型     │
                     └───────┬────────┘
                             │ 宏动作 m 及其向量 v(m)
                             v
                     ┌────────────────┐
当前画面 ───────────>│                │
目标画面 ───────────>│    低层模型     │──────> 原始动作
宏动作向量 ─────────>│                │
当前答案与光标 ──────>│                │
                     └────────────────┘
```

高层模型负责“做什么”，低层模型负责“怎么做”。最终组合系统仍然保留这两个模型，游戏环境实际接收到的是低层模型输出的原始动作。

## 3. 基本定义

### 3.1 宏动作

宏动作表示一次有明确任务含义的决策，例如：

```text
ANSWER(15)
```

宏动作必须能够表示为固定结构的向量 `v(m)`，供高层模型输出，也供低层模型作为额外观测输入。

宏动作向量不是低层模型需要自己推断的隐变量，而是高层模型明确传递给低层模型的控制意图。

低层使用 21 维分解 one-hot，而不是把整个答案编码成一个 19 类 one-hot：

```text
十位：11 类 one-hot（blank、0..9）
个位：10 类 one-hot（0..9）
```

因此目标 `2` 和 `12` 共享相同的个位特征，模型可以复用已经学到的个位控制能力。

### 3.2 目标画面

修改后的环境需要支持将宏动作结果渲染成目标画面：

```text
I_goal = Render(I_current, m)
```

例如，当前题目不变，宏动作是 `ANSWER(15)`，目标画面就是答案区域已经填写为 `15` 的画面。

目标画面只描述希望达到的结果，不提供如何达到该结果的原始动作序列。

### 3.3 当前答案状态与目标是否达成

BasicMath 的 ALE RAM 保存当前正在编辑的十位、个位和光标位置。低层观测显式加入：

```text
当前答案：22 维 one-hot（十位和个位各为 blank、0..9）
当前光标：2 维编码（十位、个位；移出这两位时均为 0）
```

这些结构化状态在训练和组合推理时都存在，因此当前低层策略不是纯视觉策略。目标答案的 21 维宏动作向量保持不变；低层网络最终接收 `21 + 22 + 2 = 45` 维结构化输入。

低层是否达到目标由答案 RAM 判断。光标位置和整帧像素差只用于观测或诊断，不影响成功语义；只要十位、个位等于宏动作目标、其余可编辑位置保持空白，并由低层输出 `FIRE`，宏动作即执行成功。

对于 `ANSWER(15)` 这种还包含提交操作的宏动作，目标画面是“提交前已经正确填写 15”的稳定检查点。低层模型到达该画面后，还必须输出原始 `FIRE` 动作；环境确认提交发生后，才认为整个宏动作完成。不能用提交后的画面做目标，因为游戏此时可能已经进入反馈动画或下一道题。

## 4. 阶段一：训练高层宏动作模型

### 4.1 环境

首先修改环境，使其能够直接接收宏动作：

```text
原始环境：step(UP)
宏动作环境：step(ANSWER(15))
```

环境直接执行宏动作，并返回游戏原有的奖励。对于 BasicMath，提交正确答案得到正奖励，提交错误答案得不到奖励。

### 4.2 高层模型

高层模型的输入是游戏画面，输出是一个宏动作：

```text
π_high(m | I_current)
```

这一阶段只训练高层模型解决题目，不要求它学习摇杆操作。

### 4.3 阶段目标

阶段一需要证明：在去除琐碎输入动作后，Agent 能够通过宏动作 RL 学会 BasicMath 的任务级决策。

阶段一的产物包括：

- 已训练的高层模型；
- 宏动作空间及其向量编码；
- 宏动作结果的目标画面渲染能力。

## 5. 阶段二：训练低层原始动作模型

### 5.1 训练任务

低层模型不负责计算题目答案。每次训练时，环境直接给出一个目标宏动作，并生成与之对应的目标画面。

`train_lo.py` 不加载高层模型。每个 episode 从 `0..18` 均匀采样目标，而不是由高层模型实时选择；这使低层能力训练不受某个高层 checkpoint 的动作分布限制。

这里的“宏动作向量”表示低层本次需要完成的目标，不是高层模型在低层训练期间实时推理得到的输出。训练好的高层模型只在第三阶段组合系统中负责产生这个目标。

低层训练不再采用一位数、两位数 curriculum stage，也不从旧低层 checkpoint 初始化。低层模型接收五部分输入：

```text
1. 当前游戏画面 I_current
2. 填写了宏动作结果的目标画面 I_goal
3. 高层宏动作的向量表示 v(m)
4. 当前答案的 22 维 one-hot
5. 当前光标的 2 维 one-hot
```

低层模型输出六个原始动作之一：

```text
π_low(a_raw | I_current, I_goal, v(m), answer_current, cursor)
```

目标画面告诉低层模型“最终画面应该是什么样”，宏动作向量进一步提供结构化的高层意图。两者同时输入，不要求低层模型仅从目标图像中反推出宏动作含义。

### 5.2 低层奖励

空白和数字 `0..9` 构成 11 状态循环。对每一位定义最短上下拨动距离，并将两位距离相加：

```text
digit_distance(current, target)
  = min((target-current) mod 11, (current-target) mod 11)

D(s) = distance_tens + distance_ones
```

非 `FIRE` 动作使用距离变化塑形：

```text
dense_reward = 0.1 * (D_before - D_after)
reward = dense_reward - 0.001
```

向目标拨近一格约得 `+0.099`，拨远一格约得 `-0.101`，没有改变数字距离的 NOOP/左右移动得 `-0.001`。距离系数由 `BasicMathConfig.distance_reward_scale` 配置；训练 CLI 的 `--distance-reward-scale 0` 可关闭塑形。

动作成本由 `BasicMathConfig.primitive_action_penalty` 配置，默认 `0.001`。最长 96 步累计成本最多 `0.096`，不会抵消成功奖励；若使用 `0.01`，长成功轨迹的回报会接近零，因此不采用。

失败 `FIRE` 会一次性补扣本 episode 尚未发生的剩余动作成本，使立即失败与超时失败的未折扣总成本都约为 `-0.096`。否则立即错误提交只损失 `0.001`，模型会利用漏洞学习尽快 `FIRE`。成功 episode 不补扣，只按实际动作数量计费，因此成本只鼓励更快成功，不鼓励更快失败。

对于 BasicMath 的 `ANSWER(value)`，成功意味着十位、个位与目标答案一致，再通过原始 `FIRE` 完成提交；光标停在十位或个位都可以成功。

低层成功与“数学答案是否正确”无关。例如当前题目是 `1+1`，低层目标是 `ANSWER(15)`：只要低层通过原始动作准确填写并提交 `15`，低层奖励就是 `+1`，即使 Atari 游戏因为答案错误而返回的原始游戏奖励是 `0`。

`FIRE` 在低层环境中始终是 episode 的最后一个 agent 动作。Agent 可以在任意时刻选择它，环境不做 action mask：若提交前两个数字与目标一致，则奖励 `+1`；否则奖励 `0`。`FIRE` 本身不计算距离变化，环境只在内部等待本题原生奖励确定，不把反馈画面暴露给策略。若超过原始动作步数上限仍未 `FIRE`，环境直接返回 truncated 并丢弃本局；最后一个 agent 动作仍保留其距离奖励。

该设计忠于原始游戏。低层使用 epsilon-greedy 完成奖励冷启动，因此高 epsilon 阶段可能产生较多过早 `FIRE` 的短失败 episode；环境不额外惩罚或屏蔽 `FIRE`，由 DQN 自行学习只在达到目标后提交。

因此低层成功标准是：

```text
提交前的十位和个位与目标答案一致
+
低层模型输出原始 FIRE 完成提交
```

而不是：

```text
正确解答当前数学题
```

逐步奖励只使用符号数字的拨动距离变化，不使用画面像素相似度。光标移动不计入距离，由结构化光标观测、动作成本和最终成功奖励学习。

### 5.3 不使用 HER

当前方案不使用 HER。

当前数字可能仍为空白或只是未完成的中间状态，并且 BasicMath 的宏动作还要求由 agent 执行最终提交。当前实现使用显式距离奖励提供中间信用，不再额外引入 HER relabeling。

### 5.4 阶段目标

阶段二需要证明：给定目标画面和宏动作向量，低层模型能够只使用原始动作，可靠地使游戏达到宏动作规定的目标状态。

阶段二主要评估：

- 宏动作目标完成率；
- 完成一个宏动作需要的原始动作数量；
- 不同宏动作参数上的泛化能力；
- 在 ALE sticky action 等控制噪声下的稳定性。

## 6. 阶段三：组合高层与低层模型

阶段三不进行训练。高层和低层模型都从已有 checkpoint 加载，并以冻结的 greedy policy 运行；这一阶段不创建 replay、不计算训练 loss，也不更新任何参数。

阶段三不再让环境直接执行高层模型选择的宏动作，而是按以下流程运行：

```text
1. 高层模型读取当前游戏画面；
2. 高层模型输出宏动作 m 和向量 v(m)；
3. 环境根据 m 渲染目标画面 I_goal；
4. 低层模型反复输出原始动作；
5. 原始环境逐步执行这些动作；
6. 当前答案数字达到目标后，低层执行宏动作要求的结束动作；
7. 对于 BasicMath，结束动作是原始 FIRE，提交后本次宏动作结束；
8. episode 结束；下一次 reset 重新随机预热并生成新的第4题。
```

组合后，任务的实际控制链为：

```text
游戏画面
  -> 高层模型
  -> 宏动作及其向量
  -> 目标画面
  -> 低层模型
  -> 原始动作
  -> 游戏环境
```

阶段三保留两个模型和内部宏动作表示，但环境真正执行的动作全部来自原始动作空间。

当前实现入口为：

```bash
python DQN/eval_combined.py \
  --high-checkpoint <high.pt> \
  --low-checkpoint <low.pt> \
  --episodes 100
```

### 6.1 阶段一致性要求

阶段一中宏动作产生的结果，与阶段三中低层模型通过原始动作产生的结果，必须在任务语义上保持一致。否则，高层模型在宏动作环境中学到的策略无法直接迁移到组合系统。

因此每个宏动作至少要满足：

- 结果可以被渲染为稳定的目标画面；
- 目标画面可以通过原始动作真实到达；
- 目标是否达成可以可靠判断；
- 提交等结束操作可以由低层模型通过原始动作完成；
- 高层、低层和组合评测必须使用相同的随机预热与单题 episode 边界。

## 7. 两个模型的职责边界

| 模型 | 输入 | 输出 | 学习目标 |
|---|---|---|---|
| 高层模型 | 当前完整游戏画面 | 宏动作向量 | 学会选择正确答案等任务级决策 |
| 低层模型 | 当前/目标画面、宏动作向量、当前答案和光标 | 一个原始动作 | 学会实现高层指定的宏动作结果 |

高层模型不学习摇杆操作，低层模型不判断答案是否正确。两个模型通过宏动作向量和目标画面连接。

## 8. 方法成立所需的假设

本方案不是对所有 RL 环境无条件适用。它要求任务满足以下条件：

1. 可以定义有明确任务意义的宏动作；
2. 宏动作可以编码为结构化向量；
3. 环境能够渲染宏动作对应的目标画面；
4. 目标画面能够由原始动作到达；
5. 环境能够提供当前执行状态，并可靠判断语义目标是否达成。

这些条件共同构成宏动作空间与原始动作空间之间的桥梁。

## 9. 当前方案暂不包含的内容

为保持研究问题清晰，当前三阶段方案暂不考虑：

- 将高层和低层模型蒸馏成单一原始动作模型；
- 第四阶段的端到端联合微调；
- 使用逐像素画面相似度作为 reward shaping；
- 将画面切分成多个宏块并分别调节距离权重；
- 在无法识别实际宏动作的情况下使用 HER；
- 自动发现宏动作或自动学习宏动作空间。

当前首先需要验证的是：宏动作高层模型和 RAM 增强的目标条件低层模型能否分别收敛，以及两者组合后能否通过原始动作完成 BasicMath。

## 10. 当前实现

```text
MathEnv/
  basic_math_env.py   一题一 episode 的 BasicMath 环境

DQN/
  model.py            Dueling Q 网络
  replay.py           n-step 与 prioritized replay
  runtime.py          CPU actor、CUDA learner、独立 evaluator
  train_hi.py         高层训练入口
  train_lo.py         低层训练入口
  eval_hi.py          高层评测入口
  eval_lo.py          低层评测入口
  eval_combined.py    第三阶段高低层组合评测入口
```

当前环境只实现：

- `mode=5` 的加法任务；
- `difficulty=0` 的低难度、无计时任务；
- 宏动作答案范围 `0..18`；
- 确定性原始控制：`frameskip=1`、sticky action 为 `0`；题目生成通过预热题提交时机随机化；
- 每次 episode 随机预热三题并使用第4题；原生十题局的其余题目不进入当前 episode。

### 10.1 环境接口

宏动作接口：

```python
from MathEnv import BasicMathEnv

env = BasicMathEnv(action_mode="macro")
observation, info = env.reset(seed=1)
observation, reward, terminated, truncated, info = env.step(15)
```

`info["question_index"]` 默认是 `3`（第4题，0-based），`info["problem_operands"]` 给出当前加法题的两个加数，`info["warmup_delays"]` 记录本次三道预热题提交前使用的随机等待帧数。低层训练和高层训练使用完全相同的题目随机化逻辑。

普通原始动作接口：

```python
env = BasicMathEnv(action_mode="raw")
observation, info = env.reset(seed=1)
observation, reward, terminated, truncated, info = env.step(env.UP)
```

带宏动作目标的原始动作接口：

```python
env = BasicMathEnv(action_mode="raw", goal_conditioned=True)
observation, info = env.reset(
    seed=1,
    options={"target_macro_action": 15},
)

current_screen = observation["current"]
goal_screen = observation["goal"]
macro_vector = observation["macro"]
current_answer_vector = observation["current_answer"]
cursor_vector = observation["cursor"]
```

`macro` 保持 21 维；`current_answer` 为 22 维；`cursor` 为 2 维。预处理按这个顺序拼接成低层网络使用的 45 维结构化输入。环境的 `current_answer_digits` 属性返回可读的 `(tens, ones)`，空白位返回 `None`。

### 10.2 题目随机性

当前使用 `mode=5`，即随机加法模式。原生游戏内部仍按十道题组织，但随机题目不是简单由 Gym `seed` 直接决定；在确定性控制设置下，重复相同的动作时序会复现相同题目。为让每个训练 episode 获得真正不同的题目，wrapper 在每次 reset 后自动完成前三道题，并在每道预热题提交前从环境 RNG 采样 `0..59` 个 NOOP 帧。这个提交时机推进 ROM 的题目生成状态，第 4 题因此覆盖全部45种低难度无序加法组合。

`info["problem_operands"]` 记录当前两个加数，`unique_problem_count` 统计一轮训练或评测中已见过的不同加数 pair 数量，最大值为45。它用于监控环境分布是否真正展开，不等同于模型成功率。关于 BasicMath 的模式和随机题定义，参见 [Atari Basic Math 手册](https://atariage.com/manual_html_page.php?SoftwareLabelID=14) 和 [ALE BasicMath 文档](https://ale.farama.org/environments/basic_math/)。

目标也可以通过 `set_target_macro_action()` 设置，或者在 episode 的第一次 `step()` 中作为可选的 `target_macro_action` 参数传入。

## 11. DDDQN 训练

高层训练：

```bash
python DQN/train_hi.py
```

低层训练：

```bash
python DQN/train_lo.py
```

低层训练不需要、也不接受高层或旧低层 checkpoint。目标 `0..18` 按 episode 均匀采样，默认运行 20M 总 transition；可通过 `--total-steps` 覆盖，通过 `--distance-reward-scale` 调整或关闭距离奖励。

低层使用一套基于总训练进度的 epsilon schedule：

```text
0.9@0% -> 0.05@5% -> 0.01@25% -> 0.001@100%
```

两个入口使用同一套类 Ape-X actor-learner 实现：

- actor 只在 CPU 上运行环境和策略；
- learner 主进程直接使用唯一 CUDA GPU，不做 GPU 探测或 CPU 回退；
- 高层 actor 的 epsilon 依据全局 transition 数量线性衰减；
- 低层 actor 按全局 transition 占总训练量的比例计算 epsilon；
- actor 只在 episode 边界同步 learner 权重，避免中途改变行为策略；
- 使用 Double DQN、Dueling 网络、n-step return 和 prioritized replay；
- learner 定期向共享 CPU 模型发布最新权重；
- greedy evaluator 在独立 CPU 进程中运行，不阻塞 learner。

所有训练默认值只在 `TrainConfig` 中定义。CLI 从该 dataclass 读取默认值，只负责声明允许覆盖的字段；修改已有参数的默认值不需要同步修改 parser。

当前默认每 `100,000` 个全局 transition 请求一次独立 greedy 评测，每 `1,000,000` 个 transition 保存一次 checkpoint。两者都可以通过 CLI 参数调整，实际默认值以 `TrainConfig` 为唯一依据。

## 12. TensorBoard 与 Checkpoint

TensorBoard 日志默认写入：

```text
runs/high_YYYYMMDD_HHMMSS_pid<PID>/
runs/low_YYYYMMDD_HHMMSS_pid<PID>/
```

查看日志：

```bash
tensorboard --logdir runs
```

日志包括：

- rollout 成功率、回合回报和 episode 长度；
- 独立 greedy 评测成功率和回报；
- 低层 rollout/eval 分别统计一位和目标 `0..9` 与两位和目标 `10..18` 的成功率、episode return、episode length、dense reward return 和最终答案距离；
- transition 吞吐量；
- 全局 epsilon 走势；
- 低层一位和/两位和各自的 episode 与 transition 计数；
- 高低层累计看到的唯一加法题数量，正常训练应最终达到全部45种低难度题；
- DQN loss、Q 值、target、TD error、梯度范数、PER beta 和 replay size。

`dqn/*` 指标严格按全局 transition 计数，每 `10,000` 步最多上报一次，避免高频 learner update 产生过密的 TensorBoard 数据。

Checkpoint 默认写入：

```text
checkpoints/<run-name>/step_000001000000.pt
checkpoints/<run-name>/final_step_<N>.pt
```

## 13. 评测

高层评测：

```bash
python DQN/eval_hi.py checkpoints/<run-name>/<checkpoint>.pt --episodes 100
```

低层评测：

```bash
python DQN/eval_lo.py checkpoints/<run-name>/<checkpoint>.pt --episodes 100
```

打开 GUI 并限制策略决策速度：

```bash
python DQN/eval_hi.py <checkpoint> --gui --fps 30
python DQN/eval_lo.py <checkpoint> --gui --fps 30
```

评测使用 greedy policy，即 `epsilon=0`。低层评测按顺序循环测试宏动作目标 `0..18`。

旧低层 checkpoint 只有 21 维目标答案条件输入，与当前 45 维结构化输入不兼容，需要重新训练；加载和组合评测时会得到明确错误。实验期间产生的 NoisyNet checkpoint 也已停用。高层网络结构未改变，已有高层 checkpoint 仍可使用。

第三阶段组合评测：

```bash
python DQN/eval_combined.py \
  --high-checkpoint checkpoints/<high-run>/<checkpoint>.pt \
  --low-checkpoint checkpoints/<low-run>/<checkpoint>.pt \
  --episodes 100
```

打开组合 GUI 并限制低层原始动作速度：

```bash
python DQN/eval_combined.py \
  --high-checkpoint <high.pt> \
  --low-checkpoint <low.pt> \
  --gui \
  --fps 30
```

组合评测分别报告低层宏动作执行成功率、Atari 原始答题成功率、两者同时成功的层级成功率、执行成功条件下的答题成功率、平均原始动作数、超时率以及高层宏动作选择分布。

## 14. 测试

```bash
pytest -q
```

测试覆盖 RAM 数字和光标编码、11 状态拨盘距离、稠密奖励、数字语义提交、题目级 episode 切分、45 维低层网络输入、统一 epsilon schedule、Dueling 网络输出以及 prioritized replay。
