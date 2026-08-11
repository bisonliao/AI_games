# 用行为克隆训练 9×9 五子棋 Agent

这套代码要做的事情，可以先用一句话概括：

> 让启发式专家观察很多棋盘并给出建议，然后用普通的监督学习，训练一个神经网络模仿这些建议。

这里不使用强化学习。训练时没有奖励、Q 值、TD target 或策略梯度，主要损失仍然是你熟悉的交叉熵。

不过，它也不只是最简单的“收集一批专家棋谱，然后训练一次”。代码还会反复让当前 Agent 自己下棋，把它容易走到、容易犯错的局面交给专家重新标注。这部分就是 DAgger。

## 1. 最基本的行为克隆

对于一个棋盘状态，我们让启发式专家选出它认为最好的动作：

```text
棋盘状态 s ──启发式专家──> 动作 a
```

把它保存成一个监督学习样本：

```text
输入：棋盘状态 s
标签：专家动作 a
```

神经网络为棋盘上的 81 个位置分别输出一个 logit，然后用交叉熵让专家动作对应的 logit 尽可能大。

已经有棋子的格子会被 mask 掉，所以 Agent 只能从空位中选择动作。

## 2. 为什么只训练一次通常不够

假设训练数据全部来自“专家对专家”：

```text
专家正确落子
  ↓
进入专家熟悉的局面
  ↓
专家继续正确落子
```

训练完成后，Agent 不可能百分之百复制专家。只要某一步走错，后面就可能发生：

```text
Agent 走错一步
  ↓
进入训练数据中很少出现的局面
  ↓
Agent 更不知道该怎么走
  ↓
继续犯错
```

这种“训练时看到的局面”和“实际部署时遇到的局面”不一致，就是行为克隆常说的分布偏移或带外局面问题。

简单增加更多“专家对专家”棋谱不一定能解决它，因为新增棋谱仍然集中在专家自己的活动范围内。

## 3. 当前代码怎样使用 DAgger

DAgger 的核心不是换一种损失函数，而是改变数据的来源。

代码会让当前 Agent 真正控制棋子：

```text
当前状态 s
  ├── Agent 选择实际动作 a_agent，用它推进棋局
  └── 专家给出建议动作 a_expert，作为监督标签保存
```

因此，一个 DAgger 样本可能是：

```text
Agent 实际走了：左上角
专家认为应该走：中央附近

棋盘下一步按“左上角”继续发展，
但训练标签保存“中央附近”。
```

这样生成的后续局面来自 Agent 自己的行为分布。即使 Agent 已经犯错，专家仍然会告诉它：“走到这个局面以后，应该怎样补救。”

完成一轮数据收集后，用新旧数据一起训练下一版 Agent，然后再让新版 Agent 下棋、重新收集数据。

```text
初始数据 → Agent 0
              ↓
       Agent 0 访问的局面 → Agent 1
                                ↓
                         Agent 1 访问的局面 → Agent 2
```

当前 pipeline 最多进行 6 轮 DAgger。如果连续两轮都达到验收标准，就提前停止。

## 4. 为什么直接训练 9×9

以前的实验主要使用 5×5。实际结果中，Agent 和专家经常把 25 个格子全部走完，所有对局都是和棋。

这会产生一个问题：即使 Agent 的落子逻辑很差，只要最后仍然和棋，从胜负上就很难区分它和专家的差距。

所以当前质量优先 pipeline 固定使用 9×9。9×9 的局面空间更大，也更容易用对战结果区分棋力。

## 5. 第一步：Bootstrap 初始数据

第一版 Agent 还不存在，因此先生成 bootstrap 数据。

代码不是只让专家确定性地重复同一盘棋，而是混合三种下法。

### 5.1 专家受控采样自博弈，占 50%

专家通常会给出几个较好的候选动作，然后在开局阶段从这些候选中采样。

这仍然是比较高质量的棋，但不会每局都完全一样。

### 5.2 扰动开局，占 25%

棋局开始后的 2～10 手故意选择一些局部合法动作，然后再交给专家继续下。

这些动作不一定是专家最喜欢的动作，目的是让专家展示如何处理不同的开局形状。

### 5.3 带少量随机动作的专家，占 25%

在非战术局面中，有 15% 概率选择棋子附近的随机合法动作，其余时候仍按专家策略下。

如果当前有明确的必胜、必堵等战术，不进行这种随机扰动。

### 5.4 Bootstrap 什么时候停止

默认至少生成 2,000 局，并持续检查真正新增的棋盘状态：

- 目标：至少 100,000 个新的 canonical 状态；
- 最多：8,000 局；
- 还必须收集足够的 win、block 和 fork 战术样本。

这里的 canonical 状态，是把旋转和镜像后等价的棋盘视为同一个状态。例如，一个左上角棋形旋转到右下角，不会被错误地计算成全新的棋形。

如果达到 8,000 局仍然没有足够的新状态或战术样本，数据会标记为 `coverage_stalled`，pipeline 会停止并报错，而不是拿覆盖不足的数据继续训练。

## 6. 后续每轮 DAgger 怎样下棋

从第一版 Agent 开始，每轮数据由四种对局组成。

### 6.1 当前 Agent 对专家，占 40%

当前 Agent 一局执黑、下一局执白，与专家对战。

它可以直接学习自己面对专家时会进入的局面。

### 6.2 当前 Agent 自博弈，占 30%

当前 Agent 同时控制黑白双方。

这样可以暴露双方都不按专家路线走时产生的连续错误。

### 6.3 当前 Agent 对历史 Agent，占 20%

从以前轮次的 checkpoint 中随机选择一个作为对手。

这样当前 Agent 不会只适应某一个固定版本的策略。

### 6.4 带少量随机动作的当前 Agent，占 10%

在非战术局面中，以 10% 概率选择附近的随机合法动作。

它的作用是主动制造一些当前 Agent 平时不容易进入，但仍然合理、合法的局面。

### 6.5 每轮什么时候停止

每轮：

- 至少生成 1,000 局；
- 目标是新增 50,000 个以前没有的 canonical 状态；
- 最多生成 4,000 局。

所有实际访问到的状态都会由同一个冻结专家重新标注。

## 7. 专家为什么给出 top-4，而不是只给一个动作

一个五子棋局面经常不只有一个合理动作。

例如专家可能认为候选动作的优先级是：

```text
位置 40：最好
位置 39：次好
位置 41：也可以
位置 31：第四选择
```

如果每次只随机选其中一个作为硬标签，同一个棋盘可能一会儿标成 40，一会儿标成 39。对监督学习来说，这相当于标签互相冲突。

当前实现会保存专家确定性的 top-4 排名：

```text
candidate_actions = [40, 39, 41, 31]
actions = 40  # 兼容旧格式的 top-1 标签
```

训练损失为：

```text
总损失 = 0.7 × top-1 交叉熵
       + 0.3 × top-4 软标签交叉熵
```

可以把它理解成：

- 主要鼓励网络选择专家认为最好的动作；
- 如果网络选择了专家排名第二或第三的合理动作，也不要像完全错误的动作那样重罚。

遇到一步必胜或必须封堵时，专家只给出明确的战术动作，不使用模糊的 top-4 标签。

## 8. 哪些样本会被重点学习

并不是所有错误都同样严重。

当前实现使用以下 loss 权重：

| 样本类型 | 权重 | 含义 |
|---|---:|---|
| 一步必胜、必须封堵 | 4 | 这种错误通常直接决定胜负 |
| 制造双杀、阻止双杀 | 2 | 重要战术局面 |
| 普通位置判断 | 1 | 一般棋形选择 |

此外，空棋盘和常见开局会在数据中重复很多次。代码会统计 canonical 状态出现频率，对高频状态进行 inverse-sqrt 降权。

直观理解是：

```text
出现 1 次的状态：正常权重
出现很多次的相同状态：每条样本权重变小
```

这样可以避免大量重复开局淹没真正稀有的中盘和带外局面。

## 9. 新旧数据怎样混合

后续训练不会只使用最新一轮数据，也不会让最早的专家数据占据全部 batch。

目标比例为：

- bootstrap：30%；
- 最近两轮 DAgger：每轮 25%，合计 50%；
- 更早的 DAgger 数据：合计 20%。

同时尽量保持：

- 黑棋样本和白棋样本各 50%；
- 前 0～15 手约占 30%；
- 第 16～35 手约占 45%；
- 第 36 手以后约占 25%。

如果某一类样本数量不足，剩余比例会分配给其他类别。

## 10. 每一轮是继续训练还是重新训练

Round 0 没有旧模型，因此从随机初始化开始训练。

后续轮次会加载上一轮的 `best.pt` 作为初始网络，这叫 warm-start：

```text
上一轮 best.pt
      ↓ 加载模型权重
新一轮模型
      ↓ 使用新旧数据继续监督训练
新的 best.pt
```

只继承网络权重，不继承 optimizer 和学习率调度器。这样既保留上一轮已经学会的棋形，又让新一轮训练有干净的优化状态。

默认网络为 128 个 hidden channels、8 个残差块：

- Round 0 学习率：`3e-4`；
- 后续轮次学习率：`1e-4`；
- batch size：256；
- 最多 100 epoch；
- 连续 10 个 epoch 没有更好的 checkpoint 时停止。

## 11. 为什么不能只看验证集准确率

普通验证集准确率回答的是：

> 对于和训练数据比较相似的单个状态，Agent 有多少次选择了专家 top-1？

它不能完整回答：

> Agent 连续下完整盘棋时，早期错误会不会积累？

所以当前实现会在 bootstrap 后建立一个固定 challenge bank，并且以后不能把这些状态加入训练集。

Challenge bank 包含：

- 20,000 个扰动开局产生的带外状态；
- 2,000 个一步必胜状态；
- 2,000 个必须封堵状态；
- 1,000 个 fork 或 block-fork 状态；
- 500 条合法对局前缀。

每轮还会在本轮 Agent 真正访问的 DAgger 数据上进行 rollout audit。

## 12. 怎样判断“达到专家水平”

必须连续两轮同时通过以下门槛：

| 指标 | 要求 | 它在检查什么 |
|---|---:|---|
| 非法动作 | 0 | 基本正确性 |
| OOD top-1 一致率 | ≥85% | 带外局面首选动作模仿程度 |
| OOD top-4 一致率 | ≥97% | 是否至少选择了专家认可的动作 |
| 必胜/必堵正确率 | ≥99.5% | 关键战术不能犯错 |
| fork top-4 | ≥98% | 双杀相关战术 |
| 当前 rollout top-4 | ≥95% | 当前 Agent 自己访问的局面 |
| 对专家 score rate | 黑白均 ≥45% | 实际对战水平 |
| score 95% 区间下界 | ≥40% | 避免少量对局造成偶然高分 |
| 决胜局比例 | ≥20% | 避免全部和棋造成假通过 |

对战评测不会总是从空棋盘开始，而是先重放 challenge bank 中的 500 条合法前缀，再让 Agent 和确定性的 top-1 专家继续下完。

每条前缀都会交换 Agent 的黑白身份。

## 13. “冻结专家”具体是什么意思

你可以把当前专家理解为一份普通的、手写的启发式程序。它不会学习，也不会因为 DAgger 进行了新一轮就自动改变。

在当前 pipeline 中，从 Bootstrap 到最后一轮 DAgger，调用的始终都是同一个 `HeuristicAgent`：

```text
同一个棋盘 + 同一个执棋方
          ↓
heuristic-v1
          ↓
相同的候选动作及排名
```

因此，“冻结”不是说程序原本会自行变化，需要在运行时把它锁住；它是一条贯穿整个实验的开发约定：**一旦开始生成这次实验的数据，就不再修改专家的决策含义。**

### 什么情况属于没有冻结

例如训练到第三轮时，开发者发现专家棋力还可以提高，于是直接修改了以下任意内容：

- 调整棋形评分权重；
- 改变“必胜、必堵、双杀、普通评分”的判断顺序；
- 增加一层搜索；
- 改变同分动作的排序规则；
- 修改 `max_candidates` 或 top-k 的含义。

这时，旧数据由旧专家标注，新数据由新专家标注。同一个棋盘就可能出现两套不同答案：

```text
旧专家：40 > 39 > 41 > 31
新专家：39 > 40 > 31 > 41
```

如果仍把它们当成同一批监督数据混合训练，标签就可能互相冲突；评测时也说不清 Agent 到底应该模仿哪个版本。这就是“专家没有冻结”。

### 当前代码怎样表示这项约定

当前专家被明确命名为 `heuristic-v1`。数据集和 checkpoint 会保存版本、`max_candidates`、top-k 及由这些配置生成的 identity hash。训练时发现身份不一致，就拒绝混用。

需要注意：这个 hash 用来识别声明的版本和配置，并不会自动扫描专家全部源代码。因此，如果确实修改了专家的决策逻辑，也必须把版本升级为 `heuristic-v2`，并重新生成与它配套的数据和 challenge bank。相关黄金局面测试则负责发现“代码变了但输出不该变”的意外回归。

允许修改的是不影响答案的性能实现。例如把重复计算改成缓存，只要对黄金局面的 top-k 排名、战术原因都完全一致，它仍然属于 `heuristic-v1`。当前缓存优化就是这种情况：它只让标注更快，不改变专家认为哪一步更好。

简而言之：

> 当前专家自始至终确实是一致的；“冻结”是在强调我们有意保证并检查这种一致性，而不是说专家本身也在训练。

## 14. 运行完整训练

先激活环境：

```bash
conda activate mygames
```

开始一个正式 run：

```bash
bash BC/run_pipeline.sh 9x9-quality-v1
```

等价命令：

```bash
python BC/pipeline.py --run-name 9x9-quality-v1
```

查看参数：

```bash
python BC/pipeline.py --help
```

默认使用 16 个数据生成进程，并要求 CUDA 可用。正式训练预计需要数天。

可以通过环境变量调整常用配置：

```bash
GEN_WORKERS=16 \
EVAL_WORKERS=16 \
TRAIN_WORKERS=4 \
EPOCHS=100 \
BATCH_SIZE=256 \
bash BC/run_pipeline.sh 9x9-quality-v1
```

## 15. 中断后怎样恢复

使用相同的 run name 重新执行即可：

```bash
bash BC/run_pipeline.sh 9x9-quality-v1
```

Pipeline 会：

- 跳过已经完整生成的数据集；
- 如果某轮存在 `latest.pt`，从该 checkpoint 恢复；
- 已完成的复合评测不会重复运行；
- 从 `pipeline_state` 中读取当前状态。

数据集完成后被视为不可变。如果要修改生成参数，应使用新的 run name，而不是覆盖旧数据。

## 16. 产物放在哪里

```text
BC/
├── data/<run>/
│   ├── round_00_bootstrap/
│   ├── round_01_dagger/
│   ├── round_02_dagger/
│   └── challenge_v1.npz
├── checkpoints/<run>/
│   ├── round_00/best.pt
│   ├── round_01/best.pt
│   └── ...
├── evaluations/<run>/
│   ├── round_00.json
│   └── ...
├── runs/<run>/              # TensorBoard 和 console.log
└── pipeline_state/<run>/state.json
```

主要观察内容：

- `data/.../metadata.json`：样本数量、来源、覆盖率和专家版本；
- `data/.../diversity.json`：棋盘状态与轨迹多样性；
- `checkpoints/.../best.pt`：本轮最佳模型；
- `evaluations/...json`：复合验收结果；
- `pipeline_state/.../state.json`：当前轮次、连续通过次数和最终状态。

TensorBoard：

```bash
tensorboard --logdir BC/runs/9x9-quality-v1
```

## 17. 训练过程中重点看什么

### 17.1 TensorBoard 中最重要的三条曲线

在每个 `round_XX_train` run 中，优先观察以下三条 `Challenge/*` 曲线：

| TensorBoard 曲线 | 目标 | 含义 |
|---|---:|---|
| `Challenge/tactical_accuracy` | ≥99.5% | 一步必胜和必须封堵局面的 top-1 正确率 |
| `Challenge/ood_accuracy` | ≥85% | 固定挑战集中带外局面的专家 top-1 一致率 |
| `Challenge/ood_top4_accuracy` | ≥97% | 带外局面中，Agent 是否至少选择了专家认可的 top-4 动作 |

它们比 `Train/loss` 或普通的 `Validation/accuracy` 更能反映 DAgger 是否真的提升了 Agent 在自身行为分布上的表现。

训练代码选择 `best.pt` 时使用以下字典序：

```text
战术正确率是否达到 99.5%
          ↓
Challenge OOD top-1
          ↓
Challenge OOD top-4
          ↓
Challenge loss
```

因此，`latest.pt` 只是最近一个 epoch 的恢复点，并不一定比 `best.pt` 更强。某一条曲线的单独最高点也不一定对应 `best.pt`，因为 checkpoint 是按上面的组合顺序选出的。

`Train/loss`、`Validation/loss` 和学习率仍然值得观察，它们主要用于判断是否收敛、是否过拟合以及学习率是否下降，不应代替上面三条 challenge 曲线。

### 17.2 TensorBoard 不是完整的棋力验收

TensorBoard 中的训练曲线只评估单个棋盘状态。每轮训练结束后，还必须查看：

```text
BC/evaluations/<run>/round_XX.json
```

这个文件还包含：

- `rollout_audit.top4_accuracy`：本轮 Agent 实际访问局面的 top-4 一致率；
- `matches.colors.black/white.score_rate`：Agent 分别执黑、执白时对专家的得分率；
- `score_rate_ci95`：得分率的 95% 置信区间；
- `decisive_game_rate`：非和棋比例；
- `passed`：本轮是否通过第 12 节定义的完整复合门槛。

最终是否达到目标应以这个 JSON 的复合评测为准，不能仅根据 loss 下降或某一条 TensorBoard 曲线上升作判断。

以仓库现有的 `9x9-quality-v1/round_01.json` 为例：战术正确率为 96.39%，OOD top-1 为 75.27%，OOD top-4 为 95.92%；Agent 对专家执黑得分率为 55.9%，执白得分率为 39.1%，所以该轮的 `passed` 为 `false`。这个例子也说明，单看执黑表现或 rollout 指标可能高估整体棋力，必须同时检查固定挑战集和黑白双方的对局结果。

## 18. 训练过程中怎样体验 checkpoint

`play.py` 可以在 pipeline 继续运行时加载已经保存的 checkpoint。为了避免和训练争抢 GPU，阶段性体验时建议使用 CPU。

例如，体验第一轮 DAgger 的最佳模型：

```bash
conda run -n mygames python BC/play.py \
  --run-name 9x9-quality-v1 \
  --stage round_01 \
  --checkpoint-name best.pt \
  --board-size 9 \
  --device cpu \
  --bc-policy greedy
```

比较 DAgger 前后的体验时，分别启动对应轮次：

```bash
# Bootstrap 模型
conda run -n mygames python BC/play.py \
  --run-name 9x9-quality-v1 \
  --stage round_00 \
  --checkpoint-name best.pt \
  --board-size 9 \
  --device cpu \
  --bc-policy greedy

# 第一轮 DAgger 模型
conda run -n mygames python BC/play.py \
  --run-name 9x9-quality-v1 \
  --stage round_01 \
  --checkpoint-name best.pt \
  --board-size 9 \
  --device cpu \
  --bc-policy greedy
```

当新一轮正在训练且已经生成 `latest.pt` 时，可以体验当前 epoch：

```bash
conda run -n mygames python BC/play.py \
  --run-name 9x9-quality-v1 \
  --stage round_02 \
  --checkpoint-name latest.pt \
  --board-size 9 \
  --device cpu \
  --bc-policy greedy
```

选择 checkpoint 时遵循以下原则：

- 阶段性验收和跨轮比较使用 `best.pt`；
- 只想感受当前训练进度时使用 `latest.pt`；
- 显式指定 `--stage`，避免新一轮 checkpoint 出现后默认解析结果发生变化；
- checkpoint 使用临时文件加原子替换保存，训练期间读取已完成的文件不会读到半写状态；
- `play.py` 启动时只加载一次 checkpoint，不会自动热更新。想体验更新后的 epoch，需要退出并重新启动。

`--bc-policy greedy` 每一步都选择最高 logit 动作，与正式 challenge 对局的 argmax 策略一致，适合严谨比较不同 checkpoint。若想体验更有变化的开局，可以使用：

```bash
conda run -n mygames python BC/play.py \
  --run-name 9x9-quality-v1 \
  --stage round_01 \
  --checkpoint-name best.pt \
  --board-size 9 \
  --device cpu \
  --bc-policy controlled \
  --play-seed 0
```

`controlled` 默认只在 Agent 的前 6 手从 top-4 中受控采样；遇到一步必胜或必须封堵时会强制使用 greedy 动作。固定 `--play-seed` 可以让采样过程可复现。

当前 `play.py` 固定人类执黑、Agent 执白。因此它适合直观感受 Agent 的白棋防守与应对能力，但不能代替同时交换黑白身份的正式评测。Agent 执黑和执白的完整结果仍应查看 `evaluations/<run>/round_XX.json`。

## 19. 各文件负责什么

| 文件 | 作用 |
|---|---|
| `heuristic_agent.py` | 启发式专家的具体棋力逻辑 |
| `oracle.py` | 固定专家版本、原因和数据编码 |
| `generate.py` | Bootstrap 和 DAgger 对局、专家标注 |
| `cache.py` | 专家查询缓存和跨轮合并 |
| `dataset.py` | 读取数据、旋转镜像增强、过滤 challenge |
| `train.py` | top-1/top-4 监督训练和 checkpoint |
| `challenge.py` | 建立固定测试集并执行复合验收 |
| `diversity.py` | 统计状态、轨迹、阶段和来源多样性 |
| `pipeline.py` | 串联多轮训练并负责恢复和停止 |
| `eval.py` | 传统的空棋盘对专家评测 |
| `play.py` | 人类与 BC Agent 对弈 |

## 20. 常见疑问

### DAgger 是不是强化学习？

不是。DAgger 改变的是训练数据从哪里来，网络更新仍然使用监督学习和交叉熵。

### Agent 实际走错的动作会不会成为标签？

不会。错误动作只负责把棋局带到 Agent 容易访问的状态，监督标签始终来自专家。

### 为什么还要保留 bootstrap 数据？

只训练最新的异常局面可能让 Agent 忘记正常专家棋形。保留 bootstrap 可以防止这种遗忘。

### top-4 会不会让 Agent 学得不坚定？

top-1 仍占损失的 70%。top-4 只是告诉网络：第二、第三候选比完全错误的动作更合理。必胜和必堵仍然使用明确硬标签。

### 为什么不一直生成更多随机局面？

完全随机的棋盘可能很不自然。当前实现只生成按照规则一步步走出来的合法局面，并优先在已有棋子附近进行扰动。

### 旧的 5×5 数据还能用吗？

旧格式仍可读取，但新的质量优先 pipeline 固定为 9×9，不会把旧 5×5 数据混入 9×9 训练。

## 21. 运行测试

```bash
python -m pytest -q
```

普通测试会检查专家输出、top-k loss、数据格式、cache、challenge 隔离和 pipeline 参数。完整的 9×9 多轮训练属于长时间任务，不会在单元测试中自动运行。
