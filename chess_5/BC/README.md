# 五子棋行为克隆（BC）

`BC/` 是一套独立于 `DQN/` 的离线模仿学习流水线。核心思想可以直接理解为一个监督学习问题：

> 给定当前棋盘和当前行棋方，预测启发式专家会落在哪个位置。

启发式机器人只在“生成数据”和“评测”时运行。GPU 训练只读取已经落盘的数据，不会调用启发式搜索，因此 9×9 下较慢的专家不会拖慢每个训练 batch。

## 1. 一条训练数据是什么

每个样本包含：

- `board`：落子前的棋盘，`int8[H,W]`，其中黑棋为 `1`、白棋为 `-1`、空位为 `0`；
- `player`：当前行棋方，取值为 `1` 或 `-1`；
- `action`：启发式专家在该局面选择的位置；
- `game_id`：样本属于哪一局棋，用于按完整棋局划分训练集和验证集。

合法动作 mask 不需要保存，因为所有 `board == 0` 的位置就是合法动作。数据按 worker 写入压缩的 `shard-*.npz`，数据版本的参数和 shard 列表记录在 `metadata.json` 中。

训练前，棋盘会转换为当前行棋方视角的三个输入通道：

```text
通道 0：当前行棋方的棋子
通道 1：对方的棋子
通道 2：空位
```

因此同一个网络可以同时学习执黑和执白，不需要分别训练两个模型。

## 2. 数据如何产生

### 2.1 初始专家数据：`--mode expert`

启发式机器人同时控制黑棋和白棋。每次落子前保存当前状态，并把专家的落子作为监督标签：

```text
局面 s ──启发式专家──> 动作 a
  └──────── 保存 (s, a) ────────┘
```

这样得到的都是专家自己会访问到的局面，适合训练第一版 BC 模型。

`generate.py` 使用多个 CPU 进程并行下完整棋局。每个进程直接写自己的 shard，避免把每一步通过进程队列传回主进程。

### 2.2 专家查询缓存

启发式搜索是数据生成阶段最昂贵的操作，尤其是在 9×9 中盘。`cache.py` 和 `symmetry.py` 会：

1. 将一个棋盘的 8 种旋转/镜像形式映射到同一个 canonical key；
2. 用 SQLite 持久化该 key 对应的专家动作；
3. 再遇到相同或对称局面时直接复用结果，并把动作坐标变换回原棋盘。

默认每个 canonical 状态缓存 4 次专家选择，再从中随机采样。这是因为专家在多个动作同分时带有随机性；只缓存一个动作会让自博弈反复生成几乎相同的轨迹。可用 `--cache-labels-per-state` 调整这个数量。

缓存会校验棋盘尺寸、专家参数、随机种子等配置，防止 5×5、9×9 或不同专家配置之间错误复用。

### 2.3 数据版本与中断恢复

- shard 先写临时文件，完成后再原子重命名；
- 产数中断后，可以使用完全相同的参数重新执行命令；已有完整 shard 会被跳过；
- 当 `metadata.json` 的状态变为 `complete` 后，该目录被视为不可变数据版本；
- 每轮聚合必须写入新的目录，checkpoint 会记录自己使用过的数据版本。

### 2.4 数据多样性监控

产数结束后，`diversity.py` 会扫描一次所有 shard。它不会再次调用专家，也不进入训练 batch 的热路径。扫描结果写入数据目录的 `diversity.json`、`metadata.json`，同时进入当前产数步骤的 TensorBoard。

保留三项互补的核心指标：

- `canonical_effective_trajectory_ratio`：先把旋转/镜像等价的完整轨迹合并，再根据轨迹频率的熵计算“有效轨迹数 ÷ 总局数”。越接近 `1`，说明每局提供的信息越独立；接近 `0` 表示大量棋局集中在少数轨迹上。这是总体多样性的主指标。
- `dominant_canonical_trajectory_fraction`：出现次数最多的 canonical 轨迹占全部棋局的比例。越低越好；如果接近 `1`，说明几乎一直在重复同一盘棋。
- `canonical_state_unique_ratio`：消除旋转/镜像后，独特棋盘状态数除以总样本数。它补充衡量局面覆盖范围；固定中心开局等合理重复会使该指标低于 `1`。

例如：

```text
canonical_effective_trajectory_ratio = 0.08
dominant_canonical_trajectory_fraction = 0.42
canonical_state_unique_ratio = 0.31
```

这表示数据虽然可能有很多对局，但有效轨迹只相当于总局数的约 8%，且一种轨迹占了 42%，需要警惕专家自博弈分支不足。当前实现只做可观测性，不用未经实测校准的固定阈值自动中止 pipeline；应先根据 5×5 正式数据建立正常区间。

## 3. 数据如何用于训练

`dataset.py` 读取 shard，并按完整 `game_id` 划分训练集和验证集。这样同一局中的相邻状态不会一部分进入训练集、一部分进入验证集。

训练样本会在线随机应用 8 种旋转/镜像变换，棋盘和动作坐标一起变换。数据文件不会预存 8 份副本，但每个专家查询可以在训练中覆盖所有对称形式。

`network.py` 定义全卷积残差策略网络：

```text
[B, 3, H, W]
      │
卷积层 + 残差块
      │
每个格子的 action logit
      │
[B, H × W]
```

网络没有依赖固定棋盘面积的全连接输出层，所以同一套结构支持 5×5 和 9×9；但两个尺寸应分别生成数据、分别训练 checkpoint。

`train.py` 的训练过程就是标准分类监督学习：

1. 网络为每个格子输出一个 logit；
2. 已经有棋子的格子被 mask 为极小值；
3. 使用专家动作作为类别标签，计算 cross-entropy；
4. 使用 AdamW、梯度裁剪和学习率调度更新网络；
5. 根据验证集 loss early stopping，并保存 `best.pt` 和 `latest.pt`。

日志中的主要指标是：

- `loss`：专家动作的交叉熵；
- `accuracy`：预测动作与这一次专家标签完全一致的比例；
- `legal_rate`：预测动作是否合法，正常情况下必须为 `1.0`；
- `samples/s`：训练吞吐量。

由于专家同分动作可能不唯一，`accuracy` 不必达到 100%。最终是否成功应以对战结果为准。

## 4. 为什么纯 BC 会产生分布偏移

第一版数据来自“专家 vs 专家”，模型只见过专家会到达的局面。但部署时模型不可能每一步都完全复制专家：

```text
专家数据：专家动作 → 专家熟悉的下一局面 → 专家动作
实际对局：模型失误 → 训练集中少见的局面 → 更容易继续失误
```

一个小错误会改变后续棋盘分布，错误可能沿整局累积。这通常称为 covariate shift，也就是这里要处理的局外/分布偏移。

## 5. 离线数据聚合如何解决分布偏移

本项目使用 DAgger 风格的分轮离线聚合，而不是在 GPU 训练过程中实时调用专家。

执行 `generate.py --mode aggregate --checkpoint ...` 时：

1. 加载一个冻结的 BC checkpoint；
2. BC 和启发式专家对局，BC 轮流被安排执黑和执白；
3. BC 行棋时，真正落到棋盘上的是 BC 自己的动作，因此后续状态来自 BC 的实际访问分布；
4. 对每一个访问到的状态都询问专家“这个状态下你会怎么走”；
5. 数据中保存的是专家动作，而不是 BC 动作；
6. 聚合结束后关闭专家进程，再用基础数据和聚合数据共同训练新模型。

最关键的一步可以表示为：

```text
状态 s ──BC──> 实际动作 a_bc ──> 决定下一状态
   │
   └──启发式专家──> 标签 a_expert ──> 保存 (s, a_expert)
```

这样模型会学到：“即使我已经走到了自己容易犯错的局面，专家会怎样纠正。”这正是聚合数据相对专家自博弈数据的价值。

基础专家数据始终保留，防止模型只学习异常局面。后续每轮聚合数据可用 `--aggregate-max-samples` 限制采样量，避免它淹没基础数据。

## 6. 完整 pipeline

```text
启发式专家自博弈
        │
        ▼
初始数据 expert-v1 ──> 训练 BC-v1 ──> 双执子评测
                           │
                    未达到目标
                           │
                           ▼
              BC-v1 与专家对局并补专家标签
                           │
                           ▼
                  聚合数据 aggregate-v2
                           │
          expert-v1 + aggregate-v2
                           │
                           ▼
                       训练 BC-v2
                           │
                 评测；必要时继续下一轮
```

建议先在 5×5 完整跑通，再使用相同流程训练独立的 9×9 数据和模型。

### 一条命令运行完整 pipeline

`run_pipeline.sh` 会依次执行下面列出的五个步骤。直接执行时会自动生成带时间戳的 run name：

```bash
bash BC/run_pipeline.sh
```

更推荐显式指定 run name。这样进程中断后，可以用同一条命令恢复：

```bash
bash BC/run_pipeline.sh my-5x5-run
```

脚本具有以下行为：

- 任一步骤失败都会立即停止，不会带着残缺数据继续下一步；
- 已完成的数据生成、评测和训练步骤会被跳过；
- 训练中断时，同一个 run name 会从对应的 `latest.pt` 继续；
- 每一步的控制台输出保存在该步骤 TensorBoard 目录下的 `console.log`；
- 数据、checkpoint、评测结果、日志和 pipeline 状态都按 run name 隔离。

常用参数通过环境变量调整。例如先跑一个较小的 5×5 实验：

```bash
BOARD_SIZE=5 \
EXPERT_GAMES=1000 \
AGGREGATE_GAMES=500 \
EPOCHS=30 \
bash BC/run_pipeline.sh 5x5-small
```

脚本支持的主要环境变量如下：

- `BOARD_SIZE`：棋盘尺寸，默认 `5`；
- `EXPERT_GAMES`：初始专家自博弈局数，默认 `10000`；
- `AGGREGATE_GAMES`：离线聚合局数，默认 `5000`；
- `GEN_WORKERS`：产数进程数，默认 `16`；
- `EVAL_GAMES`：每种执子颜色的评测局数，默认 `200`；
- `EVAL_WORKERS`：评测进程数，默认 `8`；
- `EPOCHS`、`BATCH_SIZE`、`TRAIN_WORKERS`、`DEVICE`：训练 epoch、batch 大小、DataLoader 进程数和设备；
- `CACHE_LABELS_PER_STATE`、`MAX_CANDIDATES`：专家缓存标签数和专家候选数；
- `ARTIFACT_ROOT`：全部输出的根目录，默认 `BC/`；
- `CONDA_ENV`：conda 环境名，默认 `mygames`。

### TensorBoard 日志组织

五个步骤共享同一个顶层 run name，每个步骤拥有带序号和名称的子 run：

```text
BC/runs/<run-name>/
├── 01_generate_expert/
├── 02_train_bc_v1/
├── 03_eval_bc_v1/
├── 04_generate_aggregate/
└── 05_train_bc_v2/
```

一次监控整个 pipeline：

```bash
conda run -n mygames tensorboard --logdir BC/runs/<run-name>
```

TensorBoard 会把五个子目录显示为五条带步骤名的 run，但它们都归属于同一个顶层实验目录。产数步骤记录上述三项 `Diversity/*` 指标，以及样本量、缓存命中率、专家查询吞吐和写盘耗时；训练步骤按 epoch 记录 loss、accuracy、合法动作率、吞吐和学习率；评测步骤记录黑白双方的胜负和、得分率、平均局长和非法动作数。

### 第一步：生成 5×5 初始专家数据

```bash
conda run -n mygames python BC/generate.py \
  --output BC/data/5x5-expert-v1 --mode expert --board-size 5 \
  --games 10000 --workers 16 --seed 0
```

### 第二步：训练第一版 BC

```bash
conda run -n mygames python BC/train.py \
  --data-dir BC/data/5x5-expert-v1 \
  --run-name 5x5-bc-v1 --board-size 5
```

### 第三步：分别执黑、执白评测

```bash
conda run -n mygames python BC/eval.py \
  --checkpoint BC/checkpoints/5x5-bc-v1/best.pt \
  --board-size 5 --games-per-color 200 --workers 8 --seed 10000
```

评测输出胜、负、和、原始胜率、得分率 `win + 0.5 × draw`、95% 置信区间、平均局长和非法动作数。最终确认时应再换一组固定种子，例如 `--seed 20000`。

### 第四步：生成一轮聚合数据

```bash
conda run -n mygames python BC/generate.py \
  --output BC/data/5x5-aggregate-v2 --mode aggregate --board-size 5 \
  --checkpoint BC/checkpoints/5x5-bc-v1/best.pt \
  --games 5000 --workers 16 --seed 0
```

### 第五步：混合基础数据与聚合数据训练

```bash
conda run -n mygames python BC/train.py \
  --data-dir BC/data/5x5-expert-v1 BC/data/5x5-aggregate-v2 \
  --aggregate-max-samples 100000 \
  --run-name 5x5-bc-v2 --board-size 5
```

之后重新评测 `5x5-bc-v2/best.pt`。如果仍未达到目标，就用 BC-v2 生成 `aggregate-v3`，训练时同时传入基础数据和需要保留的各轮聚合数据。

## 7. 目录职责

- `generate.py`：专家自博弈与 BC–专家聚合产数；
- `cache.py`：持久化专家查询缓存；
- `symmetry.py`：棋盘 canonicalization、旋转/镜像与动作坐标变换；
- `diversity.py`：产数后计算 canonical 轨迹和状态多样性；
- `dataset.py`：读取 shard、整局切分、三通道编码和在线增强；
- `network.py`：全卷积残差策略网络；
- `agent.py`：checkpoint 加载、批量推理和合法动作 mask；
- `train.py`：监督训练、验证、early stopping 和 checkpoint；
- `eval.py`：分别执黑、执白对战启发式专家。
- `run_pipeline.sh`：串联五个步骤、组织统一日志，并处理中断恢复。
