# afterBC：从 BC_BEST 继续训练白棋

## 1. 背景

本项目使用强化学习和模仿学习训练 9x9 五子棋 Agent，共享环境由
`gomoku_env/` 提供。

- `BC/` 使用 BC + DAgger 模仿启发式规则 Agent，是目前已经验证成功的方案。
- `BC/checkpoints/9x9-quality-v1/round_06/best.pt` 是该方案得到的较强模型，本文称为
  **BC_BEST**。
- `AZ/` 尝试在 BC 基础上用 AlphaZero 继续训练，没有取得预期结果。
- `DQN/` 尝试以自博弈为主从零训练，也没有取得预期结果。

`afterBC/` 是一条独立的新训练路线：保留 BC_BEST 已经学到的棋力，将其同时作为
白棋初始化和冻结黑棋对手，再用分布式 Dueling Double DQN 继续优化白棋。

## 2. 目标与代码边界

训练目标是让白棋在后手条件下提高对 BC_BEST 黑棋的胜率：

1. 黑棋始终由冻结的 BC_BEST 控制，不参与梯度更新。
2. 白棋不是从零开始，DQN 网络从同一份 BC_BEST 权重初始化。
3. 对白棋而言，一次环境动作就是“白棋落子”；环境随后自动完成黑棋回应。
4. 使用 Ape-X 风格的 actor-gather-learner 架构提高 CPU 吞吐和 replay 利用率。
5. 周期保存 checkpoint，并在独立 CPU 进程中执行 stochastic 评测。

除 `gomoku_env/` 外，`afterBC/` 不导入 `BC/`、`DQN/` 或 `AZ/` 的代码。为保持闭环，
BC_BEST 已复制到：

```text
afterBC/bootstrap/BC_BEST.pt
```

加载时会校验 SHA256：

```text
525e396d1ad97310101d3cd19f9eb2b647043cdfe785eb037928948c3d1df428
```

## 3. 白棋看到的环境

每个 actor 内部维护一批原始 `GomokuEnv`，但对 DQN 暴露的是白棋决策间隔：

```text
白棋状态 s_t
  -> 白棋动作 a_t
  -> GomokuEnv 执行白棋落子
  -> 若未终局，冻结 BC_BEST 黑棋自动回应
  -> 下一个白棋状态 s_(t+1) 或终局
```

因此 replay 中只保存白棋 transition。奖励保持稀疏：

```text
白胜 +1
白负 -1
和棋  0
非终局 0
```

每个环境独立维护 3-step accumulator。终局时会完整 flush 不足 3 步的尾部样本；
terminal transition 的 next state 和 next mask 为零，不进行 bootstrap。

## 4. 黑手：训练与评测统一的 stochastic 协议

早期实现中训练黑手始终 greedy，而评测黑手带随机性。这会让白棋只记住一条固定
greedy 路线，无法适应评测中的其他合理分支。当前训练和 stochastic 评测统一调用
`opponent.py` 中的同一套协议：

```text
黑棋前 6 次落子：
  - 读取冻结 BC_BEST 对全部合法动作的 logits
  - 只在排名 top-4 的动作中采样
  - rank-softmax temperature = 1.5
  - 若存在立即取胜或必须封堵的位置，强制使用 greedy BC_BEST

黑棋第 7 次及以后：
  - greedy BC_BEST
```

所有训练 actor 都使用该 stochastic 黑手。训练和评测的分布相同，但随机种子不同，
避免训练直接记忆固定评测轨迹。恢复训练时，actor seed 还会混入已完成的全局
transition 数，避免从头重复上一段随机对局。

评测仍额外保留一盘 deterministic greedy-vs-greedy 对局，用于检查已经发现的固定
主线是否遗忘；它不再是 `best.pt` 的首要选择标准。

## 5. BC 到 Dueling DQN 的初始化

BC_BEST 的输入是当前行动方视角的三通道棋盘：己方、对手、空位。白棋 DQN 沿用
相同编码，并使用与 BC 网络兼容的 128 通道、8 残差块卷积主干。

初始化规则：

- BC trunk 完整迁移到 DQN trunk。
- BC policy head 完整迁移到 Dueling 网络的 advantage head。
- 新增 value head，其最后一层初始化为零。
- target network 初始复制 online network。

该初始化保证 step 0 时 DQN 的 greedy 动作排名与 BC_BEST 完全一致，同时允许后续
TD 学习改变 Q 值。

## 6. Actor-Gather-Learner 架构

```text
Actor 0 --独立 transition queue--+
Actor 1 --独立 transition queue--+
...                              +--> Gather --> learner queue --> Learner / PER / GPU
Actor N --独立 transition queue--+

Learner --每 actor 独立权重队列-----------------------------> Actors
Actors  --共享 rollout 日志队列-----------------------------> Learner / TensorBoard
Checkpoint --------------------------------------------------> CPU Evaluator
```

### 6.1 Actor

- 默认 8 个独立 CPU 进程。
- 每个 actor 管理 16 个环境，并以 batch 方式执行白棋和黑棋网络推理。
- 每个 actor 每轮严格生成 256 条 transition。
- 每个 actor 有独立、容量为 1 的 transition queue。
- queue 满时 actor 循环阻塞等待，不能丢弃 transition。
- 发完一个 packet 后，actor 等待 gather 发放下一轮 permit。
- actor 只使用 CPU，默认每进程 1 个 PyTorch 推理线程。

### 6.2 Gather

Gather 是独立进程，严格按 actor id 顺序从每个 actor queue 读取一个完整 packet。
这种 barrier 设计确保 CPU 调度较忙时，快 actor 不会在 replay 中占据更高比例。

默认一轮数据量为：

```text
8 actors * 256 transition = 2048 global transition
```

Gather 收齐一轮后按以下顺序执行：

1. 立即向所有 actor 发送下一轮 permit，让 actor 开始采集下一批数据。
2. 将合并后的 2048 条 transition 通过 learner queue 交给 Learner。
3. 立即进入下一轮固定顺序收集。

因此 actor/gather 的下一轮采集与 Learner 的 replay/update 是并行的。learner queue
默认最多缓存 2 个 gather round；达到上限后自然形成背压，不丢数据。

### 6.3 Learner

主进程承担 Learner 职责，是唯一使用 GPU 的进程。默认直接使用 `cuda`，不执行
CUDA 可用性自动检测。Learner 负责：

- 接收 gather batch 并写入 prioritized replay。
- 从 replay 抽样并执行 Double-DQN 更新。
- 管理 online network、target network 和 optimizer。
- 定期将最新 online 权重转为 CPU NumPy snapshot，通过独立队列发给每个 actor。
- 收集 actor rollout 日志并写 TensorBoard。
- 保存 checkpoint、提交独立评测并维护 `latest.pt`/`best.pt`。

权重队列容量为 1；新权重可以替换尚未消费的旧 snapshot。该“只保留最新”规则只
适用于权重，不适用于 transition。

## 7. 探索策略

白棋使用 epsilon-greedy。不同 actor 的初始 epsilon 按 actor id 从 0.4 到 0.1
线性插值：

```text
epsilon_start(i) = 0.4 - 0.3 * i / (num_actors - 1)
```

所有 actor 使用 gather 已完成的全局 transition 数进行线性衰减，而不是使用各 actor
自己的局部计数：

```text
epsilon_i(s) = 0.05 + (epsilon_start(i) - 0.05) * max(0, 1 - s / 1_000_000)
```

达到 1M global transition 后，所有 actor 固定为 `epsilon=0.05`。epsilon 每个 actor
packet 更新一次，并在该 packet 内保持不变，避免异步调度造成探索率不一致。

黑棋不使用 epsilon-greedy；它只使用第 4 节定义的 BC top-4 受控随机协议。

## 8. DDDQN、PER 与更新比例

这里的 DDDQN 指 Dueling Double Deep Q-Network：

- online network 在 next state 上选择合法动作。
- target network 评估该动作的 Q 值。
- Dueling head 将 state value 与逐位置 advantage 合并。
- 非法位置在动作选择和 TD target 中始终 mask。
- loss 使用逐样本 Huber loss 和 PER importance weight。

Prioritized Replay 默认配置：

```text
capacity            = 1,000,000
min_replay_size     = 50,000
alpha               = 0.6
beta                = 0.4 -> 1.0（按全局训练进度线性增长）
new item priority   = 当前最大 priority
priority update     = abs(TD error) + epsilon
```

replay 存储紧凑的 int8 原始棋盘；抽样后才编码三通道输入。每次抽样在线执行随机 D4
旋转/镜像增强，并同步变换 state、next state、action 和 next action mask。

当前默认更新比例：

```text
updates_per_transition = 0.02
每轮 update credit     = 2048 * 0.02 = 40.96
learner batch size      = 256
sample-to-insert ratio  = 0.02 * 256 = 5.12
```

小数 update credit 会跨轮保留，因此长期平均每轮执行 40.96 次 optimizer update，而
不是简单固定为 40 次。replay 未预热到 50,000 条前不执行更新。

其他默认值：

| 参数 | 默认值 |
|---|---:|
| `gamma` | 0.99 |
| `n_step` | 3 |
| `lr` | 1e-4 |
| `batch_size` | 256 |
| `target_update` | 每 2,500 次 update 硬同步 |
| `weight_sync_updates` | 每 1,000 次 update 发布 actor 权重 |
| `grad_clip` | 10 |
| `total_transitions` | 20,000,000 global transition |

当前不使用 reward shaping。

## 9. TensorBoard

每次进程启动都会创建独立 TB run：

```text
afterBC/runs/<run-name>/tensorboard/
  <run-name>_YYYYMMDD_HHMMSS_pid<PID>/
```

恢复训练也会新建 event 目录，避免不同进程把 event 混写到同一文件。

首个完整 gather round 会立即上报一次；随后 `--log-interval` 按所有 actor 合计的
global transition 计数触发，默认每 100,000 条上报，而不是每 actor 100,000 条。

主要指标：

- `Rollout/*`：带白棋 epsilon 探索、stochastic 黑手的在线胜率、负率、和率、回报和局长。
- `Throughput/global_transitions_per_second`：从 Learner 视角观察的端到端全局吞吐。
- `Throughput/actor_collection_transitions_per_second`：所有 actor 原始采集能力之和。
- `Throughput/actor_<id>_collection_transitions_per_second`：单 actor 采集吞吐。
- `Throughput/learner_updates_per_second`：GPU optimizer update 速率。
- `System/actor_blocked_seconds`：actor 向 transition queue 写入时的阻塞时间。
- `System/actor_<id>_policy_lag`：Learner 与 actor 权重版本差。
- `Learner/*`：replay 大小、update 数、loss、Q、target、TD error、PER weight、梯度和 beta。
- `Exploration/*`：全局 epsilon 进度和每 actor 当前 epsilon。
- `Evaluation/stochastic_*`：checkpoint 对受控随机黑手的独立多局评测。
- `Evaluation/deterministic_white_win`：checkpoint 是否击败纯 greedy 黑手的辅助审计。

`Evaluation/stochastic_*` 包括：

```text
white_win_rate
white_loss_rate
draw_rate
white_score_rate = (white wins + 0.5 * draws) / games
mean_moves
success = white_score_rate > 0.5
```

默认 stochastic 评测只有 20 局，适合训练期间快速观察，但统计波动较大。需要判断
最终棋力时，应独立评测至少 200～500 局。

## 10. Checkpoint 与独立评测

默认在以下时机原子保存 checkpoint：

- step 0；
- 每 500,000 global transition；
- 正常训练结束时。

checkpoint 保存：online/target 网络、optimizer、全局 transition、update 数、剩余
update credit、随机状态、配置和 BC_BEST hash。checkpoint 不保存 replay buffer。

输出目录：

```text
afterBC/runs/<run-name>/
  checkpoints/step_<global_step>.pt
  evaluations/step_<global_step>.json
  tensorboard/<timestamp_pid>/events...
  latest.pt
  best.pt
  config.json
```

独立 evaluator 使用 CPU，按 FIFO 评测每个已提交 checkpoint，不跳过中间任务。
`best.pt` 的排序优先级为：

1. stochastic white score rate；
2. stochastic white win rate；
3. deterministic greedy 对局是否白胜。

旧版 evaluation JSON 中的 `statistical` 字段仍可读取；新输出统一使用更准确的
`stochastic` 命名。

## 11. 启动训练

项目运行环境为 Conda `mygames`。从仓库根目录启动：

```bash
conda run -n mygames python -m afterBC.train \
  --run-name white_apex_stochastic_v1
```

等价脚本：

```bash
bash afterBC/run_train.sh white_apex_stochastic_v1
```

查看全部参数：

```bash
conda run -n mygames python -m afterBC.train --help
```

启动 TensorBoard：

```bash
conda run -n mygames tensorboard --logdir afterBC/runs
```

## 12. 恢复训练

```bash
conda run -n mygames python -m afterBC.train \
  --run-name white_apex_stochastic_v1 \
  --resume
```

恢复内容包括模型、optimizer、global step、update step 和 epsilon 进度。由于 replay
不在 checkpoint 中，恢复后 replay 从空开始，并重新预热到 `min_replay_size` 才继续
梯度更新。这不是逐环境、逐样本的 bitwise continuation。

正在运行的 Python 进程不会热加载源码修改；修改训练协议后必须停止并重新启动进程。

## 13. 独立评测

```bash
conda run -n mygames python -m afterBC.evaluate \
  --checkpoint afterBC/runs/white_apex_stochastic_v1/best.pt \
  --stochastic-games 500 \
  --output afterBC/runs/white_apex_stochastic_v1/final_evaluation.json
```

评测白棋始终关闭 epsilon、使用 greedy DQN；随机性只来自冻结黑手的受控随机协议。
训练和评测代码共用同一个 `opponent.py`，防止协议再次发生漂移。

## 14. CPU 冒烟与测试

不使用 GPU 的短流程冒烟：

```bash
conda run -n mygames python -m afterBC.train \
  --run-name smoke \
  --device cpu \
  --num-actors 2 \
  --envs-per-actor 2 \
  --actor-batch-size 8 \
  --total-transitions 32 \
  --min-replay-size 1000 \
  --updates-per-transition 0 \
  --disable-evaluation \
  --disable-tensorboard
```

运行 afterBC 测试：

```bash
conda run -n mygames python -m pytest -q afterBC/tests
```

运行全仓回归：

```bash
conda run -n mygames python -m pytest -q
```

性能基准应在没有其他训练任务争用 CPU/GPU 时进行，否则 actor 吞吐和 GPU 利用率
不具备可比性。
