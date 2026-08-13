# Tetris RL

Gymnasium Tetris environments and a multiprocessing actor-learner Dueling Double DQN baseline.

Training uses a placement-level environment: one transition selects the current
piece's rotation and target column, then the environment moves and hard-drops the
piece. The original frame-level environment is retained for keyboard play and
low-level movement tests, but DQN checkpoints always use the 40-output placement
action space.

## Run

```bash
python -m DQN.train --config configs/ddqn_default.toml --device cuda
tensorboard --logdir runs/dddqn
```

For a small CPU smoke run:

```bash
python -m DQN.train --config configs/ddqn_default.toml --device cpu \
  --num-actors 2 --envs-per-actor 2 --total-transitions 2000
```

Play the environment with the keyboard:

```bash
python -m TetrisEnv.play --fps 8
```

Use Left/Right to move, Space to rotate clockwise, and Down to hard drop. Press Enter or R after game over to restart.

TensorBoard and checkpoints share a unique run ID. Checkpoints are stored below
`checkpoints/<run-id>/`, so rerunning training never overwrites an earlier model.
Evaluate a checkpoint without graphics (the default):

```bash
python -m DQN.evaluate checkpoints/<run-id>/dddqn_250000.pt --episodes 10
```

Show the agent while it evaluates:

```bash
python -m DQN.evaluate checkpoints/<run-id>/dddqn_250000.pt \
  --episodes 3 --max-steps 20000 --render --render-fps 10
```

## 动作空间设计：Placement 动作是关键

本项目的 DQN 使用专门设计的 **placement 动作**，而不是逐帧控制动作。每个
transition 直接决定当前方块的旋转状态和目标列（共 40 个动作），环境随后完成
移动、下落和锁定。这样一条 transition 就对应“放置一个方块”，消行、棋盘变化
和终局结果能够更直接地归因到这次落点决策。

此前使用 frame-level 动作时，agent 需要经过一长串旋转、左右移动和下落动作才
得到结果；训练 4000 万步仍完全没有学会消行，主要困难在于动作序列过长、信用
分配稀疏。改为 placement 动作后，训练约 500 万步已经出现了稳定的消行能力：

```text
mean_return: 9.35
mean_survival_pieces: 68
mean_lines: 14
```

这组结果说明，对于本任务，动作空间的抽象方式比单纯增加训练步数更关键。当前
训练和评估使用 placement 环境；原有 frame-level 环境仍保留用于键盘操作和底层
环境测试。

## TensorBoard 指标说明

启动训练后，使用下面的命令查看指标：

```bash
tensorboard --logdir runs/dddqn
```

每次训练都会写入 `runs/dddqn/<run-id>/`，可以在 TensorBoard 中选择不同
run 进行比较。所有标量图的横轴 `Step` 都表示 learner 已接收的 transition
数量；在 placement 环境中，一条 transition 就是放置一个方块，而不是一次键盘
输入或画面帧。常规指标每累计 `tb_log_every` 条 transition 刷新一次，默认值为
`10000`。

指标名后缀和统计方式需要先区分：

- 无 `cumulative` 后缀的 `train/` 指标，通常是本次写入区间内所有梯度更新的
  均值；`train/epsilon`、`train/lr`、`train/replay_size` 和
  `train/throughput` 记录区间末尾的最新值。
- `episode/` 指标是本次写入区间内所有已结束回合的均值。actor 的指标队列采用
  尽力发送策略，队列拥堵时可能少量缺失，但不会丢失 replay 中的训练样本。
- 带 `cumulative` 后缀的指标从本次 run 开始累计，适合观察长期比例或总量，
  不应当成最近一个记录区间的瞬时值。
- TensorBoard 的 smoothing 只改变曲线的显示效果，不改变事件文件里的原始值。

### `train/`：网络优化状态

| 指标 | 含义与解读 |
| --- | --- |
| `train/loss` | 当前动作 Q 值与 Double DQN TD target 之间的 Smooth L1（Huber）损失。下降通常表示拟合趋于稳定，但强化学习的目标本身会变化，因此不要求单调下降，也不能仅凭低 loss 判断策略优秀。 |
| `train/q_mean` | mini-batch 中在线网络对实际采样动作给出的 Q 值均值，即对未来折扣回报的当前估计。应结合 `target_mean` 和评估回报观察；持续无界增大可能表示价值过估计或训练不稳定。 |
| `train/target_mean` | Double DQN 目标 `reward + gamma * (1 - terminated) * next_q` 的均值。它与 `q_mean` 长期严重偏离时，通常会同时表现为 loss 较高。 |
| `train/gradient_norm` | 梯度裁剪前的总梯度范数；当前上限由 `gradient_clip_norm` 控制。长期远高于上限说明裁剪频繁发生，可能需要检查学习率、奖励尺度或异常样本。 |
| `train/epsilon` | 所有 actor 固定 epsilon 的均值。默认使用 4 个 actor，epsilon 分别为 `0.05、0.10、0.20、0.40`，因此该指标保持为 `0.1875`。 |
| `train/lr` | optimizer 当前实际使用的学习率。当前配置没有学习率退火，所以曲线应为水平线；若以后接入 scheduler，该曲线会记录退火后的实际值。 |
| `train/replay_size` | replay buffer 当前保存的 transition 数量；达到 `replay_capacity` 后保持在容量上限，旧样本会被新样本覆盖。 |
| `train/throughput` | learner 从启动至当前累计接收的 transition 数除以实际经过秒数，单位约为 transition/s。这是全程平均吞吐量，不是单个记录区间的瞬时速度。 |

### `episode/`：带探索策略的训练回合

这些数据来自 actor 正在采样的训练环境，包含各 actor 的 epsilon-greedy 探索，
因此更适合诊断训练数据和环境状态；判断模型真实性能时应优先看使用纯贪心策略的
`evaluation/`。

| 指标 | 含义与解读 |
| --- | --- |
| `episode/return` | 一个完整回合的累计 shaped reward，包含放置奖励、消行奖励、终局惩罚和棋盘势函数变化。它不是游戏原始分数；越高通常越好，但其绝对值依赖奖励配置。 |
| `episode/length` | 每回合的 placement 决策数。当前每一步放置一个方块，所以它通常与存活方块数一致。越高表示越晚堆满。 |
| `episode/lines` | 每回合消除的总行数；这是最直接的游戏能力指标之一，越高越好。 |
| `episode/survival_pieces` | 终局前成功锁定的方块总数，越高表示生存能力越强。 |
| `episode/aggregate_height` | 终局棋盘十列高度之和。较低通常更安全，但必须结合消行数判断，不能通过完全不堆叠来单独优化。 |
| `episode/max_height` | 终局棋盘最高列的高度。接近 20 表示接近顶出。 |
| `episode/holes` | 终局棋盘中被方块覆盖在上方的空格数量。洞通常难以填补，较少更好。 |
| `episode/bumpiness` | 终局相邻列高度差绝对值之和。较低表示表面更平整。 |
| `episode/wells` | 终局井深的三角加权总和，深井的惩罚大于多个浅凹槽。通常越低越稳，但为长条方块有意保留的井也会提高此值。 |

### `evaluation/`：checkpoint 的独立评估

到达 `eval_every` 时，独立 CPU 进程会用纯贪心策略评估 checkpoint。评估是异步
执行的，曲线上的 Step 使用该 checkpoint 保存时的 transition 数，所以结果即使
稍后才返回，也会落在正确的训练位置。

| 指标 | 含义与解读 |
| --- | --- |
| `evaluation/mean_return` | `eval_episodes` 个评估回合的平均累计 shaped reward。 |
| `evaluation/mean_survival_pieces` | 评估回合平均存活并锁定的方块数。 |
| `evaluation/mean_lines` | 评估回合平均消行数；比较 checkpoint 策略时应重点关注。 |
| `evaluation/mean_length` | 评估回合平均 placement 决策数。 |
| `evaluation/truncated_episodes` | 因达到每回合 `eval_max_steps` 而非自然顶出而停止的回合数。非零表示上述均值中包含被截断的长回合，可适当提高 `eval_max_steps`。 |
| `evaluation/schedule_failed` | 值为 1 表示待评估队列已满，该次 checkpoint 未能安排评估；不表示训练或 checkpoint 保存失败。 |
| `evaluation/error` | Text 面板中的评估异常信息。出现此项表示对应 checkpoint 的评估失败。 |

### `action/`：动作选择分布

动作分布汇总所有 actor 从 run 开始至今的累计选择次数，其中既包含贪心动作，也
包含 epsilon 随机探索动作。placement action 编码为
`rotation * 10 + target_column`。

| 指标 | 含义与解读 |
| --- | --- |
| `action/rotation_0_fraction_cumulative` … `rotation_3_fraction_cumulative` | 四种顺时针旋转状态各自占全部动作的累计比例，四条曲线之和约为 1。重复几何形状的旋转会被 action mask 排除，所以不同旋转并不一定均匀。 |
| `action/target_column_0_fraction_cumulative` … `target_column_9_fraction_cumulative` | 方块最左占用格落在各目标列的累计比例，十条曲线之和约为 1。明显偏向某些列可能来自策略、合法动作范围或探索共同作用。 |

### `events/`：关键 transition 比例

| 指标 | 含义与解读 |
| --- | --- |
| `events/line_clear_transition_fraction_cumulative` | 从 run 开始至今，发生至少一次消行的 transition 占比。一次同时消除多行仍只计作一个发生消行的 transition，因此它不等于“总消行数 / transition 数”。长期为 0 表示 replay 中没有消行经验，agent 很难学会消行。 |
| `events/terminal_transition_fraction_cumulative` | 以回合终止告终的 transition 累计占比。其倒数可粗略反映平均回合长度，但并行 actor、尚未结束的回合和指标上报时机会带来小幅偏差。 |

### `communication/`：多进程采样与 learner 通信

这些累计指标用于判断 actor、transition queue 和 learner 之间是否存在吞吐瓶颈，
不直接表示策略质量。

| 指标 | 含义与解读 |
| --- | --- |
| `communication/actors_transition_put_wait_seconds_cumulative` | 所有 actor 为把 transition batch 放入队列所花时间的累计和。增长斜率明显变大，通常表示队列经常满、learner 消费速度低于 actor 生产速度。 |
| `communication/actors_transition_put_poll_timeout_count_cumulative` | actor 因队列满而等待满一个 `transition_put_poll_timeout` 轮询周期的累计次数。超时后会继续重试，不会因此丢弃 transition。 |
| `communication/actors_transition_put_message_count_cumulative` | actor 成功发送的 IPC batch 消息累计数。用当前横轴上的 transition 数除以消息数，可粗略估算每条 IPC 消息承载的 transition 数；通常越接近 `transition_batch_size`，通信批处理越充分。 |
| `communication/learner/transition_get_wait_seconds_cumulative` | learner 所有取队列尝试耗时的累计和，包括成功取得 batch 和空轮询。默认非阻塞配置下主要反映调用开销。 |
| `communication/learner/transition_get_poll_timeout_count_cumulative` | learner 使用正数 `transition_get_poll_timeout` 时，等待后仍未取得数据的累计次数；默认值为 0 时通常一直为 0。 |
| `communication/learner/transition_get_empty_seconds_cumulative` | learner 未取到 transition 的队列查询所消耗时间累计值，不包含随后 `learner_idle_sleep` 的睡眠时间。持续快速增长通常表示 actor 供给不足。 |
| `communication/learner/transition_get_empty_poll_count_cumulative` | learner 查询 transition queue 为空的累计次数。该值受 `learner_idle_sleep` 和轮询配置影响，不宜单独跨配置比较。 |

实际观察时，可以先用 `evaluation/mean_lines`、`evaluation/mean_return` 和
`evaluation/mean_survival_pieces` 判断策略是否进步，再结合 `episode/` 与
`events/` 判断训练数据是否包含有效消行经验；若训练速度异常，再查看
`train/throughput` 和 `communication/` 定位 actor 或 learner 的供需瓶颈。

The environments use standard 10x20 SRS Tetris and 7-bag generation. Placement
action `rotation * 10 + column` combines a clockwise rotation state with the
leftmost occupied target column. Each observation includes a 40-element
`action_mask`; duplicate rotations and placements that cannot be executed from
spawn are excluded from greedy selection, random exploration, and Double DQN
bootstrapping. Episodes terminate only when a piece cannot be spawned or locks
above the visible board.

Reward shaping is applied once per placement. Ordinary placement contributes
`0.01`, cleared lines contribute `0.75` each, and board potential penalizes
aggregate height, maximum height, holes, bumpiness, and wells. These reward
magnitudes can be changed in `configs/ddqn_default.toml`.

`total_transitions` now counts placed pieces rather than input frames. The
default is therefore 5 million placement decisions; old 6-action checkpoints
are intentionally incompatible with this training and evaluation path.

Actors use fixed geometrically spaced exploration rates. With four actors,
epsilon remains `0.05, 0.10, 0.20, 0.40` throughout training.

Each actor/environment pair receives a distinct deterministic seed. Every
configured checkpoint interval is saved; checkpoints reaching `eval_every` are
evaluated by a separate CPU process with `eval_max_steps` as a per-episode
truncation limit.

Transition communication uses separate policies. Actors use `transition_put_poll_timeout` only as a polling interval while waiting for a full queue; the transition is retained and eventually enqueued, so a poll timeout never drops data. The learner defaults to non-blocking `transition_get_poll_timeout = 0`, and when no transition is available it processes metrics/evaluation messages and sleeps for `learner_idle_sleep` before retrying.

Each actor also accumulates transitions locally before IPC. `transition_batch_size = 256` means an actor with 8 vector environments usually sends one message every 32 vector steps, with 256 transitions per message, instead of sending one message per vector step. `transition_batch_max_wait = 0.1` provides a latency bound for short runs: a partially filled batch is sent after 100 ms, so small smoke runs cannot wait forever for 256 transitions.
