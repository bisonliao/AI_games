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

Resume a model and optimizer from an existing checkpoint:

```bash
python -m DQN.train --config configs/ddqn_default.toml --device cuda \
  --resume-from checkpoints/<run-id>/dddqn_14002568.pt \
  --total-transitions 50000000
```

`--total-transitions` is an absolute target, not the number of additional
transitions. The default run uses a 1,000,000-entry replay buffer. LR and every
actor's epsilon are linearly interpolated over the first 5 million absolute
transitions. With four actors, epsilon changes from `0.05、0.10、0.20、0.40` to
`0、0.01、0.01、0.01`; LR changes from the configured value to 10% of that value
(`1e-4 → 1e-5` by default). `final_epsilon` configures the non-greedy actors'
final epsilon. `update_every` and `gamma` remain unchanged throughout training.

Checkpoints do not contain replay samples. A resumed run therefore restores the
online/target networks, Adam state, counters, and RNG states, then
freezes optimization while collecting a fresh `replay_capacity` transitions
(1,000,000 by default). Schedule progress is reconstructed directly from the
restored absolute transition count. A resumed job starts a new run directory
and never overwrites the source
checkpoint. This is a warm restart rather than bit-for-bit continuation.

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

## Epsilon/LR 衰减消融结论

三条同配置实验比较了能力达标后立即衰减、前 5M transition 线性衰减和完全不衰减。
在共同的约 8.53M 观察区间末端，三者 `evaluation/mean_lines` 分别为
`417.7、1262.6、331.6`；线性衰减实验随后达到 `1836.0`。5M以后，线性衰减实验
的 `episode/lines` 均值约 `352`、`holes` 均值约 `22.8`，明显优于立即衰减的
`151.6、30.1`。不衰减虽然曲线波动较小，但训练 episode 能力很低，不属于有效的
“稳定”。

这说明本任务对衰减时序高度敏感。一个合理解释是：骤降会突然改变 replay 数据分布，
完全不降又会让高探索数据长期占据 replay；平滑线性衰减在探索多样性和后期策略质量
之间取得了更好平衡。因此训练代码只保留前 5M transition 线性衰减。由于三次运行
时长不同且多进程训练并非逐 bit 确定，这一结果是明确的工程选择，而不是跨随机种子
的统计显著性结论。



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

## 奖励计算：事件奖励与棋盘势函数

当前 DQN 训练使用 placement 环境，因此每个 transition 都会完成并锁定一个方块。
该步的 shaped reward 由事件奖励和相邻棋盘状态之间的势函数差分组成：

```text
r_t = r_event + gamma * Phi(s_{t+1}) - Phi(s_t)

r_event = 0.01 * I(成功放置方块)
        + 0.75 * 消除行数
        - 1.00 * I(回合终止)
```

在 placement 环境中，每个合法步骤都会成功放置一个方块，所以普通未消行、未终止
步骤的事件奖励是 `0.01`。一次消除多行时，`0.75` 会按实际消除行数累加；导致顶出
的最后一步还会额外减去 `1.00`。这些默认值分别由
`configs/ddqn_default.toml` 中的 `piece_placed_reward`、`line_clear_reward` 和
`terminal_penalty` 控制。

棋盘势函数 `Phi` 把以下五种结构特征组合为一个非正的棋盘质量分数：

```text
Phi(s) = -(
    0.45 * aggregate_height / 200
  + 0.20 * max_height       / 20
  + 4.00 * holes            / 200
  + 0.30 * bumpiness        / 180
  + 0.30 * wells            / 200
)
```

| 因素 | 计算方法 | 奖励中考虑它的原因 |
| --- | --- | --- |
| `aggregate_height` | 10 列高度之和 | 抑制棋盘整体越堆越高；它能区分“整面抬高”和只有一列较高的情况。 |
| `max_height` | 所有列中的最大高度 | 反映最高点离顶部还有多远，用于控制近期顶出风险。 |
| `holes` | 每列最高已占格下方的空格总数 | 洞通常需要额外操作或消行才能填补，因此使用五项中最大的系数重点惩罚。 |
| `bumpiness` | 相邻列高度差绝对值之和 | 抑制过于崎岖的表面，使后续方块更容易找到安全落点。 |
| `wells` | 每列井深的三角加权和 `depth * (depth + 1) / 2` | 同时考虑凹槽数量和深度，并让深井的代价高于多个浅凹槽。 |

因为 `Phi` 是“负代价”，落子后棋盘变平、变低或减少洞时，势函数通常会上升，
差分项会提供正反馈；棋盘结构恶化时则通常提供负反馈。默认 `gamma` 是 `0.99`，
它同时用于 Double DQN 的 TD target 和这里的势函数差分，始终使用配置中的
`gamma`（默认 `0.99`）。终局步骤仍使用真实终局棋盘的 `Phi(s_{t+1})`，不会把它强制
置零，避免顶出时凭空获得“逃离负势函数棋盘”的正奖励。

`episode/return` 和 `evaluation/mean_return` 是上述 shaped reward 的逐步直接求和，
不是俄罗斯方块的原始游戏分数，也没有在统计时再次乘折扣；网络的 Q 值和 TD target
才按 `gamma` 估计未来折扣回报。原有 frame-level 环境只在方块锁定或回合终止时
应用势函数差分，避免同一方块移动期间被重复 shaping。

## TensorBoard 指标说明

启动训练后，使用下面的命令查看指标：

```bash
tensorboard --logdir runs/dddqn
```

每次训练都会写入 `runs/dddqn/<run-id>/`，可以在 TensorBoard 中选择不同
run 进行比较。所有标量图的横轴 `Step` 都表示 learner 已接收的 transition
数量；在 placement 环境中，一条 transition 就是放置一个方块，而不是一次键盘
输入或画面帧。常规指标每累计 `tb_log_every` 条 transition 刷新一次，默认值为
`100000`。

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
| `train/epsilon` | 所有 actor 当前实际 epsilon 的均值。前 5M transition 内从初始 profile 线性下降；默认四 actor 从 `0.05、0.10、0.20、0.40` 变为 `0、0.01、0.01、0.01`。 |
| `train/actor_0_epsilon` … `train/actor_3_epsilon` | 每个 actor 当前实际使用的 epsilon，用于确认线性衰减和最终一个贪心 actor 的 profile。 |
| `train/lr` | optimizer 当前实际使用的学习率；前 5M transition 内从配置初值线性下降到初值的 1/10，默认是 `1e-4 → 1e-5`。 |
| `train/gamma` | 当前 TD target、actor reward shaping 和 checkpoint evaluator 共同使用的有效 gamma；它始终使用配置中的 `gamma`，默认是 `0.99`。 |
| `train/updates_per_transition` | 当前每条 transition 对应的目标梯度更新频率，始终由配置中的 `update_every` 决定，默认是 `0.25`。这是配置 cadence，不是已完成 update 的实测计数；断点续训重新填充 replay 时暂为 0。 |
| `train/gradient_updates_cumulative` | learner 实际完成的 `optimizer.step()` 累计次数。新训练在 `learning_starts` 后应按 `update_every` 稳定增长；CPU/GPU 变慢只应改变墙钟速度，不应减少相同 transition step 下的该计数。断点续训会包含 checkpoint 已有的累计次数，并在 replay 预热期间保持不变。 |
| `train/replay_warming_up` | 断点续训是否正在重建未保存的 replay；1 表示网络冻结并只采样，0 表示允许优化。新训练始终为 0。 |
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
稍后才返回，也会落在正确的训练位置。各 checkpoint 使用同一组互不相同的固定
环境 seed；每个回合的 7-bag 仍有随机性，但不同 checkpoint 面对相同的随机场景，
适合作为低噪声的 checkpoint 对比集。最终比较模型时仍建议另外更换 seed
做离线 holdout 评测。

| 指标 | 含义与解读 |
| --- | --- |
| `evaluation/mean_return` | `eval_episodes` 个评估回合的平均累计 shaped reward。 |
| `evaluation/mean_survival_pieces` | 评估回合平均存活并锁定的方块数。 |
| `evaluation/mean_lines` | 评估回合平均消行数；比较 checkpoint 策略时应重点关注。 |
| `evaluation/mean_length` | 评估回合平均 placement 决策数。 |
| `evaluation/truncated_episodes` | 因达到每回合 `eval_max_steps` 而非自然顶出而停止的回合数。非零表示上述均值中包含被截断的长回合，可适当提高 `eval_max_steps`。 |
| `evaluation/queue_full` | 值为 1 表示待评估队列已满，该次 checkpoint 未能安排评估；不表示训练或 checkpoint 保存失败。 |
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

训练使用 gather–learner 流水线。独立 gather 进程通过每个 actor 的结果队列和
round 命令队列，严格收齐每个 actor 恰好 `transition_batch_size` 条 transition，
再按固定 actor 顺序合并。完整 round 通过容量为 1 的可靠队列交给 learner；成功
入队后 gather 立即统一放行下一轮，因此 actor rollout 可以和 learner 对上一轮的
replay 插入及梯度更新并行。各 actor 的样本贡献仍严格相同，流水线速度由最慢的
actor 采样阶段与 learner 更新阶段中较慢的一侧决定。

权重不参与 round barrier。learner 为每个 actor 维护容量为 1 的 latest-wins
mailbox；actor 只在每轮开始时检查并加载当前可见的最新版本，没有新权重就继续使用
旧版本。因此单个 actor 的一轮内权重一致，但不同 actor 在同一轮可能看到不同版本。

这些指标用于确认 round 均衡性和定位慢 actor，不直接表示策略质量。

| 指标 | 含义与解读 |
| --- | --- |
| `communication/synchronous_rounds_cumulative` | 本次 run 已被 learner 完整接收并写入 replay 的同步 round 数。每轮总样本数固定为 `num_actors * transition_batch_size`。 |
| `communication/actor_N/accepted_transitions_cumulative` | learner 已从 actor N 接受的 transition 累计数。所有 actor 的曲线在每个已完成 round 后应完全重合；这是均衡性的权威指标。 |
| `communication/actor_N/transitions_sent_cumulative` | actor N 自报的成功发送累计数。它走低优先级指标队列，可能滞后或缺点；判断 replay 实际构成应以前一个 accepted 指标为准。 |
| `communication/actor_N/round_arrival_seconds` | 最近记录 round 中，从统一放行到 actor N 的 batch 被 gather 取出的耗时。长期明显高于其他 actor 表示它是 straggler。 |
| `communication/synchronous_round_collection_seconds` | gather 最近记录 round 收齐全部 actor batch 的耗时，通常接近最慢 actor 的 arrival 时间。 |
| `communication/actors_transition_put_wait_seconds_cumulative` | 所有 actor 为把本轮 batch 放入各自独立结果队列所花时间的累计和。正常同步协议下通常很低；持续增长表示 gather 没有及时取走上一轮结果或进程调度严重拥堵。 |
| `communication/actors_transition_put_poll_timeout_count_cumulative` | actor 因队列满而等待满一个 `transition_put_poll_timeout` 轮询周期的累计次数。超时后会继续重试，不会因此丢弃 transition。 |
| `communication/actors_transition_put_message_count_cumulative` | 所有 actor 成功发送的 round batch 消息累计数；正常情况下约等于 `num_actors * synchronous_rounds_cumulative`。 |
| `communication/gather/batch_queue_wait_seconds_cumulative` | gather 为把完整大 batch 放入容量为 1 的 learner 队列所等待的累计时间。持续增长表示 learner 更新慢于 actor round 生产速度，也说明流水线背压正在生效。 |
| `communication/gather/batch_queue_wait_timeout_count_cumulative` | gather 等满一个 `transition_put_poll_timeout` 仍无法提交大 batch 的累计次数；超时只会重试，不会丢训练数据。 |
| `communication/learner/round_update_seconds` | learner 最近一轮补齐全部到期 mini-batch update 的耗时。与 round collection time 对比可判断流水线由 rollout 还是 update 限制。 |
| `communication/learner/transition_get_wait_seconds_cumulative` | learner 查询 gather 大 batch 队列所花时间的累计和，不包含 `learner_idle_sleep`。 |
| `communication/learner/transition_get_empty_seconds_cumulative` | learner 没取得 gather 大 batch 时，队列查询本身所消耗的累计时间，不包含随后睡眠。 |
| `communication/learner/transition_get_empty_poll_count_cumulative` | gather 大 batch 尚未可用时 learner 的空轮询次数。该值受 `learner_idle_sleep` 影响，不宜单独跨配置比较。 |

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

`total_transitions` now counts placed pieces rather than input frames. The
default configuration runs for 30 million placement decisions; old 6-action checkpoints
are intentionally incompatible with this training and evaluation path.

Actors start with geometrically spaced exploration rates. Both epsilon and LR
are linearly decayed over the first 5 million absolute transitions. With four
actors, epsilon moves from `0.05, 0.10, 0.20, 0.40` to
`0, 0.01, 0.01, 0.01`, while LR reaches 10% of its configured initial value.

Each actor/environment pair receives a distinct deterministic seed. Every
configured checkpoint interval is saved; checkpoints reaching `eval_every` are
evaluated by a separate CPU process with `eval_max_steps` as a per-episode
truncation limit.

Transition collection uses a gather–learner pipeline. The gather process owns
the actor round barrier: every actor produces exactly `transition_batch_size`
transitions, and gather concatenates one batch per actor in stable order. It
reliably submits the complete round through a queue of size one, then releases
all actors to collect the next round while the learner inserts and updates from
the previous one. The batch size must be divisible by `envs_per_actor`; there is
no wall-clock partial-batch flush.

Replay remains private to the learner, and its transition-based update cursor
keeps `update_every` independent of IPC timing. Actor weights use separate
latest-wins one-slot mailboxes and are loaded only at actor round boundaries;
weight delivery never joins the gather barrier. Training stops only at a complete
round, so `total_transitions` may be exceeded by less than
`num_actors * transition_batch_size` transitions.
