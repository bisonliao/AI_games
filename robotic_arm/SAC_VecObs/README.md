# SAC with vector observations

本目录实现基于精确状态向量的两个 SAC 任务：

- `reach`：Panda 夹爪到达红色物体；
- `pick_place`：抓起红色物体并放到绿色目标区域。

`RobotEnv/` 负责共享的 PyBullet 场景、Panda IK 和夹爪控制；本目录负责向量 observation、任务阶段、reward、并行 rollout、SAC learner、评估和 TensorBoard。

未来的 `SAC_PixelObs/` 可以复用相同的物理环境和动作定义，把精确状态向量换成相机观测。阶段判断和成功条件应保持一致，避免出现“向量任务”和“视觉任务”实际目标不同的问题。

## 并行 rollout 架构

核心要求是多个 PyBullet 环境并行采样，而不是绑定某个 RL 库。当前使用 Stable-Baselines3 管理 SAC、replay buffer 和 checkpoint，PyTorch 是计算后端；SB3 是可替换的实现组件。

```text
                    batched actions
Central SAC policy ─────────────────────────┐
                                            ▼
                ┌─────────────── SubprocVecEnv ───────────────┐
                │ worker 0   worker 1   ...    worker N-1      │
                │ PyBullet   PyBullet          PyBullet        │
                └────────────────────┬─────────────────────────┘
                                     │ parallel transitions
                                     ▼
                              central replay buffer
                                     │
                                     ▼
                              SAC learner updates
```

这是同步的并行环境 rollout：中央进程批量推理动作，多个环境进程同时执行物理仿真，然后中央 learner 更新。用 `--n-actors` 指定环境进程数量。参数名称沿用 actor，但当前这些子进程准确来说只是 environment worker，并不是 Ape-X 意义上各自持有 policy 的独立 actor。

### 当前不是异步 actor-learner

当前没有独立 learner 进程，也没有多个 actor 持续向统一 transition queue 写数据。`SubprocVecEnv` 使用进程间 pipe 与环境 worker 通信，每一轮严格按照以下顺序执行：

```text
主进程用同一个 SAC policy 批量生成 N 个 action
                      ↓
分别发送给 N 个 PyBullet environment worker
                      ↓
N 个环境并行执行一个 RL step
                      ↓
主进程等待最慢的环境完成，即同步 barrier
                      ↓
一次收回 N 条 transition，中央 replay buffer 各写入一条
                      ↓
主进程执行 SAC learner 更新
```

因此每个 worker 每轮固定贡献一条 transition。CPU 快的 worker 不会多写，CPU 慢的 worker 不会少写；慢 worker 只会降低整批吞吐量，不会像异步 Ape-X 那样造成快 actor 数据过采样。episode 提前终止时，该 worker 会自动 reset，但下一轮仍然只贡献一条 transition。

各 worker 的任务分布和 policy 完全相同，仅使用 `seed + rank` 产生独立的环境随机序列。当前不存在 per-actor epsilon，也没有 actor 专属探索强度。

### SAC 的探索方式

SAC 不使用 epsilon-greedy。当前探索分成两段：

1. `num_timesteps < learning_starts` 时，每个并行环境使用动作空间内独立采样的均匀随机动作；
2. warm-up 结束后，中央 SAC actor 根据每个 observation 输出高斯分布，并以 `deterministic=False` 采样，再经过 `tanh` 限制到动作范围。

所有环境共享同一个 actor 网络和一个全局自动熵系数：

```python
ent_coef="auto"
```

worker 间动作不同来自 observation 不同和高斯采样噪声不同，而不是不同的 epsilon。评估脚本默认使用 `deterministic=True`，不进行随机探索。

### 多核利用率和吞吐量

PyBullet environment step 确实在多个 CPU 进程中并行运行，但仿真和 learner 更新没有流水线重叠：环境运行时 learner 等待，learner 更新时环境 worker 等待。因此它能利用多核加速物理仿真，却不能像真正异步 actor-learner 一样持续占满全部 CPU/GPU。

在当前开发机的 10 物理核、20 逻辑 CPU、Microsoft hypervisor 环境上，`reach`、`action_repeat=8` 的纯环境基准如下：

| environment worker 数 | RL transitions/s | Bullet physics steps/s |
|---:|---:|---:|
| 1 | 592 | 4,739 |
| 2 | 875 | 7,002 |
| 4 | 1,328 | 10,627 |
| 8 | 2,019 | 16,149 |

结果表明多进程环境并发确实生效，但由于 IPC、同步 barrier、Python 包装层和最慢 worker 等因素，不会线性扩展。

SB3 日志中的 `time/fps` 是完整训练吞吐量，不是纯 PyBullet 吞吐量，它包含：

```text
环境采样
+ observation normalization
+ replay buffer 写入
+ actor 前向
+ twin critic 更新
+ actor 更新
+ entropy coefficient 更新
+ target network 更新
+ callback、evaluation 和 checkpoint 开销
```

当前 `action_repeat=8`，所以 `300 FPS` 表示约每秒 300 条 RL transition、2,400 个 Bullet physics step。若纯环境达到约 1,328 transitions/s，而完整训练不到 300 FPS，主要瓶颈已经是中央 learner，而不是环境没有并行。

增加 `--n-actors` 可以继续提高环境吞吐，但 learner 不变时会逐渐收益递减。如果未来改成仿真和学习重叠的异步 actor-learner，必须同时设计 per-actor 配额、round-robin/分层 replay、policy lag 监控和 actor 数据占比指标，避免快 actor 主导 transition 分布。

## SAC 实现归属

当前项目没有自行编写 SAC 的数学实现，而是直接引用 Stable-Baselines3：

```python
from stable_baselines3 import SAC

model = SAC(
    "MlpPolicy",
    train_env,
    learning_rate=args.learning_rate,
    buffer_size=args.buffer_size,
    learning_starts=args.learning_starts,
    batch_size=args.batch_size,
    tau=args.tau,
    gamma=args.gamma,
    train_freq=(args.train_freq, "step"),
    gradient_steps=args.gradient_steps,
    ent_coef="auto",
    policy_kwargs={"net_arch": [256, 256]},
)
```

SB3/PyTorch 当前负责：

- stochastic Gaussian actor；
- twin Q critics 和 target critic；
- replay buffer；
- critic loss、actor loss 和自动 entropy coefficient loss；
- target network soft update；
- rollout 收集和 transition 写入；
- 模型、replay buffer 和 VecNormalize checkpoint 基础设施。

本项目明确实现的是：

- PyBullet 机械臂、物体和桌面环境；
- 52 维 vector observation；
- reach 和 pick-place 的 action、reward 与成功条件；
- pick-place 阶段状态机；
- 多环境进程创建和随机 seed；
- 训练/评估命令行；
- TensorBoard 中的任务阶段、抓取率、抬起率和成功率指标。

所以当前方案的准确描述是：

```text
SB3/PyTorch SAC
+ 项目自定义 RobotEnv 和任务语义
+ 同步多进程 PyBullet environment rollout
+ 项目自定义 TensorBoard 任务指标
```

SB3 是当前实现依赖，但不是架构前提。若以后需要真正的异步 actor-learner，可以保留环境、reward 和监控代码，替换中央 SAC/replay/参数同步部分。

## Reach

当前 `reach` 的目标是红色物体，而不是绿色目标标记。红色物体在该任务中被冻结，防止碰撞后目标移动；绿色标记会被隐藏。

动作是三维连续增量：

```text
[dx, dy, dz]
```

夹爪由环境保持打开。奖励为：

```text
2.0 × (上一步距离 - 当前距离)
- 0.1 × 当前距离
- 时间成本
- 动作成本
+ 成功奖励
```

夹爪与物体中心距离小于 `4.5 cm` 时成功并结束 episode。

## Pick-place

整个任务只训练一个 SAC policy、一个 replay buffer，不分别训练四个 agent。环境内部使用阶段状态机，为同一个 policy 提供与当前阶段匹配的 reward：

```text
APPROACH → GRASP → TRANSPORT → PLACE → RELEASE → SUCCESS
```

阶段含义如下：

| 阶段 | 判断 | 主要奖励 |
|---|---|---|
| `approach` | 尚未形成双指抓取 | 夹爪接近物体 |
| `grasp` | 双指接触且夹爪闭合 | 首次抓取奖励、向上抬起进度 |
| `transport` | 物体已离开桌面 | 物体 XY 接近目标 |
| `place` | 被抓物体已到目标上方 | 稳定向桌面降低 |
| `release` | 目标内松开夹爪 | 等待物体稳定 |

成功必须同时满足：

- episode 内确实发生过抓取和抬起；
- 红色物体位于绿色区域；
- 物体回到桌面；
- 夹爪已打开；
- 物体线速度、角速度足够低，并连续稳定 4 个环境 step。

掉落在目标外会收到失败惩罚并终止。阶段机只允许向前推进：确认进入 `grasp` 后不会退回 `approach`，后续阶段也不会回退。接触判定使用少量连续 step 防抖；确认抓取后持续失去接触、物体离开目标区域、重新抓取或当前阶段超时，都会得到失败惩罚并终止 episode。抓取、抬起、到达目标和释放奖励只发放一次，避免反复切换阶段刷 reward。

阶段是 history-dependent 的，因此 observation 显式包含阶段 one-hot、`ever_grasped`、`ever_lifted` 和稳定计数，保持任务状态对 policy 可见。

阶段 progress 使用有符号差分：抬升获得正的高度变化，下降产生负的高度变化；不再把负向变化截断为零。这样即使在同一阶段上下振荡，也不能持续获得净 progress reward。

当前阶段预算（单位为 RL step）为 `approach=50`、`grasp=30`、`transport=75`、`place=100`、`release=20`。进入 `grasp` 需要连续 2 个 step 的有效双指接触；抓取丢失、掉落或释放异常连续 4 个 step 才触发失败，以过滤瞬时接触抖动。进入 `place` 使用约 `7.5 cm` 的目标判定带，离开阶段则使用约 `18 cm` 的更宽退出带，形成迟滞而不允许回退。

### 奖励防投机设计

Reward shaping 的目标是帮助 agent 发现完整动作序列，而不是让 shaping reward 取代真实任务目标。早期实现曾出现一种典型的 reward hacking：policy 在 `approach` 和 `grasp` 之间反复切换，每次重新接近物体都能领取 progress；同时 `grasp` 只奖励物体向上的高度变化，却把向下变化截断为零。因此 episode reward 和 eval reward 持续上升，但 `lift_rate` 最终回到零，`success_rate` 始终为零。

当前设计遵守以下约束：从一个任务状态出发，经过一组动作又回到相同或更差的任务状态时，这个循环不能产生正的净 shaping reward。具体通过以下机制实现：

1. **阶段单向推进**：状态机只允许 `approach → grasp → transport → place → release`。确认进入某阶段后不能退回前一阶段重新领取 progress；持续回退会被解释为抓取丢失、掉落或释放失败，并终止 episode。
2. **使用有符号进度差分**：接近、抬升、搬运和降低奖励都基于前后状态的误差变化。向目标前进为正，退步为负；尤其抬升奖励不再使用 `max(0, Δz)`。在同一阶段往返运动时，正负 progress 会相互抵消，时间和动作成本使循环的净回报为负。
3. **里程碑奖励只发放一次**：抓取、首次抬起、到达目标、释放和最终成功由 history flag 保护，不能通过抖动或重复触发事件反复领取 bonus。
4. **违反任务顺序会失败终止**：确认抓取后持续丢失接触、运输中掉落、放置后明显离开目标、释放后重新闭合夹爪，以及阶段超时，都会产生 `-5` 的 `reward/failure` 或 `reward/drop` 并设置 `terminated=True`。失败后不能继续利用剩余 horizon 刷分。
5. **防抖与迟滞避免误罚**：物理接触和目标边界可能在相邻 simulation step 间抖动，因此进入阶段需要连续确认，异常也需要连续多个 RL step 才判失败；目标区域采用不同的进入/退出阈值。严格的任务顺序与对 PyBullet 瞬时噪声的容忍并不冲突。
6. **阶段预算只作为停滞安全阀**：预算用于防止 policy 长期停在某阶段而不尝试推进，并减少 replay buffer 中的低价值 transition；它不是修复循环刷分的主要机制。真正保证循环无利可图的是阶段不可回退、有符号差分和一次性事件奖励。全局 `max_episode_steps` 仍是 Gymnasium truncation 上限。
7. **奖励内部状态对 policy 可见**：阶段 one-hot、历史 flag、阶段预算进度、接触确认进度和异常容忍进度都包含在 observation 中，避免 reward/termination 依赖 policy 看不到的隐藏计数器。

判断训练是否真的进步时，不应只看 `rollout/ep_rew_mean` 或 `eval/mean_reward`。首要观察 `eval/success_rate`，并用 `task/lift_rate` 判断是否突破抓取抬升瓶颈、用 `task/final_stage` 判断策略是否从 `approach/grasp` 推进到 `transport/place/release`。如果 reward 上升而这三项长期没有改善，应优先怀疑新的奖励投机路径，而不是简单增加训练步数。

## Observation

两个任务都使用长度为 52 的一维 `float32` 向量。位置、线速度使用 PyBullet 世界坐标系，长度单位为米；关节角单位为弧度。

| 下标 | 维数 | 中文含义 |
|---|---:|---|
| `[0:7]` | 7 | Panda 七个机械臂关节的角度，顺序对应 `panda_joint1` 到 `panda_joint7`，单位 rad |
| `[7:14]` | 7 | 上述七个关节的角速度，单位 rad/s |
| `[14:17]` | 3 | 末端抓取参考点在世界坐标系中的 `(x, y, z)` 位置 |
| `[17:20]` | 3 | 红色物体中心在世界坐标系中的 `(x, y, z)` 位置 |
| `[20:24]` | 4 | 红色物体姿态四元数 `(qx, qy, qz, qw)` |
| `[24:27]` | 3 | 绿色放置目标中心的 `(x, y, z)` 位置；reach 没有独立放置目标，因此这三维与红色物体位置相同 |
| `[27]` | 1 | 两个夹爪指关节位置之和，可理解为夹爪总开口宽度，约 `0` 表示闭合、约 `0.08` 表示完全打开 |
| `[28:31]` | 3 | `末端位置 - 物体位置`，即从物体指向末端的相对位移向量 |
| `[31:34]` | 3 | `物体位置 - 目标位置`，即从绿色目标指向红色物体的相对位移；reach 中恒为零 |
| `[34:37]` | 3 | 红色物体在世界坐标系中的线速度 `(vx, vy, vz)`，单位 m/s |
| `[37:40]` | 3 | 红色物体在世界坐标系中的角速度 `(wx, wy, wz)`，单位 rad/s |
| `[40:45]` | 5 | pick-place 阶段 one-hot，顺序固定为 `approach, grasp, transport, place, release`；reach 始终为 `approach` |
| `[45]` | 1 | 当前是否形成有效双指接触：`1` 表示左右手指均接触物体且夹爪已收拢，否则为 `0` |
| `[46]` | 1 | 本 episode 是否曾经形成过有效抓取，布尔历史标记 |
| `[47]` | 1 | 本 episode 是否曾经把物体抬离桌面，布尔历史标记 |
| `[48]` | 1 | 在目标区域稳定放置的进度，等于 `stable_steps / 4` 并截断到 `[0, 1]` |
| `[49]` | 1 | 当前阶段已消耗步数相对于该阶段预算的比例，截断到 `[0, 1]` |
| `[50]` | 1 | `approach` 阶段双指接触确认进度，等于连续接触步数除以确认所需步数 |
| `[51]` | 1 | 当前阶段异常状态的容忍进度，例如抓取后失去接触或释放后离开目标 |

之所以同时提供绝对坐标和相对位移，是为了让当前 privileged-state baseline 尽量容易学到控制规律。绝对坐标保留完整状态，相对向量减少神经网络自行学习坐标相减的负担。后续做消融实验时，可以再比较是否删除重复字段。

阶段 one-hot、历史标记和阶段计时/防抖进度不是额外“答案泄漏”，而是为了保证环境的奖励状态对 policy 可见。例如，同样是物体放在桌面上，`ever_lifted=0` 表示尚未抓取，`ever_lifted=1` 则可能表示已经搬运并释放；如果隐藏该信息，依赖历史的成功判定就不再满足 Markov 假设。

训练端使用 `VecNormalize` 维护 observation 均值和方差。评估模型时必须同时加载对应的 `vecnormalize.pkl`。

## 安装

项目环境已经具有 PyTorch、PyBullet、Gymnasium 和 TensorBoard。当前实现额外使用：

```bash
python -m pip install -r SAC_VecObs/requirements.txt
```

## 训练

Reach：

```bash
python -m SAC_VecObs.train \
    --task reach \
    --n-actors 4 \
    --total-timesteps 200000
```

Pick-place：

```bash
python -m SAC_VecObs.train \
    --task pick_place \
    --n-actors 8 \
    --total-timesteps 1000000
```

常用参数：

```text
--n-actors              并行 PyBullet 环境数量，默认 4
--device                auto/cpu/cuda
--max-episode-steps     episode 最大步数，默认 150
--action-repeat         每个 action 的 physics step 数，默认 8
--learning-starts       开始 learner 更新前的 transition 数
--eval-freq             评估间隔，以全局 transition 计
--checkpoint-freq       checkpoint 间隔，以全局 transition 计
--run-name              自定义运行目录名
```

默认 multiprocessing start method 是 `forkserver`，它比在已初始化 CUDA 的进程中直接 `fork` 更安全。在受限容器中如果 Unix socket 不可用，可以显式指定：

```bash
python -m SAC_VecObs.train --task reach --n-actors 4 --start-method fork
```

每次训练产生独立目录：

```text
SAC_VecObs/runs/<run_name>/
├── config.json
├── final_model.zip
├── vecnormalize.pkl
├── tensorboard/
├── checkpoints/
├── best_model/
├── eval/
└── monitor/
```

## TensorBoard

启动：

```bash
tensorboard --logdir SAC_VecObs/runs
```

除 SB3 自带的 actor loss、critic loss、entropy coefficient、Q 更新等指标外，还会上报：

```text
task/success_rate
task/failure_rate
task/grasp_rate
task/lift_rate
task/final_stage
task/truncation_rate
task/grasp_lost_rate
task/drop_rate
task/release_regression_rate
task/stage_timeout_rate
task/stage_fraction_*
reward/progress
reward/event
reward/drop
reward/failure
reward/time
reward/action
eval/mean_reward
eval/success_rate
```

`task/truncation_rate` 表示 episode 因达到 `max_episode_steps` 而结束的比例。环境适配层会独立检查时间上限，factory 外还使用 Gymnasium `TimeLimit` 二次保护，避免底层和上层任务成功条件不一致时产生永不结束的 episode。判断是否需要增大 horizon 时，应结合最终阶段观察该指标：大量 episode 在 `place/release` 阶段截断才说明时间可能不够；如果始终停在 `approach/grasp`，增加 horizon 通常不能解决探索或奖励问题。

Monitor 的 terminal info 还会保存 `failure_reason`，常见值包括 `grasp_lost`、`object_dropped`、`object_left_goal`、`release_regressed` 和各阶段的 `*_timeout`，便于区分探索失败、物理掉落与阶段停滞。

## 评估和 GUI

```bash
python -m SAC_VecObs.evaluate \
    --task pick_place \
    --checkpoint SAC_VecObs/runs/<run_name>/final_model.zip \
    --vecnormalize SAC_VecObs/runs/<run_name>/vecnormalize.pkl \
    --episodes 10 \
    --gui \
    --fps 15
```

`evaluate` 会尝试在 checkpoint 附近自动查找 `vecnormalize.pkl`，但显式传入最可靠。

## Curriculum

### 当前状态：未启用

当前代码没有实施 Curriculum，也没有使用脚本控制器为训练构造中间阶段。所有训练 actor 都直接面对完整任务，先验证增加并行环境数量、训练步数和随机探索能否“大力出奇迹”。

每次 `reset()` 的实际行为是：

1. Panda 七个关节恢复到固定初始姿态，夹爪完全打开；
2. 红色物体中心在台面范围内均匀采样：`x ∈ [0.42, 0.72]`、`y ∈ [-0.28, 0.28]`、`z = 0.025`；
3. pick-place 的绿色目标在相同 XY 范围内独立采样，并保证它与红色物体的平面距离至少为 `0.12 m`；
4. pick-place 总是从 `approach` 阶段开始，`ever_grasped=0`、`ever_lifted=0`、`stable_steps=0`；
5. reach 会冻结红色物体，并把共享 observation 中的目标坐标设成红色物体坐标；
6. 不会预先移动机械臂、闭合夹爪、抬起物体，也不会向 replay buffer 注入脚本控制器 transition；
7. 各个并行 actor 使用 `seed + rank` 的独立随机序列，因此 reset 分布相同，但具体物体和目标位置不同。

因此当前训练结果能够直接回答：在完整随机初始状态下，单一 SAC policy 配合阶段 reward 是否可以自行发现整条动作序列。

### 何时考虑启用

建议先用多个 seed 做完整任务 baseline，并重点观察 TensorBoard 中的分阶段指标：

```text
task/grasp_rate
task/lift_rate
task/stage_fraction_transport
task/stage_fraction_place
task/success_rate
```

如果增加训练量和并行 actor 后仍长期卡在某一阶段，例如 `grasp_rate` 已经上升但 `lift_rate` 接近零，或者能够抬起却始终到不了 `place`，再引入 Curriculum。判断依据应是阶段指标持续形成瓶颈，而不只是某一次训练的最终 success 为零。

### 保留的实现方案：反向课程

Pick-place 的课程难度适合通过改变 `reset()` 的起始物理状态来递增。课程从最接近成功的状态开始，然后逐级把起点向任务开头后退：

| 难度 | reset 后的物理状态 | 初始阶段和历史标记 | policy 需要学习的新增能力 |
|---|---|---|---|
| L0 | 红色物体已被夹爪持有，并位于绿色目标上方 | `place`，`ever_grasped=1`，`ever_lifted=1` | 降低物体、松开夹爪、等待稳定 |
| L1 | 物体已被夹住并抬起，但与目标还有随机平面距离 | `transport`，两个历史标记均为 `1` | 搬运到目标，再完成放置 |
| L2 | 夹爪已经在物体两侧形成有效接触，物体仍在桌面 | `grasp`，`ever_grasped=1`、`ever_lifted=0` | 抬起物体，并完成搬运和放置 |
| L3 | 夹爪打开并位于物体上方较近位置，物体仍在桌面 | `approach`，两个历史标记均为 `0` | 下降、对准、抓取，再完成后续流程 |
| L4 | 当前实现的完整随机 reset：机械臂回到固定初始姿态，物体和目标全范围随机 | `approach`，两个历史标记均为 `0` | 完整 pick-place |

从 L0 到 L4，任务难度增加体现在三个方面：episode 所需动作序列变长、需要自行探索的阶段增多、初始状态分布变宽。整个过程中仍然只有一个 policy 和一个 replay buffer，不会训练五套模型。

如果将来实现，应遵守以下约束，保证 Curriculum 与任务定义一致：

- 中间状态应通过真实接触的脚本控制器或预先验证的物理状态快照构造，不能简单把物体瞬移进夹爪造成穿模；
- 构造 reset 状态所执行的脚本动作不进入 replay buffer，也不占用 episode 的 step 预算；
- 从中间阶段开始时，observation 中的阶段 one-hot、`ever_grasped` 和 `ever_lifted` 必须与物理状态一致；
- reset 前已经完成的抓取、抬起等事件要标记为奖励已发放，避免 agent 凭课程初始状态白拿 event bonus；
- 晋级后仍应保留少量较简单 reset，例如 70% 当前难度、30% 旧难度，降低灾难性遗忘；
- 晋级条件优先使用最近一段窗口内的成功率和阶段通过率，而不是只按照固定 timestep；
- 常规 evaluation 始终使用 L4 完整随机 reset，避免课程内的高成功率掩盖完整任务尚未学会。

这个方案目前只作为后备设计保留，代码和命令行中没有 Curriculum 开关。
