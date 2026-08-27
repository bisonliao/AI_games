# Atari H.E.R.O. 长回合课程学习

这是一个用课程学习解决 Atari H.E.R.O. 长回合探索问题的强化学习项目。项目只保留游戏画面作为 agent 的观测，通过人工示范采集一组由近及远的 ALE 环境 checkpoint，再让 Dueling Double DQN 从靠近矿工的短任务逐步学习到完整任务。

实验表明，这套设计能够稳定学会 Stage 1～3。尤其是 Stage 3 训练得到的 `checkpoint_000002500000.pt`，在完整回合评测中可以从 Level 1 或 Level 2 起点出发并救出矿工。Stage 4 尚未收敛：rollout 成功率曾上升，但随后突然暴跌，同时出现大量超时，agent 常在原地徘徊并反复射击。这个失败现象也被如实保留，作为长时序训练中策略退化与信用分配问题的案例。

## 环境准备

运行环境需要 Python、ALE/Gymnasium、PyTorch、Pillow、NumPy 和 pygame；训练使用 CUDA learner，评测脚本可以在 CPU 上运行。一个常用的安装方式是：

```bash
python -m pip install ale-py gymnasium numpy pillow pygame torch tensorboard
```

还需要自行准备与项目匹配的 H.E.R.O. ROM。ROM 不随代码分发，且课程清单会校验 ROM MD5。

## 任务定义

项目在原版 H.E.R.O. 之上构造了两个任务难度不同的关卡：

- Level 1 有 2 个房间；
- Level 2 有 4 个房间；
- 矿工位于每个 Level 的最深处；
- 救出当前 Level 的矿工即为成功；
- 掉命或达到时间上限即为失败。

在完整回合训练和独立评测中，Level 1 与 Level 2 以相同概率随机出现，并且是两个独立的 episode：

```text
reset ──50%──> Level 1 / Room 1 ──救出矿工──> episode 成功结束
      └─50%──> Level 2 / Room 1 ──救出矿工──> episode 成功结束
```

因此，本项目中的“完整 episode”不是连续完成 Level 1 和 Level 2。Level 1 进入 Level 2 表示 Level 1 的矿工已经获救，episode 随即结束；同理，Level 2 进入 Level 3 时成功结束，Level 3 的画面不会暴露给 agent。

原子课程训练的语义与此一致：从某个课程 checkpoint 恢复后，只需完成该 checkpoint 所属的 Level。

## 观测与算法

Agent 不接收关卡号、房间号、坐标、剩余距离等向量化或结构化输入。环境提供的 RGB 游戏帧只在 wrapper 内做标准 Atari 图像预处理：

1. 转为灰度图；
2. 缩放到 `84 × 84`；
3. 每个动作重复 4 个 ALE frame，并对末两帧做 max pooling；
4. 堆叠最近 4 帧，最终观测形状为 `4 × 84 × 84`。

关卡和房间信息只用于环境终止判断、课程 checkpoint 管理与训练指标，不会进入神经网络。学习器使用 Dueling Double DQN、1-step TD target、统一的 uniform replay、AMP、梯度裁剪和周期性 target network 更新；多个 CPU actor 负责并行采样，CUDA learner 负责更新网络。

## 奖励设计

原始 ALE 分数先被解析为 H.E.R.O. 事件，再转换为训练奖励。当前 `train.py` 的默认值如下：

| 事件 | 训练奖励 |
| --- | ---: |
| 炸毁墙壁 | `+0.5` |
| 击毙生物 | `+0.5` |
| 救出矿工 | `+10.0` |
| 每个 DQN decision 的时间成本 | `-0.002` |
| 超时 | `-10.0 - 0.002` |
| 掉命 | `-10.0 - 0.002` |

每个 episode 最多运行 500 个 DQN decision steps。达到上限会作为明确的失败终止，而不是可 bootstrap 的普通 time-limit truncation。逐步时间成本和超时惩罚共同用于抑制无效停留。

事件解析遵循游戏分值语义：原始 `+75` 记为炸墙，矿工奖励出现前的 `+50` 记为击杀，`+1000` 或 `1000 + 50 × n` 记为营救矿工及剩余炸药奖励。其中只有矿工事件换成 RL reward，剩余炸药数量作为诊断事件保留。未识别分值只进入诊断指标，不参与 RL reward；矿工奖励在一次 episode 中只发放一次。

奖励参数可以通过 `--wall-event-reward`、`--creature-event-reward`、`--miner-event-reward` 和 `--decision-step-penalty` 覆盖。

## 课程学习设计

长回合任务的主要困难是：随机探索几乎不可能在有限时间内穿过多个房间并获得最终的矿工奖励。本项目没有给 agent 额外的导航状态，而是改变训练 episode 的起点，先把最终目标变成一个很短的任务，再逐步向关卡入口回退。

### Stage 如何定义

人工从每个 Level 的入口一路玩到矿工处，在每次向下一层、进入适合作为课程起点的位置时记录深度 checkpoint。每个课程 episode 的深度从 `-1` 开始；按一次 `D` 才加一，程序不会根据 RAM、房间切换或画面变化自动创建 checkpoint。冻结课程时，系统在每个 Level 内独立地反向编号：

```text
stage = 当前 Level 的最大深度 - checkpoint 深度 + 1
```

所以距离矿工最近的 checkpoint 属于 Stage 1；数字越大的 Stage，起点越靠近关卡入口，所需完成的有效时域越长。Stage 不是游戏 Level，也不是把多个 Level 串成一个 episode。

当前 frozen curriculum 共包含 6 个起点：

| Stage | Level 1 课程起点 | Level 2 课程起点 |
| --- | --- | --- |
| 1 | Room 2，距矿工最近 | Room 4，距矿工最近 |
| 2 | Room 1，Level 1 完整起点 | Room 3 |
| 3 | — | Room 2 |
| 4 | — | Room 1，Level 2 完整起点 |

<table>
  <tr>
    <th>Stage 1 / Level 1</th>
    <th>Stage 1 / Level 2</th>
    <th>Stage 2 / Level 1</th>
    <th>Stage 2 / Level 2</th>
  </tr>
  <tr>
    <td><img src="../stage_img/1_1.jpg" width="160" alt="Stage 1 Level 1 起点"></td>
    <td><img src="../stage_img/1_2.jpg" width="160" alt="Stage 1 Level 2 起点"></td>
    <td><img src="../stage_img/2_1.jpg" width="160" alt="Stage 2 Level 1 起点"></td>
    <td><img src="../stage_img/2_2.jpg" width="160" alt="Stage 2 Level 2 起点"></td>
  </tr>
  <tr>
    <th>Stage 3 / Level 2</th>
    <th>Stage 4 / Level 2</th>
    <th></th>
    <th></th>
  </tr>
  <tr>
    <td><img src="../stage_img/3_1.jpg" width="160" alt="Stage 3 Level 2 起点"></td>
    <td><img src="../stage_img/4_1.jpg" width="160" alt="Stage 4 Level 2 起点"></td>
    <td></td>
    <td></td>
  </tr>
</table>

### `teacher.py`：人工生成课程起点

[`HeroEnv/teacher.py`](../HeroEnv/teacher.py) 是课程数据采集工具。运行后由人工直接控制游戏，并在每次向下一层、进入一个适合作为课程起点的位置时按 `D`：程序先把深度加一，再同时保存精确的 ALE system state 和当前画面。

```bash
python HeroEnv/teacher.py
```

主要按键如下：

- 方向键：移动；
- `Space` 或 `Ctrl`：射击；
- `D`：深度加一并缓存当前 checkpoint；
- `U`：撤销最近一次尚未提交的 `D`；
- `R` 或 `F2`：放弃本回合未提交的数据并重新开始；
- `P`：暂停；
- `Q` 或 `Esc`：退出。

采集并不是按下 `D` 后立即永久写入。候选 checkpoint 只有在人工随后不掉命并成功救出该 Level 的矿工后才会提交；掉命、重开或未完成的示范都会丢弃本回合候选点。这样可以保证每个课程起点都来自一条真实可完成的后续轨迹。

提交前还会检查生命、能量、NOOP 生存时间、动作响应性和 sticky-action smoke seeds。不合格的状态进入 quarantine。人工演示从 checkpoint 到营救所需的剩余帧数也会写入元数据，便于审计课程难度。

采集完成后冻结课程清单：

```bash
python HeroEnv/teacher.py --freeze-curriculum
```

该命令会生成带内容哈希、checkpoint 哈希和版本号的 `HeroEnv/checkpoints/curriculum-vNNNN.json`，并更新训练使用的 `curriculum.json`。可进一步检查每个 checkpoint 的精确 reset 画面：

```bash
python HeroEnv/check.py
```

更完整的采集和校验规则见 [`HeroEnv/README.md`](../HeroEnv/README.md)。

### `train.py`：逐个训练原子课程

[`train.py`](train.py) 的一次调用只训练一个目标 Stage。例如：

```bash
python -m curri_DQN.train \
  --stage 1 \
  --run-dir curri_DQN/runs/hero_dddqn \
  --actors 16 \
  --total-transitions 2000000
```

训练 Stage 2 及后续 Stage 时，用前一 Stage 的 checkpoint 初始化网络权重：

```bash
python -m curri_DQN.train \
  --stage 2 \
  --load-checkpoint \
    curri_DQN/chkpt/hero_dddqn_stage-01/checkpoint_000002000000.pt
```

`--load-checkpoint` 只迁移网络权重，用于开始一个新的 Stage；`--resume latest` 则恢复同一 Stage 的 optimizer、replay 和训练进度，两者不能同时使用。

默认训练采样中，80% 的 episode 从当前 Stage 开始，20% 从更早的 Stage 中均匀采样，用于减轻学习新课程时对旧能力的遗忘。一个 Stage 含多个 Level 任务时，先均匀选择 task，再选择该 task 的健康 checkpoint variant。Evaluator 使用固定的 checkpoint/seed 组合，并同时报告整体、task 和 checkpoint 级成功率。

### 课程之后的完整回合训练

`--after-curri` 从 Level 1 / Room 1 和 Level 2 / Room 1 各以 50% 概率开始，每个 episode 只营救当前 Level 的矿工。该模式必须从课程 checkpoint 初始化：

```bash
python -m curri_DQN.train \
  --after-curri \
  --load-checkpoint \
    curri_DQN/chkpt/hero_dddqn_stage-03/checkpoint_000002500000.pt \
  --run-dir curri_DQN/runs/hero_dddqn
```

此模式不再使用 Stage 间的 80/20 reset 混合，输出目录自动使用 `_afterCurri` 后缀。

## 实验结果

本项目当前最重要的实验结论不是“所有课程都能收敛”，而是课程学习确实把一个仅凭稀疏终局奖励难以探索的长任务，分解成了多个能够逐步学会的原子任务。

| 训练阶段 | 结果 |
| --- | --- |
| Stage 1 | 能够学会从最靠近矿工的位置完成营救 |
| Stage 2 | 能够继续向前扩展，并覆盖完整的 Level 1 |
| Stage 3 | 能够完成；`checkpoint_000002500000.pt` 可在完整评测中完成 Level 1、Level 2 两类 episode |
| Stage 4 | 未稳定收敛；成功率上升后突然暴跌，并伴随大量超时 |

Stage 4 失败时的典型行为是 agent 在局部区域来回徘徊、持续射击，直到 episode 超时。这说明课程已经显著延长了可学习时域，但从 Level 2 / Room 1 出发的四房间任务仍可能出现策略退化。它是当前实现的已知局限，也为后续研究保留了一个明确问题：如何在继续延长 horizon 时维持旧课程能力，并避免策略坍缩到局部动作循环。

综合来看，Stage 1～3 的可训练性和 Stage 3 checkpoint 的完整任务表现，已经验证了“人工可达状态 checkpoint + 反向课程训练”对纯视觉长 episode 任务的有效性。

## 评测

[`eval.py`](eval.py) 使用 greedy（`epsilon=0`）策略评测指定 checkpoint。每个 episode 随机选择 Level 1 或 Level 2 的 Room 1 起点，因此可以直接检验模型是否真正完成两类完整任务：

```bash
python -m curri_DQN.eval \
  --checkpoint \
    curri_DQN/chkpt/hero_dddqn_stage-03/checkpoint_000002500000.pt \
  --episodes 20
```

增加 `--gui` 可以显示游戏过程：

```bash
python -m curri_DQN.eval \
  --checkpoint \
    curri_DQN/chkpt/hero_dddqn_stage-03/checkpoint_000002500000.pt \
  --episodes 5 --gui --fps 30
```

成功、掉命和超时分别报告为 `miner-rescued`、`life-lost` 和 `timeout`。

训练期间可使用 TensorBoard 查看指标：

```bash
tensorboard --logdir curri_DQN/runs
```

重点关注：

- `success/rollout_current_stage` 与 `success/eval_current_stage`；
- `success/rollout_task/*` 与 `success/eval_task/*`；
- `train/timeout_rate` 与 `timeout/eval_rate`；
- `train/episode_return_mean` 与各类墙壁、敌人、矿工事件统计。

## 代码结构

```text
HeroEnv/
├── teacher.py       # 人工示范、深度标记与 ALE checkpoint 采集
├── curriculum.py    # 课程清单、Stage 编号、校验与冻结
├── hero_env.py      # Level 1～2 的 episode 语义和 checkpoint reset
└── check.py         # 检查冻结课程的 reset 画面

curri_DQN/
├── train.py         # 多 actor 的课程训练入口
├── eval.py          # 独立 checkpoint 评测
├── envs.py          # 图像预处理、frame stack、事件奖励和超时
├── reward.py        # ALE 分值到 H.E.R.O. 事件的解析
├── model.py         # Dueling DQN 网络
├── actor.py         # CPU actor 与 rollout 采集
├── evaluator.py     # 固定任务评测
└── replay.py        # 统一经验回放
```

> H.E.R.O. ROM 不属于本项目代码；运行者需要自行以合法方式准备与 ALE 兼容的 ROM。冻结课程会校验 ROM MD5，以避免 checkpoint 与游戏版本不一致。
