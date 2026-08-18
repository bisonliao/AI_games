# MPE 多智能体强化学习：MADDPG 与 QMIX

本仓库在 MPE（Multi-Agent Particle Environment）上提供两套相互独立的
PyTorch 实现：

- **MADDPG**：现有 faithful PyTorch 迁移版，入口位于 `experiments/` 和
  `maddpg/`。默认使用 archived OpenAI MPE 的 `legacy + official` 接口，也保留
  PettingZoo 对照模式。
- **QMIX**：位于 `QMIX/`，面向共享奖励的纯合作任务，固定使用 archived
  OpenAI MPE 的 `legacy + official` 接口；agent 使用共享 GRU，mixer 使用单调
  mixing network。

两种算法的 TensorBoard 日志和 checkpoint 都按算法分开。默认场景均为
`simple_spread`，默认 episode 长度均为 25 步。

> **项目结论：QMIX和MADDPG都 可以成功完成 `simple_spread`。QMIX需要大力出奇迹：训练至少100万个episode,也就是25M步才有效果。**
>
> 使用默认 QMIX 配置训练得到的 `state_steps_32250000.pt` 已经学会三个 agent
> 对三个 landmark 的分工覆盖。在默认 seed 10000～10009 的 GUI 回放中，10 个
> episode 都曾同时实现三个目标的完整覆盖；按当前“只检查第 25 步终局”的严格
> 指标为 6/10。两者差异来自部分 agent 最后一步因惯性略微滑出覆盖区域，而不是
> QMIX 没有学会任务或几何阈值计算错误。阈值和成功语义见本文最后一章。

## 安装与运行约定

建议从仓库根目录执行所有命令：

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

`requirements.txt` 固定了 archived OpenAI MPE commit `6ed7cac`。默认训练使用
CUDA；QMIX 默认使用 `cuda:0`，可用 `--cuda-device N` 选择 GPU。仅在 CPU
调试时添加 `--no-cuda`。

启动 TensorBoard：

```bash
tensorboard --logdir runs
```

## MADDPG

MADDPG 为每个 agent 使用独立 actor；中央 critic 输入所有 agent 的 observation
和 action。当前实现不使用 RNN，也不进行 BPTT。

### 训练

默认 faithful 配置为 archived MPE、official categorical vector action、
`simple_spread`、25 步/episode、20 万个 episode：

```bash
python -m experiments.train_torch \
  --env-backend legacy \
  --policy-mode official \
  --scenario simple_spread \
  --num-episodes 200000 \
  --save-dir chkpt \
  --tb-log-interval 10000 \
  --checkpoint-eval-episodes 10 \
  --checkpoint-eval-seed 10000
```

这些参数目前也都是默认值，因此以下命令等价于启动默认训练：

```bash
python -m experiments.train_torch
```

从相同配置的最新 checkpoint 恢复训练：

```bash
python -m experiments.train_torch \
  --env-backend legacy \
  --policy-mode official \
  --scenario simple_spread \
  --save-dir chkpt \
  --restore
```

### 独立评测与 GUI

`play_torch.py` 从 checkpoint metadata 恢复环境和网络配置，默认执行确定性、
无 GUI 评测。建议明确指定 `.pt` 文件：

```bash
python -m experiments.play_torch \
  --checkpoint chkpt/maddpg/legacy/official/simple_spread/state_steps_5000000.pt \
  --episodes 100 \
  --seed 20000 \
  --report-json reports/maddpg_simple_spread.json
```

第一个 episode 使用 `--seed`，后续 episode 使用连续 seed。开启 GUI：

```bash
python -m experiments.play_torch \
  --checkpoint chkpt/maddpg/legacy/official/simple_spread/state_steps_5000000.pt \
  --episodes 10 \
  --seed 30000 \
  --render \
  --fps 10
```

默认使用 deterministic argmax 动作；只有显式添加 `--stochastic` 才会采样动作。

### MADDPG 日志与 checkpoint 目录

TensorBoard run：

```text
runs/legacy_official_simple_spread_YYYYMMDD_HHMMSS/
```

checkpoint 默认目录：

```text
chkpt/maddpg/legacy/official/simple_spread/
├── state_steps_<steps>.pt
├── evaluation_steps_<steps>.json
└── evaluation.json
```

通用布局为：

```text
<save-dir>/maddpg/<env-backend>/<policy-mode>/<scenario>/
```

`--checkpoint` 也可以传入上述 scenario 目录，此时自动选择步数最大的
`state_steps_<N>.pt`。当多个训练共享同一 checkpoint 目录时，同算法、同 step
的文件仍可能互相覆盖，因此正式评测最好明确指定 `.pt`。

### MADDPG 关键 TensorBoard 指标

| 指标 | 含义与判断方式 |
|---|---|
| `reward/episode_reward` | 一个 episode 内所有 agent reward 的总和；`simple_spread` 中越高（越不负）越好。 |
| `reward/episode_reward_roll10` | 最近 10 个 episode 的训练回报，用于观察短期趋势。 |
| `reward/agent<N>_episode_reward` | 各 agent 的 episode reward。共享奖励场景中各 agent 通常接近。 |
| `task/covered_landmarks` | 终局被覆盖的 landmark 数量的区间平均。 |
| `task/coverage_ratio` | 终局覆盖比例的区间平均。 |
| `task/episode_success` | 当前日志区间内严格终局成功的比例。 |
| `task/success_rate` | 从训练开始累计的严格终局成功率。 |
| `task/success_rate_roll100` | 最近 100 个 episode 的严格终局成功率，最适合观察业务能力变化。 |
| `eval/episode_reward_mean`、`eval/episode_reward_std` | 每次保存 checkpoint 后，在固定连续 seed 上执行 deterministic 策略的回报均值和标准差。判断泛化时优先于带探索的训练 reward。 |
| `eval/task_coverage_ratio`、`eval/task_episode_success` | checkpoint 固定 seed 评测的终局覆盖率和业务成功率。 |
| `loss/q_loss`、`loss/p_loss`、`loss/pg_loss` | critic、actor 总损失和策略梯度部分；主要用于诊断，不应单独作为任务收敛依据。 |
| `grad/q_grad_norm`、`grad/p_grad_norm` | critic 和 actor 梯度规模；持续异常增大通常表示训练不稳定。 |
| `q/mean_q`、`q/mean_target_q` | critic 当前 Q 与 Bellman target 的均值，用于识别 Q 值漂移。 |
| `policy/action_entropy` | official categorical policy 的动作熵；下降通常表示策略逐渐确定。 |

训练曲线按 `--tb-log-interval` 聚合区间内所有 episode 和 optimizer update，
不是只采样区间最后一个值。

## QMIX

QMIX 使用参数共享的 `FC -> GRUCell -> FC` agent 网络，输入为局部 observation、
上一动作和 agent ID；完整 episode 从初始 hidden state 展开并进行 BPTT。中央
mixer 使用所有局部 observation 的固定顺序拼接作为 state，并通过非负 mixing
权重满足 QMIX 单调性约束。

### 训练

当前默认配置为 `simple_spread`、25 步/episode、200 万个 episode、CUDA 0：

```bash
python -m QMIX.train \
  --scenario simple_spread \
  --num-episodes 2000000 \
  --save-dir QMIX/checkpoints \
  --runs-dir runs \
  --tb-log-interval 10000 \
  --checkpoint-eval-episodes 10 \
  --checkpoint-eval-seed 10000
```

直接启动全部默认值：

```bash
python -m QMIX.train
```

选择其他 GPU：

```bash
python -m QMIX.train --cuda-device 1
```

恢复与 checkpoint metadata 完全匹配的训练：

```bash
python -m QMIX.train \
  --restore \
  --save-dir QMIX/checkpoints
```

QMIX checkpoint 严格校验环境和 learner 配置。恢复后 episode replay 不落盘，
会从空 replay 重新 warm up。

### 独立评测与 GUI

`QMIX.play` 完全从 `.pt` metadata 构建网络，不依赖当前训练默认超参数。无 GUI
评测：

```bash
python -m QMIX.play \
  --checkpoint QMIX/checkpoints/qmix/legacy/official/simple_spread/state_steps_32250000.pt \
  --episodes 100 \
  --env-seed 20000 \
  --report-json reports/qmix_simple_spread.json
```

`--env-seed N` 表示第一个 episode 使用 seed `N`，后续使用 `N+1`、`N+2`……，
可以通过更换 seed 区间测试泛化能力。开启 GUI：

```bash
python -m QMIX.play \
  --checkpoint QMIX/checkpoints/qmix/legacy/official/simple_spread/state_steps_32250000.pt \
  --episodes 10 \
  --env-seed 10000 \
  --render \
  --fps 10
```

`QMIX.play` 始终使用 greedy deterministic 策略。默认使用 CUDA 0；CPU 调试添加
`--no-cuda`。

### QMIX 日志与 checkpoint 目录

TensorBoard run：

```text
runs/qmix_legacy_official_simple_spread[_<exp-name>]_YYYYMMDD_HHMMSS_pid<PID>/
```

checkpoint 默认目录：

```text
QMIX/checkpoints/qmix/legacy/official/simple_spread/
├── state_steps_<steps>.pt
├── evaluation_steps_<steps>.json
└── evaluation.json
```

通用布局为：

```text
<save-dir>/qmix/legacy/official/<scenario>/
```

算法目录 `maddpg/` 与 `qmix/` 保证两种算法不会互相覆盖。同一个 QMIX save-dir
中的并发或重复训练仍可能覆盖同 step 文件；run 日志因时间戳和 PID 可以并存，
但 checkpoint 正式评测仍建议明确指定 `.pt`。

### QMIX 关键 TensorBoard 指标

| 指标 | 含义与判断方式 |
|---|---|
| `reward/episode_reward` | 所有 agent 广播 reward 的 episode 总和，口径与 MADDPG 协作任务一致。 |
| `reward/team_episode_reward` | legacy 环境的一份原始共享团队回报；越高（越不负）越好。 |
| `reward/scaled_team_episode_reward` | learner 写入 replay、用于 TD target 的缩放团队回报，默认缩放为 `1 / n_agents`。 |
| `task/covered_landmarks`、`task/coverage_ratio` | 终局覆盖数量和比例的日志区间平均。 |
| `task/episode_success` | 当前日志区间内严格终局成功的比例。 |
| `task/success_rate` | 从训练开始累计的严格终局成功率。 |
| `task/success_rate_roll100` | 最近 100 个 episode 的严格终局成功率；业务能力的首要训练指标。 |
| `eval/team_episode_reward_mean`、`eval/team_episode_reward_std` | checkpoint 固定连续 seed、greedy 策略的团队回报均值和标准差。 |
| `eval/task_coverage_ratio`、`eval/task_episode_success` | checkpoint 固定 seed 评测的严格终局覆盖率和成功率。 |
| `loss/td_loss` | Huber 或 MSE TD loss；默认 Huber。应结合 Q、梯度和 reward 判断。 |
| `loss/td_error_abs` | 有效 transition 的平均绝对 TD error。 |
| `q/chosen_total`、`q/target_total` | mixer 对已选动作的联合 Q 与 Bellman target 均值；两者长期分离表示拟合异常。 |
| `q/chosen_total_abs_max`、`q/target_total_abs_max` | Q 绝对值峰值；持续快速增长通常是价值漂移的先兆。 |
| `grad/pre_clip_global_norm` | 梯度裁剪前的全局范数，用于观察原始训练压力。 |
| `grad/post_clip_global_norm` | 梯度裁剪后的全局范数，应不超过 `--grad-norm-clip`。 |
| `policy/epsilon` | ε-greedy 当前探索率。`--epsilon-anneal-steps` 的单位是 environment steps。 |
| `replay/episodes`、`learner/updates` | replay 中 episode 数量和累计 learner update 次数。 |
| `learner/last_target_update_episode` | 最近一次 hard target network 同步发生的 episode。 |
| `system/env_steps_per_second` | 本次进程的采样吞吐。 |
| `config/json` | 本次运行完整参数及 checkpoint metadata，排查多次实验混淆时应首先查看。 |

## `simple_spread` 覆盖阈值与成功语义

### 圆形大小决定了为什么阈值是 0.1

Legacy `simple_spread` 中：

- agent 圆半径为 `0.15`；
- landmark 圆半径为 `0.05`；
- 设两者圆心距离为 `d`。

不同“覆盖”语义对应不同阈值：

| 几何语义 | 条件 | 等价圆心距离 |
|---|---|---:|
| landmark 整圆完全处于 agent 圆内 | `d + 0.05 < 0.15` | `d < 0.10` |
| landmark 圆心进入 agent 圆 | `d < 0.15` | `d < 0.15` |
| 两个圆发生任意相交 | `d < 0.15 + 0.05` | `d < 0.20` |

当前 `task/*` 指标使用 `d < 0.1`，与 archived legacy MPE 自带的
`benchmark_data()` 一致；从圆形几何看，它也恰好表示“landmark 整圆被 agent
完整包含”，并非任意经验阈值。严格 `<` 与 `<=` 只影响精确落在边界的状态，
不是当前 60% 与视觉结果差异的原因。

当前实现还使用最大二分图匹配，保证一个 agent 不能同时替两个 landmark 计数。
这比原始 benchmark 的“每个 landmark 找最近 agent”更符合“三个 agent 分别覆盖
三个目标”的业务定义。在 seed 10000～10009 的分析中，每个 landmark 的最近
agent 本来就不同，因此一对一匹配也不是 60% 差异的来源。

### 当前成功率是“终局成功”，不是“回合内曾成功”

训练日志、checkpoint 自动评测以及两个 `play` 入口都在 episode 结束后读取
`task/episode_success`。对于默认 25 步 episode，它回答的是：

> 第 25 步结束时，是否仍有三个不同 agent 分别完整包含三个 landmark？

对 QMIX `state_steps_32250000.pt`、seed 10000～10009 的逐步分析结果为：

- 10/10 episode 都曾在回合内同时完整覆盖三个 landmark；
- 终局仍保持完整覆盖的是 6/10；
- 四个终局失败 episode 最后一次完整覆盖分别发生在第 24、24、23、24 步；
- 它们第 25 步最差圆心距离为 `0.1198`、`0.1220`、`0.1021`、`0.1015`。

后两局只超出完整包含边界 `0.0021` 和 `0.0015`，在 GUI 中不足或接近一个
像素；前两局也仍然高度重叠。因此视觉上看到“每局都完成覆盖”与标准输出 60%
并不矛盾：人眼观察了整段轨迹，而当前指标只截取最后一帧。

### 沉淀结论

1. **保留 `0.1` 作为严格完整覆盖阈值是合理的。** 把阈值放宽到 `0.15` 表示
   “圆心进入”，放宽到 `0.20` 只表示“相交”，不能再称为完整包含。
2. **QMIX 已经学会并能够成功完成 `simple_spread`。** 当前 32.25M-step
   checkpoint 在所检查的 10 个连续 seed 中全部至少完成过一次三目标同时完整
   覆盖，且 60% 能在终局继续保持。
3. **“完成过”与“最终稳定驻留”是两个不同业务指标。** 当前
   `task/episode_success` 应继续解释为严格终局成功率；如果业务只要求回合内完成
   一次，未来应新增独立的 `episode_ever_success`，而不是修改或混淆现有指标。
4. 若关注稳定停靠，还可以额外定义“连续 K 步保持完整覆盖”或“最后 K 步覆盖
   比例”，它们比单帧终局指标更能描述控制稳定性。这些属于后续指标扩展，当前
   代码尚未实现。

## 测试

MADDPG 与公共组件：

```bash
python -m unittest discover -s tests -v
```

QMIX：

```bash
python -m unittest discover -s QMIX/tests -v
```

长时间、多 seed 的收敛实验不属于单元测试范围。
