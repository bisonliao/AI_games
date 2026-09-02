# DQN Actor-Learner

Ms. Pac-Man 的多进程 Double-Dueling DQN 实现。架构借鉴 Ape-X 的 CPU Actor、
集中 Learner 和优先经验回放，但所有 Actor 使用相同的全局 epsilon 退火策略。

## 进程结构

```text
Actor 0 --\
Actor 1 ----> rollout_queue --> Learner(cuda:0, replay, TensorBoard, checkpoint)
...       /                         |                    |
                                parameter queues   eval request
                                             |                v
                                          Actors        Evaluator(CPU)
                                                             |
                                                        eval result
                                                             v
                                                          Learner
```

- 每个 Actor 只在 CPU 上执行批量推理，并拥有独立的 `AsyncVectorEnv`。
- Actor 每采集 64 条 transition 才发送一个 chunk。rollout 队列满时持续阻塞重试，
  不丢弃数据。
- Actor 成功发送 chunk 后才检查自己的参数队列，drain 所有旧版本并加载最新版本。
- Learner 是唯一使用 GPU、replay buffer 和 `SummaryWriter` 的进程。
- Evaluator 只使用 CPU；它读取原子 checkpoint，完成 50 局 greedy 完整游戏后把
  结果发回 Learner，由 Learner 写 TensorBoard。

## 默认训练配置

所有常量集中在 `DQN/config.py` 的 `DQNConfig`，包括嵌套的环境配置。主要默认值：

| 配置 | 默认值 |
| --- | ---: |
| Actor 数 | 2 |
| 每 Actor 环境数 | 8 |
| Actor chunk | 64 transitions |
| replay capacity | 100,000 |
| learner batch size | 64 |
| learning starts | 20,000 transitions |
| SGD update/transition | 0.25 |
| epsilon | 0.9 -> 0.05 over 1M global transitions |
| TensorBoard 间隔 | 200K global transitions |
| checkpoint 间隔 | 1M global transitions |
| checkpoint 评测局数 | 50 |
| 单局评测最大长度 | 30,000 decisions |
| 总训练量 | 10M global transitions |

`updates_per_transition=0.25` 按优化器更新次数解释：Learner 每消费一个 64 条
chunk，在 replay warmup 完成后累计执行 16 个 minibatch 更新，然后广播一次最新
参数。修改 Actor 环境数时，`actor_transition_batch_size` 必须仍能被它整除。

训练和评测环境本身固定返回 ALE 原始奖励：`step_cost=0` 且不做环境内裁剪。
Actor 和 Evaluator 在收到原始奖励后统一执行：

```text
score_reward = clip(log1p(max(raw_reward, 0) / 10), 0, 5)
shaped_reward = score_reward - 0.01
if life_lost:
    shaped_reward = -5
if game_over:
    shaped_reward = -10
```

因此 `rollout/eval` 的 `episode_return_mean` 表示塑形后的训练目标回报；
`raw_score_mean` 始终是未塑形、未裁剪的业务游戏分数。对当前 ROM 使用 10 个随机
种子实测了 30 次 lost-life 事件，ALE 当步原始奖励全部为 `0.0`，不会自行给出负
奖励，所以显式死亡惩罚是必要的。Game Over 覆盖普通丢命惩罚，不与 `-5` 叠加。
对数尺度、裁剪上下限、决策成本和两类死亡惩罚都在 `DQNConfig` 中集中配置。

## Epsilon 与多样性

整个系统只维护一个原子 global transition 计数。唯一写入者是 Learner：每当
Learner 从 rollout queue 成功收到一个 Actor chunk，就把该 chunk 的 transition
数量加入全局计数。Actor 只读取这个值，并使用完全相同的函数：

```text
epsilon(t) = 0.9 + min(t / 1,000,000, 1) * (0.05 - 0.9)
```

这里没有按 Actor 分层。多样性来自以下独立来源：

- Actor `i` 的 ALE seed 区间从 `seed + i * actor_seed_stride` 开始；每个向量子环境
  又使用连续且唯一的 seed。
- 每个 Actor 的 epsilon 随机数生成器使用另一个互不重叠的 seed。
- ALE 保留随机 no-op 和 `repeat_action_probability=0.25`。
- `rollout/observation_unique_fraction` 持续监控每个 chunk 中观测哈希的唯一比例，
  可用于发现环境意外复制同一轨迹。

## TensorBoard 与 checkpoint

运行目录格式为：

```text
runs/YYYYMMDD-HHMMSS-pid<PID>/
chkpt/YYYYMMDD-HHMMSS-pid<PID>/checkpoint_step_XXXXXXXXXXXX.pt
```

核心指标包括：

- `rollout/transitions_per_second`
- `learner/consumed_transitions_per_second`
- `rollout/episode_length_mean`
- `rollout/episode_return_mean`
- `rollout/raw_score_mean`
- `rollout/observation_unique_fraction`
- `train/epsilon`、`train/loss`、`train/q_mean`
- `train/td_error_abs_mean`、`train/gradient_norm`
- `eval/episode_length_mean`、`eval/episode_return_mean`
- `eval/raw_score_mean`、`eval/raw_score_median/p25/p75`
- `eval/capped_episode_count`、`eval/capped_episode_fraction`

Evaluator 最多运行 `evaluation_max_episode_steps=30,000` 个决策步；默认
`frame_skip=4` 时约为 120,000 个模拟器帧。达到上限的局会记录截止时的回报和
原始分，并单独计入 capped 指标，不会伪装成自然 Game Over。

checkpoint 保存 online/target 网络、优化器和训练计数；为控制 checkpoint 大小，不
保存 replay 内容。恢复训练会重新 warmup replay，但 epsilon 和 checkpoint/TB 计数
从 checkpoint 的 global transition 继续。

## 运行

在项目根目录执行：

```bash
python -m DQN.train
```

常用临时覆盖：

```bash
python -m DQN.train \
  --num-actors 2 \
  --envs-per-actor 8 \
  --total-transitions 10000000 \
  --learner-device cuda:0
```

恢复 checkpoint：

```bash
python -m DQN.train --resume chkpt/<run>/checkpoint_step_000001000000.pt
```

查看日志：

```bash
tensorboard --logdir runs
```

加载 checkpoint 做多轮 greedy 评测：

```bash
python -m DQN.play \
  --checkpoint chkpt/<run>/checkpoint_step_000001000000.pt \
  --episodes 50
```

显示 ALE 游戏窗口：

```bash
python -m DQN.play \
  --checkpoint chkpt/<run>/checkpoint_step_000001000000.pt \
  --episodes 5 \
  --gui \
  --fps 15
```

`play.py` 使用 checkpoint 中保存的环境、reward shaping、评测 seed 和最大 Episode
长度配置，在 CPU 上执行 greedy 策略，并输出每局及汇总的塑形回报、业务原始分、
Episode 长度和 capped 状态。`--fps` 是 GUI 下的目标决策画面刷新率，只节流播放
速度，不改变 ALE 模拟器帧数、frame skip 或 transition；ALE 自身的实时渲染上限
仍然生效。

## 验证

```bash
python -m pytest DQN/tests
```
