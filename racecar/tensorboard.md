# RaceCar PPO TensorBoard 指标说明

本文描述当前 `ppo/train.py` 产生的 TensorBoard 指标。旧实验目录可能还保留已经删除的
tag，应以本文和当前代码为准。

## 统计口径

- 横轴是 learner 的 `global_steps`，代表所有 actor、所有环境累计产生的 transition 数。
- `window/*`、`actions/*`、`ppo/*` 和性能指标约每 200,000 环境步汇总一次，汇报后清空，
  **不是从训练开始累计**。因为一次 PPO batch 有固定步数，实际横轴可能略微越过 200,000
  的整数倍。
- 一个跨越汇报边界的回合在结束时才进入 `window/*`；它的 return 和 length 是完整回合值，
  不是只统计落在当前窗口内的部分。
- TensorBoard 页面上的 smoothing 只改变显示曲线，不改变日志原始值。稀疏的 eval 指标建议
  先用较低 smoothing 查看。

## 最值得先看的指标

建议先联合观察以下几组，而不是只看 reward：

1. `window/success_rate`、`eval/stochastic_success_rate`、
   `eval/greedy_success_rate`：训练采样策略和两种部署方式是否真的成功。
2. `window/mean_terminal_distance`：成功率尚未变化时，车辆是否已经在接近终点。
3. `window/collision_rate`、`window/time_limit_rate`：失败方式是撞墙还是路线低效。
4. `ppo/entropy`、`actions/*_rate`：策略是在合理探索，还是已经动作坍缩。
5. `ppo/approx_kl`、`ppo/clip_fraction`、`ppo/update_epochs_used`：一次 PPO update 的实际
   修改幅度以及是否频繁触发 KL 提前停止。
6. `ppo/explained_variance`、`ppo/value_loss`：critic 是否能提供可信的 advantage 基线。

## `window/*`：最近一个训练窗口内完成的回合

训练 actor 使用 `Categorical.sample()` 选择动作，因此这里反映的是 **stochastic policy**，
并且窗口内可能包含多个相邻 policy version 的回合。

| 指标 | 计算口径 | 如何解读 |
| --- | --- | --- |
| `window/success_rate` | 成功回合数 / 窗口内结束回合数 | 核心训练成功率。比 eval 样本更多，但初态、动作采样和策略版本都在变化，不宜与 eval 数值强行相等。 |
| `window/episodes` | 窗口内结束的回合数 | 判断其他回合指标的样本量。长回合阶段该值小，成功率自然更抖。 |
| `window/successes` | 窗口内成功回合的绝对数量 | 和 `episodes` 一起判断 success rate 的统计置信度；不能脱离窗口回合数单看。 |
| `window/collision_rate` | 撞墙结束数 / 结束回合数 | 上升通常表示控制更激进或局部转向错误；下降但 timeout 上升可能只是从“撞墙”变成“走不完”。 |
| `window/time_limit_rate` | 达到 1500 步数 / 结束回合数 | 高值表示路线低效、卡住或未学会推进。time-limit 是 truncation，训练 GAE 会从 terminal observation bootstrap。 |
| `window/mean_episode_length` | 所有结束回合的平均步数 | 下降可能是更快成功，也可能是更快撞墙，必须和 success/collision 联看。 |
| `window/mean_success_length` | 只对成功回合统计平均步数 | 成功率稳定后越低通常表示路线更高效。成功样本很少时噪声很大，且可能有幸存者偏差。 |
| `window/mean_episode_return` | 完整结束回合的平均 shaped return | 当前奖励是距离进展、step cost 和终奖的组合。适合诊断，不应替代成功率。 |
| `window/mean_terminal_distance` | 所有结束回合结束位置到终点 `(14.5, 7.5)` 的平均欧氏距离 | 很适合发现“还没成功但在进步”。成功率走平时若它继续下降，策略可能正在改善后半程。成功率提高也会机械地拉低它。 |
| `window/mean_step_reward` | 当前窗口所有 transition 的 shaped reward 平均值 | 比 episode return 更少受回合长度影响；仍会受成功、碰撞和 timeout 终奖比例影响。 |

当前 shaped reward 不是 `RaceCarEnv` 的原始正距离奖励：普通步为
`5 × (旧距离 - 新距离) - 0.01`，到达终点为 `+20`，碰撞为 `-5`，超时为 `-2`；终止步
使用终奖替换普通步奖励。因此 reward 上升可能来自距离推进、失败类型改变或成功数增加。

## `eval/*`：固定条件下的 checkpoint 评测

每次约 1,000,000 步 checkpoint、恢复起点和最终点都会评测。两种模式使用相同的 32 个
固定且互异的初态 seed、相同的起点位置/朝向扰动和 1500 步上限；使用 8 个并行环境。

| 指标 | 动作方式 | 含义 |
| --- | --- | --- |
| `eval/greedy_success_rate` | 每步取 logits 的 `argmax` | 确定性部署策略在固定初态集上的成功率。 |
| `eval/stochastic_success_rate` | 每步按 categorical 概率采样 | 与 actor 和默认 `ppo.play` 更一致。每个评测回合拥有固定、独立的动作 RNG seed，checkpoint 之间可比，也不受并行调度影响。 |

32 回合意味着成功率最小变化单位是 `1/32 = 0.03125`。两条曲线差异大并不自动表示 eval
有 bug：离散动作是满左、直行、满右，stochastic 连续采样可以形成时间平均转向，而逐步
argmax 可能长期固定在一个动作上。`window/success_rate` 高而 greedy 低时，应优先比较
`eval/stochastic_success_rate`；同时这也说明策略概率分布本身有价值，不能只看最大概率动作。

`checkpoint_best_greedy.pt` 首先按 greedy 成功率选择；greedy 相同时才用 stochastic
成功率打破平局。

## `actions/*`：训练采样动作比例

| 指标 | 动作 ID | 环境转向 |
| --- | ---: | ---: |
| `actions/straight_rate` | 0 | `0` |
| `actions/left_rate` | 1 | `-1` |
| `actions/right_rate` | 2 | `+1` |

三者之和应约为 1。单一动作比例接近 1 且 entropy 很低，通常表示策略坍缩；但 S 型赛道在
不同训练阶段本来就可能偏向某一方向，因此应结合成功率判断。动作比例接近均匀也不一定好，
它可能表示尚未学到状态相关控制。

## `ppo/*`：learner 更新状态

这些指标是在一个汇报窗口内，对所有 PPO update/minibatch 结果再次取平均。

| 指标 | 含义与解读 |
| --- | --- |
| `ppo/policy_loss` | PPO clipped surrogate policy loss。绝对正负值不能直接代表策略好坏；advantage 已标准化，重点看是否异常发散，并结合 KL、clip fraction 和成功率。 |
| `ppo/value_loss` | critic 的 clipped value loss，代码中包含 `0.5` 系数。量级受 shaped reward 和 return 尺度影响；突然增大表示 critic 跟不上新的回报分布或发生不稳定。 |
| `ppo/entropy` | 三动作 categorical 分布的平均熵，单位为 nat；三动作均匀分布的上限是 `ln(3) ≈ 1.099`。下降表示动作更确定。它是策略实际熵，不等于 entropy coefficient。 |
| `ppo/approx_kl` | 新旧策略的近似 KL：`mean((ratio - 1) - logratio)`。越大表示单次更新改变策略越多；超过 target KL 会在当前 epoch 结束后提前停止。 |
| `ppo/clip_fraction` | 概率比率落在 `[1-clip, 1+clip]` 之外的样本比例。长期接近 0 可能更新过弱；长期很高表示大量样本依赖 clipping，更新较激进。没有脱离任务表现的统一最佳值。 |
| `ppo/learning_rate` | 当前窗口实际采用的 LR 平均值，由 `LEARNING_RATE_POINTS` 插值得到。跨越调度锚点的窗口会显示区间平均值。 |
| `ppo/entropy_coef` | loss 中 entropy bonus 的权重，不是 entropy 本身。它为 0 后策略仍可能保留非零熵，只是不再被 loss 主动奖励探索。 |
| `ppo/clip_coef` | PPO ratio clipping 半径，当前固定为 `0.2`。 |
| `ppo/target_kl` | KL 提前停止阈值，当前固定为 `0.01`。它不会直接截断单个 minibatch，而是在 epoch 结束后判断。 |
| `ppo/update_epochs_used` | 每批 rollout 实际执行的 epoch 数，最多 8。因为 target KL 提前停止，窗口平均值可以是小数。持续明显小于 8 表示 KL 经常限制实际更新次数。 |
| `ppo/explained_variance` | `1 - Var(return - value) / Var(return)`。接近 1 表示 critic 很好地解释 GAE return；接近 0 表示不优于常数预测；负值表示比常数基线更差。return 方差极小时代码记为 0。 |

诊断更新是否“不够”时，不要只看 GPU 利用率：如果 `update_epochs_used` 已接近 8、
`approx_kl` 和 `clip_fraction` 仍很低，才更支持单次更新偏弱；如果 epochs 经常因 KL 提前结束，
增加最大 epochs 并不会带来等比例更新。

## `perf/*` 与 `queue/*`：吞吐和瓶颈

| 指标 | 计算口径 | 如何解读 |
| --- | --- | --- |
| `perf/mean_rollout_collect_seconds` | actor 加载新权重后，直接驱动其多个 PyBullet 环境完成一个 rollout 的平均墙钟时间 | 越低表示 actor 环境采样越快。记录的是 actor 平均值，不是最慢 actor。 |
| `queue/mean_learner_rollout_wait_seconds` | learner 从开始接收，到收齐当前 policy version 的所有 actor rollout 的平均等待时间 | 同步框架中它主要包含 actor 采样和最慢 actor 拖尾，不只是队列 IPC。高值通常说明 learner 在等环境。 |
| `perf/steps_per_second` | 窗口环境步数 / 窗口总墙钟时间 | 端到端吞吐，包含采样、learner 更新和其他训练主线程耗时。 |

瓶颈判断可使用下面的组合：

- rollout collect 和 learner wait 都高、steps/s 低：优先怀疑 PyBullet/actor/慢 actor。
- learner wait 低但 steps/s 仍低：优先检查 PPO 更新、数据搬运或 checkpoint eval。
- learner wait 明显高于平均 collect：可能存在 actor 拖尾、调度不均或队列开销；平均 collect
  会掩盖最慢 actor。
- checkpoint 现在会连续执行 greedy 和 stochastic 两次评测。评测发生在汇报计时重置之后
  时，其耗时可能压低下一个窗口的 `steps_per_second`，这不代表常规 rollout 突然变慢；
  `mean_rollout_collect_seconds` 通常不会受到这段评测时间影响。

## 常见曲线组合

- success rate 不变、terminal distance 下降：策略在接近终点，继续训练可能仍有价值。
- return 上升但 success/distance 不改善：检查 reward hacking 或失败类型比例变化。
- entropy 快速下降、某动作比例趋近 1、success 下降：探索过早消失或策略坍缩。
- value loss 高且 explained variance 为负：critic 不可靠，advantage 噪声可能拖累 policy。
- approx KL 接近 target、epochs used 经常降低：KL 正在限制更新，不能把“配置为8 epochs”
  理解成每次一定执行8轮。
- stochastic eval 稳定而 greedy eval 很差：策略依赖动作混合；默认 stochastic 回放更能代表
  actor 训练时的真实使用方式。
