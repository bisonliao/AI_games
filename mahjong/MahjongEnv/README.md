# MahjongEnv 环境接口

本文档记录当前麻将环境的实际接口。环境规则是湖南红中麻将：只能使用万、条、饼、红中；红中作为万能牌；不吃；可以碰、杠、胡；训练目标是尽量多自摸胡牌。

## 环境与训练对手

训练入口创建 `MahjongEnv(observation_mode="vector")`，并通过 `SubprocessVectorEnv` 并行运行多个环境。玩家 1、2、3 是启发式机器人：它们摸牌后先检查自摸，然后使用 `MahjongEnv.bots.choose_discard` 出牌。该机器人根据对子、刻子和相邻牌给牌评分，优先打孤张；默认 `robot_randomness=0.35`，因此在近似最差牌中保留随机性。当前机器人没有实现自己的碰、杠决策，训练 agent（玩家 0）才执行碰、杠、胡和过。

## 观测空间

向量观测为 `float32[306]`，同时提供 `uint8[141]` 的 `action_mask`。mask 中 1 表示当前动作合法，牌编号统一为 0--26 的普通牌和 27 的红中。

306 维向量按以下顺序拼接；牌计数特征除以 4，标量特征使用代码中的固定归一化：

| 区段 | 维度 | 内容 |
| --- | ---: | --- |
| 手牌 | 28 | 玩家 0 当前手牌各牌计数 `/4` |
| 自有副露 | 28 | 玩家 0 的碰、明杠、暗杠、加杠牌计数 `/4` |
| 各家弃牌 | 112 | 4 个玩家 × 28 种牌的弃牌计数 `/4` |
| 公共副露 | 112 | 4 个玩家 × 28 种牌的可见副露计数 `/4`；玩家 0 的暗杠不对自己以外泄露 |
| 副露类型 | 16 | 4 个玩家 × 4 种类型（碰、明杠、暗杠、加杠）计数 `/4` |
| 状态标量 | 4 | 剩余牌墙 `len(wall)/112`、当前玩家 `/3`、步数 `/max_steps`、上一张弃牌 `/27`（无弃牌为 `-1/27`） |
| 阶段标志 | 4 | `phase == play`、`phase == claim`、是否抢杠胡、当前手牌是否已胡 |
| 牌效特征 | 2 | `potential(hand, meld_count, visible_counts)/10`、玩家 0 弃牌数 `/30` |

因此 `28+28+112+112+16+4+4+2=306`。环境也支持 RGB 或 combined observation，但 DDDQN 训练只使用上述向量。

## 动作空间

动作空间是固定的 `Discrete(141)`。每个牌相关动作占 28 个编号，即“动作类型偏移 + 牌编号”：

| 编号 | 动作 | 说明 |
| ---: | --- | --- |
| `0..27` | `discard(tile)` | 打出牌 `tile` |
| `28..55` | `peng(tile)` | 对上家弃牌碰牌 |
| `56..83` | `ming_kong(tile)` | 对上家弃牌明杠 |
| `84..111` | `concealed_kong(tile)` | 暗杠 |
| `112..139` | `added_kong(tile)` | 碰后摸到同牌加杠 |
| `140` | `pass` | 对可碰/可杠的弃牌过牌 |

摸牌后如果玩家已经胡牌，环境自动结束 episode 并发放胡牌奖励，不需要 agent 额外选择“胡”动作。非法动作由环境返回 `-0.2`，不推进游戏。正常训练中 action mask 会阻止 agent 选择非法动作。终局 observation 的 mask 为空；向量环境随后自动 reset，并在 transition 的 info 中保留终局信息。

## 奖励空间与奖励设计

每次 `step` 返回一个标量 `float` 奖励。当前奖励由以下部分相加：

```text
r = discard_shaping
    + event_reward（发生 discard/peng/kong 时）
    + terminal_reward（仅终局）
```

默认 `event_reward=0`，所以实际训练中不会因为出牌、碰牌或杠牌事件本身得到额外奖励。

### 弃牌后的稠密 shaping

`potential` 只关注残句质量，不直接奖励已经完成的成句或副露：

```text
potential = - isolated_count
            + 0.5 * ukeire_count
```

`isolated_count` 尝试拆出若干成句和一对将牌，红中作为万能牌参与拆解；然后对剩余未成句的牌计分：

| 残子类型 | 例子 | 分值 |
| --- | --- | --- |
| 天然对子 | 8万、8万 | 整对 `PAIR_WEIGHT=0.5` |
| 未使用的红中 | 红中 | 每张 `RED_RESIDUAL_WEIGHT=0.25` |
| 两面搭子（同花色 23、34、45、56、67、78） | 2万、3万，可进 1万或4万 | 每张 `OPEN_TATSU_TILE_WEIGHT=0.5`，合计 1.0 |
| 边张搭子（同花色 12、89） | 8万、9万，只可进 7万 | 每张 `CLOSED_TATSU_TILE_WEIGHT=0.75`，合计 1.5 |
| 嵌张搭子（同花色点数相差 2） | 7万、9万，只可进 8万 | 每张 `CLOSED_TATSU_TILE_WEIGHT=0.75`，合计 1.5 |
| 其他孤子 | 无法归入以上组合的残子 | 每张 1.0 |

在不同拆法中取最低残子分；每张牌只能用一次，且搭子不能跨花色。例如 `1万3万5万` 只能拆成
一个嵌张搭子和一个孤子，计 2.5，不能让 3万同时参与两个搭子。成句及选定的一对将不计残子分。
这项计算在 `MahjongEnv/rules.py` 中，由 DQN 和 DDDQN 共用，同时影响潜势观测和弃牌后的潜势差奖励。
修改后需要重新启动训练进程；已经运行的进程不会自动加载新代码，旧日志和旧数据不会被重算。

`ukeire_count` 只对听牌状态生效。对于已经确定的残手牌，只有存在能够直接胡牌的摸牌时才统计这些
牌的剩余张数；如果当前不是听牌状态，`ukeire_count=0`。有效进张根据残句生成，再按 4 张总数扣除
自己手牌和公开可见的弃牌、副露。因此海里已经出现的牌不会被重复计入听牌张数。`ukeire()` 接收
已经确定的残手牌，不负责枚举应该打哪一张牌。

环境只在 agent 完成 `discard` 动作后计算 shaping：

```text
new_potential = potential(打牌后的残手牌, 当前可见死牌)
delta = new_potential - last_discard_potential
discard_shaping = potential_scale * delta
```

`last_discard_potential` 保存上一次 agent 完成弃牌后的完整潜势，并在每次弃牌后更新。每局 reset 后还没有
上一次弃牌基准，因此本局第一次弃牌的 `discard_shaping` 为 0；从第二次弃牌开始比较。摸牌形成
的 14 张手牌只是动作前的临界状态，奖励计算点是打完牌后的残手牌，通常为 13 张。碰、杠、过和胡
动作不会触发这项 shaping。弃牌后的 ukeire 会在三家机器人行动和玩家下一次摸牌之前计算，避免把
下一轮环境变化混入本次弃牌的计算时刻。

潜势本身同时综合 `isolated_count` 和 `ukeire_count`：

```text
potential = -isolated_count + 0.5 * ukeire_count
```

因此 shaping 是完整潜势的差值，不是单独的 ukeire 差值。`isolated_count` 越小、`ukeire_count`
越大，潜势越高。该分数不因 `meld_count` 本身增加而增加；副露数量只作为胡牌判定所需的结构参数。当前有效进张只使用玩家可见的死牌信息，
不扣除对手暗手中的未知牌。

需要区分奖励 shaping 和观测向量中的牌效特征：306 维观测仍然会把当前状态的
`potential(hand, meld_count, visible_counts)/10` 写入牌效特征，但这不是额外奖励；奖励 shaping
只使用上面定义的“完成弃牌后的完整 potential 差分”。

### 非法动作与终局奖励

对非终局状态执行非法动作时，环境返回固定 `-0.2`，不推进游戏，也不产生 shaping。终局状态没有
合法动作；如果误收到动作，环境直接返回对应终局奖励并标记 `terminated=True`，避免环境停在终局状态
反复产生非法动作 transition。

默认终局奖励来自 `EnvConfig`：

| 结果 | 奖励 |
| --- | ---: |
| 玩家 0 胡牌（包括抢杠胡） | `+10.0` |
| 其他玩家自摸，玩家 0 输 | `-10.0` |
| 牌墙耗尽或达到最大步数 | `-1.0` |

终局奖励可以和弃牌 shaping 同时出现。例如 agent 弃牌后其他玩家立即自摸，该 step 的奖励为：

```text
potential_scale * (new_potential - last_discard_potential) + loss_reward
```

如果摸牌后自动检测到玩家已经胡牌，则只有终局胡牌奖励（默认 `+10.0`），不会额外计算弃牌 shaping。

默认参数为 `potential_scale=0.5`、`ukeire_weight=0.5`、`event_reward=0.0`、`win_reward=10.0`、`loss_reward=-10.0`、`draw_reward=-1.0`。这些参数属于 `MahjongEnv.config.EnvConfig`，不是 DDDQN 网络的超参数。
