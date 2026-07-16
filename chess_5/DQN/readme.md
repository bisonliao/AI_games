# DQN 五子棋训练

## 训练架构

```text
Rollout A（CPU，多进程）── 黑棋 transition ──> Replay A ──┐
                                                           ├─> Learner（单进程/GPU）
Rollout B（CPU，独立旁路）── 双方 transition ──> Replay B ──┘
                                         │
Learner 定期同步最新 DQN 参数 <───────────┘

保存 checkpoint ──> 独立 CPU Evaluator ──> DQN执黑 vs 启发式白棋 ──> TensorBoard
```

- Rollout A 是主数据源：每个 actor 在多个同步环境中批量决策，黑棋使用最新 DQN，白棋使用随机或近期历史 checkpoint，只生成黑棋 transition。
- Rollout B 与 A 完全隔离：最新 DQN 和启发式机器人对弈，黑白双方 transition 都转换为当前玩家视角后写入 Replay B。
- Learner 独占网络、optimizer、Replay A/B 和 GPU；Replay B 预热后，每个 minibatch 按配置混合 A/B 数据，但更新次数仍由 Rollout A black steps 决定。
- 每次保存 checkpoint 后，训练进程立即继续；独立 evaluator 按 FIFO 在 CPU 上完成评测，再由 learner 写入日志。

## 关键实体（7个）

1. **训练控制器（`train.py`）**：启动/关闭各进程，消费 transition，调度更新、权重同步、checkpoint 和 TensorBoard。
2. **DQNAgent（`agent.py`）**：持有 online/target network、optimizer、Replay A，并实现混合采样、梯度更新和 checkpoint。
3. **Rollout A Actor（`async_train.py`）**：主 self-play 采样进程，只产生黑棋 decision-interval transition。
4. **Replay A/B（`ReplayBuffer`）**：A 保存主 self-play 数据；B 保存启发式旁路的双向数据。
5. **HeuristicSidecar（`heuristic_sidecar.py`）**：独立的 Rollout B 多进程控制器和采样循环，不阻塞 Rollout A。
6. **PlayerTransitionCollector（`player_transitions.py`）**：按单一玩家视角维护 pending/n-step 状态；黑白各自实例化，避免状态交叉。
7. **HeuristicEvaluator（`evaluator.py`）**：异步加载新 checkpoint，在 CPU 上对战启发式机器人并返回胜负和耗时。

训练入口只有 actor–learner 模式；`--num-actors` 必须至少为 1。Rollout B 可通过 `--num-sidecar-actors 0` 关闭。
