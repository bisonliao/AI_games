# DQN 五子棋训练

## 训练架构

```text
Rollout A（CPU，多进程 self-play）── 黑棋 transition ──> Replay A ──┐
                                                           ├─> Learner（单进程/GPU）
Rollout B（CPU，与规则专家对弈）── 双方 transition ──> Replay B ──┘
                                         │
Learner 定期同步最新 DQN 参数 <───────────┘

保存 checkpoint ──> 独立 CPU Evaluator ──> DQN执黑 vs 规则专家 ──> TensorBoard
```

- **Rollout A 是主 self-play 数据源**：黑方是被训练的 DQN agent，白方策略与棋盘状态共同被视作环境。每个 actor 在多个同步环境中批量推进，白方使用随机策略或近期历史 checkpoint，只生成黑棋 transition。
- **Rollout B 是独立的“与专家对弈”旁路**：最新 DQN 与启发式规则机器人对弈，黑白双方 transition 都转换为当前玩家视角后写入 Replay B。引入它的初衷是让 DQN 接触并学习更高质量的专家动作，同时学习如何应对专家，避免 Replay 中长期充斥 self-play 双方产生的低质量“馊招”。
- Learner 独占网络、optimizer、Replay A/B 和 GPU；Replay B 预热后，每个 minibatch 按配置混合 A/B 数据，但更新次数仍由 Rollout A black steps 决定。
- **每次保存 checkpoint 后都会异步进行专家评测**：独立 evaluator 按 FIFO 在 CPU 上让该 checkpoint 执黑、以 greedy 策略对弈启发式机器人，并把胜、负、和及耗时写入 TensorBoard。该评测提供跨 checkpoint 相对稳定的外部标尺，是判断棋力是否真正进步的重要参考指标；评测异步执行，不阻塞训练。

## 启发式机器人（规则专家）

启发式机器人由 `heuristic_agent.py` 实现，只依赖 NumPy。它会优先寻找本方立即获胜点、封堵对方必胜点，再识别双杀等战术威胁，并结合有限候选动作的浅层搜索、棋形、开放端、中心位置和落子距离进行评分；同分动作使用带种子的随机选择，以兼顾可复现性和对局多样性。

它不参与梯度更新，主要承担两个角色：为 Rollout B 提供相对高质量的专家经验，以及作为 checkpoint 评测中固定、稳定的对手。这里的“专家”表示它通常明显强于随机策略，并不代表完美棋力、严格 minimax 策略或完整禁手规则实现。

## 关键实体（7个）

1. **训练控制器（`train.py`）**：启动/关闭各进程，消费 transition，调度更新、权重同步、checkpoint 和 TensorBoard。
2. **DQNAgent（`agent.py`）**：持有 online/target network、optimizer、Replay A，并实现混合采样、梯度更新和 checkpoint。
3. **Rollout A Actor（`async_train.py`）**：主 self-play 采样进程，只产生黑棋 decision-interval transition。
4. **Replay A/B（`ReplayBuffer`）**：A 保存主 self-play 数据；B 保存启发式旁路的双向数据。
5. **HeuristicSidecar（`heuristic_sidecar.py`）**：独立的 Rollout B 多进程控制器和“与规则专家对弈”的采样循环，不阻塞 Rollout A。
6. **PlayerTransitionCollector（`player_transitions.py`）**：按单一玩家视角维护 pending/n-step 状态；黑白各自实例化，避免状态交叉。
7. **HeuristicEvaluator（`evaluator.py`）**：异步加载新 checkpoint，在 CPU 上对战启发式机器人并返回重要的棋力进展指标。

训练入口只有 actor–learner 模式；`--num-actors` 必须至少为 1。Rollout B 可通过 `--num-sidecar-actors 0` 关闭。
