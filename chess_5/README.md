# 五子棋智能体训练项目

本项目使用强化学习和行为克隆训练五子棋智能体，运行环境为 Conda `mygames`。

项目包含三套并列且相互独立的训练方案：

- `DQN/`：在线强化学习主线，通过 self-play、历史 checkpoint 陪练和启发式专家旁路训练 DQN；
- `BC/`：离线行为克隆流水线，使用启发式专家生成的数据进行监督学习；
- `afterBC/`：以 BC_BEST 初始化白棋，用冻结 BC_BEST 黑棋构成环境，再通过 Ape-X 风格的 Dueling Double DQN 继续提高白棋能力。

三套方案共享同一个 Gymnasium 五子棋环境 `gomoku_env/`。各训练目录保持代码独立；
`afterBC/` 在自己的目录中保存一份经过 hash 校验的 BC_BEST 权重副本，不导入 `BC/`、
`DQN/` 或 `AZ/` 的代码。

## 目录结构

```text
chess_5/
├── DQN/                 # DQN 在线强化学习、评测和人机对弈
├── BC/                  # 行为克隆的数据生成、训练、评测和人机对弈
├── afterBC/             # 从 BC_BEST 继续训练白棋的 Ape-X DDDQN 方案
├── gomoku_env/          # 三套算法共享的五子棋 Gymnasium 环境
├── tests/               # 自动化测试
├── setup_mygames_env.sh # Conda 环境初始化脚本
└── README.md
```

## DQN 强化学习主线

`DQN/train.py` 是强化学习训练入口。Learner 持有唯一的待训练 DQN、optimizer、Replay Buffer 和 GPU，采样进程只负责生成经验。

整体架构如下：

```text
Rollout A：DQN 黑棋 vs 陪练白棋 ──黑棋 transition──> Replay A ──┐
                                                                ├─> DQN Learner
Rollout B：当前 DQN vs 启发式专家 ──双方 transition──> Replay B ──┘

Learner ──定期同步最新参数──> Rollout A / Rollout B
Learner ──定期保存 checkpoint──> 历史 checkpoint / 异步评测
```

### Rollout A：主 self-play 数据源

- 黑棋由当前 DQN 控制，是主训练对象；
- 白棋是陪练，不持有独立的 optimizer，也不会被单独更新；
- 白棋可以使用随机策略或近期历史 checkpoint；
- 每隔配置的训练步数保存一次 checkpoint；
- 每隔配置的训练步数，从近期 checkpoint 中随机选择一个加载给白棋；
- Rollout A 只向 Replay A 写入黑棋 decision-interval transition；
- Learner 的更新节奏由 Rollout A 收到的黑棋步数决定。

这种设计使白棋策略随训练历史变化，避免黑棋长期只适应某个固定对手。

### Rollout B：启发式专家旁路

Rollout B 由 `DQN/heuristic_sidecar.py` 实现。它让当前 DQN 与 `DQN/heuristic_agent.py` 中的规则专家对弈，并把双方经验转换到当前玩家视角后写入 Replay B。

Rollout B 的目的包括：

- 为 Replay Buffer 提供质量高于纯随机对局的动作和局面；
- 让 DQN 学习专家的有效走法；
- 让 DQN 学习如何应对专家；
- 减少早期低质量 self-play 经验占据 Replay Buffer 的影响。

Replay B 预热后，Learner 会按配置比例混合 Replay A 和 Replay B，但二者最终都使用 DQN 的 TD target 和 Q-learning loss 训练。

### Checkpoint 与异步评测

训练过程中会定期保存 checkpoint。保存后，`DQN/evaluator.py` 可以在独立 CPU 进程中让该 checkpoint 对战固定启发式专家，并返回胜、负、和棋及耗时指标。

`DQN/evaluator.py` 是训练内部使用的模块，不是命令行入口。离线评测应使用：

```bash
python DQN/eval.py --help
```

## BC 行为克隆流水线

`BC/` 是独立于 DQN 的多轮监督模仿学习方案。当前质量优先流程直接面向 9×9，通过 bootstrap 和最多六轮 DAgger 持续收集当前策略实际访问的带外局面。

```text
棋盘状态 ──> BC 策略网络 ──> 专家会选择的落子位置
```

主要流程为：

```text
启发式专家自博弈
       ↓
生成 (board, player, expert_action) 数据集
       ↓
BC 策略网络使用 top-1 + top-k 加权交叉熵训练
       ↓
生成独立的 BC checkpoint
```

BC 的主要特点：

- 专家只在数据生成和评测阶段运行；
- GPU 训练只读取已经落盘的数据；
- 同一个策略网络学习黑棋和白棋；
- 使用动作分类和 top-k cross-entropy，而不是 TD loss；
- 拥有独立的数据集、网络、agent、checkpoint 和评测入口。

完整 BC 流水线可使用：

```bash
bash BC/run_pipeline.sh RUN_NAME
```

具体参数和数据目录约定参见 `BC/README.md`。

## afterBC：从 BC_BEST 继续训练白棋

`afterBC/` 的目标不是从零学习五子棋，而是在成功的 BC + DAgger 模型上继续做强化
学习。冻结的 BC_BEST 始终执黑先手；待训练白棋也从 BC_BEST 初始化，但只有白棋
Dueling DQN 会更新。对白棋来说，一次 transition 覆盖“白棋落子 + 黑棋自动回应”
的完整决策间隔。

训练采用 Ape-X 风格的 actor-gather-learner 架构：多个 CPU actor 并行运行环境，
gather 按 actor id 严格等份收集 transition，GPU Learner 使用 prioritized replay、
3-step return 和 Double-DQN target 进行更新。训练和评测中的冻结黑棋使用同一套受控
随机协议：前6次黑棋落子在 BC_BEST top-4 中按温度1.5采样，立即胜/必须堵时以及
后续回合使用 greedy。

当前 `white_apex_v1/latest.pt` 的独立评测结果表明该路线取得了明显效果：

- 对纯 greedy BC_BEST 的 deterministic 对局，白棋获胜；
- 对受控随机 BC_BEST 的128局评测中，白棋123胜、1负、4和；
- 白棋胜率为96.1%，计和棋后的得分率为97.7%。

上述结果针对冻结 BC_BEST 及项目定义的受控随机协议，不表示对任意五子棋对手都有
相同胜率。完整架构、训练参数、TensorBoard 指标和恢复语义参见 `afterBC/README.md`。

启动训练：

```bash
conda run -n mygames python -m afterBC.train --run-name EXPERIMENT_NAME
```

独立评测：

```bash
conda run -n mygames python -m afterBC.evaluate \
  --checkpoint afterBC/runs/EXPERIMENT_NAME/latest.pt \
  --stochastic-games 128
```

## DQN、Rollout B 与 BC 的关系

Rollout B 和 BC 都使用“启发式专家”，但接入方式完全不同。

| 项目 | DQN Rollout B | BC |
|---|---|---|
| 专家实现 | `DQN/heuristic_agent.py` | `BC/heuristic_agent.py` |
| 专家运行时间 | DQN 在线训练期间 | 离线数据生成和评测期间 |
| 数据形式 | `(state, action, reward, next_state, done)` | `(state, expert_action)` |
| 数据去向 | Replay B | 磁盘数据集 |
| 学习目标 | TD target / Q-learning | 专家动作分类 / cross-entropy |
| 网络 | `DuelingGomokuQNet` | `GomokuPolicyNet` |
| checkpoint | DQN 格式 | BC 格式 |

DQN 当前没有导入任何 `BC` 模块：

```text
DQN/train.py
  └── DQN/heuristic_sidecar.py
        ├── DQN/heuristic_agent.py
        ├── DQN/agent.py
        └── DQN/player_transitions.py
```

因此：

- Rollout B 确实参考了专家经验；
- Rollout B 不读取 BC 数据集；
- Rollout B 不加载 BC checkpoint；
- DQN 和 BC 的网络参数不能直接互换；
- BC 目前不是 DQN 的预训练阶段，也不是 DQN 白棋陪练的一部分。

这里描述的是 `DQN/` 主线本身；`afterBC/` 已经单独实现了 BC policy trunk/head 到
Dueling DQN trunk/advantage head 的兼容迁移，不改变 `DQN/` 的边界。

## 公共五子棋环境

三套训练方案共享 `gomoku_env`：

```python
from gomoku_env import GomokuEnv, make_vector_env
```

使用项目专属包名 `gomoku_env`，而不是通用顶层包名 `env`，是为了避免同一个 Python/Conda 环境中其他项目也存在 `env` 包时发生错误导入。例如，平级 bicycle 项目以 editable 模式安装后，其 `src/env` 也可能出现在 `sys.path` 中；唯一包名可以从根本上避免两个项目串包。

Gymnasium 环境注册入口为：

```text
gomoku_env.gomoku_env:GomokuEnv
```

## 常用命令

首先激活运行环境：

```bash
conda activate mygames
```

查看 DQN 训练参数：

```bash
python DQN/train.py --help
```

启动一个 DQN 训练任务：

```bash
python DQN/train.py --run-name EXPERIMENT_NAME
```

查看 DQN 离线评测参数：

```bash
python DQN/eval.py --help
```

启动 DQN 人机对弈：

```bash
python DQN/play.py --help
```

查看 BC 训练、评测和对弈参数：

```bash
python BC/train.py --help
python BC/eval.py --help
python BC/play.py --help
```

查看 afterBC 训练和独立评测参数：

```bash
python -m afterBC.train --help
python -m afterBC.evaluate --help
```

运行完整测试：

```bash
python -m pytest -q
```

## 总结

- `DQN/` 是当前在线强化学习主线；
- Rollout A 负责黑棋与历史/随机白棋的主要 self-play 采样；
- Rollout B 直接使用 DQN 自己的启发式专家生成辅助强化学习经验；
- `BC/` 是独立的离线专家模仿流水线，不被 DQN 导入；
- `afterBC/` 从 BC_BEST 出发，已训练出相对冻结黑棋具有显著优势的白棋 DDDQN checkpoint；
- 三套算法只共享 `gomoku_env/` 棋盘环境，训练代码仍分别闭环；
- DQN、BC 与 afterBC 的学习目标、训练流程和 checkpoint 格式各自独立。
