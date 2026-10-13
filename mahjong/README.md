# 红中麻将 RL / IL 实验

本项目以湖南红中麻将为背景，训练 agent 提高自摸胡牌率，并通过人类对局收集专家数据，供后续模仿学习（IL）使用。
牌种为万、饼、条和红中，红中可作万能牌；不吃、不放炮，支持碰、杠，自摸后自动结束本局。
不考虑打漂、抓龙等计分规则。当前对手是三家启发式出牌机器人，其碰杠决策及完整抢杠胡流程尚未实现。

训练使用 `306` 维向量观测和 `141` 维合法动作掩码。奖励结合终局结果与潜势差；潜势考虑残子结构和听牌时可胡的剩余牌数。
环境也支持图像观测，但当前两套 DQN 实现均使用向量输入。接口和奖励细节见 [MahjongEnv/README.md](MahjongEnv/README.md)。

## 目录与关键代码

| 路径 | 用途 |
| --- | --- |
| `MahjongEnv/` | 两套训练共用的环境。`env.py` 推进牌局；`rules.py` 判断胡牌、计算残子与听牌张数；`reward.py` 组合奖励；`bots.py` 实现对手启发式出牌；`renderer.py` 绘制牌桌；`dataset.py` 保存专家数据。 |
| `DDDQN/` | 手写 PyTorch Dueling Double DQN。`network.py` 定义网络，`agent.py` 选择动作并更新参数，`replay.py` 管理经验回放，`subproc_vec_env.py` 并行采样。详见 [训练说明](DDDQN/README.md)。 |
| `DQN/` | 基于 Stable-Baselines3 的实现，加入合法动作掩码、Dueling 网络和 Double DQN 目标。`policy.py` 定义策略，`algorithm.py` 实现更新，`callbacks.py` 记录指标和触发评测。详见 [训练说明](DQN/README.md)。 |
| `tests/` | 环境规则、奖励、观测、数据存储及训练相关测试。 |
| `assets/` | GUI 所需资源，包括中文字体。 |
| `expert_data/` | 人类对局采集的逐局 `.npz` 专家数据。 |
| `checkpoints/` | 训练模型：DDDQN 使用 `.pt`，SB3 使用 `.zip`，二者不能直接互换。 |
| `tb_logs/` | TensorBoard 训练与评测日志。 |
| `play.py` | 人类 GUI 对局与专家数据采集入口。 |
| `HANDOFF.md` / `AGENTS.md` | 项目交接记录与开发背景约定。 |

## 常用入口

以下命令均在项目根目录运行。核心依赖为 NumPy、Gymnasium、PyTorch、TensorBoard；SB3 版本还需要 Stable-Baselines3，GUI 对局需要 Pygame。

### 人类对局、采集专家数据

```bash
python play.py --episodes 3 --output expert_data
```

通过鼠标或键盘出牌，逐局保存专家数据；自摸或其他终局发生后停留在结果画面，按 Enter / Space 进入下一局。
`--episodes 0` 表示持续对局。`--headless` 会用随机合法动作运行，无 GUI，也不是训练模型演示。

### 训练模型

```bash
# 手写 DDDQN
python -m DDDQN.train --device auto

# SB3 版本
python -m DQN.train --device auto
```

默认参数分别位于 `DDDQN/config.py`、`DQN/config.py`。各入口支持 `--help` 查看可覆盖参数。

### 在标准输出查看模型如何出牌

```bash
python -m DDDQN.evaluate --checkpoint checkpoints/best.pt --episodes 3 --num-envs 1
```

`DDDQN/evaluate.py` 默认逐步打印观测手牌、当前阶段和模型动作，无需 GUI。手牌按“红中、万、饼、条”排序，新摸到的牌单独放在最后。例如：

```text
手牌：红中 1万 2万 3万 …  | 摸到：9万
模型动作：打出 2饼
```

响应牌友弃牌时还会显示对方打出的牌；每局结束打印结果和累计奖励。加 `--quiet` 可只查看评测汇总：

```bash
python -m DDDQN.evaluate --checkpoint checkpoints/best.pt --episodes 500 --quiet
```

`DQN/evaluate.py` 用于 SB3 模型，目前输出评测汇总，不逐步打印手牌：

```bash
python -m DQN.evaluate --model checkpoints/best/best_model.zip --episodes 500
```

### 随机基线、日志和测试

```bash
python -m DQN.random_baseline --episodes 500
tensorboard --logdir tb_logs
python -m unittest discover -s tests -v
```

当前 `eval/self_draw_rate` 与 `eval/win_rate` 都是我方获胜局数占评测局数的比例，分母包括流局。
`return` 是每局所有奖励之和的平均值，包含 shaping；它可以为负，不能直接当作胜率。
评估策略时，应同时查看多局成功率和具体出牌记录。
