# PyBullet 跳一跳 + 单步 TD3

这是一个类似微信“跳一跳”的最小强化学习工程。两个等高平台出现在固定区域内，
agent 观察归一化平台距离并输出一个连续蓄力动作。一次 `env.step(action)` 会在
PyBullet 内部完成整段抛物线飞行，因此每个回合恰好包含一个决策。

这个任务严格来说是 contextual bandit。默认算法是针对该结构简化的 TD3：critic
直接拟合即时奖励，不进行下一状态 bootstrap，也不创建 target network。双 critic、
延迟 actor 更新和探索噪声仍然保留。

源码按职责拆成两个同级包：`src/env` 保存 PyBullet 环境、并行环境和人工操作工具，
`src/td3` 保存 TD3 agent、actor-learner、训练 CLI 和模型 GUI 评估工具。后续 PPO 等
算法可以继续作为新的同级包加入，而无需依赖 TD3 包。

## 代码文件说明

| 文件 | 一句话说明 |
| --- | --- |
| `src/env/jump_env.py` | 定义平台、角色、弹道、奖励、终止条件以及 GUI/RGB 渲染的 Gymnasium 环境。 |
| `src/env/vector_env.py` | 创建基于多进程 `AsyncVectorEnv` 的同步批量环境，并解析 `SAME_STEP` 终局信息。 |
| `src/env/evaluation.py` | 提供通用策略、解析 oracle 和随机策略的无 GUI 评测函数。 |
| `src/env/benchmark.py` | 测量不同 PyBullet worker 数量下的环境采样吞吐量。 |
| `src/env/gui_keys.py` | 统一不同 PyBullet 版本的空格、Escape 和退出按键兼容逻辑。 |
| `src/env/play.py` | 打开 PyBullet GUI，将人工按住空格的真实时长转换为跳跃动作。 |
| `src/env/__init__.py` | 导出环境配置、环境类和向量环境工厂。 |
| `src/td3/agent.py` | 定义 actor、双 critic 以及不使用 bootstrap 的单步 TD3 更新逻辑。 |
| `src/td3/replay.py` | 实现由 learner 独占的固定容量 NumPy replay buffer。 |
| `src/td3/trainer.py` | 实现 CPU actor、多进程环境、队列通信、GPU/CPU learner、TensorBoard 和 checkpoint。 |
| `src/td3/cli.py` | 提供环境检查、基准、训练和无 GUI checkpoint 评测命令。 |
| `src/td3/eval.py` | 查找并加载 TD3 checkpoint，以可调速度在 GUI 中演示 agent。 |
| `src/td3/__main__.py` | 允许通过 `python -m td3` 进入 TD3 CLI。 |
| `src/td3/__init__.py` | 导出 TD3 agent、训练配置和训练入口。 |
| `tests/test_env.py` | 检查 Gymnasium 契约、弹道单调性、reward、渲染和蓄力动作映射。 |
| `tests/test_vector_env.py` | 检查多进程向量环境、worker seed 和 `SAME_STEP` 终局信息。 |
| `tests/test_td3.py` | 检查 replay buffer、单步 critic target 和设备解析。 |
| `tests/test_trainer.py` | 运行 actor-learner smoke test，并验证时间戳目录及 TensorBoard tags。 |
| `tests/test_gui_tools.py` | 无需打开窗口即可检查 GUI 参数、按键兼容和 checkpoint 名称查找。 |
| `pyproject.toml` | 声明依赖、Python 包发现、命令行入口和 pytest 配置。 |

## 安装与快速检查

项目配置的解释器为 `/home/bison/.conda/envs/mygames/bin/python`。依赖已经存在时可直接：

```bash
PYTHONPATH=src /home/bison/.conda/envs/mygames/bin/python -m td3.cli check
```

也可以安装为 editable package：

```bash
/home/bison/.conda/envs/mygames/bin/pip install -e '.[dev]'
td3-cli check
```

解析策略和随机策略基准：

```bash
PYTHONPATH=src /home/bison/.conda/envs/mygames/bin/python -m td3.cli baseline oracle
PYTHONPATH=src /home/bison/.conda/envs/mygames/bin/python -m td3.cli baseline random
```

向量环境吞吐量比较：

```bash
PYTHONPATH=src /home/bison/.conda/envs/mygames/bin/python -m td3.cli benchmark \
  --env-counts 1,4,8 --transitions 2000
```

## 训练

默认结构是 2 个非 daemon CPU actor，每个 actor 管理 4 个使用 `spawn` 启动的
PyBullet `AsyncVectorEnv` worker。actor 批量发送 transition，learner 独占 replay
buffer、优化器和指定设备。

训练数据流为：actor 批量推理并驱动各自的环境 worker，将 transition chunk 放入有界
队列；learner 消费 chunk、写入 replay buffer 并按 update-to-data 比例更新网络，再通过
每个 actor 独立的 latest-only 参数队列发布新权重。主进程同时负责队列健康统计、周期
评测、TensorBoard 和安全停机。

```bash
PYTHONPATH=src /home/bison/.conda/envs/mygames/bin/python -m td3.cli train \
  --device cuda --actors 2 --envs-per-actor 4 \
  --transitions 100000 --run-name td3-main
```

每次训练会建立独立目录，例如
`runs/20260730-081530-pid12345-td3-main/`。目录名包含本地年月日时分秒和进程
PID；即使同一秒并行启动多个训练也不会冲突。checkpoint 默认保存在该目录中。

启动 TensorBoard：

```bash
tensorboard --logdir runs --port 6006
```

TensorBoard 包含采样/评测成功率、reward、落点误差和 learner loss，同时包含队列健康
指标：transition queue 长度/占用率、各 actor 阻塞时间和 queue-full 次数、actor 丢弃
transition、learner 等待数据的均值/最大值/长等待次数/超时次数，以及训练停止时 learner
丢弃的预取数据量。

其中 `queue/actor_dropped_transitions_total` 表示 actor 已采样但未能送入队列的数据（通常
只会在停止训练时出现）；`queue/learner_discarded_prefetch_transitions_total` 表示已经进入
队列、但 learner 达到训练预算后不再消费的预取数据。`actors/actor_N/*` 可定位具体发生
阻塞或丢弃的 actor。

当前机器上的 PyTorch 若无法访问 CUDA，请先用 CPU 训练或 smoke test：

```bash
PYTHONPATH=src /home/bison/.conda/envs/mygames/bin/python -m td3.cli train \
  --device cpu --actors 1 --envs-per-actor 2 \
  --transitions 10000 --run-name cpu-smoke
```

评测 checkpoint 时始终关闭探索噪声：

```bash
PYTHONPATH=src /home/bison/.conda/envs/mygames/bin/python -m td3.cli evaluate \
  runs/20260730-081530-pid12345-td3-main/checkpoint.pt --episodes 1000
```

## GUI 人工操作

打开 PyBullet GUI 后，按下空格开始蓄力，松开空格跳跃；按 Esc、Q 或 Ctrl+C 退出：

```bash
PYTHONPATH=src /home/bison/.conda/envs/mygames/bin/python -m env.play \
  --speed 1.0
```

`--speed 0.5` 以半速播放物理飞行，`--episodes 10` 可限定游戏局数。人工动作使用真实
空格按住时长，超过环境最大蓄力时间的部分会被裁剪。

## GUI 模型演示

输入 checkpoint 路径，让 TD3 agent 自动游戏并渲染每次跳跃：

```bash
PYTHONPATH=src /home/bison/.conda/envs/mygames/bin/python -m td3.eval \
  runs/20260730-081530-pid12345-td3-main/checkpoint.pt \
  --device cpu --speed 0.5 --episodes 20
```

也可以只输入 run 名称或 checkpoint 文件名，程序会在 `--run-root` 下递归查找并选取
最近修改的匹配模型：

```bash
PYTHONPATH=src /home/bison/.conda/envs/mygames/bin/python -m td3.eval \
  td3-main --run-root runs --speed 0.5
```

`--speed` 控制物理播放倍速，`--pre-jump-delay` 控制起跳前停留时间，
`--result-delay` 控制成功/失败画面的保留时间，`--no-show-charge` 可跳过模拟蓄力等待。

## 环境语义

- observation：`Box(0, 1, (1,))`，即目标距离除以最大采样距离。
- action：`Box(-1, 1, (1,))`，线性映射为 0–1 秒蓄力。
- `v_xy = charge_scale * hold_time`，竖直初速度固定，方向自动指向 B。
- 默认 reward 是中心落点误差的稠密项加成功奖励；`info["is_success"]` 是独立二值指标。
- 正常成功和失败均为 `terminated=True`；只有内部安全超时为 `truncated=True`。
- 向量环境使用 `SAME_STEP` autoreset。返回 observation 已属于下一回合，终止回合
  的诊断信息位于 `infos["final_info"]`。
