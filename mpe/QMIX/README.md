# QMIX：legacy OpenAI MPE

本目录是独立的 PyTorch QMIX 训练实现，只使用仓库依赖中固定到
`6ed7cac` 的 archived OpenAI MPE，不导入或调用 PettingZoo，也不修改现有
MADDPG 代码。

## 直接训练

在仓库根目录执行：

```bash
python -m QMIX.train
```

默认任务是 `simple_spread`，3 个 agent、每回合 25 步、训练 200000 回合。
CUDA 是默认且不会静默回退到 CPU；指定其他卡用 `--cuda-device N`。仅做本机
CPU 调试时显式添加 `--no-cuda`。

TensorBoard 日志默认写入：

```text
runs/qmix_legacy_official_simple_spread_YYYYMMDD_HHMMSS_pid<PID>/
```

实验名可以用 `--exp-name name` 加入目录名。查看日志：

```bash
tensorboard --logdir runs
```

## 评测与 GUI 回放

`play.py` 直接从指定 `.pt` checkpoint 的 metadata 恢复场景和网络超参数，
不使用当前训练默认值。无 GUI 的确定性评测示例：

```bash
python -m QMIX.play \
  --checkpoint QMIX/checkpoints/qmix/legacy/official/simple_spread/state_steps_4000000.pt \
  --episodes 100 \
  --env-seed 20000
```

第一个 episode 使用 `--env-seed`，后续 episode 依次使用连续 seed。通过更换
seed 区间可以检查对未见初始状态的泛化能力。开启 archived MPE GUI：

```bash
python -m QMIX.play \
  --checkpoint QMIX/checkpoints/qmix/legacy/official/simple_spread/state_steps_4000000.pt \
  --episodes 10 \
  --env-seed 30000 \
  --render \
  --fps 10
```

可用 `--report-json reports/qmix_eval.json` 保存逐回合和汇总指标；在没有 CUDA
的调试环境中显式添加 `--no-cuda`。`--checkpoint` 也可指向直接包含
`state_steps_<N>.pt` 的目录，此时选择步数最大的文件；多个训练共享 checkpoint
目录时建议始终指定确切 `.pt` 文件。

主曲线：

- `reward/episode_reward`：每步对所有 agent 的 reward 求和后再按回合累加，
  口径与现有 MADDPG 的协作任务曲线一致。
- `reward/team_episode_reward`：legacy 环境广播的原始共享团队回报；
  `reward/scaled_team_episode_reward` 是 QMIX TD target 实际使用的缩放回报。
- `loss/td_loss`、`loss/td_error_abs`、`grad/pre_clip_global_norm`、
  `grad/post_clip_global_norm`、`q/*`、`policy/epsilon`：学习状态。前者是
  裁剪前诊断，后者应不超过 `--grad-norm-clip`。
- `task/covered_landmarks`、`task/coverage_ratio`、`task/episode_success`、
  `task/success_rate` 和 `task/success_rate_roll100`：与 MADDPG 完全相同口径的
  `simple_spread` 终局覆盖与业务成功率。
- `eval/*`：保存 checkpoint 时用固定种子执行的 greedy 策略评测。

## 与现有项目保持一致的部分

- 环境固定为 `legacy + official`，默认场景 `simple_spread`。
- `max_episode_len=25`、`num_episodes=200000`、网络隐层 64、seed 0。
- TensorBoard 聚合间隔 10000 environment steps，checkpoint 间隔 10000
  episodes，checkpoint 评测 10 回合且基准 seed 为 10000。
- 动作选择虽然是离散的，但传入旧 MPE 时转换成 hard one-hot vector，走原
  official MADDPG 使用的 vector action path。因此物理分支保持
  `u_x=a[1]-a[2]`、`u_y=a[3]-a[4]` 的语义，而不是旧环境的整数 action path。
- 默认把 25 步时间上限作为有限回合 termination，避免最后一步的 Q bootstrap
  形成正反馈；传入 `--bootstrap-time-limit` 才恢复旧版行为。

## QMIX 专用默认值

这些值针对 legacy MPE 的 reward 尺度做了稳定化，而没有照搬不适合当前
数值范围的配置：RMSProp、learning rate `1e-4`、`gamma=0.95`、Huber TD loss、
32 个完整 episode/batch、5000 episode replay、每 200 episode 硬更新 target、
Double Q，以及 epsilon 从 1.0 在 50000 environment steps 内线性退火到 0.05。
默认 learner reward scale 为 `1 / n_agents`；可用 `--reward-scale` 覆盖。
每个 agent 使用共享的 `FC -> GRUCell -> FC` Q 网络，输入含
局部 observation、上一动作和 agent id；mixer 的 central state 是按固定 agent
顺序拼接的全部局部 observation，hypernetwork 权重取绝对值来满足 QMIX 的
单调性约束。默认 `--max-abs-q 1000`，超过阈值或出现非有限 TD/Q 值会在更新前
中止训练，避免继续覆盖可用 checkpoint。

常用覆盖示例：

```bash
python -m QMIX.train \
  --num-episodes 200000 \
  --batch-size 32 \
  --buffer-size 5000 \
  --tb-log-interval 10000 \
  --save-dir QMIX/checkpoints \
  --exp-name seed0
```

恢复训练：

```bash
python -m QMIX.train --restore --save-dir QMIX/checkpoints
```

checkpoint 在
`<save-dir>/qmix/legacy/official/<scenario>/state_steps_<steps>.pt`。恢复时严格校验
环境、动作规格和网络/优化配置。为避免产生超大 checkpoint，episode replay
不落盘，恢复后会重新 warm up 满一个 batch 再继续更新。新稳定配置保存为 v2；
旧版 v1 checkpoint 仍可读取，但需要显式传入与旧 checkpoint 相同的 `--lr`、
`--gamma` 等配置。旧版不带 `qmix/`
算法目录的 checkpoint 仍可作为恢复时的兼容回退，但所有新保存都进入
`qmix/`，不会和 MADDPG 相互覆盖。

当前入口有意限定为 observation/action 同构且提供 shared reward 的纯合作
legacy MPE 场景；若 reward 不共享会立即报错，避免把非合作任务静默地用错误
的 QMIX target 训练。
