# RaceCar PPO 项目约定

本项目训练 PPO agent 驾驶 PyBullet 赛车从 S 型赛道起点到达终点。修改环境、
采样器或训练代码前，请先阅读 `env/RaceCarEnv.py` 顶部的环境契约。

## 回合长度与任务目标

- 人工键盘实测约 **1400 simulation steps** 可以从起点成功到达终点。
- 训练和正式评估的默认回合上限是 **1500 steps**。这是任务难度和路径效率的
  一部分，不应为了提高成功率而无说明地放宽。
- 调试、课程学习或消融实验可以显式传入不同的 `max_steps`，但配置和实验指标中
  必须记录该值；最终结果仍须在 1500-step 上限下评估。
- 必须区分 `terminated` 与 `truncated`：撞墙/到达终点是 terminal；达到 1500 步是
  time-limit truncation。计算 return/GAE 时，time-limit 和 rollout chunk 边界应从
  critic value bootstrap，真正 terminal 才把后继 value 置零。

## 长序列 PPO 训练要求

1500 步属于长回合。实现或修改训练流程时应明确处理以下事项：

- 使用多个 actor 和多个独立环境提高有效样本吞吐；PyBullet 环境必须在 worker
  进程内创建，不能把已连接的 client fork 给子进程。
- rollout 可以按固定长度分块，不必等待完整回合，但每个环境的 episode 状态必须
  连续，并在分块边界正确 bootstrap。使用 GAE、advantage 标准化，并记录实际采用的
  `gamma`、`gae_lambda`、rollout length 和 batch size。
- 默认 `gamma=0.99` 的有效信用跨度相对 1400 步很短。需要结合当前稠密奖励进行调参，
  并至少比较更高 gamma（如 0.995/0.999）对成功率和 critic 稳定性的影响；不要只根据
  总 reward 判断效果。
- 监控并汇报 success rate、成功回合步数、collision rate、time-limit rate、episode
  return、回合结束时到终点的平均距离、value loss/explained variance、entropy 和
  approximate KL。成功率、成功步数和结束距离是核心指标。
- 当前非终止奖励 `1 / (1 + distance_to_goal)` 始终为正，理论上可能鼓励靠近终点后拖延，
  与“高效到达”目标冲突。设计训练配置时应检查这种 reward hacking；若修改奖励，优先
  考虑距离进展量、适度 step cost 和明确的终点奖励，并用实验验证，而不是静默改变任务。
- 观测、reward 或 advantage 的归一化统计应由 learner 聚合或以明确方式同步；不要在
  actor 之间共享非进程安全的 TensorBoard writer。

## 并发环境

- `RaceCarEnv` 的每个实例拥有独立 `physicsClientId`。
- `env/actor_env.py::DirectRaceCarVectorEnv` 是训练默认后端：actor 在自身进程内直接同步
  驱动多个 client，避免逐步 IPC。
- actor--learner 之间的 rollout/权重队列由训练器负责；不要重新引入逐 environment
  step 的子进程队列，除非有明确 benchmark 证明有收益。
- worker 返回的数据应保持 NumPy/CPU 形式。learner 持有 master model 和 optimizer；
  actor 可持有仅用于推理的 policy 副本，并通过控制/权重队列在明确的 rollout 边界更新。

## 当前 PPO 实现

- 入口是 `python -m ppo.train`；默认 4 个 actor、每个 actor 2 个环境、512 步 rollout，
  actor 默认直接持有 8 个 PyBullet client，learner 使用 CUDA。14 核机器上可从这些
  默认值开始，再根据实际 CPU 利用率调整 `--num-actors` 和 `--envs-per-actor`。
- 离散动作定义固定为 `0=straight`、`1=left`、`2=right`，在 `ppo/model.py` 映射到环境
  的连续转向值 `[0, -1, +1]`。
- `ppo/model.py` 的训练适配层使用距离进展奖励、轻微步耗以及终点/碰撞终奖，避免原环境
  的全正距离奖励诱导拖延；TensorBoard 的 `window/*` 指标每 200000 环境步清零一次。
- 性能监控只保留 `perf/mean_rollout_collect_seconds` 和
  `queue/mean_learner_rollout_wait_seconds` 两个关键时间指标，另结合
  `perf/steps_per_second` 判断 actor 采样是否成为瓶颈，避免高频队列探测反过来影响性能。
- 训练前先用 `--device cpu --total-timesteps ...` 做小规模 smoke test；正式训练使用默认
  `--device cuda`。默认日志和 checkpoint 使用相同的
  `racecar_ppo_YYYYMMDD_HHMMSS_mmm_pidN` 唯一 run 名，支持并发实验；显式传入
  `--log-dir`/`--checkpoint-dir` 时则完全采用指定路径。
- 使用 `python -m ppo.play checkpoints/<run>/checkpoint_final.pt` 可在 PyBullet GUI 中
  回放策略；默认确定性选取最大概率动作，按 `q` 退出，回合结束后自动循环。可以用
  `--stochastic` 进行采样式回放。
- PPO 调度是 absolute `global_steps` 的纯函数，以保证从 0 训练和任意 checkpoint 恢复
  完全一致：0～5M 使用 learning rate `3e-4`、clip `0.2`、最多 8 epochs、target KL
  `0.01`；5M 后切入精调，learning rate `5e-5 -> 1e-5`、entropy coefficient
  `0.01 -> 0.001` 线性退火，并使用 clip `0.1`、最多 4 epochs、target KL `0.003`。
- greedy 评测只在每次 1000000 步周期 checkpoint（以及恢复起点/最终点）运行；默认用
  32 个固定、互异的 evaluation seeds 和 8 个环境，记录 `eval/greedy_*` 聚合指标。
  固定评测 seed 集保证 checkpoint 间可比，同时初始位置/朝向扰动保证回合不重复。
  最优策略保存为 `checkpoint_best_greedy.pt`，不要假设 final 一定最佳。
- 训练默认对起始 x/y 各加入 `±0.05m`、heading 加入 `±0.03rad` 的 seeded uniform
  扰动。actor seed 由 base seed、resume global step 和 actor id 共同确定；actor 内每个
  环境使用不同 seed。策略的 Categorical 采样由各 actor 独立 torch RNG 驱动。
- 周期 checkpoint 的纯数字 step 序号按 `total_timesteps` 位数左补零，保证文件名字典序
  等于 step 数值顺序；`checkpoint_best_greedy.pt` 和 `checkpoint_final.pt` 保持语义名称。
