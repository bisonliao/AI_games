# MPE 动作语义与 MADDPG PyTorch 策略输出

## 两种环境后端

`train_torch.py` 现在明确区分两套 MPE：

- `--env-backend legacy`（默认）：固定到 OpenAI MPE `6ed7cac`，用于与原 TF 代码对照。
- `--env-backend pettingzoo`：使用固定版本的 PettingZoo，用于现代 API 实验。其 reward 和部分动力学已与旧 MPE 不同，回报不能和 TF/Legacy 曲线直接比较。

`simple` 是 1 agent 的调试任务；`simple_spread` 才是 3 agent 的协作导航任务，二者不再互相映射。

## 官方策略的动作形式

`--policy-mode official` 是默认值。Actor 对每个离散动作分支输出 logits，并使用可微的 Gumbel-Softmax 样本：

```text
observation -> MLP -> branch logits -> Gumbel-Softmax -> soft action vector
```

可移动 agent 的 physical 分支在 2D MPE 中为 5 维；可通信 agent 还有一个大小为 `world.dim_c` 的 communication 分支。多分支动作是各自独立 softmax 后拼接，因此 Actor/Critic 使用的动作维度是各分支大小之和，不是乘积。

旧 MPE 虽然声明 `Discrete(5)`，但官方 MADDPG 实际传入 5 维 soft vector，环境直接计算差值：

```text
u_x = action[1] - action[2]
u_y = action[3] - action[4]
```

这个 vector path 的索引方向和旧 MPE 的整数 discrete path 恰好相反。PettingZoo 修正了 continuous path 的索引顺序，所以 PettingZoo 适配器会交换 physical 分支的 `1/2` 和 `3/4`，使同一个策略向量在两个后端产生同方向的力。

## Gaussian 消融模式

`--policy-mode gaussian` 仅在 `pettingzoo` 后端可用，用来复现和对照原 PyTorch 版的“可学习对角高斯 + tanh”实现。Actor 对 d 维 Box 动作输出 `2*d` 个参数（mean 和 logstd）。该模式不是官方 TF MADDPG 的动作语义，不应用它与官方曲线做算法级复现。

## Checkpoint 隔离

checkpoint 路径为：

```text
<root>/maddpg/<env-backend>/<policy-mode>/<scenario>/state_steps_<steps>.pt
```

文件保存算法名、online/target 网络、两个 Adam 状态、训练 step、完成 episode 数和动作规格元数据。恢复时严格校验算法、后端、场景、策略模式和规格。已有的 v2/v3 checkpoint 及其不带 `maddpg/` 的旧目录布局仍可兼容加载；更早的格式不兼容，也不会部分静默加载。
