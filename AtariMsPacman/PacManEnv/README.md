# PacManEnv

为 Actor-Learner 训练提供进程并行的 Ms. Pac-Man Gymnasium 环境。

## 环境约定

- 观测为 `uint8[N, 4, 84, 84]` 灰度帧，时间顺序从最旧到最新。
- 使用 9 个动作的 ALE minimal action set。
- 只有 Game Over 返回 `terminated=True`，丢命通过 `info["life_lost"]` 报告。
- ALE 内部时间上限和 Gymnasium `TimeLimit` 均关闭，`truncated` 必须始终为
  `False`。
- 默认返回原始游戏奖励；训练奖励塑形不会修改 `info["raw_reward"]` 和
  `info["raw_score"]`。

## 奖励计算

ALE 在每个模拟器帧返回该帧产生的游戏分数增量。`AtariPreprocessing` 会将一次
Actor 决策中重复执行的所有帧奖励相加。设本次决策实际执行了 `K` 个模拟器帧，
则：

```text
raw_reward_t = sum(ALE_frame_reward_1 ... ALE_frame_reward_K)
```

通常 `K == frame_skip`；如果在中途发生 Game Over，则可能少于
`frame_skip`。当前默认 `frame_skip=4`，所以 `info["raw_reward"]` 表示一次
Actor 决策所跨越的最多 4 个模拟器帧的原始得分增量，而不是单个模拟器帧的奖励。

环境实际通过 `step()` 返回给训练代码的 `reward` 按以下顺序计算：

```text
shaped_reward_t = raw_reward_t - step_cost
clip_training_reward=False: reward_t = shaped_reward_t
clip_training_reward=True:  reward_t = clip(shaped_reward_t, -1, 1)
```

默认配置为 `step_cost=0.0`、`clip_training_reward=False`，因此默认情况下：

```text
reward_t == info["raw_reward"]
```

环境没有额外添加丢命惩罚或 Game Over 惩罚。训练用的 step cost 和奖励裁剪只
影响返回的 `reward`，不会修改 `info["raw_reward"]` 或
`info["raw_score"]`。

### `raw_reward` 与 `raw_score`

| 字段 | 含义 | reset 后的值 |
| --- | --- | ---: |
| `info["raw_reward"]` | 当前这一次 `step()` 产生的原始游戏得分增量 | `0.0` |
| `info["raw_score"]` | 当前完整游戏中历次 `raw_reward` 的累计值 | `0.0` |

两者关系为：

```text
raw_score_t = raw_score_(t-1) + raw_reward_t
```

例如三次决策分别得到 `10、0、50` 分，则第三次返回的
`info["raw_reward"] == 50`，`info["raw_score"] == 60`。Game Over 的
terminal transition 仍会先计入本次奖励；之后显式 masked reset 才把该环境的
`raw_score` 清零。

`raw_score` 遵循项目约定的 `sum(raw_reward)`，不是直接读取游戏画面中的 HUD
分数。随机 no-op 在 `reset()` 内部执行，Gymnasium 的 reset 接口不会返回这些
no-op 的奖励，因此它们不会计入 `raw_score`。这保证了它严格等于 Actor 实际收到
的所有原始 step 奖励之和。

向量环境中这些字段都是按环境排列的数组，例如
`infos["raw_reward"][i]` 和 `infos["raw_score"][i]` 分别属于第 `i` 个 ALE
实例。

## 安装

```bash
python -m pip install -r PacManEnv/requirements.txt
```

## 使用

`spawn` 多进程要求创建向量环境的代码位于 `if __name__ == "__main__"` 中：

```python
import numpy as np

from PacManEnv import MsPacmanEnvConfig, make_vector_env


def main():
    env = make_vector_env(MsPacmanEnvConfig(num_envs=8))
    try:
        observations, infos = env.reset(seed=42)
        actions = np.zeros(env.num_envs, dtype=np.int64)
        next_observations, rewards, terminated, truncated, infos = env.step(actions)

        # 必须先保存 terminal transition，再重置结束的行。
        if terminated.any():
            next_observations, reset_infos = env.reset(
                options={"reset_mask": terminated.astype(np.bool_)}
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
```

完整采集顺序见 `PacManEnv/examples/actor_loop.py`。

## RAM 展示诊断

`include_ram_metrics=True` 会在 `info` 中加入已确认的 `dots_eaten`。当前不从
RAM 推断 `level` 或 `mazes_cleared`，这些值不能参与观测、奖励或终止逻辑。

以下命令记录候选 RAM 变化和附近截图，用于以后校准过关事件：

```bash
python -m PacManEnv.ram_diagnostics --output /tmp/ms_pacman_ram
```

## 验证

```bash
python -m pytest PacManEnv/tests
```

`frame_skip=4` 是当前默认实验参数，并已通过下述首张迷宫典型路口实验。仍可在
配置中切换到 `1` 或 `2`，接口和其余语义保持不变。

## 观测与 frame skip 实测结论

以下结论基于当前安装的 `ale-py 0.12.1`、打包的 Ms. Pac-Man ROM
`87e79cd41ce136fd4f72cc6e2c161bee`、标准 `mode=0` 和 `difficulty=0`。

### 可食幽灵在灰度图中仍可区分

实验使用 `frameskip=1` 并关闭 sticky action，通过固定动作序列让 Ms. Pac-Man
吃到左下角能量豆，分别读取同一 ALE 实例的 RGB 和原生灰度帧。该 ROM 中可食
幽灵实际变为蓝色，而不是绿色。捕获到的精确调色板及 ALE 灰度值如下：

| 幽灵状态 | RGB | ALE 灰度值 |
| --- | --- | ---: |
| 普通橙色 | `(180, 122, 48)` | 131 |
| 普通青色 | `(84, 184, 153)` | 151 |
| 普通粉色 | `(198, 89, 179)` | 132 |
| 普通红色 | `(200, 72, 72)` | 110 |
| 可食状态 | `(66, 114, 194)` | 109 |

前三种普通幽灵变为可食状态时亮度变化明显。红色幽灵主体只相差 1 个灰度级，
但可食状态还会改变精灵身体、眼睛和后期闪烁图案；对齐后的局部灰度块并不相同。
经过 `84x84` 缩放后差异仍然存在，连续 4 帧又保留了颜色切换和闪烁的时序信息。

因此当前环境采用灰度观测 `uint8[N, 4, 84, 84]`。它保留了可食状态信号，同时
把观测、进程通信和 replay 帧存储降低到 RGB 方案的三分之一。真实 ROM 回归测试
见 `PacManEnv/tests/test_observation_research.py`。

### `frame_skip=4` 不会错过当前典型路口

实验同样使用 `frameskip=1`、关闭 sticky action，并用
`cloneSystemState/restoreSystemState` 从相同接近状态逐帧延迟转向输入。路口通过
时间按角色中心穿过路口中心前后各 8 像素计算；有效转向窗口是从进入该区域到最晚
仍能在目标路口成功转向的模拟器帧数。

| 进入方向和转向 | 通过路口所需帧数 | 有效转向窗口 |
| --- | ---: | ---: |
| 向左行驶，在 `(53,99)` 向上转 | 28 | 14 |
| 向右行驶，在 `(97,99)` 向上转 | 28 | 16 |
| 向上行驶，在 `(53,51)` 向右转 | 20 | 10 |

最小窗口为 10 帧，大于项目约定的 8 帧验收标准。对每个路口额外检查了
`frame_skip=4` 的四种决策边界相位 `0/1/2/3`，全部成功转向。因此在当前 ROM、
标准模式和首张迷宫的典型路口上，保留 `frame_skip=4`。

这里关闭 sticky action 是为了隔离 frame skip 的几何可控性。正式配置中的
`repeat_action_probability=0.25` 仍可能随机沿用上一动作，这是评测协议引入的独立
随机性，不能归因于 frame skip。当前数据尚未覆盖后续关卡；如果后续发现高速关卡
转向退化，应使用同一回归方法增加对应关卡样本，而不是直接根据训练分数猜测原因。
