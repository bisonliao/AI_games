# PyTorch 独立项目文件与依赖

`scripts/extract_torch_project.sh` 按本文档的边界生成不含 TensorFlow
实现的独立 MADDPG 项目。MPE、Gym/Gymnasium、PettingZoo 和 PyTorch 均通过
Python 包管理器安装，不拷贝进新项目。

## 运行时文件

| 文件 | 用途 |
|---|---|
| `experiments/train_torch.py` | 训练、checkpoint 保存/恢复、独立评测 |
| `experiments/play_torch.py` | 指定 checkpoint 的无 GUI 评测或 GUI 展示 |
| `maddpg/common/distributions_torch.py` | Gumbel-Softmax/多分支离散动作和 Gaussian 兼容模式 |
| `maddpg/common/env_adapters_torch.py` | Legacy MPE/PettingZoo 双后端适配 |
| `maddpg/common/tensorboard_logger_torch.py` | TensorBoard 区间聚合与上报 |
| `maddpg/common/tf_util_torch.py` | device 选择和 checkpoint I/O（不依赖 TensorFlow） |
| `maddpg/trainer/maddpg_torch.py` | MADDPG agent/trainer 主实现 |
| `maddpg/trainer/replay_buffer_torch.py` | replay buffer |
| `maddpg/common/scenario_metrics.py` | `simple_spread`/`simple_adversary` 条件式场景指标插件 |
| `maddpg/__init__.py` | `AgentTrainer` 基类 |

前 8 个是项目中全部的 `*_torch.py`。后两个虽然文件名没有 `_torch`，
但是 PyTorch 运行时的直接依赖，不能遗漏。抽取脚本还会生成三个空的
`__init__.py`，使 `experiments`/`common`/`trainer` 在源码运行和打包时都是普通
Python package。

## 测试文件

- `tests/test_checkpoint_evaluation.py`
- `tests/test_distributions_torch.py`
- `tests/test_env_adapters_torch.py`
- `tests/test_maddpg_torch_parity.py`
- `tests/test_scenario_metrics.py`
- `tests/test_tensorboard_logger_torch.py`

## Python 依赖

| 类别 | 包 | 原因 |
|---|---|---|
| 算法 | `torch`, `numpy` | 网络、优化器、数值计算 |
| 日志 | `tensorboard` | `torch.utils.tensorboard.SummaryWriter` 的运行时依赖 |
| PettingZoo | `gymnasium`, `pettingzoo[mpe]` | 现代 MPE 对照后端；`mpe` extra 带入 `pygame` |
| Legacy MPE | `gym`, `multiagent`, `numpy-stl` | 官方 TF 版所用的归档 MPE 环境 |
| Legacy GUI | `pyglet`, `six` | 归档 MPE `rendering.py` 的直接依赖 |

`gymnasium` 和 `gym` 不是二选一：保留双后端时，PettingZoo 使用前者，
Legacy MPE 使用后者。依赖版本以项目根目录的 `requirements.txt` 为准。
由于 Gym 0.26 不支持 NumPy 2，独立项目将 NumPy 限制在 2.0 以下。

安装脚本中固定的 `multiagent` Git commit 需要系统有 `git` 命令且安装时
能访问 GitHub。Legacy GUI 还需要操作系统提供 OpenGL/GLX 和可用的
X11/WSLg display；这些不是 Python 包。无 GUI 训练和评测不需要 display。

## 不抽取的内容

- TF1 训练入口、trainer、distribution 和 TensorFlow 工具代码。
- `experiments/minimal_rollout.py` 等非 MADDPG PyTorch 主干示例。
- 已安装的 MPE/Gym/PettingZoo/PyTorch 库源码。
- 旧 checkpoint、TensorBoard runs 和训练输出。
