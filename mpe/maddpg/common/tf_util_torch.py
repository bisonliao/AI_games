"""
PyTorch 工具：设备管理、模型保存/加载。对应原 tf_util.py 的保存/加载与会话相关逻辑。
"""
import os
import re
import torch


_STEP_CHECKPOINT_PATTERN = re.compile(r"^state_steps_(\d+)\.pt$")


def get_device(use_cuda=True):
    """获取计算设备"""
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_state_path(fname):
    """Resolve a file or select the greatest-step checkpoint in a directory."""

    path = os.path.expanduser(os.fspath(fname))
    if not os.path.isdir(path):
        return path

    step_checkpoints = []
    for entry in os.scandir(path):
        if not entry.is_file():
            continue
        match = _STEP_CHECKPOINT_PATTERN.match(entry.name)
        if match is not None:
            step_checkpoints.append((int(match.group(1)), entry.path))
    if step_checkpoints:
        return max(step_checkpoints, key=lambda item: item[0])[1]
    return os.path.join(path, "state.pt")


def load_state(fname, map_location=None):
    """
    从路径加载 checkpoint。
    fname: 目录或 checkpoint 文件路径；若为目录，优先加载 steps 最大的
           state_steps_<steps>.pt，并兼容回退到 state.pt。
    返回加载的 state dict（若有 'trainers' 等则返回完整 checkpoint）。
    """
    if map_location is None:
        map_location = torch.device("cpu")
    path = resolve_state_path(fname)
    if not os.path.isfile(path):
        raise FileNotFoundError("No checkpoint at {}".format(path))
    return torch.load(path, map_location=map_location)


def save_state(fname, obj):
    """
    保存 checkpoint 到路径。
    fname: 目录或文件路径；若为目录则保存为 fname/state.pt。
    """
    path = fname
    if os.path.isdir(path) or not path.endswith(".pt"):
        path = os.path.join(path.rstrip("/"), "state.pt")
    # 确保最终保存文件的父目录存在（例如传入 ../chkpt/simple_push 时需要创建 simple_push）
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(obj, path)
    return path
