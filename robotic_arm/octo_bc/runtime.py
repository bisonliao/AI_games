"""Shared PyTorch runtime; checkpoint files must come from trusted sources."""

import os

os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import random
from pathlib import Path
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec

import numpy as np
import torch


class ForbiddenBackendLoader(Loader):
    def create_module(self, spec):
        raise RuntimeError(f"Torch-only verification forbids importing {spec.name}")

    def exec_module(self, module):
        raise RuntimeError(f"Torch-only verification forbids importing {module.__name__}")


class TorchOnlyImports(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"jax", "jaxlib", "flax", "tensorflow"}:
            return ModuleSpec(fullname, ForbiddenBackendLoader())
        return None


if os.environ.get("OCTO_TORCH_ONLY") == "1":
    if any(name in sys.modules for name in ["jax", "flax", "tensorflow"]):
        raise RuntimeError("Forbidden backend was imported before Torch-only verification")
    sys.meta_path.insert(0, TorchOnlyImports())

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "expert_showcase_v4"
CHECKPOINT_DIR = ROOT / "octo" / "checkpoints"
LOG_DIR = ROOT / "tb_logs" / "octo"
INSTRUCTION = "Pick up the red block and place it at the green target."
VIEWS = ("xy", "xz", "yz")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(value, device):
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return value


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, model, metadata, **training):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(format_version=1, config=model.config,
                   example_batch=to_device(model.example_batch, "cpu"),
                   state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
                   metadata=metadata, training=training)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    from octo.model.octo_model_pt import OctoModelPt

    # 读取本地 PyTorch 检查点；base.pt 的预训练参数来自 convert.py 的 HF/JAX 转换流程。
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["format_version"] != 1:
        raise ValueError("Unsupported checkpoint version")
    # 适配自定义任务：按转换时保存的三视角、proprio 和 4 维动作头配置重建结构；此时尚未载入保存的权重。
    model = OctoModelPt.from_config(payload["config"], payload["example_batch"])
    # 真正把检查点中的参数赋给模型；strict=True 要求参数名称和形状完全匹配。
    model.load_state_dict(payload["state_dict"], strict=True)
    for name, parameter in model.named_parameters():
        if ".hf_model." in name:
            # 适配自定义任务：保留预训练 T5 的语言能力，冻结参数以减少微调开销。
            parameter.requires_grad_(False)
    return model.to(device), payload


def frozen_text_eval(model):
    for tokenizer in model.module.octo_transformer.task_tokenizers.values():
        if getattr(tokenizer, "hf_model", None) is not None:
            tokenizer.hf_model.eval()
