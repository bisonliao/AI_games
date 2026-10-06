"""One-time JAX checkpoint bridge. All subsequent commands use Torch files."""

import argparse
import copy
import json
import os

from .runtime import CHECKPOINT_DIR, DATA_DIR, INSTRUCTION, VIEWS, save_checkpoint, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", default="hf://rail-berkeley/octo-small-1.5")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--output", type=str, default=str(CHECKPOINT_DIR / "base.pt"))
    args = parser.parse_args()
    os.environ["USE_FLAX"] = "1"
    os.environ["JAX_PLATFORMS"] = "cpu"

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from octo.model.octo_model_pt import OctoModelPt
    from octo.utils.spec import ModuleSpec
    from octo.model.components.tokenizers_pt import LowdimObsTokenizerPt
    from octo.model.components.action_heads_pt import L1ActionHeadPt
    from .data import make_manifest, action_statistics, ExpertDataset

    seed_everything(42)
    manifest = make_manifest(args.data_dir)
    statistics = action_statistics(args.data_dir, manifest["train"])
    # HF 原始预训练权重的加载入口：读取下载缓存/本地 JAX 检查点并转换为 OctoModelPt 参数。
    loaded = OctoModelPt.load_pretrained_from_jax(args.pretrained)
    base = loaded["octo_model"]
    missing = list(loaded["missing_keys"])
    unexpected_missing = [name for name in missing if "hf_model" not in name]
    if unexpected_missing:
        raise RuntimeError(f"Pretrained weights missing outside T5: {unexpected_missing}")
    config = copy.deepcopy(base.config)
    config["text_processor"] = None
    model_config = config["model"]
    # 适配自定义任务：将 primary 相机配置复制成 xy/xz/yz 三个独立视觉分支，后面再复制其预训练参数。
    primary = model_config["observation_tokenizers"]["primary"]
    observations = {}
    for view in VIEWS:
        specification = copy.deepcopy(primary)
        specification["kwargs"]["obs_stack_keys"] = [f"image_{view}"]
        specification["kwargs"]["task_stack_keys"] = [f"image_{view}"]
        observations[view] = specification
    # 适配自定义任务：为环境的 26 维自身状态新增分箱 tokenizer；对应可学习投影从头训练。
    observations["proprio"] = ModuleSpec.create(LowdimObsTokenizerPt, obs_keys=["proprio"], discretize=True,
                                                n_bins=256, bin_type="normal")
    model_config["observation_tokenizers"] = observations
    language = base.module.octo_transformer.task_tokenizers["language"]
    encoder_config = language.hf_model.config.to_dict()
    model_config["task_tokenizers"]["language"]["kwargs"]["encoder_config"] = encoder_config
    model_config["task_tokenizers"]["language"]["kwargs"]["finetune_encoder"] = False
    embedding_size = model_config["token_embedding_size"]
    # 适配自定义任务：替换原动作头，预测单步 [dx, dy, dz, gripper]，使用新的 4 维 L1 回归头。
    model_config["heads"] = dict(action=ModuleSpec.create(L1ActionHeadPt, input_dim=embedding_size,
                                        action_dim=4, action_horizon=1, readout_key="readout_action"))
    # 适配自定义任务：匹配三个 256×256 视图、26 个状态 token 和两帧历史，更新 token 数和时间窗口。
    model_config["num_tokens_dict"] = {view: 256 for view in VIEWS}
    model_config["num_tokens_dict"].update(proprio=26, language=16, action=1)
    model_config["max_horizon"] = 2
    tokenizer = AutoTokenizer.from_pretrained(language.encoder)
    # 适配自定义任务：将固定 pick-place 指令编码后保存，训练/评估复用相同文本条件。
    text = tokenizer(INSTRUCTION, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
    task = dict(language_instruction=dict(text), pad_mask_dict=dict(language_instruction=torch.ones(1, dtype=torch.bool)))
    dataset = ExpertDataset(args.data_dir, manifest, "train", statistics)
    example = next(iter(DataLoader(dataset, batch_size=1)))
    example["task"] = copy.deepcopy(task)
    # 适配自定义任务：全零 goal RGB 占位保留原 stem 的六通道形状，不提供目标图像或坐标真值。
    for view in VIEWS:
        example["task"][f"image_{view}"] = torch.zeros(1, 3, 256, 256)
        example["task"]["pad_mask_dict"][f"image_{view}"] = torch.zeros(1, dtype=torch.bool)
    model = OctoModelPt.from_config(config, example)
    source_state = base.state_dict()
    target_state = model.state_dict()
    copied, new = {}, []
    # 适配自定义任务：复用 Transformer/T5；三个视图分别复制 primary 的编码器、投影和位置参数。
    for name, value in target_state.items():
        source_name = name
        for view in VIEWS:
            source_name = source_name.replace(f"observation_tokenizers.{view}.", "observation_tokenizers.primary.")
            source_name = source_name.replace(f"obs_{view}_projection", "obs_primary_projection")
            source_name = source_name.replace(f"obs_{view}_pos_embedding", "obs_primary_pos_embedding")
        if ".heads." in name or "proprio" in name:
            # 适配自定义任务：新动作头和 proprio 模块保留新建值，不从原任务拷贝不匹配的参数。
            new.append(name)
        elif source_name in source_state and value.shape == source_state[source_name].shape:
            target_state[name] = source_state[source_name]
            copied[name] = source_name
        else:
            raise RuntimeError(f"Unexpected unmatched target weight: {name} {tuple(value.shape)}")
    # 适配自定义任务：装入重映射后的参数；随后保存为 base.pt，供 train.py 直接加载。
    model.load_state_dict(target_state, strict=True)
    text_keys = [name for name in copied if ".hf_model." in name]
    if not text_keys:
        raise RuntimeError("No pretrained T5 weights copied")
    metadata = dict(manifest=manifest, statistics=statistics, task=task, instruction=INSTRUCTION,
                    views=list(VIEWS), history=2, image_size=256, data_dir=str(args.data_dir),
                    pretrained=args.pretrained, weight_report=dict(copied=copied, new=new, source_missing=missing,
                    source_skipped=list(loaded["skipped_keys"]), t5_pretrained=True))
    save_checkpoint(args.output, model, metadata)
    from pathlib import Path
    Path(args.output).with_suffix(".report.json").write_text(json.dumps(metadata["weight_report"], indent=2))
    print(f"Converted {len(copied)} weights, initialized {len(new)} new tensors -> {args.output}")
    print(f"Train episodes: {len(manifest['train'])}; validation: {len(manifest['validation'])}; failures: {len(manifest['diagnostic'])}")


if __name__ == "__main__":
    main()
