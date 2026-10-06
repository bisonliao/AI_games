"""Validated NPZ episodes and synchronized three-view Torch observations."""

from collections import OrderedDict
import hashlib
import json
from pathlib import Path

from .runtime import VIEWS
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torch.nn import functional as functional


def validate_episode(path):
    try:
        with np.load(path, allow_pickle=False) as episode:
            required = {"format_version", "image", "proprio", "action", "reward", "stage",
                        "terminated", "truncated", "success", "episode_metadata"}
            if not required.issubset(episode.files):
                raise ValueError(f"Missing fields: {required - set(episode.files)}")
            size = len(episode["action"])
            if size < 1 or int(episode["format_version"]) != 2:
                raise ValueError("Empty episode or unsupported format")
            shapes = dict(image=(size, 18, 96, 96), proprio=(size, 26), action=(size, 4),
                          reward=(size,), stage=(size,), terminated=(size,), truncated=(size,))
            for key, shape in shapes.items():
                value = episode[key]
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"Invalid {key}: {value.shape}")
            if episode["image"].dtype != np.uint8:
                raise ValueError("Images must be uint8")
            if not np.isin(episode["stage"], np.arange(5)).all():
                raise ValueError("Stage must be in [0, 4]")
            if np.any(np.diff(episode["stage"].astype(int)) < 0):
                raise ValueError("Stages must not go backwards")
            if np.abs(episode["action"]).max() > 1.00001:
                raise ValueError("Actions outside [-1, 1]")
            if not np.allclose(episode["proprio"][1:, 22:26], episode["action"][:-1], atol=1e-5):
                raise ValueError("Proprio previous-action fields do not match pre-action observation alignment")
            if episode["success"].shape != () or episode["success"].dtype != np.bool_:
                raise ValueError("Success must be a boolean scalar")
            for key in ["terminated", "truncated"]:
                if episode[key].dtype != np.bool_:
                    raise ValueError(f"{key} must be boolean")
            if (episode["terminated"][:-1] | episode["truncated"][:-1]).any():
                raise ValueError("Episode contains samples after a terminal transition")
            images = episode["image"]
            if not np.array_equal(images[0, :9], images[0, 9:]):
                raise ValueError("Initial history must repeat the reset image")
            if not np.array_equal(images[1:, :9], images[:-1, 9:]):
                raise ValueError("Image history is not oldest-first or contains discontinuities")
            json.loads(str(episode["episode_metadata"].item()))
            digest = hashlib.sha256()
            for key in ["image", "proprio", "action", "reward", "stage", "terminated", "truncated", "episode_metadata"]:
                digest.update(episode[key].tobytes())
            return dict(name=Path(path).name, success=bool(episode["success"]), sha256=digest.hexdigest(),
                        length=size, stages=episode["stage"].tolist())
    except Exception as error:
        raise ValueError(f"Invalid episode {path}: {error}") from error


def make_manifest(directory, seed=42):
    paths = sorted(Path(directory).glob("episode_*.npz"))
    if not paths:
        raise ValueError(f"No episodes in {directory}")
    records = [validate_episode(path) for path in paths]
    successes = [record["name"] for record in records if record["success"]]
    if len(successes) < 2:
        raise ValueError("Need at least two successful episodes")
    np.random.default_rng(seed).shuffle(successes)
    count = max(1, int(len(successes) * 0.9))
    manifest = dict(seed=seed, train=sorted(successes[:count]), validation=sorted(successes[count:]),
                    diagnostic=[record["name"] for record in records if not record["success"]],
                    records=records)
    manifest["fingerprint"] = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    return manifest


def action_statistics(directory, names):
    actions = []
    for name in names:
        with np.load(Path(directory) / name) as episode:
            actions.append(episode["action"])
    actions = np.concatenate(actions).astype(np.float64)
    return dict(mean=actions.mean(0).tolist(), std=np.maximum(actions.std(0), 1e-6).tolist())


def normalize_action(action, statistics):
    return (action - np.asarray(statistics["mean"], dtype=np.float32)) / np.asarray(statistics["std"], dtype=np.float32)


def denormalize_action(action, statistics):
    return action * np.asarray(statistics["std"], dtype=np.float32) + np.asarray(statistics["mean"], dtype=np.float32)


def observation_tensors(image, proprio):
    # 适配自定义任务：18 通道拆成旧/当前两帧，每帧 xy/xz/yz 三个 RGB 视图，不重复堆叠历史。
    image = torch.as_tensor(np.asarray(image).copy()).reshape(2, 3, 3, 96, 96)
    observation = {}
    for index, view in enumerate(VIEWS):
        # 适配自定义任务：resize 到预训练视觉输入尺寸，保留 0..255，由 Octo 编码器内部归一化。
        observation[f"image_{view}"] = functional.interpolate(
            image[:, index].float(), size=(256, 256), mode="bilinear", align_corners=False)
    observation["proprio"] = torch.as_tensor(np.asarray(proprio).copy(), dtype=torch.float32)
    observation["timestep_pad_mask"] = torch.ones(2, dtype=torch.bool)
    observation["pad_mask_dict"] = {key: torch.ones(2, dtype=torch.bool) for key in [*(f"image_{view}" for view in VIEWS), "proprio"]}
    return observation


class ExpertDataset(Dataset):
    def __init__(self, directory, manifest, split, statistics):
        self.directory = Path(directory)
        self.statistics = statistics
        records = {record["name"]: record for record in manifest["records"]}
        self.indices = [(name, step, stage) for name in manifest[split]
                        for step, stage in enumerate(records[name]["stages"])]
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        name, step, stage = self.indices[index]
        if name not in self.cache:
            with np.load(self.directory / name) as episode:
                self.cache[name] = {key: episode[key] for key in ["image", "proprio", "action"]}
            if len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(name)
        episode = self.cache[name]
        # 适配自定义任务：状态历史不跨 episode；标签为当前观测对应的专家动作，用训练集统计归一化。
        previous = max(0, step - 1)
        proprio = episode["proprio"][[previous, step]]
        action = normalize_action(episode["action"][step], self.statistics)
        return dict(observation=observation_tensors(episode["image"][step], proprio),
                    action=torch.from_numpy(action), stage=torch.tensor(stage))

    def balanced_sampler(self, count, generator=None):
        stages = np.array([entry[2] for entry in self.indices])
        counts = np.bincount(stages, minlength=5)
        if np.any(counts == 0):
            raise ValueError("Training split must contain all five stages")
        return WeightedRandomSampler(1.0 / counts[stages], count, replacement=True, generator=generator)
