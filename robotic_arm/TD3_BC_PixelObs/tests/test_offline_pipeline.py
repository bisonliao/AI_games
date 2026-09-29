from pathlib import Path

import pytest
import torch

from TD3_BC_PixelObs.algorithm import OfflineTD3BC
from TD3_BC_PixelObs.bc_init import load_bc_actor, transfer_to_td3
from TD3_BC_PixelObs.config import EnvConfig
from TD3_BC_PixelObs.dataset import OfflineDataset
from TD3_BC_PixelObs.augmentation import ConsistentMultiViewAugmentation
from TD3_BC_PixelObs.train import SpaceOnlyEnv
from TD3_BC_PixelObs.visual import MultiViewCombinedExtractor
from stable_baselines3.common.vec_env import DummyVecEnv


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "SAC_BC_PixelObs" / "runs" / "expert_showcase_v3"
BC_MODEL = ROOT / "SAC_BC_PixelObs" / "runs" / "bc_model.pt"


def make_model(config: EnvConfig):
    env = DummyVecEnv([lambda: SpaceOnlyEnv(config)])
    kwargs = {
        "net_arch": [256, 256],
        "features_extractor_class": MultiViewCombinedExtractor,
        "features_extractor_kwargs": {
            "n_views": 3,
            "frame_stack": config.frame_stack,
            "visual_head_version": 2,
        },
        "share_features_extractor": False,
    }
    model = OfflineTD3BC(
        "MultiInputPolicy", env, policy_kwargs=kwargs, batch_size=4,
        learning_rate=3e-4, device="cpu", verbose=0,
    )
    return model, env


def test_success_dataset_and_transition_shapes():
    dataset = OfflineDataset(DATASET, EnvConfig())
    assert len(dataset.episodes) == 395
    assert len(dataset) == 11531
    batch = dataset.sample(4, "cpu")
    assert tuple(batch.observations["image"].shape) == (4, 18, 96, 96)
    assert tuple(batch.observations["proprio"].shape) == (4, 26)
    assert tuple(batch.actions.shape) == (4, 4)


def test_shift_is_consistent_across_views_and_history():
    augmentation = ConsistentMultiViewAugmentation(shift_pixels=2)
    base = torch.arange(2 * 8 * 8, dtype=torch.uint8).reshape(2, 1, 8, 8)
    image = base.repeat(1, 18, 1, 1)
    offsets = torch.tensor([[1, -1], [-2, 2]])
    shifted = augmentation.apply(image, offsets)
    assert shifted.shape == image.shape
    assert augmentation.shift_pixels == 2
    assert torch.equal(shifted[:, 0], shifted[:, 3])
    assert torch.equal(shifted[:, 0], shifted[:, 6])


def test_space_only_environment_rejects_interaction():
    env = SpaceOnlyEnv(EnvConfig())
    with pytest.raises(RuntimeError):
        env.reset()
    with pytest.raises(RuntimeError):
        env.step(env.action_space.sample())


def test_bc_actor_transfer_and_offline_update():
    config = EnvConfig()
    dataset = OfflineDataset(DATASET, config)
    model, env = make_model(config)
    bc_actor, _ = load_bc_actor(BC_MODEL, config, "cpu")
    assert transfer_to_td3(bc_actor, model.policy.actor, "cpu") < 1e-5
    metrics = model.offline_update(dataset, 4)
    assert torch.isfinite(torch.tensor(metrics["critic_loss"]))
    model.offline_update(dataset, 4)
    assert model._n_updates == 2
    env.close()
