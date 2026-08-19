import numpy as np
import torch

from SAC_PixelObs.env import PixelTaskEnv
from SAC_PixelObs.policy import MultiViewCombinedExtractor


def test_three_view_observation_is_seeded_and_has_expected_shape():
    env = PixelTaskEnv(task="pick_place", image_size=32, frame_stack=3, action_repeat=1)
    try:
        first, _ = env.reset(seed=17)
        second, _ = env.reset(seed=17)
        assert first["image"].shape == (27, 32, 32)
        assert first["image"].dtype == np.uint8
        assert first["proprio"].shape == (20,)
        np.testing.assert_array_equal(first["image"], second["image"])
        np.testing.assert_allclose(first["proprio"], second["proprio"])
    finally:
        env.close()


def test_multiview_extractor_forward():
    env = PixelTaskEnv(task="reach", image_size=32, frame_stack=2, action_repeat=1)
    try:
        extractor = MultiViewCombinedExtractor(
            env.observation_space,
            n_views=3,
            frame_stack=2,
            visual_feature_dim=16,
            proprio_feature_dim=8,
        )
        observation, _ = env.reset(seed=3)
        batch = {key: torch.as_tensor(value[None]) for key, value in observation.items()}
        output = extractor(batch)
        assert output.shape == (1, extractor.features_dim)
        assert torch.isfinite(output).all()
        extractor.eval()
        deterministic = extractor(batch)
        extractor.train()
        augmented = extractor(batch)
        assert deterministic.shape == augmented.shape
        assert not torch.equal(deterministic, augmented)
    finally:
        env.close()


def test_sb3_visual_sac_smoke():
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv([
        lambda: PixelTaskEnv(task="reach", image_size=32, frame_stack=2, action_repeat=1)
    ])
    try:
        model = SAC(
            "MultiInputPolicy",
            env,
            learning_starts=1,
            batch_size=2,
            buffer_size=32,
            train_freq=1,
            gradient_steps=1,
            policy_kwargs={
                "features_extractor_class": MultiViewCombinedExtractor,
                "features_extractor_kwargs": {"n_views": 3, "frame_stack": 2},
            },
            device="cpu",
            verbose=0,
        )
        model.learn(total_timesteps=6)
    finally:
        env.close()
