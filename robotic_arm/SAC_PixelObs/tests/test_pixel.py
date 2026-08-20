import numpy as np
import torch
from types import SimpleNamespace
from queue import Queue

from stable_baselines3.common.logger import configure
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from SAC_PixelObs.async_eval import _put_latest
from SAC_PixelObs.callbacks import (
    PixelTaskMetricsCallback,
    RolloutActionEntropyCallback,
    VisualHealthCallback,
    write_tensorboard_scalars,
)
from SAC_PixelObs.env import PixelTaskEnv
from SAC_PixelObs.policy import MultiViewCombinedExtractor
from SAC_PixelObs.train import parse_args


def test_three_view_observation_is_seeded_and_has_expected_shape():
    env = PixelTaskEnv(task="pick_place", image_size=32, frame_stack=3, action_repeat=1)
    try:
        first, _ = env.reset(seed=17)
        second, _ = env.reset(seed=17)
        assert first["image"].shape == (27, 32, 32)
        assert first["image"].dtype == np.uint8
        assert first["proprio"].shape == (26,)
        np.testing.assert_array_equal(first["image"], second["image"])
        np.testing.assert_allclose(first["proprio"], second["proprio"])
    finally:
        env.close()


def test_cube_and_goal_remain_visible_at_default_96_resolution():
    """Guard against camera projections that collapse tabletop objects."""

    env = PixelTaskEnv(task="pick_place", image_size=96, frame_stack=1, action_repeat=1)
    try:
        for seed in range(10):
            observation, _ = env.reset(seed=seed)
            views = observation["image"].reshape(3, 3, 96, 96)
            red_counts = []
            green_counts = []
            for view in views:
                red, green, blue = view.astype(np.int16)
                red_counts.append(
                    int(((red > green + 35) & (red > blue + 35) & (red > 100)).sum())
                )
                green_counts.append(
                    int(((green > red + 30) & (green > blue + 30) & (green > 80)).sum())
                )
            # Each marker may be thin or occluded in a side projection, but
            # the three-camera observation must contain a usable projection.
            assert max(red_counts) >= 4
            assert max(green_counts) >= 8
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
            visual_head_version=2,
        )
        observation, _ = env.reset(seed=3)
        batch = {key: torch.as_tensor(value[None]) for key, value in observation.items()}
        output = extractor(batch)
        assert output.shape == (1, extractor.features_dim)
        assert torch.isfinite(output).all()
        extractor.train()
        first = extractor(batch)
        second = extractor(batch)
        assert first.shape == second.shape
        torch.testing.assert_close(first, second)
        visual = extractor.encode_visual(batch["image"])
        assert torch.count_nonzero(visual).item() > 0
        assert extractor.visual_head[1].elementwise_affine is False
        assert isinstance(extractor.visual_head[2], torch.nn.LeakyReLU)
        visual.sum().backward()
        assert extractor.visual_head[0].weight.grad is not None
        assert torch.count_nonzero(extractor.visual_head[0].weight.grad).item() > 0
    finally:
        env.close()


def test_sb3_visual_sac_smoke():
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CallbackList
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
                "features_extractor_kwargs": {
                    "n_views": 3,
                    "frame_stack": 2,
                    "visual_head_version": 2,
                },
                "share_features_extractor": False,
            },
            device="cpu",
            verbose=0,
        )
        model.learn(
            total_timesteps=6,
            callback=CallbackList(
                [
                    VisualHealthCallback(check_freq=2),
                    RolloutActionEntropyCallback(log_freq=2),
                ]
            ),
        )
        assert (
            model.actor.features_extractor
            is not model.critic.features_extractor
        )
    finally:
        env.close()


def test_reach_callback_omits_pick_place_metrics():
    class RecordingLogger:
        def __init__(self):
            self.names = []

        def record_mean(self, name, value):
            del value
            self.names.append(name)

    callback = PixelTaskMetricsCallback(task="reach")
    logger = RecordingLogger()
    callback.model = SimpleNamespace(logger=logger)
    callback.locals = {
        "infos": [
            {
                "episode": {
                    "success": True,
                    "failure": False,
                    "time_limit_reached": False,
                    "ever_grasped": False,
                    "ever_lifted": False,
                    "stage_index": 0,
                    "failure_reason": "",
                }
            }
        ]
    }

    assert callback._on_step()
    assert set(logger.names) == {
        "task/success_rate",
        "task/failure_rate",
        "task/truncation_rate",
    }


def test_rollout_action_entropy_uses_sampled_action_log_probability():
    class RecordingLogger:
        output_formats = []

        def __init__(self):
            self.values = {}

        def record(self, name, value):
            self.values[name] = value

    class Distribution:
        def proba_distribution(self, mean_actions, log_std, **kwargs):
            del mean_actions, log_std, kwargs
            return self

        def log_prob(self, actions):
            assert actions.shape == (2, 3)
            return torch.tensor([-1.0, -3.0])

    class Actor:
        action_dist = Distribution()

        def get_action_dist_params(self, observation):
            batch_size = observation["image"].shape[0]
            zeros = torch.zeros((batch_size, 3))
            return zeros, zeros, {}

    logger = RecordingLogger()
    callback = RolloutActionEntropyCallback(log_freq=2)
    fake_env = SimpleNamespace(num_envs=2)
    callback.model = SimpleNamespace(
        logger=logger,
        learning_starts=0,
        _last_obs={
            "image": np.zeros((2, 9, 4, 4), dtype=np.uint8),
            "proprio": np.zeros((2, 26), dtype=np.float32),
        },
        device=torch.device("cpu"),
        actor=Actor(),
        get_env=lambda: fake_env,
    )
    callback.locals = {"buffer_actions": np.zeros((2, 3), dtype=np.float32)}
    callback.num_timesteps = 2

    assert callback._on_step()
    assert logger.values["rollout/action_entropy"] == 2.0


def test_latest_eval_request_replaces_stale_pending_request():
    requests = Queue(maxsize=1)
    requests.put_nowait({"step": 100})
    dropped = _put_latest(requests, {"step": 200})
    assert dropped == 1
    assert requests.get_nowait()["step"] == 200


def test_explicit_tensorboard_step_is_preserved(tmp_path):
    logger = configure(str(tmp_path), ["tensorboard"])
    try:
        assert write_tensorboard_scalars(
            logger,
            {"eval/success_rate": 0.75},
            step=123_456,
        )
    finally:
        logger.close()
    events = EventAccumulator(str(tmp_path))
    events.Reload()
    scalar = events.Scalars("eval/success_rate")
    assert len(scalar) == 1
    assert scalar[0].step == 123_456
    assert scalar[0].value == 0.75


def test_training_defaults_match_updates_to_collected_transitions(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train", "--task", "reach"])
    args = parse_args()
    assert args.gradient_steps == -1
    assert args.share_features_extractor is False
    assert args.eval_freq == 100_000
