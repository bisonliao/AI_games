from pathlib import Path

import torch

from DQN.evaluator import evaluate_checkpoint
from DQN.model import DuelingDQN


def test_evaluation_truncates_long_episodes(tmp_path: Path):
    model = DuelingDQN()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"online": model.state_dict()}, checkpoint)
    result = evaluate_checkpoint(checkpoint, episodes=2, max_steps=1, seed=3, device="cpu")
    assert result["truncated_episodes"] == 2
    assert result["mean_length"] == 1
    assert result["env_mode"] == "placement"


def test_evaluation_detects_placement_checkpoint(tmp_path: Path):
    model = DuelingDQN()
    checkpoint = tmp_path / "placement.pt"
    torch.save({"online": model.state_dict(), "config": {"env_mode": "placement"}}, checkpoint)
    result = evaluate_checkpoint(checkpoint, episodes=1, max_steps=2, seed=3, device="cpu")
    assert result["env_mode"] == "placement"
    assert result["mean_length"] == 2


def test_evaluation_uses_configured_gamma_from_checkpoint(tmp_path: Path):
    model = DuelingDQN()
    checkpoint = tmp_path / "gamma.pt"
    torch.save(
        {
            "online": model.state_dict(),
            "config": {"gamma": 0.97},
        },
        checkpoint,
    )

    result = evaluate_checkpoint(
        checkpoint, episodes=1, max_steps=1, seed=3, device="cpu"
    )

    assert result["gamma"] == 0.97


def test_evaluation_ignores_legacy_stability_controls(tmp_path: Path):
    model = DuelingDQN()
    checkpoint = tmp_path / "legacy-controls.pt"
    torch.save(
        {
            "online": model.state_dict(),
            "config": {"gamma": 0.99},
            "stability_controls": {"gamma": 0.97},
        },
        checkpoint,
    )

    result = evaluate_checkpoint(checkpoint, episodes=1, max_steps=1, seed=3, device="cpu")

    assert result["gamma"] == 0.99
