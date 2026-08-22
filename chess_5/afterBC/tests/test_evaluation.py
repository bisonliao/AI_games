from __future__ import annotations

from pathlib import Path

import torch

from afterBC.common import DEFAULT_BC_CHECKPOINT
from afterBC.evaluator import evaluate_checkpoint
from afterBC.learner import DQNLearner


def test_initial_checkpoint_evaluates_as_white_without_illegal_moves(tmp_path: Path) -> None:
    checkpoint = tmp_path / "initial.pt"
    learner = DQNLearner(
        DEFAULT_BC_CHECKPOINT, device="cpu", batch_size=2,
        replay_size=8, min_replay_size=2, seed=4,
    )
    learner.global_step = 500_000
    learner.save_checkpoint(checkpoint, {"test": True})
    restored = DQNLearner(
        DEFAULT_BC_CHECKPOINT, device="cpu", batch_size=2,
        replay_size=8, min_replay_size=2, seed=5,
    )
    restored.load_checkpoint(checkpoint)
    assert restored.global_step == 500_000
    result = evaluate_checkpoint(
        checkpoint, DEFAULT_BC_CHECKPOINT, stochastic_games=2, seed=12,
    )
    assert result["deterministic"]["winner"] == "draw"
    stochastic = result["stochastic"]
    assert stochastic["white_wins"] + stochastic["white_losses"] + stochastic["draws"] == 2
    assert 0.0 <= stochastic["white_score_rate"] <= 1.0
    assert result["success"] == (stochastic["white_score_rate"] > 0.5)
    assert result["protocol"]["black_training"] == result["protocol"]["black_stochastic"]
