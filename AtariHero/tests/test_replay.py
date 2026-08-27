from __future__ import annotations

import torch

from curri_DQN.messages import PackedTransition, SuccessfulEpisode
from curri_DQN.replay import ReplayBuffer


def _transition(marker: int, *, stage: int = 1) -> PackedTransition:
    return PackedTransition(
        observations=bytes([marker]),
        action=marker,
        reward=float(marker),
        terminated=False,
        stage=stage,
        actor_id=0,
    )


def test_single_replay_contains_success_and_failure_transitions_together() -> None:
    replay = ReplayBuffer(capacity=4, seed=7)
    failure = _transition(1, stage=4)
    success_terminal = PackedTransition(
        observations=b"success",
        action=2,
        reward=99.998,
        terminated=True,
        stage=4,
        actor_id=0,
    )
    replay.add(failure)
    replay.add(success_terminal)

    assert len(replay) == 2
    assert replay.stage_sizes() == {4: 2}
    assert {item.observations for item in replay.sample(100)} == {
        failure.observations,
        success_terminal.observations,
    }


def test_replay_state_round_trip_preserves_sampling_rng() -> None:
    first = ReplayBuffer(capacity=4, seed=7)
    for marker in range(4):
        first.add(_transition(marker))
    state = first.state_dict()

    second = ReplayBuffer(capacity=4, seed=99)
    second.load_state_dict(state)
    assert second.sample(16) == first.sample(16)


def test_legacy_success_episode_checkpoint_can_be_unpickled(tmp_path) -> None:
    legacy = SuccessfulEpisode(
        reset_stage=4,
        task_id="legacy-task",
        checkpoint_id="legacy-checkpoint",
        transitions=(_transition(1, stage=4),),
    )
    path = tmp_path / "legacy.pt"
    torch.save(
        {"online_model": {}, "success_replay": {"episodes": [legacy]}},
        path,
    )

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["success_replay"]["episodes"][0].checkpoint_id == (
        "legacy-checkpoint"
    )
