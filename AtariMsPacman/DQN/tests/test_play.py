from dataclasses import asdict, replace

import torch

from DQN.config import DQNConfig
from DQN.network import DuelingQNetwork
from DQN.play import evaluate_checkpoint


def test_play_loads_checkpoint_and_runs_multiple_greedy_episodes(tmp_path) -> None:
    base = DQNConfig()
    config = replace(
        base,
        actor_env=replace(
            base.actor_env,
            num_envs=1,
            noop_max=0,
            repeat_action_probability=0.0,
        ),
        evaluation_max_episode_steps=3,
    )
    model = DuelingQNetwork(config.observation_shape, config.action_count)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()  # Greedy action is always NOOP.
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "online_state_dict": model.state_dict(),
            "config": asdict(config),
        },
        checkpoint_path,
    )

    result = evaluate_checkpoint(
        checkpoint_path,
        episodes=2,
        gui=False,
        fps=30.0,
    )
    assert result.episode_lengths == [3, 3]
    assert result.capped_episodes == 2
    assert len(result.episode_returns) == 2
    assert result.episode_raw_scores == [0.0, 0.0]
