from dataclasses import asdict, replace

import torch

from _R2D2.config import R2D2Config
from _R2D2.network import RecurrentDuelingQNetwork
from _R2D2.play import evaluate_checkpoint, load_model_and_config


def test_play_loads_recurrent_checkpoint_and_runs_greedy_episodes(tmp_path) -> None:
    base = R2D2Config()
    config = replace(
        base,
        hidden_size=32,
        actor_env=replace(
            base.actor_env,
            num_envs=1,
            noop_max=0,
            repeat_action_probability=0.0,
        ),
        evaluation_max_episode_steps=5,
    )
    model = RecurrentDuelingQNetwork(
        config.observation_shape, config.action_count, config.hidden_size
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "online_state_dict": model.state_dict(),
            "config": asdict(config),
        },
        checkpoint_path,
    )

    loaded_model, loaded_config = load_model_and_config(checkpoint_path)
    assert loaded_model.hidden_size == 32
    assert loaded_config.evaluation_max_episode_steps == 5
    result = evaluate_checkpoint(checkpoint_path, episodes=2, gui=False)
    assert result.episode_lengths == [5, 5]
    assert result.capped_episodes == 2
    assert len(result.episode_returns) == len(result.episode_raw_scores) == 2
