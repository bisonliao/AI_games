"""Transfer an existing SAC/BC actor into the deterministic TD3 actor."""
from __future__ import annotations

from pathlib import Path

import torch
from stable_baselines3.sac.policies import Actor as SACActor

from .config import EnvConfig
from .visual import MultiViewCombinedExtractor


def load_bc_actor(
    path: Path,
    config: EnvConfig,
    device: torch.device | str,
):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format_version") != 2 or "actor_state" not in payload:
        raise ValueError("unsupported BC checkpoint format")
    saved = EnvConfig(**payload["env_config"])
    if saved != config:
        raise ValueError(f"BC env config {saved} does not match requested {config}")
    extractor = MultiViewCombinedExtractor(
        config.observation_space(), frame_stack=config.frame_stack
    ).to(device)
    actor = SACActor(
        config.observation_space(), config.action_space(), [256, 256], extractor,
        extractor.features_dim,
    ).to(device)
    actor.load_state_dict(payload["actor_state"], strict=True)
    actor.set_training_mode(False)
    return actor, payload


def transfer_to_td3(bc_actor, td3_actor, device: torch.device | str) -> float:
    """Copy extractor/MLP/mean layers and verify action equivalence."""
    td3_actor.features_extractor.load_state_dict(
        bc_actor.features_extractor.state_dict(), strict=True
    )
    with torch.no_grad():
        td3_actor.mu[0].weight.copy_(bc_actor.latent_pi[0].weight)
        td3_actor.mu[0].bias.copy_(bc_actor.latent_pi[0].bias)
        td3_actor.mu[2].weight.copy_(bc_actor.latent_pi[2].weight)
        td3_actor.mu[2].bias.copy_(bc_actor.latent_pi[2].bias)
        td3_actor.mu[4].weight.copy_(bc_actor.mu.weight)
        td3_actor.mu[4].bias.copy_(bc_actor.mu.bias)
    td3_actor.set_training_mode(False)
    observation = {
        "image": torch.randint(
            0, 256, (3, *bc_actor.observation_space["image"].shape), device=device,
            dtype=torch.uint8,
        ),
        "proprio": torch.randn(
            3, *bc_actor.observation_space["proprio"].shape, device=device
        ),
    }
    with torch.inference_mode():
        expected = bc_actor(observation, deterministic=True)
        actual = td3_actor(observation)
    error = float((expected - actual).abs().max().cpu())
    if error > 1e-5:
        raise RuntimeError(f"BC to TD3 action transfer mismatch: {error}")
    return error
