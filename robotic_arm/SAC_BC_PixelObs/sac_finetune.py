"""Ape-X style visual SAC fine-tuning.

The learner is the only process that owns SAC, replay, GPU tensors, logging,
and checkpoints. CPU actor processes only step PyBullet environments and send
bounded transition chunks through a multiprocessing queue.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import DictReplayBuffer, DictReplayBufferSamples
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.vec_env import DummyVecEnv

from .common import (
    EnvConfig,
    load_actor,
    make_actor,
    new_tensorboard_run,
    output_path,
    policy_kwargs,
    setup,
)
from .dataset import load_episodes
from .randomized_env import make_randomized_env_factory
from .visual_augmentation import ConsistentMultiViewAugmentation


STOP = "__stop__"


class MixedSB3Replay(DictReplayBuffer):
    """SB3 replay with separate online and expert-prior sampling.

    The parent ``DictReplayBuffer`` remains the online store. Actor processes
    never access this object; the learner calls ``add`` after receiving a
    chunk. Successful expert episodes are kept in ``self.prior`` because
    inserting them into SB3's circular arrays would advance the online cursor
    and make the learner's online transition count ambiguous.

    A prior row has the tuple layout
    ``(obs, action, reward, next_obs, terminated, truncated)``. The observation
    dictionaries contain ``image`` and ``proprio`` arrays. ``sample`` converts
    prior arrays to torch tensors and concatenates them with a normal SB3 batch,
    so SAC receives its standard ``DictReplayBufferSamples`` object.

    ``prior_ratio`` is the desired fraction of prior rows once both sources
    contain enough data. For batch size 128 and ratio 0.25, sampling requests
    32 prior and 96 online rows. During warm-up, missing online rows are
    replaced by prior rows when available. No row is duplicated: if there are
    not enough distinct rows, the learner waits before updating.
    """

    def __init__(
        self,
        *args: Any,
        prior_ratio: float = 0.25,
        augmentation=None,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        # SB3 allocates online observation/action/reward/timeout arrays and
        # initializes the circular cursor used by DictReplayBuffer.add().
        super().__init__(*args, **kwargs)
        if not 0.0 <= prior_ratio <= 1.0:
            raise ValueError("prior_ratio must be in [0, 1]")
        self.prior_ratio = float(prior_ratio)
        # Kept for BC prior-action sampling; replay.sample itself returns raw
        # observations so DrQ can draw independent K/M transforms per update.
        self.augmentation = augmentation
        self.rng = np.random.default_rng(seed)
        self.prior: list[tuple] = []

    def online_size(self) -> int:
        """Return valid online rows, including after circular wraparound."""
        return self.buffer_size if self.full else self.pos

    def can_sample(self, batch_size: int) -> bool:
        """Return whether a no-replacement mixed batch can be constructed."""
        if self.prior_ratio == 0.0:
            return self.online_size() >= batch_size
        return self.online_size() + len(self.prior) >= batch_size

    def add_prior_episode(self, episode: dict) -> None:
        """Convert one saved expert episode into prior transitions."""
        last_index = len(episode["action"]) - 1
        for index in range(last_index + 1):
            next_index = min(index + 1, last_index)
            observation = {
                "image": episode["image"][index],
                "proprio": episode["proprio"][index],
            }
            next_observation = {
                "image": episode["image"][next_index],
                "proprio": episode["proprio"][next_index],
            }
            self.prior.append(
                (
                    observation,
                    episode["action"][index],
                    episode["reward"][index],
                    next_observation,
                    episode["terminated"][index],
                    episode["truncated"][index],
                )
            )

    def _sample_prior(
        self, count: int
    ) -> tuple[dict, dict, np.ndarray, np.ndarray, np.ndarray]:
        """Sample prior rows and stack them in SB3's batch layout."""
        if count <= 0:
            raise ValueError("prior sample count must be positive")
        indices = self.rng.choice(len(self.prior), count, replace=False)
        rows = [self.prior[int(index)] for index in indices]
        observations = {
            key: np.stack([row[0][key] for row in rows]) for key in rows[0][0]
        }
        next_observations = {
            key: np.stack([row[3][key] for row in rows]) for key in rows[0][3]
        }
        actions = np.stack([row[1] for row in rows])
        rewards = np.asarray([row[2] for row in rows], dtype=np.float32).reshape(-1, 1)
        dones = np.asarray(
            [row[4] or row[5] for row in rows], dtype=np.float32
        ).reshape(-1, 1)
        return observations, next_observations, actions, rewards, dones

    def sample(self, batch_size: int, env=None) -> DictReplayBufferSamples:
        """Return one SAC batch containing prior and online transitions.

        ``self.pos`` is a write position, not a size after the circular buffer
        wraps. This method therefore uses ``online_size()``. The learner calls
        ``can_sample`` before entering this method, and the final count check
        protects against a race-free programming error in future changes.
        """
        available_online = self.online_size()
        requested_prior = int(round(batch_size * self.prior_ratio))
        prior_count = min(requested_prior, len(self.prior))
        online_count = min(batch_size - prior_count, available_online)
        prior_count = min(batch_size - online_count, len(self.prior))
        if prior_count + online_count < batch_size:
            raise ValueError("Insufficient distinct prior and online rows")

        if prior_count == 0:
            return super().sample(batch_size, env)

        if online_count == 0:
            prior_obs, prior_next, prior_actions, prior_rewards, prior_dones = (
                self._sample_prior(batch_size)
            )
            batch = DictReplayBufferSamples(
                observations={key: self.to_torch(value) for key, value in prior_obs.items()},
                actions=self.to_torch(prior_actions),
                next_observations={key: self.to_torch(value) for key, value in prior_next.items()},
                dones=self.to_torch(prior_dones),
                rewards=self.to_torch(prior_rewards),
            )
            return batch

        online = super().sample(online_count, env)
        prior_obs, prior_next, prior_actions, prior_rewards, prior_dones = (
            self._sample_prior(prior_count)
        )
        observations = {
            key: torch.cat([online.observations[key], self.to_torch(value)])
            for key, value in prior_obs.items()
        }
        next_observations = {
            key: torch.cat([online.next_observations[key], self.to_torch(value)])
            for key, value in prior_next.items()
        }
        batch = DictReplayBufferSamples(
            observations=observations,
            actions=torch.cat([online.actions, self.to_torch(prior_actions)]),
            next_observations=next_observations,
            dones=torch.cat([online.dones, self.to_torch(prior_dones)]),
            rewards=torch.cat([online.rewards, self.to_torch(prior_rewards)]),
        )
        return batch

    def sample_prior_observations(self, count: int, device) -> tuple[dict, np.ndarray]:
        """Sample prior state/action pairs for BC actor regularization."""
        if count <= 0 or len(self.prior) < count:
            raise ValueError("not enough prior rows for BC regularization")
        indices = self.rng.choice(len(self.prior), count, replace=False)
        rows = [self.prior[int(index)] for index in indices]
        observations = {
            key: torch.as_tensor(np.stack([row[0][key] for row in rows]), device=device)
            for key in rows[0][0]
        }
        actions = np.stack([row[1] for row in rows]).astype(np.float32)
        return observations, actions


class ProtectedSAC(SAC):
    """SAC with a BC warmup and decaying prior-action regularizer."""

    def __init__(
        self,
        *args,
        bc_state: dict[str, torch.Tensor],
        replay: MixedSB3Replay,
        bc_warmup_transitions: int,
        bc_regularization_transitions: int,
        bc_action_coef: float,
        bc_action_coef_final: float,
        drq_k: int = 2,
        drq_m: int = 2,
        augmentation=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.bc_state = {key: value.detach().cpu().clone() for key, value in bc_state.items()}
        self.bc_replay = replay
        self.bc_warmup_transitions = int(bc_warmup_transitions)
        self.bc_regularization_transitions = int(bc_regularization_transitions)
        self.bc_action_coef = float(bc_action_coef)
        self.bc_action_coef_final = float(bc_action_coef_final)
        self.drq_k = int(drq_k)
        self.drq_m = int(drq_m)
        self.drq_augmentation = augmentation
        if self.drq_k < 1 or self.drq_m < 1:
            raise ValueError("drq_k and drq_m must be positive")
        self.last_bc_action_loss = 0.0
        self.last_bc_coef = 0.0
        self.last_actor_frozen = 0.0
        self.last_target_q_std = 0.0
        self.last_current_q_std = 0.0

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """Run SAC updates with independent DrQ target/current transforms."""
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)
        ent_coef_losses, ent_coefs, actor_losses, critic_losses = [], [], [], []
        transition_count = int(self.num_timesteps)
        frozen = transition_count < self.bc_warmup_transitions
        self.last_actor_frozen = float(frozen)

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma
            raw_obs = replay_data.observations
            raw_next_obs = replay_data.next_observations
            augmentation = self.drq_augmentation

            # Entropy temperature update uses one independently augmented state.
            actor_obs = self._augment_observation(raw_obs, augmentation)
            _, log_prob = self.actor.action_log_prob(actor_obs)
            log_prob = log_prob.reshape(-1, 1)
            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = torch.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(
                    self.log_ent_coef * (log_prob + self.target_entropy).detach()
                ).mean()
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
                ent_coef_losses.append(float(ent_coef_loss.detach().cpu()))
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(float(ent_coef.detach().cpu()))

            # K target augmentations: average the soft double-Q targets.
            target_values = []
            with torch.no_grad():
                for _ in range(self.drq_k):
                    next_obs = self._augment_observation(raw_next_obs, augmentation)
                    next_actions, next_log_prob = self.actor.action_log_prob(next_obs)
                    next_q = torch.cat(self.critic_target(next_obs, next_actions), dim=1)
                    next_q = next_q.min(dim=1, keepdim=True).values
                    target_values.append(next_q - ent_coef * next_log_prob.reshape(-1, 1))
                averaged_target = torch.stack(target_values, dim=0).mean(dim=0)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * averaged_target
            self.last_target_q_std = float(torch.stack(target_values, dim=0).std(dim=0, unbiased=False).mean().cpu()) if self.drq_k > 1 else 0.0

            # M current-Q augmentations: accumulate mean critic gradients without
            # retaining all M graphs simultaneously.
            self.critic.optimizer.zero_grad()
            current_values = []
            for _ in range(self.drq_m):
                obs = self._augment_observation(raw_obs, augmentation)
                q_values = self.critic(obs, replay_data.actions)
                current_values.append(torch.cat(q_values, dim=1).detach())
                loss = 0.5 * sum(F.mse_loss(q, target_q_values) for q in q_values)
                critic_losses.append(float(loss.detach().cpu()))
                (loss / self.drq_m).backward()
            self.critic.optimizer.step()
            self.last_current_q_std = float(torch.stack(current_values, dim=0).std(dim=0, unbiased=False).mean().cpu()) if self.drq_m > 1 else 0.0

            if not frozen:
                actor_obs = self._augment_observation(raw_obs, augmentation)
                actions_pi, actor_log_prob = self.actor.action_log_prob(actor_obs)
                q_pi = torch.cat(self.critic(actor_obs, actions_pi), dim=1)
                sac_actor_loss = (ent_coef * actor_log_prob.reshape(-1, 1) - q_pi.min(dim=1, keepdim=True).values).mean()
                progress = min(1.0, max(0.0, (transition_count - self.bc_warmup_transitions) / max(self.bc_regularization_transitions, 1)))
                coefficient = self.bc_action_coef + progress * (self.bc_action_coef_final - self.bc_action_coef)
                prior_obs, prior_actions = self.bc_replay.sample_prior_observations(min(batch_size, len(self.bc_replay.prior)), self.device)
                prior_obs = self._augment_observation(prior_obs, augmentation)
                prior_target = torch.as_tensor(prior_actions, dtype=torch.float32, device=self.device)
                bc_loss = F.mse_loss(self.actor(prior_obs, deterministic=True), prior_target)
                actor_loss = sac_actor_loss + coefficient * bc_loss
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
                self.actor.optimizer.step()
                actor_losses.append(float(sac_actor_loss.detach().cpu()))
                self.last_bc_action_loss = float(bc_loss.detach().cpu())
                self.last_bc_coef = coefficient
            else:
                self.policy.actor.load_state_dict(self.bc_state, strict=True)
                self.last_bc_action_loss = 0.0
                self.last_bc_coef = self.bc_action_coef

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates)
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses) if actor_losses else 0.0)
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if ent_coef_losses:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    @staticmethod
    def _augment_observation(observation: dict, augmentation):
        if augmentation is None:
            return observation
        result = dict(observation)
        result["image"] = augmentation(result["image"])
        return result

    def _excluded_save_params(self) -> list[str]:
        """Do not serialize replay/prior objects through ``bc_replay``.

        SB3 already excludes ``replay_buffer``.  ``ProtectedSAC`` keeps a
        second reference to the same replay in ``bc_replay`` for BC action
        regularization; without excluding it, every model ZIP embeds the full
        multi-gigabyte Python replay graph.
        """
        return [*super()._excluded_save_params(), "bc_replay", "bc_state"]


def _send_latest(control_queue, message: tuple | str) -> None:
    """Replace a pending actor command without blocking the learner."""
    try:
        while True:
            control_queue.get_nowait()
    except queue.Empty:
        pass
    try:
        control_queue.put_nowait(message)
    except queue.Full:
        pass


def _actor_process(
    actor_id: int,
    config: EnvConfig,
    seed: int,
    rollout_chunk_size: int,
    transition_queue,
    control_queue,
    status_queue,
    stop_event,
) -> None:
    """Run one CPU rollout worker and send bounded transition chunks."""
    torch.set_num_threads(1)
    env = None
    try:
        # The learner places the initial (version, actor_state_dict) before
        # starting this process, so actors never need a GPU or a checkpoint file.
        policy_version, state = control_queue.get()
        actor = make_actor(config, device="cpu")
        actor.load_state_dict(_state_dict_from_numpy(state), strict=True)
        # Fine-tuning intentionally uses the same nominal environment as the
        # original PixelObs task.  No object, goal, joint, or camera jitter is
        # applied to actor rollouts; task/* metrics therefore describe this
        # fixed reset distribution.
        env = config.make_env(seed=seed + actor_id, randomize=False)
        observation, _ = env.reset(seed=seed + actor_id)
        rows: list[tuple] = []
        summaries: list[dict] = []

        while not stop_event.is_set():
            try:
                while True:
                    command = control_queue.get_nowait()
                    if command == STOP:
                        stop_event.set()
                        break
                    policy_version, state = command
                    actor.load_state_dict(_state_dict_from_numpy(state), strict=True)
            except queue.Empty:
                pass
            if stop_event.is_set():
                break

            with torch.inference_mode():
                tensors, _ = actor.obs_to_tensor(observation)
                action = actor(tensors, deterministic=False).cpu().numpy()[0]
            next_observation, reward, terminated, truncated, info = env.step(action)
            rows.append(
                (observation, action, reward, next_observation, terminated, truncated)
            )
            observation = next_observation

            if terminated or truncated:
                summaries.append(
                    {
                        "actor_id": actor_id,
                        "success": bool(info.get("success", False)),
                        "ever_grasped": bool(info.get("ever_grasped", False)),
                        "ever_lifted": bool(info.get("ever_lifted", False)),
                        "stage_index": int(info.get("stage_index", -1)),
                        "failure_reason": str(info.get("failure_reason", "")),
                        "premature_release": bool(
                            info.get("premature_release", False)
                        ),
                        "release_height_ready": bool(
                            info.get("release_height_ready", False)
                        ),
                    }
                )
                observation, _ = env.reset()

            if len(rows) >= rollout_chunk_size or terminated or truncated:
                _put_chunk(transition_queue, actor_id, policy_version, rows, summaries, stop_event)
                rows, summaries = [], []

        if rows:
            _put_chunk(transition_queue, actor_id, policy_version, rows, summaries, stop_event)
        status_queue.put(("stopped", actor_id))
    except Exception as error:
        status_queue.put(("crashed", actor_id, repr(error)))
    finally:
        if env is not None:
            env.close()


def _put_chunk(queue_, actor_id, policy_version, rows, summaries, stop_event) -> None:
    """Pack rows into arrays and block only when the bounded queue is full."""
    payload = {
        "image": np.stack([row[0]["image"] for row in rows]),
        "proprio": np.stack([row[0]["proprio"] for row in rows]),
        "action": np.stack([row[1] for row in rows]).astype(np.float32),
        "reward": np.asarray([row[2] for row in rows], dtype=np.float32),
        "next_image": np.stack([row[3]["image"] for row in rows]),
        "next_proprio": np.stack([row[3]["proprio"] for row in rows]),
        "terminated": np.asarray([row[4] for row in rows], dtype=np.bool_),
        "truncated": np.asarray([row[5] for row in rows], dtype=np.bool_),
        "actor_id": actor_id,
        "policy_version": policy_version,
        "episode_summaries": summaries,
    }
    while not stop_event.is_set():
        try:
            queue_.put(payload, timeout=0.5)
            return
        except queue.Full:
            continue


def _build_run_dir(requested: Path | None) -> Path:
    """Create learner-owned run artifacts inside this package."""
    run_dir = output_path(
        requested
        or Path(__file__).resolve().parent
        / "runs"
        / f"sac_finetune_{datetime.now():%Y%m%d_%H%M%S}_pid{os.getpid()}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir()
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-model", type=Path, required=True)
    parser.add_argument("--expert-data", type=Path, required=True)
    parser.add_argument("--total-timesteps", type=int, default=10_100_000)
    parser.add_argument("--n-actors", type=int, default=8)
    parser.add_argument("--utd-ratio", type=float, default=0.10)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--buffer-size", type=int, default=10_000)
    parser.add_argument("--prior-ratio", type=float, default=0.25)
    parser.add_argument("--bc-warmup-transitions", type=int, default=50_000)
    parser.add_argument("--bc-regularization-transitions", type=int, default=200_000)
    parser.add_argument("--bc-action-coef", type=float, default=1.0)
    parser.add_argument("--bc-action-coef-final", type=float, default=0.1)
    parser.add_argument("--augmentation-shift", type=int, default=4)
    parser.add_argument("--drq-k", type=int, default=2)
    parser.add_argument("--drq-m", type=int, default=2)
    parser.add_argument("--rollout-chunk-size", type=int, default=32)
    parser.add_argument("--actor-queue-size", type=int, default=16)
    parser.add_argument("--policy-sync-interval", type=int, default=1_000)
    parser.add_argument("--checkpoint-freq", type=int, default=1_000_000)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument("--log-freq", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--frame-stack", type=int, default=2)
    parser.add_argument("--max-episode-steps", type=int, default=150)
    parser.add_argument("--action-repeat", type=int, default=8)
    parser.add_argument("--start-method", choices=("spawn", "forkserver", "fork"), default="spawn")
    args = parser.parse_args()
    positive = (
        "total_timesteps", "n_actors", "learning_starts", "batch_size", "buffer_size",
        "rollout_chunk_size", "actor_queue_size", "policy_sync_interval", "checkpoint_freq",
        "log_freq", "image_size", "frame_stack", "max_episode_steps", "action_repeat",
        "bc_warmup_transitions", "bc_regularization_transitions",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.utd_ratio <= 0:
        parser.error("--utd-ratio must be positive")
    if not 0.0 <= args.prior_ratio <= 1.0:
        parser.error("--prior-ratio must be in [0, 1]")
    if args.bc_action_coef < 0.0 or args.bc_action_coef_final < 0.0:
        parser.error("BC coefficients must be non-negative")
    if args.augmentation_shift < 0:
        parser.error("--augmentation-shift must be non-negative")
    if args.drq_k < 1 or args.drq_m < 1:
        parser.error("--drq-k and --drq-m must be positive")
    if args.keep_checkpoints < 1:
        parser.error("--keep-checkpoints must be positive")
    return args


def _state_dict_on_cpu(model: SAC) -> dict[str, np.ndarray]:
    """Copy actor parameters to plain numpy arrays before multiprocessing."""
    return {
        key: value.detach().cpu().numpy().copy()
        for key, value in model.policy.actor.state_dict().items()
    }


def _state_dict_from_numpy(state: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """Rebuild a torch state dict without sending torch storage over a queue."""
    return {key: torch.from_numpy(value) for key, value in state.items()}


def _send_policy(controls: list, model: SAC, version: int) -> None:
    state = _state_dict_on_cpu(model)
    for control_queue in controls:
        _send_latest(control_queue, (version, state))


def _add_chunk_to_replay(replay: MixedSB3Replay, payload: dict) -> int:
    """Insert actor rows through the normal SB3 replay ``add`` interface."""
    count = len(payload["action"])
    for index in range(count):
        observation = {
            "image": payload["image"][index : index + 1],
            "proprio": payload["proprio"][index : index + 1],
        }
        next_observation = {
            "image": payload["next_image"][index : index + 1],
            "proprio": payload["next_proprio"][index : index + 1],
        }
        replay.add(
            observation,
            next_observation,
            payload["action"][index : index + 1],
            np.asarray([payload["reward"][index]], dtype=np.float32),
            np.asarray(
                [payload["terminated"][index] or payload["truncated"][index]],
                dtype=np.float32,
            ),
            [{"TimeLimit.truncated": bool(payload["truncated"][index])}],
        )
    return count


def _record_summaries(logger, summaries: list[dict]) -> None:
    """Aggregate completed-episode business metrics in the learner logger."""
    for summary in summaries:
        reason = summary["failure_reason"]
        logger.record_mean("task/success_rate", float(summary["success"]))
        logger.record_mean("task/grasp_rate", float(summary["ever_grasped"]))
        logger.record_mean("task/lift_rate", float(summary["ever_lifted"]))
        logger.record_mean("task/failure_rate", float(not summary["success"]))
        logger.record_mean("task/stage_timeout_rate", float(reason.endswith("_timeout")))
        logger.record_mean(
            "task/drop_rate", float(reason in {"object_dropped", "object_left_goal"})
        )
        logger.record_mean(
            "task/premature_release_rate",
            float(summary.get("premature_release", reason == "premature_release")),
        )
        logger.record_mean(
            "task/release_height_ready_rate",
            float(summary.get("release_height_ready", False)),
        )
        logger.record_mean("task/final_stage", float(summary["stage_index"]))


def _queue_size(queue_) -> int:
    try:
        return int(queue_.qsize())
    except (NotImplementedError, AttributeError):
        return -1


def _drain_status(status_queue, logger) -> int:
    """Consume actor lifecycle messages and log crashes on the learner."""
    crashed = 0
    while True:
        try:
            status = status_queue.get_nowait()
        except queue.Empty:
            break
        if status[0] == "crashed":
            crashed += 1
            logger.record(f"actors/{status[1]}/crashed", 1.0)
            print(f"Actor {status[1]} crashed: {status[2]}")
        elif status[0] == "stopped":
            logger.record(f"actors/{status[1]}/stopped", 1.0)
    return crashed


def _save_checkpoint(model, run_dir, transition_count, n_updates, args, policy_version, tensorboard_dir, final=False):
    """Save one compact SB3 ZIP and retain only the newest periodic files."""
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    name = "final_model" if final else f"sac_finetune_{transition_count}_steps"
    path = checkpoint_dir / name
    model.save(str(path))
    if not final:
        periodic = sorted(
            checkpoint_dir.glob("sac_finetune_*_steps.zip"),
            key=lambda item: item.stat().st_mtime,
        )
        for old_path in periodic[:-args.keep_checkpoints]:
            old_path.unlink()
    metadata = {
        "learner_transition_count": transition_count,
        "n_updates": n_updates,
        "utd_ratio": args.utd_ratio,
        "n_actors": args.n_actors,
        "policy_version": policy_version,
        "prior_ratio": args.prior_ratio,
        "batch_size": args.batch_size,
        "rollout_chunk_size": args.rollout_chunk_size,
        "tensorboard_dir": str(tensorboard_dir),
        "drq_k": args.drq_k,
        "drq_m": args.drq_m,
        "drq_shift_pixels": args.augmentation_shift,
        "environment_randomize": False,
        "object_position_jitter": 0.0,
        "goal_position_jitter": 0.0,
        "initial_joint_jitter": 0.0,
        "camera_jitter": 0.0,
        "args": vars(args),
    }
    (run_dir / "checkpoint_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def _actor_parameter_delta(model: ProtectedSAC) -> float:
    """Return RMS actor parameter change relative to the BC initialization."""
    squared = []
    for name, parameter in model.policy.actor.state_dict().items():
        initial = model.bc_state[name].to(parameter.device, dtype=parameter.dtype)
        squared.append((parameter.detach() - initial).pow(2).mean())
    return float(torch.stack(squared).mean().sqrt().detach().cpu())


def main() -> None:
    """Start actors, run the learner loop, then shut down all processes."""
    args = parse_args()
    setup(args.seed, args.device)
    run_dir = _build_run_dir(args.output)
    config = EnvConfig(
        image_size=args.image_size,
        frame_stack=args.frame_stack,
        max_episode_steps=args.max_episode_steps,
        action_repeat=args.action_repeat,
    )
    if args.resume is None:
        _, saved_config, bc_payload = load_actor(args.bc_model, device="cpu")
    else:
        saved_config = config
        resume_model = SAC.load(args.resume, device=args.device)
        bc_payload = {"actor_state": resume_model.policy.actor.state_dict()}
    # The learner env is never stepped. It only gives SB3 the spaces needed to
    # construct policy/critic modules and the replay buffer.
    learner_env = DummyVecEnv(
        [make_randomized_env_factory(
            # This environment is not stepped; it only supplies SB3 spaces.
            # Keep it consistent with the no-jitter actor environments anyway.
            task="pick_place", rank=0, seed=args.seed, config=config, randomize=False
        )]
    )
    augmentation = ConsistentMultiViewAugmentation(args.augmentation_shift)
    replay = MixedSB3Replay(
        args.buffer_size,
        learner_env.observation_space,
        learner_env.action_space,
        device="cpu",
        n_envs=1,
        prior_ratio=args.prior_ratio,
        seed=args.seed,
        augmentation=augmentation,
    )
    model = ProtectedSAC(
        "MultiInputPolicy", learner_env,
        policy_kwargs=policy_kwargs(config), learning_rate=3e-4,
        buffer_size=args.buffer_size, batch_size=args.batch_size,
        learning_starts=args.learning_starts, train_freq=1, gradient_steps=1,
        gamma=0.99, tau=0.005, seed=args.seed, device=args.device,
        bc_state=bc_payload["actor_state"], replay=replay,
        bc_warmup_transitions=args.bc_warmup_transitions,
        bc_regularization_transitions=args.bc_regularization_transitions,
        bc_action_coef=args.bc_action_coef,
        bc_action_coef_final=args.bc_action_coef_final,
        drq_k=args.drq_k,
        drq_m=args.drq_m,
        augmentation=augmentation,
    )
    if args.resume is not None:
        loaded = SAC.load(args.resume, env=learner_env, device=args.device)
        model.set_parameters(loaded.get_parameters(), exact_match=True)
        match = re.search(r"sac_finetune_(\d+)_steps", args.resume.name)
        if match:
            learner_transition_count = int(match.group(1))
        else:
            learner_transition_count = 0
        model.num_timesteps = learner_transition_count
    replay.device = model.device
    # Only the actor is initialized from BC; critics learn from the new replay.
    model.policy.actor.load_state_dict(bc_payload["actor_state"], strict=True)
    model._total_timesteps = args.total_timesteps

    for episode in load_episodes(args.expert_data, successful_only=True):
        replay.add_prior_episode(episode)
    model.replay_buffer = replay

    tensorboard_dir = new_tensorboard_run("sac_finetune")
    logger = configure(str(tensorboard_dir), format_strings=["stdout", "tensorboard"])
    model.set_logger(logger)
    print(f"Run directory: {run_dir}")
    print(f"TensorBoard logs: {tensorboard_dir}")
    print(f"Prior transitions: {len(replay.prior)}")

    context = mp.get_context(args.start_method)
    transition_queue = context.Queue(maxsize=args.actor_queue_size)
    status_queue = context.Queue()
    stop_event = context.Event()
    controls = [context.Queue(maxsize=1) for _ in range(args.n_actors)]
    actors = []
    initial_state = _state_dict_on_cpu(model)
    for actor_id, control_queue in enumerate(controls):
        control_queue.put((0, initial_state))
        process = context.Process(
            target=_actor_process,
            args=(actor_id, config, args.seed, args.rollout_chunk_size,
                  transition_queue, control_queue, status_queue, stop_event),
            name=f"pixel_actor_{actor_id}",
        )
        process.start()
        actors.append(process)

    learner_transition_count = locals().get("learner_transition_count", 0)
    actor_batches_received = 0
    update_budget = 0.0
    policy_version = 0
    n_updates = 0
    last_sync = last_log = last_checkpoint = 0
    started = time.perf_counter()

    try:
        while learner_transition_count < args.total_timesteps:
            _drain_status(status_queue, logger)
            try:
                payload = transition_queue.get(timeout=1.0)
            except queue.Empty:
                if all(not actor.is_alive() for actor in actors):
                    raise RuntimeError("All actor processes exited before training completed")
                continue

            received = _add_chunk_to_replay(replay, payload)
            learner_transition_count += received
            actor_batches_received += 1
            update_budget += received * args.utd_ratio
            _record_summaries(logger, payload["episode_summaries"])

            while (
                update_budget >= 1.0
                and learner_transition_count >= args.learning_starts
                and replay.can_sample(args.batch_size)
            ):
                model.num_timesteps = learner_transition_count
                model.train(gradient_steps=1, batch_size=args.batch_size)
                n_updates += 1
                update_budget -= 1.0

            if learner_transition_count - last_sync >= args.policy_sync_interval:
                policy_version += 1
                _send_policy(controls, model, policy_version)
                last_sync = learner_transition_count

            now = time.perf_counter()
            if learner_transition_count - last_log >= args.log_freq:
                elapsed = max(now - started, 1e-6)
                logger.record("learner/online_transitions", learner_transition_count)
                logger.record("learner/queue_size", _queue_size(transition_queue))
                logger.record("learner/actor_batches_received", actor_batches_received)
                logger.record("rollout/transitions_per_second", learner_transition_count / elapsed)
                logger.record("train/n_updates", n_updates)
                logger.record("train/utd_target", args.utd_ratio)
                logger.record("train/utd_actual", n_updates / max(learner_transition_count, 1))
                logger.record("train/drq_k", args.drq_k)
                logger.record("train/drq_m", args.drq_m)
                logger.record("train/drq_shift_pixels", args.augmentation_shift)
                logger.record("train/drq_target_q_std", model.last_target_q_std)
                logger.record("train/drq_current_q_std", model.last_current_q_std)
                logger.record("train/bc_action_loss", model.last_bc_action_loss)
                logger.record("train/bc_action_coef", model.last_bc_coef)
                logger.record("train/bc_actor_frozen", model.last_actor_frozen)
                logger.record(
                    "train/actor_parameter_delta_from_bc",
                    _actor_parameter_delta(model),
                )
                logger.dump(learner_transition_count)
                last_log = learner_transition_count

            if learner_transition_count - last_checkpoint >= args.checkpoint_freq:
                _save_checkpoint(model, run_dir, learner_transition_count, n_updates,
                                 args, policy_version, tensorboard_dir)
                last_checkpoint = learner_transition_count
    finally:
        stop_event.set()
        for control_queue in controls:
            _send_latest(control_queue, STOP)
        for actor in actors:
            actor.join(timeout=10.0)
        for actor in actors:
            if actor.is_alive():
                actor.terminate()
                actor.join()
        _save_checkpoint(model, run_dir, learner_transition_count, n_updates,
                         args, policy_version, tensorboard_dir, final=True)
        # The requested run may finish before the next periodic log interval.
        # Dump once at the exact final learner transition count so the last
        # business-rate aggregation and update counters are visible in TB.
        logger.record("learner/online_transitions", learner_transition_count)
        logger.record("train/n_updates", n_updates)
        logger.record("train/utd_target", args.utd_ratio)
        logger.record(
            "train/utd_actual",
            n_updates / max(learner_transition_count, 1),
        )
        logger.record("train/bc_action_loss", model.last_bc_action_loss)
        logger.record("train/bc_action_coef", model.last_bc_coef)
        logger.record("train/bc_actor_frozen", model.last_actor_frozen)
        logger.record("train/actor_parameter_delta_from_bc", _actor_parameter_delta(model))
        logger.record("train/drq_k", args.drq_k)
        logger.record("train/drq_m", args.drq_m)
        logger.record("train/drq_shift_pixels", args.augmentation_shift)
        logger.record("train/drq_target_q_std", model.last_target_q_std)
        logger.record("train/drq_current_q_std", model.last_current_q_std)
        logger.dump(learner_transition_count)
        logger.close()
        learner_env.close()
        transition_queue.close()
        status_queue.close()
        for control_queue in controls:
            control_queue.close()


if __name__ == "__main__":
    main()
