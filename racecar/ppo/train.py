"""Train RaceCar with a GPU learner and CPU actor processes.

Run from the project root with ``python -m ppo.train``.  The learner collects
one rollout from every actor at a policy version, performs PPO updates on the
GPU, then publishes the next CPU policy state through each actor command
queue. TensorBoard window metrics are reset after every report interval.
"""

from __future__ import annotations

import argparse
import datetime
import math
import os
import queue
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from env.actor_env import DirectRaceCarVectorEnv
from ppo.actor import actor_process
from ppo.model import ACTION_NAMES, ActorCritic, actions_to_steering


@dataclass
class Config:
    total_timesteps: int = 20_000_000
    num_actors: int = 8
    envs_per_actor: int = 2
    rollout_steps: int = 512
    max_episode_steps: int = 1500
    fps: int = 20
    seed: int = 1
    start_position_noise: float = 0.15
    start_heading_noise: float = 0.08
    evaluation_episodes: int = 32
    evaluation_envs: int = 8
    # Deliberately outside the actor seed ranges used by the default setup.
    evaluation_seed: int = 2_000_000_000
    evaluation_action_seed: int = 2_100_000_000
    fine_tune_start_steps: int = 5_000_000
    entropy_anneal_end_steps: int = 12_000_000
    fine_tune_tighten_steps: int = 15_000_000
    learning_rate: float = 3e-4
    fine_tune_learning_rate: float = 5e-5
    tightened_learning_rate: float = 1e-5
    final_learning_rate: float = 5e-6
    gamma: float = 0.995
    gae_lambda: float = 0.95
    update_epochs: int = 8
    fine_tune_update_epochs: int = 4
    final_update_epochs: int = 2
    minibatch_size: int = 512
    clip_coef: float = 0.2
    fine_tune_clip_coef: float = 0.1
    final_clip_coef: float = 0.05
    entropy_coef: float = 0.01
    final_entropy_coef: float = 0.0
    target_kl: float = 0.01
    fine_tune_target_kl: float = 0.003
    final_target_kl: float = 0.001
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    report_interval: int = 200_000
    checkpoint_interval: int = 1_000_000
    # Empty means: create racecar_ppo_YYYYMMDD_HHMMSS_mmm_pidN under the root.
    log_dir: str = ""
    checkpoint_dir: str = ""
    device: str = "cuda"
    resume: str = ""


@dataclass(frozen=True)
class PPOSettings:
    """Step-dependent PPO settings; derived solely from absolute global steps."""

    learning_rate: float
    entropy_coef: float
    clip_coef: float
    update_epochs: int
    target_kl: float


def _settings_at_step(config: Config, global_steps: int) -> PPOSettings:
    """Return a re-entrant schedule that is identical after any checkpoint resume."""
    if global_steps < config.fine_tune_start_steps:
        learning_rate = config.learning_rate
        clip_coef = config.clip_coef
        update_epochs = config.update_epochs
        target_kl = config.target_kl
    elif global_steps < config.fine_tune_tighten_steps:
        fine_tune_progress = float(np.clip(
            (global_steps - config.fine_tune_start_steps)
            / max(1, config.fine_tune_tighten_steps
                  - config.fine_tune_start_steps),
            0.0, 1.0,
        ))
        learning_rate = (config.fine_tune_learning_rate
                         + fine_tune_progress
                         * (config.tightened_learning_rate
                            - config.fine_tune_learning_rate))
        clip_coef = config.fine_tune_clip_coef
        update_epochs = config.fine_tune_update_epochs
        target_kl = config.fine_tune_target_kl
    else:
        tighten_progress = float(np.clip(
            (global_steps - config.fine_tune_tighten_steps)
            / max(1, config.total_timesteps - config.fine_tune_tighten_steps),
            0.0, 1.0,
        ))
        learning_rate = (config.tightened_learning_rate
                         + tighten_progress
                         * (config.final_learning_rate
                            - config.tightened_learning_rate))
        clip_coef = (config.fine_tune_clip_coef
                     + tighten_progress
                     * (config.final_clip_coef - config.fine_tune_clip_coef))
        update_epochs = config.final_update_epochs
        target_kl = (config.fine_tune_target_kl
                     + tighten_progress
                     * (config.final_target_kl - config.fine_tune_target_kl))

    entropy_progress = float(np.clip(
        (global_steps - config.fine_tune_start_steps)
        / max(1, config.entropy_anneal_end_steps - config.fine_tune_start_steps),
        0.0, 1.0,
    ))
    entropy_coef = (config.entropy_coef
                    + entropy_progress * (config.final_entropy_coef
                                          - config.entropy_coef))
    return PPOSettings(
        learning_rate=learning_rate,
        entropy_coef=entropy_coef,
        clip_coef=clip_coef,
        update_epochs=update_epochs,
        target_kl=target_kl,
    )


def _cpu_state_dict(model: ActorCritic) -> dict[str, np.ndarray]:
    # Send NumPy arrays rather than torch tensors. torch's multiprocessing
    # reducer uses shared file descriptors, which is fragile with nested
    # spawned queues and unnecessary for this small policy network.
    return {name: parameter.detach().cpu().numpy().copy()
            for name, parameter in model.state_dict().items()}


def _compute_gae(rollout: dict, gamma: float, gae_lambda: float):
    rewards = rollout["rewards"]
    values = rollout["values"]
    terminated = rollout["terminated"]
    truncated = rollout["truncated"]
    timeout_values = rollout["timeout_values"]
    last_values = rollout["last_values"]
    steps, envs = rewards.shape
    advantages = np.zeros_like(rewards, dtype=np.float32)
    next_advantage = np.zeros(envs, dtype=np.float32)

    for step in reversed(range(steps)):
        if step == steps - 1:
            next_value = last_values
        else:
            next_value = values[step + 1]
        # For a time-limit, bootstrap from V(terminal observation); for a
        # true terminal, the bootstrap is zero. Non-terminal transitions use
        # the next rollout value as usual.
        bootstrap = next_value.copy()
        timeout_mask = truncated[step] & ~terminated[step]
        bootstrap[timeout_mask] = timeout_values[step, timeout_mask]
        bootstrap[terminated[step]] = 0.0
        delta = rewards[step] + gamma * bootstrap - values[step]
        continuation = ~(terminated[step] | truncated[step])
        advantages[step] = delta + gamma * gae_lambda * continuation * next_advantage
        next_advantage = advantages[step]
    returns = advantages + values
    return advantages, returns


def _ppo_update(model: ActorCritic, optimizer: torch.optim.Optimizer,
                batch: dict[str, torch.Tensor], config: Config,
                settings: PPOSettings) -> dict[str, float]:
    for group in optimizer.param_groups:
        group["lr"] = settings.learning_rate

    size = batch["observations"].shape[0]
    metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
               "approx_kl": 0.0, "clip_fraction": 0.0}
    update_count = 0
    epochs_used = 0
    for epoch in range(settings.update_epochs):
        epoch_kls = []
        indices = np.random.permutation(size)
        for start in range(0, size, config.minibatch_size):
            minibatch = torch.as_tensor(indices[start:start + config.minibatch_size],
                                        device=batch["observations"].device)
            _, new_logprob, entropy, new_value = model.get_action_and_value(
                batch["observations"][minibatch], batch["actions"][minibatch]
            )
            logratio = new_logprob - batch["logprobs"][minibatch]
            ratio = logratio.exp()
            policy_loss_1 = -batch["advantages"][minibatch] * ratio
            policy_loss_2 = -batch["advantages"][minibatch] * ratio.clamp(
                1 - settings.clip_coef, 1 + settings.clip_coef
            )
            policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()

            old_value = batch["values"][minibatch]
            value_unclipped = (new_value - batch["returns"][minibatch]).pow(2)
            value_clipped = old_value + (new_value - old_value).clamp(
                -settings.clip_coef, settings.clip_coef
            )
            value_clipped_loss = (value_clipped - batch["returns"][minibatch]).pow(2)
            value_loss = 0.5 * torch.max(value_unclipped, value_clipped_loss).mean()
            entropy_mean = entropy.mean()
            loss = (policy_loss + config.value_coef * value_loss
                    - settings.entropy_coef * entropy_mean)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            metrics["policy_loss"] += float(policy_loss.detach())
            metrics["value_loss"] += float(value_loss.detach())
            metrics["entropy"] += float(entropy_mean.detach())
            approx_kl = float(((ratio - 1.0) - logratio).mean().detach())
            metrics["approx_kl"] += approx_kl
            epoch_kls.append(approx_kl)
            metrics["clip_fraction"] += float(
                ((ratio - 1.0).abs() > settings.clip_coef).float().mean().detach()
            )
            update_count += 1
        epochs_used = epoch + 1
        if settings.target_kl > 0 and np.mean(epoch_kls) > settings.target_kl:
            break
    for name in metrics:
        metrics[name] /= max(1, update_count)
    metrics["learning_rate"] = settings.learning_rate
    metrics["entropy_coef"] = settings.entropy_coef
    metrics["clip_coef"] = settings.clip_coef
    metrics["target_kl"] = settings.target_kl
    metrics["update_epochs_used"] = float(epochs_used)
    metrics["explained_variance"] = _explained_variance(
        batch["values"], batch["returns"]
    )
    return metrics


def _explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float:
    values = values.detach().float().cpu().numpy()
    returns = returns.detach().float().cpu().numpy()
    variance = np.var(returns)
    if variance < 1e-8:
        return 0.0
    return float(1.0 - np.var(returns - values) / variance)


def _resolve_output_dirs(config: Config) -> str:
    """Assign one collision-resistant name to this training process."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    run_name = f"racecar_ppo_{timestamp}_pid{os.getpid()}"
    if not config.log_dir:
        config.log_dir = str(Path("runs") / run_name)
    if not config.checkpoint_dir:
        config.checkpoint_dir = str(Path("checkpoints") / run_name)
    return run_name


def _format_checkpoint_suffix(suffix: str, total_timesteps: int) -> str:
    """Pad numeric steps so filename order equals numeric step order."""
    if str(suffix).isdigit():
        return f"{int(suffix):0{len(str(total_timesteps))}d}"
    return str(suffix)


class Learner:
    def __init__(self, config: Config):
        self.run_name = _resolve_output_dirs(config)
        self.config = config
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --device cpu explicitly for a smoke test")
        self.device = torch.device(config.device)
        self.model = ActorCritic().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate, eps=1e-5)
        self.global_steps = 0
        self.policy_version = 0
        self.best_greedy_score = None
        self.last_evaluation_step = None
        self._last_rollout_wait_seconds = 0.0
        self._load_resume()
        if self.global_steps:
            resumed_seed = config.seed + self.global_steps
            np.random.seed(resumed_seed % (2 ** 32))
            torch.manual_seed(resumed_seed)

        context = torch.multiprocessing.get_context("spawn")
        self.rollout_queue = context.Queue(maxsize=config.num_actors * 2)
        self.command_queues = [context.Queue(maxsize=2) for _ in range(config.num_actors)]
        actor_config = asdict(config)
        actor_config["resume_global_steps"] = self.global_steps
        self.actors = [
            context.Process(
                target=actor_process,
                args=(actor_id, actor_config, self.command_queues[actor_id], self.rollout_queue),
                name=f"racecar-actor-{actor_id}",
            )
            for actor_id in range(config.num_actors)
        ]
        for actor in self.actors:
            actor.start()
        self.writer = SummaryWriter(config.log_dir)
        self.writer.add_text("config", str(asdict(config)), 0)
        print(f"run={self.run_name}\nTensorBoard: {config.log_dir}\ncheckpoints: {config.checkpoint_dir}")

    def _load_resume(self):
        if not self.config.resume:
            return
        checkpoint = torch.load(self.config.resume, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        for state in self.optimizer.state.values():
            for name, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[name] = value.to(self.device)
        self.global_steps = int(checkpoint.get("global_steps", 0))
        self.policy_version = int(checkpoint.get("policy_version", 0))
        saved_score = checkpoint.get("best_greedy_score")
        if saved_score is not None:
            self.best_greedy_score = tuple(saved_score)

    def _broadcast_weights(self):
        state_dict = _cpu_state_dict(self.model)
        command = {"type": "weights", "version": self.policy_version,
                   "state_dict": state_dict}
        for command_queue in self.command_queues:
            command_queue.put(command)

    def _receive_rollouts(self):
        rollouts = {}
        wait_started = time.perf_counter()
        while len(rollouts) < self.config.num_actors:
            try:
                payload = self.rollout_queue.get(timeout=300.0)
            except queue.Empty as error:
                dead = [actor.name for actor in self.actors if not actor.is_alive()]
                raise RuntimeError(f"timed out waiting for actor rollouts; dead={dead}") from error
            except (EOFError, OSError) as error:
                dead = [actor.name for actor in self.actors if not actor.is_alive()]
                raise RuntimeError(f"timed out waiting for actors; dead={dead}") from error
            if payload.get("type") == "error":
                raise RuntimeError(f"actor {payload['actor_id']} failed:\n{payload['traceback']}")
            actor_id = int(payload["actor_id"])
            if int(payload["version"]) != self.policy_version:
                raise RuntimeError("received a rollout from an unexpected policy version")
            rollouts[actor_id] = payload
        self._last_rollout_wait_seconds = time.perf_counter() - wait_started
        return [rollouts[index] for index in range(self.config.num_actors)]

    def _make_batch(self, rollouts):
        advantages = []
        returns = []
        for rollout in rollouts:
            rollout_advantages, rollout_returns = _compute_gae(
                rollout, self.config.gamma, self.config.gae_lambda
            )
            advantages.append(rollout_advantages)
            returns.append(rollout_returns)
        observations = np.concatenate([item["observations"] for item in rollouts], axis=1)
        actions = np.concatenate([item["actions"] for item in rollouts], axis=1)
        logprobs = np.concatenate([item["logprobs"] for item in rollouts], axis=1)
        values = np.concatenate([item["values"] for item in rollouts], axis=1)
        advantages = np.concatenate(advantages, axis=1)
        returns = np.concatenate(returns, axis=1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        def tensor(array, dtype):
            return torch.as_tensor(array.reshape(-1), dtype=dtype, device=self.device)

        return {
            "observations": torch.as_tensor(observations.reshape(-1, observations.shape[-1]),
                                             dtype=torch.float32, device=self.device),
            "actions": tensor(actions, torch.long),
            "logprobs": tensor(logprobs, torch.float32),
            "values": tensor(values, torch.float32),
            "advantages": tensor(advantages, torch.float32),
            "returns": tensor(returns, torch.float32),
        }

    def _save_checkpoint(self, suffix: str):
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        # Numeric checkpoint names use a fixed width so lexicographic file
        # ordering is identical to training-step ordering. Semantic names
        # such as "best_greedy" and "final" remain readable.
        suffix = _format_checkpoint_suffix(suffix, self.config.total_timesteps)
        path = Path(self.config.checkpoint_dir) / f"checkpoint_{suffix}.pt"
        torch.save({"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                    "global_steps": self.global_steps, "policy_version": self.policy_version,
                    "best_greedy_score": self.best_greedy_score,
                    "config": asdict(self.config)}, path)
        return path

    def _evaluate_success_rate(self, *, stochastic: bool):
        evaluation_model = ActorCritic().cpu().eval()
        evaluation_model.load_state_dict({
            name: torch.from_numpy(value)
            for name, value in _cpu_state_dict(self.model).items()
        })
        episode_count = self.config.evaluation_episodes
        num_envs = min(self.config.evaluation_envs, episode_count)
        environments = DirectRaceCarVectorEnv(
            num_envs, render=False, fps=self.config.fps,
            max_steps=self.config.max_episode_steps,
            start_position_noise=self.config.start_position_noise,
            start_heading_noise=self.config.start_heading_noise,
        )
        # One fixed action RNG stream per evaluation episode avoids coupling
        # stochastic results to vector scheduling or other episodes' lengths.
        action_generators = [torch.Generator(device="cpu") for _ in range(num_envs)]
        for index, generator in enumerate(action_generators):
            generator.manual_seed(self.config.evaluation_action_seed + index)
        try:
            next_episode = num_envs
            observations, _ = environments.reset([
                self.config.evaluation_seed + index for index in range(num_envs)
            ])
            active = np.ones(num_envs, dtype=bool)
            results = []
            while len(results) < episode_count:
                with torch.no_grad():
                    observation_tensor = torch.from_numpy(observations)
                    logits = evaluation_model.policy(
                        evaluation_model.trunk(observation_tensor)
                    )
                    if stochastic:
                        probabilities = logits.softmax(dim=-1)
                        actions = np.asarray([
                            int(torch.multinomial(
                                probabilities[index], num_samples=1,
                                replacement=True, generator=action_generators[index],
                            ).item())
                            for index in range(num_envs)
                        ], dtype=np.int64)
                    else:
                        actions = logits.argmax(dim=-1).numpy()
                next_observations, _, terminated, truncated, infos = environments.step(
                    actions_to_steering(actions)
                )
                done_indices = np.flatnonzero(terminated | truncated)
                reset_indices = []
                reset_seeds = []
                for index in done_indices:
                    if active[index]:
                        results.append(bool(infos[index].get("is_success", False)))
                    reset_indices.append(int(index))
                    if next_episode < episode_count:
                        episode_id = next_episode
                        reset_seeds.append(self.config.evaluation_seed + episode_id)
                        action_generators[index].manual_seed(
                            self.config.evaluation_action_seed + episode_id
                        )
                        next_episode += 1
                        active[index] = True
                    else:
                        # Keep the fixed-size vector valid while the remaining
                        # active evaluation episodes finish; ignore this dummy.
                        reset_seeds.append(self.config.evaluation_seed + episode_count + int(index))
                        action_generators[index].manual_seed(
                            self.config.evaluation_action_seed + episode_count + int(index)
                        )
                        active[index] = False
                observations = next_observations
                for index, (reset_observation, _) in environments.reset_at(
                    reset_indices, reset_seeds
                ).items():
                    observations[index] = reset_observation

            return float(sum(results) / len(results))
        finally:
            environments.close()

    def _evaluate_and_save_best(self):
        greedy_success_rate = self._evaluate_success_rate(stochastic=False)
        stochastic_success_rate = self._evaluate_success_rate(stochastic=True)
        step = self.global_steps
        self.last_evaluation_step = step
        self.writer.add_scalar("eval/greedy_success_rate", greedy_success_rate, step)
        self.writer.add_scalar("eval/stochastic_success_rate", stochastic_success_rate, step)
        # Prefer deployment-time greedy success; use stochastic success as tie-breaker.
        score = (greedy_success_rate, stochastic_success_rate)
        if self.best_greedy_score is None or score > self.best_greedy_score:
            self.best_greedy_score = score
            self._save_checkpoint("best_greedy")
        return {
            "greedy_success_rate": greedy_success_rate,
            "stochastic_success_rate": stochastic_success_rate,
        }

    def train(self):
        config = self.config
        self._broadcast_weights()
        batch_steps = config.num_actors * config.envs_per_actor * config.rollout_steps
        remaining_steps = max(0, config.total_timesteps - self.global_steps)
        updates = math.ceil(remaining_steps / batch_steps)
        next_report = ((self.global_steps // config.report_interval) + 1) * config.report_interval
        next_checkpoint = ((self.global_steps // config.checkpoint_interval) + 1) * config.checkpoint_interval
        report_window = {"episodes": [], "metrics": [], "action_counts": np.zeros(3, np.int64),
                         "env_steps": 0, "reward_sum": 0.0, "reward_count": 0,
                         "rollout_seconds": [], "learner_rollout_wait_seconds": []}
        report_started = time.perf_counter()

        try:
            if self.global_steps:
                initial_evaluation = self._evaluate_and_save_best()
                print(f"resume checkpoint evaluation: {initial_evaluation}")
            for _ in range(updates):
                rollouts = self._receive_rollouts()
                batch = self._make_batch(rollouts)
                settings = _settings_at_step(config, self.global_steps)
                metrics = _ppo_update(self.model, self.optimizer, batch, config, settings)
                self.global_steps += batch_steps
                self.policy_version += 1
                self._broadcast_weights()

                report_window["env_steps"] += batch_steps
                report_window["metrics"].append(metrics)
                report_window["rollout_seconds"].extend(
                    float(rollout.get("rollout_collect_seconds", 0.0)) for rollout in rollouts
                )
                report_window["learner_rollout_wait_seconds"].append(
                    self._last_rollout_wait_seconds
                )
                for rollout in rollouts:
                    report_window["episodes"].extend(rollout["episodes"])
                    report_window["action_counts"] += rollout["action_counts"]
                    report_window["reward_sum"] += float(rollout["rewards"].sum())
                    report_window["reward_count"] += rollout["rewards"].size

                if self.global_steps >= next_report:
                    self._report(report_window, report_started)
                    report_window = {"episodes": [], "metrics": [], "action_counts": np.zeros(3, np.int64),
                                     "env_steps": 0, "reward_sum": 0.0, "reward_count": 0,
                                     "rollout_seconds": [], "learner_rollout_wait_seconds": []}
                    report_started = time.perf_counter()
                    next_report += config.report_interval
                if self.global_steps >= next_checkpoint:
                    evaluation = self._evaluate_and_save_best()
                    print(f"checkpoint evaluation at steps={self.global_steps}: {evaluation}")
                    self._save_checkpoint(str(self.global_steps))
                    next_checkpoint += config.checkpoint_interval
            if report_window["env_steps"]:
                self._report(report_window, report_started)
            if self.last_evaluation_step != self.global_steps:
                evaluation = self._evaluate_and_save_best()
                print(f"final evaluation at steps={self.global_steps}: {evaluation}")
            self._save_checkpoint("final")
        finally:
            self.close()

    def _report(self, window: dict, started: float):
        step = self.global_steps
        episodes = window["episodes"]
        count = len(episodes)
        successes = sum(item["success"] for item in episodes)
        collisions = sum(item["reason"] == "collision" for item in episodes)
        timeouts = sum(item["reason"] == "time_limit" for item in episodes)
        lengths = [item["length"] for item in episodes]
        successful_lengths = [item["length"] for item in episodes if item["success"]]
        returns = [item["return"] for item in episodes]
        terminal_distances = [item["terminal_distance"] for item in episodes]
        self.writer.add_scalar("window/success_rate", successes / count if count else 0.0, step)
        self.writer.add_scalar("window/episodes", count, step)
        self.writer.add_scalar("window/successes", successes, step)
        self.writer.add_scalar("window/collision_rate", collisions / count if count else 0.0, step)
        self.writer.add_scalar("window/time_limit_rate", timeouts / count if count else 0.0, step)
        self.writer.add_scalar("window/mean_episode_length", np.mean(lengths) if lengths else 0.0, step)
        self.writer.add_scalar("window/mean_success_length", np.mean(successful_lengths) if successful_lengths else 0.0, step)
        self.writer.add_scalar("window/mean_episode_return", np.mean(returns) if returns else 0.0, step)
        self.writer.add_scalar("window/mean_terminal_distance",
                              np.mean(terminal_distances) if terminal_distances else 0.0, step)
        self.writer.add_scalar("window/mean_step_reward",
                              window["reward_sum"] / max(1, window["reward_count"]), step)
        self.writer.add_scalar("perf/mean_rollout_collect_seconds",
                              np.mean(window["rollout_seconds"]) if window["rollout_seconds"] else 0.0,
                              step)
        self.writer.add_scalar("queue/mean_learner_rollout_wait_seconds",
                              np.mean(window["learner_rollout_wait_seconds"])
                              if window["learner_rollout_wait_seconds"] else 0.0, step)
        for index, name in enumerate(ACTION_NAMES):
            total = max(1, int(window["action_counts"].sum()))
            self.writer.add_scalar(f"actions/{name}_rate", window["action_counts"][index] / total, step)
        for name in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
                     "learning_rate", "entropy_coef", "clip_coef", "target_kl",
                     "update_epochs_used", "explained_variance"):
            values = [metric[name] for metric in window["metrics"]]
            self.writer.add_scalar(f"ppo/{name}", np.mean(values) if values else 0.0, step)
        self.writer.add_scalar("perf/steps_per_second",
                              window["env_steps"] / max(1e-6, time.perf_counter() - started), step)
        self.writer.flush()
        print(f"steps={step} episodes={count} success_rate={successes / count if count else 0:.3f} "
              f"mean_success_steps={np.mean(successful_lengths) if successful_lengths else 0:.1f}")

    def close(self):
        for command_queue in self.command_queues:
            try:
                command_queue.put_nowait({"type": "stop"})
            except queue.Full:
                pass
        for actor in self.actors:
            actor.join(timeout=10)
            if actor.is_alive():
                actor.terminate()
        if hasattr(self, "writer"):
            self.writer.close()


def parse_args() -> Config:
    defaults = Config()
    parser = argparse.ArgumentParser(description="Actor-learner PPO for RaceCarEnv")
    for field in defaults.__dataclass_fields__.values():
        argument = f"--{field.name.replace('_', '-')}"
        kwargs = {"default": getattr(defaults, field.name), "type": type(getattr(defaults, field.name))}
        if field.name in {"resume", "log_dir", "checkpoint_dir", "device"}:
            kwargs["type"] = str
        parser.add_argument(argument, **kwargs)
    values = vars(parser.parse_args())
    return Config(**values)


def main():
    config = parse_args()
    if config.num_actors < 1 or config.envs_per_actor < 1:
        raise ValueError("num_actors and envs_per_actor must be positive")
    if not (0 <= config.fine_tune_start_steps
            < config.entropy_anneal_end_steps
            <= config.fine_tune_tighten_steps
            < config.total_timesteps):
        raise ValueError(
            "schedule must satisfy 0 <= fine-tune start < entropy end "
            "<= tighten start < total"
        )
    if min(config.learning_rate, config.fine_tune_learning_rate,
           config.tightened_learning_rate, config.final_learning_rate) <= 0:
        raise ValueError("all learning rates must be positive")
    if min(config.entropy_coef, config.final_entropy_coef) < 0:
        raise ValueError("entropy coefficients must be non-negative")
    if config.evaluation_episodes < 1 or config.evaluation_envs < 1:
        raise ValueError("evaluation_episodes and evaluation_envs must be positive")
    if config.start_position_noise < 0 or config.start_heading_noise < 0:
        raise ValueError("start pose noise must be non-negative")
    if min(config.clip_coef, config.fine_tune_clip_coef,
           config.final_clip_coef) <= 0:
        raise ValueError("all clip coefficients must be positive")
    if min(config.update_epochs, config.fine_tune_update_epochs,
           config.final_update_epochs) < 1:
        raise ValueError("all update epoch counts must be positive")
    if min(config.target_kl, config.fine_tune_target_kl,
           config.final_target_kl) <= 0:
        raise ValueError("all target KL values must be positive")
    Learner(config).train()


if __name__ == "__main__":
    main()
