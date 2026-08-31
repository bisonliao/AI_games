from __future__ import annotations

import os
import queue
import random
import signal
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from MathEnv import BasicMathConfig, BasicMathEnv

from .model import DuelingDQN, NetworkSpec
from .preprocessing import batch_to_tensors, encode_observation, observation_to_tensors
from .replay import NStepAccumulator, PrioritizedReplayBuffer, Transition


TaskName = Literal["high", "low"]


@dataclass
class TrainConfig:
    task: TaskName
    total_transitions: int = 20_000_000
    low_distance_reward_scale: float = 0.1
    low_epsilon_fraction_points: tuple[tuple[float, float], ...] = (
        (0.0, 0.9),
        (0.05, 0.05),
        (0.25, 0.01), #训练进行到25%，就线性衰减到0.01
        (1.0, 0.001),
    )
    num_actors: int = 16
    seed: int = 1
    replay_capacity: int = 100_000
    replay_warmup: int = 20_000
    batch_size: int = 32
    gamma: float = 0.99
    n_step: int = 3
    learning_rate: float = 1e-4
    updates_per_transition: float = 0.25
    target_update_interval: int = 2_500
    shared_model_update_interval: int = 250
    actor_sync_interval: int = 1_000
    max_grad_norm: float = 10.0
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    epsilon_start: float = 0.9
    epsilon_end: float = 0.05
    epsilon_decay_transitions: int = 1_000_000
    transition_queue_size: int = 4_096
    report_interval: int = 10_000
    dqn_log_interval: int = 10_000
    checkpoint_interval: int = 1_000_000
    evaluation_interval: int = 100_000
    evaluation_episodes: int = 100
    runs_dir: str = "runs"
    checkpoints_dir: str = "checkpoints"

    @property
    def network_spec(self) -> NetworkSpec:
        if self.task == "high":
            return NetworkSpec(input_channels=1, macro_dim=0, num_actions=19)
        return NetworkSpec(input_channels=2, macro_dim=45, num_actions=6)

    @property
    def goal_conditioned(self) -> bool:
        return self.task == "low"

    @property
    def epsilon_decay(self) -> int:
        return self.epsilon_decay_transitions or self.total_transitions


def epsilon_at(global_transitions: int, config: TrainConfig) -> float:
    progress = min(max(global_transitions / max(config.epsilon_decay, 1), 0.0), 1.0)
    return config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)


def piecewise_epsilon(
    position: float,
    points: tuple[tuple[float, float], ...],
) -> float:
    if not points:
        raise ValueError("epsilon schedule must contain at least one point")
    if position <= points[0][0]:
        return points[0][1]
    for (left_step, left_value), (right_step, right_value) in zip(
        points,
        points[1:],
        strict=False,
    ):
        if position == right_step:
            return right_value
        if position < right_step:
            progress = (position - left_step) / max(right_step - left_step, 1e-12)
            return left_value + progress * (right_value - left_value)
    return points[-1][1]


def low_epsilon_at(
    config: TrainConfig,
    global_transitions: int,
) -> float:
    fraction = min(
        max(global_transitions / max(config.total_transitions, 1), 0.0),
        1.0,
    )
    return piecewise_epsilon(fraction, config.low_epsilon_fraction_points)


def _sample_low_target(_config: TrainConfig, rng: np.random.Generator) -> int:
    return int(rng.integers(0, BasicMathConfig.max_answer + 1))


def _reset_actor_env(
    env: BasicMathEnv,
    config: TrainConfig,
    rng: np.random.Generator,
    seed: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    options = None
    if config.task == "low":
        options = {"target_macro_action": _sample_low_target(config, rng)}
    return env.reset(seed=seed, options=options)


def _make_env(config: TrainConfig, render_mode: str | None = "rgb_array") -> BasicMathEnv:
    if config.task == "high":
        return BasicMathEnv(action_mode="macro", render_mode=render_mode)
    env_config = BasicMathConfig(distance_reward_scale=config.low_distance_reward_scale)
    return BasicMathEnv(
        action_mode="raw",
        goal_conditioned=True,
        render_mode=render_mode,
        config=env_config,
    )


def _copy_model(source: DuelingDQN, destination: DuelingDQN) -> None:
    destination.load_state_dict(source.state_dict())


def _publish_model(gpu_model: DuelingDQN, shared_model: DuelingDQN, lock: Any) -> None:
    with lock, torch.no_grad():
        for shared_parameter, gpu_parameter in zip(
            shared_model.parameters(), gpu_model.parameters(), strict=True
        ):
            shared_parameter.copy_(gpu_parameter.detach().cpu())
        for shared_buffer, gpu_buffer in zip(
            shared_model.buffers(), gpu_model.buffers(), strict=True
        ):
            shared_buffer.copy_(gpu_buffer.detach().cpu())


def _actor_process(
    actor_id: int,
    config: TrainConfig,
    shared_model: DuelingDQN,
    model_lock: Any,
    transition_queue: Any,
    stats_queue: Any,
    global_transitions: Any,
    one_digit_transitions: Any,
    two_digit_transitions: Any,
    stop_event: Any,
) -> None:
    torch.set_num_threads(1)
    actor_seed = config.seed + actor_id * 10_007
    random.seed(actor_seed)
    np.random.seed(actor_seed)
    rng = np.random.default_rng(actor_seed)

    local_model = DuelingDQN(config.network_spec).cpu().eval()
    with model_lock:
        _copy_model(shared_model, local_model)
    env = _make_env(config)
    accumulator = NStepAccumulator(config.n_step, config.gamma)
    local_steps = 0
    last_model_sync_step = 0

    try:
        observation, _ = _reset_actor_env(env, config, rng, seed=actor_seed)
        state = encode_observation(observation, config.goal_conditioned)
        target_macro_action = env.target_macro_action
        episode_return = 0.0
        episode_dense_return = 0.0
        episode_length = 0

        while not stop_event.is_set():
            current_global = int(global_transitions.value)
            if current_global >= config.total_transitions:
                break
            if config.task == "low":
                if target_macro_action is None:
                    raise RuntimeError("Low-level actor has no target macro action")
                epsilon = low_epsilon_at(config, current_global)
            else:
                epsilon = epsilon_at(current_global, config)

            if rng.random() < epsilon:
                action = int(rng.integers(config.network_spec.num_actions))
            else:
                images, macro = observation_to_tensors(state, "cpu")
                with torch.inference_mode():
                    action = int(local_model(images, macro).argmax(dim=1).item())

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            next_state = encode_observation(next_observation, config.goal_conditioned)
            transitions = accumulator.append(state, action, float(reward), next_state, done)
            for transition in transitions:
                while not stop_event.is_set():
                    try:
                        transition_queue.put(transition, timeout=0.2)
                        break
                    except queue.Full:
                        continue

            with global_transitions.get_lock():
                global_transitions.value += 1
                updated_global = int(global_transitions.value)
            if target_macro_action is not None:
                category_counter = (
                    one_digit_transitions
                    if target_macro_action < 10
                    else two_digit_transitions
                )
                with category_counter.get_lock():
                    category_counter.value += 1

            local_steps += 1
            episode_return += float(reward)
            episode_dense_return += float(info.get("distance_reward", 0.0))
            episode_length += 1
            state = next_state

            if done:
                stat = {
                    "actor_id": actor_id,
                    "global_transitions": updated_global,
                    "return": episode_return,
                    "dense_return": episode_dense_return,
                    "final_answer_distance": info.get("answer_distance"),
                    "length": episode_length,
                    "success": float(bool(info.get("success", reward > 0))),
                    "epsilon": epsilon,
                    "target_macro_action": info.get("target_macro_action"),
                    "problem_operands": info.get("problem_operands"),
                }
                try:
                    stats_queue.put_nowait(stat)
                except queue.Full:
                    pass
                if local_steps - last_model_sync_step >= config.actor_sync_interval:
                    with model_lock:
                        _copy_model(shared_model, local_model)
                    local_model.eval()
                    last_model_sync_step = local_steps
                observation, _ = _reset_actor_env(env, config, rng)
                state = encode_observation(observation, config.goal_conditioned)
                target_macro_action = env.target_macro_action
                episode_return = 0.0
                episode_dense_return = 0.0
                episode_length = 0
    finally:
        env.close()


def evaluate_policy(
    model: DuelingDQN,
    config: TrainConfig,
    episodes: int,
    seed: int,
    gui: bool = False,
    fps: float = 0.0,
) -> dict[str, float]:
    torch.set_num_threads(1)
    model = model.cpu().eval()
    env = _make_env(config, "human" if gui else "rgb_array")
    returns: list[float] = []
    successes: list[float] = []
    one_digit_successes: list[float] = []
    two_digit_successes: list[float] = []
    one_digit_returns: list[float] = []
    two_digit_returns: list[float] = []
    one_digit_lengths: list[float] = []
    two_digit_lengths: list[float] = []
    dense_returns: list[float] = []
    one_digit_dense_returns: list[float] = []
    two_digit_dense_returns: list[float] = []
    final_distances: list[float] = []
    one_digit_final_distances: list[float] = []
    two_digit_final_distances: list[float] = []
    problem_pairs: set[tuple[int, int]] = set()
    lengths: list[int] = []
    started = time.monotonic()

    try:
        for episode in range(episodes):
            options = None
            if config.task == "low":
                options = {
                    "target_macro_action": episode % (BasicMathConfig.max_answer + 1)
                }
            observation, _ = env.reset(
                seed=seed if episode == 0 else None,
                options=options,
            )
            state = encode_observation(observation, config.goal_conditioned)
            episode_return = 0.0
            episode_dense_return = 0.0
            length = 0
            info: dict[str, Any] = {}
            done = False
            while not done:
                images, macro = observation_to_tensors(state, "cpu")
                with torch.inference_mode():
                    action = int(model(images, macro).argmax(dim=1).item())
                observation, reward, terminated, truncated, info = env.step(action)
                state = encode_observation(observation, config.goal_conditioned)
                episode_return += float(reward)
                episode_dense_return += float(info.get("distance_reward", 0.0))
                length += 1
                done = bool(terminated or truncated)
                if fps > 0:
                    time.sleep(1.0 / fps)
            returns.append(episode_return)
            success = float(bool(info.get("success", episode_return > 0)))
            successes.append(success)
            problem_operands = info.get("problem_operands")
            if problem_operands is not None:
                problem_pairs.add(
                    (int(problem_operands[0]), int(problem_operands[1]))
                )
            if config.task == "low":
                target_macro_action = int(info["target_macro_action"])
                final_distance = float(info["answer_distance"])
                dense_returns.append(episode_dense_return)
                final_distances.append(final_distance)
                if target_macro_action < 10:
                    one_digit_successes.append(success)
                    one_digit_returns.append(episode_return)
                    one_digit_lengths.append(float(length))
                    one_digit_dense_returns.append(episode_dense_return)
                    one_digit_final_distances.append(final_distance)
                else:
                    two_digit_successes.append(success)
                    two_digit_returns.append(episode_return)
                    two_digit_lengths.append(float(length))
                    two_digit_dense_returns.append(episode_dense_return)
                    two_digit_final_distances.append(final_distance)
            lengths.append(length)
    finally:
        env.close()

    elapsed = max(time.monotonic() - started, 1e-6)
    metrics = {
        "success_rate": float(np.mean(successes)),
        "mean_return": float(np.mean(returns)),
        "mean_length": float(np.mean(lengths)),
        "episodes_per_second": episodes / elapsed,
        "unique_problem_count": float(len(problem_pairs)),
    }
    if config.task == "low":
        metrics["mean_dense_reward_return"] = (
            float(np.mean(dense_returns)) if dense_returns else 0.0
        )
        metrics["mean_final_answer_distance"] = (
            float(np.mean(final_distances)) if final_distances else 0.0
        )
        metrics["episodes_one_digit_sum"] = float(len(one_digit_successes))
        metrics["episodes_two_digit_sum"] = float(len(two_digit_successes))
        metrics["success_rate_one_digit_sum"] = (
            float(np.mean(one_digit_successes)) if one_digit_successes else 0.0
        )
        metrics["success_rate_two_digit_sum"] = (
            float(np.mean(two_digit_successes)) if two_digit_successes else 0.0
        )
        metrics["episode_return_one_digit_sum"] = (
            float(np.mean(one_digit_returns)) if one_digit_returns else 0.0
        )
        metrics["episode_return_two_digit_sum"] = (
            float(np.mean(two_digit_returns)) if two_digit_returns else 0.0
        )
        metrics["episode_length_one_digit_sum"] = (
            float(np.mean(one_digit_lengths)) if one_digit_lengths else 0.0
        )
        metrics["episode_length_two_digit_sum"] = (
            float(np.mean(two_digit_lengths)) if two_digit_lengths else 0.0
        )
        metrics["dense_reward_return_one_digit_sum"] = (
            float(np.mean(one_digit_dense_returns)) if one_digit_dense_returns else 0.0
        )
        metrics["dense_reward_return_two_digit_sum"] = (
            float(np.mean(two_digit_dense_returns)) if two_digit_dense_returns else 0.0
        )
        metrics["final_answer_distance_one_digit_sum"] = (
            float(np.mean(one_digit_final_distances))
            if one_digit_final_distances
            else 0.0
        )
        metrics["final_answer_distance_two_digit_sum"] = (
            float(np.mean(two_digit_final_distances))
            if two_digit_final_distances
            else 0.0
        )
    return metrics


def _evaluator_process(
    config: TrainConfig,
    shared_model: DuelingDQN,
    model_lock: Any,
    request_queue: Any,
    result_queue: Any,
    stop_event: Any,
) -> None:
    torch.set_num_threads(1)
    local_model = DuelingDQN(config.network_spec).cpu().eval()
    while not stop_event.is_set():
        try:
            request = request_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if request is None:
            break
        evaluation_step = int(request)
        with model_lock:
            _copy_model(shared_model, local_model)
        metrics = evaluate_policy(
            local_model,
            config,
            config.evaluation_episodes,
            seed=config.seed + 900_001 + evaluation_step,
        )
        result_queue.put((evaluation_step, metrics))


def _learn_batch(
    online_model: DuelingDQN,
    target_model: DuelingDQN,
    optimizer: torch.optim.Optimizer,
    replay: PrioritizedReplayBuffer,
    config: TrainConfig,
    beta: float,
    device: torch.device,
) -> dict[str, float]:
    batch = replay.sample(config.batch_size, beta)
    images, macro = batch_to_tensors(batch["pixels"], batch["macro"], device)
    next_images, next_macro = batch_to_tensors(
        batch["next_pixels"], batch["next_macro"], device
    )
    actions = torch.from_numpy(batch["actions"]).to(device=device, dtype=torch.int64)
    rewards = torch.from_numpy(batch["rewards"]).to(device=device, dtype=torch.float32)
    discounts = torch.from_numpy(batch["discounts"]).to(device=device, dtype=torch.float32)
    weights = torch.from_numpy(batch["weights"]).to(device=device, dtype=torch.float32)

    q_values = online_model(images, macro)
    chosen_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_actions = online_model(next_images, next_macro).argmax(dim=1)
        next_q = target_model(next_images, next_macro).gather(
            1, next_actions.unsqueeze(1)
        ).squeeze(1)
        targets = rewards + discounts * next_q

    td_errors = targets - chosen_q
    per_item_loss = F.smooth_l1_loss(chosen_q, targets, reduction="none")
    loss = (weights * per_item_loss).mean()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = clip_grad_norm_(online_model.parameters(), config.max_grad_norm)
    optimizer.step()
    replay.update_priorities(batch["indices"], td_errors.detach().cpu().numpy())

    return {
        "loss": float(loss.item()),
        "q_mean": float(chosen_q.mean().item()),
        "q_max": float(q_values.max().item()),
        "target_mean": float(targets.mean().item()),
        "td_error_mean": float(td_errors.abs().mean().item()),
        "grad_norm": float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
        "beta": beta,
    }


def _cpu_state_dict(model: DuelingDQN) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def _save_checkpoint(
    path: Path,
    online_model: DuelingDQN,
    target_model: DuelingDQN,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    global_transitions: int,
    learner_updates: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "task": config.task,
            "config": asdict(config),
            "network_spec": config.network_spec.to_dict(),
            "model": _cpu_state_dict(online_model),
            "target_model": _cpu_state_dict(target_model),
            "optimizer": optimizer.state_dict(),
            "global_transitions": global_transitions,
            "learner_updates": learner_updates,
        },
        temporary,
    )
    os.replace(temporary, path)


def load_checkpoint(checkpoint_path: str | Path) -> tuple[DuelingDQN, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    spec_data = dict(checkpoint["network_spec"])
    if spec_data.pop("noisy", False):
        raise ValueError(
            "This checkpoint uses the discontinued NoisyNet architecture and "
            "cannot be loaded by the current standard Dueling DQN"
        )
    spec = NetworkSpec(**spec_data)
    model = DuelingDQN(spec)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def run_training(config: TrainConfig) -> Path:
    if config.task not in {"high", "low"}:
        raise ValueError(f"Unsupported task: {config.task}")
    if config.low_distance_reward_scale < 0.0:
        raise ValueError("low_distance_reward_scale must be non-negative")

    mp.set_start_method("spawn", force=True)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.task}_{timestamp}_pid{os.getpid()}"
    run_dir = Path(config.runs_dir) / run_name
    checkpoint_dir = Path(config.checkpoints_dir) / run_name
    writer = SummaryWriter(log_dir=run_dir)

    # The learner intentionally binds directly to CUDA. Actors and evaluator
    # only receive CPU models and never inspect or initialize CUDA.
    device = torch.device("cuda")
    online_model = DuelingDQN(config.network_spec).to(device)
    target_model = DuelingDQN(config.network_spec).to(device)
    target_model.load_state_dict(online_model.state_dict())
    target_model.eval()
    optimizer = torch.optim.Adam(online_model.parameters(), lr=config.learning_rate)

    context = mp.get_context("spawn")
    shared_model = DuelingDQN(config.network_spec).cpu()
    shared_model.load_state_dict(_cpu_state_dict(online_model))
    shared_model.share_memory()
    model_lock = context.Lock()
    transition_queue = context.Queue(maxsize=config.transition_queue_size)
    stats_queue = context.Queue(maxsize=10_000)
    global_transitions = context.Value("q", 0)
    one_digit_transitions = context.Value("q", 0)
    two_digit_transitions = context.Value("q", 0)
    stop_event = context.Event()
    eval_stop_event = context.Event()
    eval_request_queue = context.Queue()
    eval_result_queue = context.Queue()

    actors = [
        context.Process(
            target=_actor_process,
            args=(
                actor_id,
                config,
                shared_model,
                model_lock,
                transition_queue,
                stats_queue,
                global_transitions,
                one_digit_transitions,
                two_digit_transitions,
                stop_event,
            ),
            name=f"actor-{actor_id}",
        )
        for actor_id in range(config.num_actors)
    ]
    evaluator = context.Process(
        target=_evaluator_process,
        args=(
            config,
            shared_model,
            model_lock,
            eval_request_queue,
            eval_result_queue,
            eval_stop_event,
        ),
        name="greedy-evaluator",
    )

    for actor in actors:
        actor.start()
    evaluator.start()

    replay = PrioritizedReplayBuffer(
        config.replay_capacity,
        alpha=config.per_alpha,
        seed=config.seed,
    )
    returns_window: deque[float] = deque(maxlen=200)
    successes_window: deque[float] = deque(maxlen=200)
    one_digit_successes_window: deque[float] = deque(maxlen=200)
    two_digit_successes_window: deque[float] = deque(maxlen=200)
    one_digit_returns_window: deque[float] = deque(maxlen=200)
    two_digit_returns_window: deque[float] = deque(maxlen=200)
    one_digit_lengths_window: deque[float] = deque(maxlen=200)
    two_digit_lengths_window: deque[float] = deque(maxlen=200)
    one_digit_dense_returns_window: deque[float] = deque(maxlen=200)
    two_digit_dense_returns_window: deque[float] = deque(maxlen=200)
    one_digit_final_distances_window: deque[float] = deque(maxlen=200)
    two_digit_final_distances_window: deque[float] = deque(maxlen=200)
    epsilons_window: deque[float] = deque(maxlen=200)
    lengths_window: deque[float] = deque(maxlen=200)
    learner_updates = 0
    update_budget = 0.0
    next_report = config.report_interval
    next_dqn_log = config.dqn_log_interval
    next_checkpoint = config.checkpoint_interval
    next_evaluation = config.evaluation_interval
    pending_evaluations: set[int] = set()
    last_report_step = 0
    last_report_time = time.monotonic()
    latest_dqn_metrics: dict[str, float] = {}
    one_digit_episodes = 0
    two_digit_episodes = 0
    problem_pairs_seen: set[tuple[int, int]] = set()
    interrupted = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True
        stop_event.set()

    old_sigint = signal.signal(signal.SIGINT, request_stop)
    old_sigterm = signal.signal(signal.SIGTERM, request_stop)

    try:
        while not stop_event.is_set():
            try:
                first_transition: Transition = transition_queue.get(timeout=0.2)
                transitions = [first_transition]
                for _ in range(127):
                    try:
                        transitions.append(transition_queue.get_nowait())
                    except queue.Empty:
                        break
                for transition in transitions:
                    replay.add(transition)
                if len(replay) >= config.replay_warmup:
                    update_budget += len(transitions) * config.updates_per_transition
            except queue.Empty:
                transitions = []

            global_step = int(global_transitions.value)
            if global_step >= config.total_transitions:
                stop_event.set()

            while update_budget >= 1.0 and len(replay) >= config.replay_warmup:
                beta_progress = min(global_step / max(config.total_transitions, 1), 1.0)
                beta = config.per_beta_start + beta_progress * (1.0 - config.per_beta_start)
                latest_dqn_metrics = _learn_batch(
                    online_model,
                    target_model,
                    optimizer,
                    replay,
                    config,
                    beta,
                    device,
                )
                learner_updates += 1
                update_budget -= 1.0

                if learner_updates % config.target_update_interval == 0:
                    target_model.load_state_dict(online_model.state_dict())
                if learner_updates % config.shared_model_update_interval == 0:
                    _publish_model(online_model, shared_model, model_lock)
            while True:
                try:
                    stat = stats_queue.get_nowait()
                except queue.Empty:
                    break
                returns_window.append(float(stat["return"]))
                success = float(stat["success"])
                successes_window.append(success)
                epsilons_window.append(float(stat["epsilon"]))
                lengths_window.append(float(stat["length"]))
                target_macro_action = stat.get("target_macro_action")
                problem_operands = stat.get("problem_operands")
                if problem_operands is not None:
                    problem_pairs_seen.add(
                        (int(problem_operands[0]), int(problem_operands[1]))
                    )
                if target_macro_action is not None:
                    if int(target_macro_action) < 10:
                        one_digit_episodes += 1
                        one_digit_successes_window.append(success)
                        one_digit_returns_window.append(float(stat["return"]))
                        one_digit_lengths_window.append(float(stat["length"]))
                        one_digit_dense_returns_window.append(float(stat["dense_return"]))
                        one_digit_final_distances_window.append(
                            float(stat["final_answer_distance"])
                        )
                    else:
                        two_digit_episodes += 1
                        two_digit_successes_window.append(success)
                        two_digit_returns_window.append(float(stat["return"]))
                        two_digit_lengths_window.append(float(stat["length"]))
                        two_digit_dense_returns_window.append(float(stat["dense_return"]))
                        two_digit_final_distances_window.append(
                            float(stat["final_answer_distance"])
                        )

            if latest_dqn_metrics and global_step >= next_dqn_log:
                # Log once at the most recent 10k-aligned threshold. Do not
                # backfill missed thresholds with duplicate metrics.
                dqn_log_step = (global_step // config.dqn_log_interval) * config.dqn_log_interval
                for name, value in latest_dqn_metrics.items():
                    writer.add_scalar(f"dqn/{name}", value, dqn_log_step)
                writer.add_scalar("dqn/replay_size", len(replay), dqn_log_step)
                writer.add_scalar("dqn/learner_updates", learner_updates, dqn_log_step)
                next_dqn_log = dqn_log_step + config.dqn_log_interval

            while True:
                try:
                    evaluation_step, metrics = eval_result_queue.get_nowait()
                except queue.Empty:
                    break
                pending_evaluations.discard(int(evaluation_step))
                for name, value in metrics.items():
                    writer.add_scalar(f"eval/{name}", value, int(evaluation_step))
                print(
                    f"[eval step={evaluation_step}] success={metrics['success_rate']:.4f} "
                    f"return={metrics['mean_return']:.4f} length={metrics['mean_length']:.2f}",
                    flush=True,
                )

            while global_step >= next_evaluation:
                _publish_model(online_model, shared_model, model_lock)
                eval_request_queue.put(next_evaluation)
                pending_evaluations.add(next_evaluation)
                next_evaluation += config.evaluation_interval

            while global_step >= next_checkpoint:
                checkpoint_path = checkpoint_dir / f"step_{next_checkpoint:012d}.pt"
                _save_checkpoint(
                    checkpoint_path,
                    online_model,
                    target_model,
                    optimizer,
                    config,
                    global_step,
                    learner_updates,
                )
                print(f"[checkpoint] {checkpoint_path}", flush=True)
                next_checkpoint += config.checkpoint_interval

            if global_step >= next_report:
                now = time.monotonic()
                throughput = (global_step - last_report_step) / max(now - last_report_time, 1e-6)
                rollout_return = float(np.mean(returns_window)) if returns_window else 0.0
                rollout_success = float(np.mean(successes_window)) if successes_window else 0.0
                rollout_length = float(np.mean(lengths_window)) if lengths_window else 0.0
                epsilon = (
                    float(np.mean(epsilons_window))
                    if epsilons_window
                    else (
                        low_epsilon_at(config, global_step)
                        if config.task == "low"
                        else epsilon_at(global_step, config)
                    )
                )
                writer.add_scalar("rollout/success_rate", rollout_success, global_step)
                if config.task == "low":
                    one_digit_step = int(one_digit_transitions.value)
                    two_digit_step = int(two_digit_transitions.value)
                    writer.add_scalar(
                        "exploration/transitions_one_digit_sum",
                        one_digit_step,
                        global_step,
                    )
                    writer.add_scalar(
                        "exploration/transitions_two_digit_sum",
                        two_digit_step,
                        global_step,
                    )
                    writer.add_scalar(
                        "rollout/episodes_one_digit_sum",
                        one_digit_episodes,
                        global_step,
                    )
                    writer.add_scalar(
                        "rollout/episodes_two_digit_sum",
                        two_digit_episodes,
                        global_step,
                    )
                    one_digit_success = (
                        float(np.mean(one_digit_successes_window))
                        if one_digit_successes_window
                        else 0.0
                    )
                    two_digit_success = (
                        float(np.mean(two_digit_successes_window))
                        if two_digit_successes_window
                        else 0.0
                    )
                    writer.add_scalar(
                        "rollout/success_rate_one_digit_sum",
                        one_digit_success,
                        global_step,
                    )
                    writer.add_scalar(
                        "rollout/success_rate_two_digit_sum",
                        two_digit_success,
                        global_step,
                    )
                    writer.add_scalar(
                        "rollout/episode_return_one_digit_sum",
                        float(np.mean(one_digit_returns_window))
                        if one_digit_returns_window
                        else 0.0,
                        global_step,
                    )
                    writer.add_scalar(
                        "rollout/episode_return_two_digit_sum",
                        float(np.mean(two_digit_returns_window))
                        if two_digit_returns_window
                        else 0.0,
                        global_step,
                    )
                    writer.add_scalar(
                        "rollout/episode_length_one_digit_sum",
                        float(np.mean(one_digit_lengths_window))
                        if one_digit_lengths_window
                        else 0.0,
                        global_step,
                    )
                    writer.add_scalar(
                        "rollout/episode_length_two_digit_sum",
                        float(np.mean(two_digit_lengths_window))
                        if two_digit_lengths_window
                        else 0.0,
                        global_step,
                    )
                    writer.add_scalar(
                        "rollout/dense_reward_return_one_digit_sum",
                        float(np.mean(one_digit_dense_returns_window))
                        if one_digit_dense_returns_window
                        else 0.0,
                        global_step,
                    )
                    writer.add_scalar(
                        "rollout/dense_reward_return_two_digit_sum",
                        float(np.mean(two_digit_dense_returns_window))
                        if two_digit_dense_returns_window
                        else 0.0,
                        global_step,
                    )
                    writer.add_scalar(
                        "rollout/final_answer_distance_one_digit_sum",
                        float(np.mean(one_digit_final_distances_window))
                        if one_digit_final_distances_window
                        else 0.0,
                        global_step,
                    )
                    writer.add_scalar(
                        "rollout/final_answer_distance_two_digit_sum",
                        float(np.mean(two_digit_final_distances_window))
                        if two_digit_final_distances_window
                        else 0.0,
                        global_step,
                    )
                writer.add_scalar(
                    "rollout/unique_problem_count",
                    len(problem_pairs_seen),
                    global_step,
                )
                writer.add_scalar("rollout/episode_return", rollout_return, global_step)
                writer.add_scalar("rollout/episode_length", rollout_length, global_step)
                writer.add_scalar("system/transitions_per_second", throughput, global_step)
                writer.add_scalar("exploration/epsilon", epsilon, global_step)
                print(
                    f"[train step={global_step}] success={rollout_success:.4f} "
                    f"return={rollout_return:.4f} eps={epsilon:.4f} "
                    f"throughput={throughput:.1f}/s replay={len(replay)} updates={learner_updates}",
                    flush=True,
                )
                last_report_step = global_step
                last_report_time = now
                skipped_reports = (global_step - next_report) // config.report_interval + 1
                next_report += skipped_reports * config.report_interval

            failed_actors = [
                actor for actor in actors if actor.exitcode not in {None, 0}
            ]
            if failed_actors and global_step < config.total_transitions:
                failures = ", ".join(
                    f"{actor.name}(exit={actor.exitcode})" for actor in failed_actors
                )
                raise RuntimeError(f"Rollout actors failed before training completed: {failures}")

        # Actors can reach the global limit while n-step transitions are still
        # buffered in the multiprocessing queue. Preserve those samples before
        # writing the final checkpoint.
        for actor in actors:
            actor.join(timeout=5)
        drained_transitions = 0
        while True:
            try:
                replay.add(transition_queue.get_nowait())
                drained_transitions += 1
            except queue.Empty:
                break
        if len(replay) >= config.replay_warmup:
            update_budget += drained_transitions * config.updates_per_transition
        while update_budget >= 1.0 and len(replay) >= config.replay_warmup:
            final_global_step = int(global_transitions.value)
            beta_progress = min(final_global_step / max(config.total_transitions, 1), 1.0)
            beta = config.per_beta_start + beta_progress * (1.0 - config.per_beta_start)
            latest_dqn_metrics = _learn_batch(
                online_model,
                target_model,
                optimizer,
                replay,
                config,
                beta,
                device,
            )
            learner_updates += 1
            update_budget -= 1.0
            if learner_updates % config.target_update_interval == 0:
                target_model.load_state_dict(online_model.state_dict())

        _publish_model(online_model, shared_model, model_lock)
        final_step = int(global_transitions.value)
        final_path = checkpoint_dir / f"final_step_{final_step:012d}.pt"
        _save_checkpoint(
            final_path,
            online_model,
            target_model,
            optimizer,
            config,
            final_step,
            learner_updates,
        )

        deadline = time.monotonic() + 120.0
        while pending_evaluations and time.monotonic() < deadline:
            try:
                evaluation_step, metrics = eval_result_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            pending_evaluations.discard(int(evaluation_step))
            for name, value in metrics.items():
                writer.add_scalar(f"eval/{name}", value, int(evaluation_step))
        writer.flush()
        return final_path
    finally:
        stop_event.set()
        eval_stop_event.set()
        try:
            eval_request_queue.put_nowait(None)
        except queue.Full:
            pass
        for actor in actors:
            actor.join(timeout=10)
            if actor.is_alive():
                actor.terminate()
                actor.join(timeout=5)
        evaluator.join(timeout=10)
        if evaluator.is_alive():
            evaluator.terminate()
            evaluator.join(timeout=5)
        transition_queue.close()
        stats_queue.close()
        eval_request_queue.close()
        eval_result_queue.close()
        writer.close()
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        if interrupted:
            print("Training interrupted; final checkpoint saved when possible.", flush=True)
