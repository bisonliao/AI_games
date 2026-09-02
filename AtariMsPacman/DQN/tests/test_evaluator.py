from dataclasses import replace
import multiprocessing as mp
from queue import Empty

import torch

from DQN.config import DQNConfig
from DQN.evaluator import evaluator_process
from DQN.messages import EvaluationRequest, EvaluationResult, EvaluatorStop
from DQN.network import DuelingQNetwork


def test_async_cpu_evaluator_runs_a_complete_greedy_episode(tmp_path) -> None:
    base = DQNConfig()
    config = replace(
        base,
        actor_env=replace(
            base.actor_env,
            num_envs=1,
            noop_max=0,
            repeat_action_probability=0.0,
        ),
        evaluation_episodes=1,
        learner_device="cpu",
    )
    model = DuelingQNetwork(config.observation_shape, config.action_count)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.advantage_stream[-1].bias[1] = 1.0  # Greedy action is always UP.

    checkpoint_path = tmp_path / "constant_up.pt"
    torch.save({"online_state_dict": model.state_dict()}, checkpoint_path)
    context = mp.get_context("spawn")
    request_queue = context.Queue()
    result_queue = context.Queue()
    error_queue = context.Queue()
    process = context.Process(
        target=evaluator_process,
        args=(config, request_queue, result_queue, error_queue),
    )
    process.start()
    request_queue.put(EvaluationRequest(str(checkpoint_path), 123))
    request_queue.put(EvaluatorStop())
    result: EvaluationResult = result_queue.get(timeout=30.0)
    process.join(timeout=10.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5.0)

    try:
        error = error_queue.get_nowait()
    except Empty:
        error = None
    assert error is None
    assert process.exitcode == 0
    assert result.checkpoint_transition == 123
    assert len(result.episode_lengths) == 1
    assert result.episode_lengths[0] > 0
    assert len(result.episode_returns) == 1
    assert len(result.episode_raw_scores) == 1
    assert result.capped_episodes == 0
    assert result.episode_returns[0] != result.episode_raw_scores[0]


def test_evaluator_caps_non_terminating_episode_and_reports_it(tmp_path) -> None:
    base = DQNConfig()
    config = replace(
        base,
        actor_env=replace(
            base.actor_env,
            num_envs=1,
            noop_max=0,
            repeat_action_probability=0.0,
        ),
        evaluation_episodes=1,
        evaluation_max_episode_steps=5,
        learner_device="cpu",
    )
    model = DuelingQNetwork(config.observation_shape, config.action_count)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()  # Greedy action 0 keeps the game at its start screen.

    checkpoint_path = tmp_path / "constant_noop.pt"
    torch.save({"online_state_dict": model.state_dict()}, checkpoint_path)
    context = mp.get_context("spawn")
    request_queue = context.Queue()
    result_queue = context.Queue()
    error_queue = context.Queue()
    process = context.Process(
        target=evaluator_process,
        args=(config, request_queue, result_queue, error_queue),
    )
    process.start()
    request_queue.put(EvaluationRequest(str(checkpoint_path), 456))
    request_queue.put(EvaluatorStop())
    result: EvaluationResult = result_queue.get(timeout=30.0)
    process.join(timeout=10.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5.0)

    assert process.exitcode == 0
    assert result.episode_lengths == [5]
    assert result.capped_episodes == 1
