"""Launch CPU Actors, one GPU Learner, and one asynchronous CPU Evaluator."""

from __future__ import annotations

import argparse
from dataclasses import replace
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import sys
import time

from DQN.actor import actor_process
from DQN.config import DQNConfig
from DQN.evaluator import evaluator_process
from DQN.learner import learner_process
from DQN.messages import EvaluatorStop, ProcessError


def build_processes(config: DQNConfig):
    context = mp.get_context("spawn")
    rollout_queue = context.Queue(maxsize=config.rollout_queue_capacity)
    metrics_queue = context.Queue(maxsize=config.metrics_queue_capacity)
    parameter_queues = [context.Queue(maxsize=1) for _ in range(config.num_actors)]
    evaluation_request_queue = context.Queue()
    evaluation_result_queue = context.Queue()
    error_queue = context.Queue()
    global_transition_counter = context.Value("q", 0)
    stop_event = context.Event()

    evaluator = context.Process(
        name="evaluator",
        target=evaluator_process,
        args=(
            config,
            evaluation_request_queue,
            evaluation_result_queue,
            error_queue,
        ),
    )
    learner = context.Process(
        name="learner",
        target=learner_process,
        args=(
            config,
            rollout_queue,
            metrics_queue,
            parameter_queues,
            evaluation_request_queue,
            evaluation_result_queue,
            global_transition_counter,
            stop_event,
            error_queue,
        ),
    )
    actors = [
        context.Process(
            name=f"actor-{actor_id}",
            target=actor_process,
            args=(
                actor_id,
                config,
                rollout_queue,
                metrics_queue,
                parameter_queues[actor_id],
                global_transition_counter,
                stop_event,
                error_queue,
            ),
        )
        for actor_id in range(config.num_actors)
    ]
    return (
        evaluator,
        learner,
        actors,
        evaluation_request_queue,
        error_queue,
        stop_event,
        (
            rollout_queue,
            metrics_queue,
            parameter_queues,
            evaluation_result_queue,
            global_transition_counter,
        ),
    )


def stop_processes(processes, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    for process in processes:
        remaining = max(deadline - time.monotonic(), 0.0)
        process.join(timeout=remaining)
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=5.0)


def run_training(config: DQNConfig) -> None:
    (
        evaluator,
        learner,
        actors,
        evaluation_request_queue,
        error_queue,
        stop_event,
        _ipc_resources,
    ) = build_processes(config)
    # Keep every Queue/Value alive until spawn has rebuilt all synchronization
    # primitives and every process has exited.
    processes = [evaluator, learner, *actors]
    evaluator.start()
    learner.start()
    for actor in actors:
        actor.start()

    failure: ProcessError | None = None
    try:
        while learner.is_alive():
            try:
                failure = error_queue.get(timeout=1.0)
            except Empty:
                for process in [evaluator, *actors]:
                    if (
                        not process.is_alive()
                        and process.exitcode is not None
                        and not stop_event.is_set()
                    ):
                        failure = ProcessError(
                            process_name=process.name,
                            traceback=(
                                f"Process exited unexpectedly with code "
                                f"{process.exitcode}"
                            ),
                        )
                        stop_event.set()
                        break
                if failure is None:
                    continue
                break
            else:
                stop_event.set()
                break
        learner.join(timeout=1.0)
        if failure is None:
            try:
                failure = error_queue.get_nowait()
            except Empty:
                pass
        if failure is None and learner.exitcode not in (0, None):
            failure = ProcessError(
                process_name="learner",
                traceback=f"Learner exited with code {learner.exitcode}",
            )
    except KeyboardInterrupt:
        print("Interrupted; stopping training processes...", file=sys.stderr)
        stop_event.set()
    finally:
        stop_event.set()
        if evaluator.is_alive():
            evaluation_request_queue.put(EvaluatorStop())
        stop_processes(processes, config.shutdown_timeout_seconds)

    if failure is not None:
        raise RuntimeError(
            f"{failure.process_name} failed:\n{failure.traceback}"
        )


def parse_config() -> DQNConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-transitions", type=int)
    parser.add_argument("--num-actors", type=int)
    parser.add_argument("--envs-per-actor", type=int)
    parser.add_argument("--learner-device")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    config = DQNConfig()
    actor_env = config.actor_env
    if args.envs_per_actor is not None:
        actor_env = replace(actor_env, num_envs=args.envs_per_actor)
    overrides = {"actor_env": actor_env}
    if args.total_transitions is not None:
        overrides["total_transitions"] = args.total_transitions
    if args.num_actors is not None:
        overrides["num_actors"] = args.num_actors
    if args.learner_device is not None:
        overrides["learner_device"] = args.learner_device
    if args.resume is not None:
        overrides["resume_checkpoint"] = args.resume.resolve()
    return replace(config, **overrides)


def main() -> None:
    mp.freeze_support()
    run_training(parse_config())


if __name__ == "__main__":
    main()
