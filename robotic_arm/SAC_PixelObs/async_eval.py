"""Asynchronous CPU-only evaluation for visual SAC training."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from .callbacks import write_tensorboard_scalars
from .utils import make_env_factory


def _put_latest(request_queue, request: Optional[Dict[str, Any]]) -> int:
    """Put a request in a one-slot queue, replacing any stale pending item."""

    dropped = 0
    while True:
        try:
            request_queue.put_nowait(request)
            return dropped
        except queue.Full:
            try:
                stale = request_queue.get_nowait()
                dropped += int(stale is not None)
            except queue.Empty:
                # multiprocessing.Queue uses a feeder thread; Full and Empty
                # can briefly overlap while ownership is being transferred.
                time.sleep(0.001)


def _evaluate_checkpoint_worker(request_queue, result_queue) -> None:
    """Process target: load and evaluate checkpoints entirely on CPU."""

    # This is intentionally set before constructing/loading the model. The
    # explicit ``device='cpu'`` below is the authoritative safeguard; hiding
    # CUDA here also prevents accidental CUDA context creation in this worker.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        import torch

        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    except ImportError:  # pragma: no cover - torch is required by SB3
        pass

    while True:
        request = request_queue.get()
        if request is None:
            return
        try:
            result = _run_evaluation(request)
            result_queue.put(result)
        except Exception as exc:  # return failures to the learner process
            result_queue.put(
                {
                    "ok": False,
                    "step": int(request["step"]),
                    "checkpoint": request["checkpoint"],
                    "error": repr(exc),
                }
            )


def _run_evaluation(request: Dict[str, Any]) -> Dict[str, Any]:
    checkpoint = str(request["checkpoint"])
    env = DummyVecEnv(
        [
            make_env_factory(
                task=request["task"],
                rank=0,
                seed=int(request["seed"]),
                image_size=int(request["image_size"]),
                frame_stack=int(request["frame_stack"]),
                max_episode_steps=int(request["max_episode_steps"]),
                action_repeat=int(request["action_repeat"]),
                camera_scale=float(request["camera_scale"]),
            )
        ]
    )
    try:
        # Never use the learner's CUDA device for evaluation. This process is
        # deliberately CPU-only so it cannot contend with the training GPU.
        model = SAC.load(checkpoint, env=env, device="cpu")
        observation = env.reset()
        episodes = int(request["episodes"])
        successes = []
        rewards = []
        lengths = []
        final_stages = []
        lifts = []
        completed = 0
        episode_reward = 0.0
        while completed < episodes:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, dones, infos = env.step(action)
            episode_reward += float(reward[0])
            if dones[0]:
                info = infos[0]
                episode = info.get("episode", {})
                successes.append(float(info.get("success", False)))
                lifts.append(float(info.get("ever_lifted", False)))
                rewards.append(float(episode_reward))
                lengths.append(float(episode.get("l", 0)))
                final_stages.append(float(info.get("stage_index", -1)))
                completed += 1
                episode_reward = 0.0
        return {
            "ok": True,
            "task": str(request["task"]),
            "step": int(request["step"]),
            "checkpoint": checkpoint,
            "success_rate": float(np.mean(successes)),
            "mean_reward": float(np.mean(rewards)),
            "mean_ep_length": float(np.mean(lengths)),
            "lift_rate": float(np.mean(lifts)),
            "mean_final_stage": float(np.mean(final_stages)),
            "episodes": episodes,
        }
    finally:
        env.close()


class AsyncCpuEvalCallback(BaseCallback):
    """Evaluate saved model snapshots in one persistent CPU worker.

    The learner only performs a quick ``model.save`` and queue operation at an
    evaluation interval. CNN inference and PyBullet stepping happen in the
    standby process, so the training process continues collecting rollouts
    and updating SAC while evaluation is running.
    """

    def __init__(
        self,
        *,
        eval_freq: int,
        n_eval_episodes: int,
        checkpoint_dir: Path,
        eval_dir: Path,
        task: str,
        seed: int,
        image_size: int,
        frame_stack: int,
        max_episode_steps: int,
        action_repeat: int,
        camera_scale: float,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.eval_dir = Path(eval_dir)
        self.task = task
        self.seed = int(seed)
        self.image_size = int(image_size)
        self.frame_stack = int(frame_stack)
        self.max_episode_steps = int(max_episode_steps)
        self.action_repeat = int(action_repeat)
        self.camera_scale = float(camera_scale)
        self._context: Optional[mp.context.BaseContext] = None
        self._requests = None
        self._results = None
        self._worker: Optional[mp.Process] = None
        self._next_eval = self.eval_freq
        self._best_success = -float("inf")
        self._records = []

    def _init_callback(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        # Spawn avoids inheriting the learner's CUDA context. It also works
        # when the training SubprocVecEnv uses a different start method.
        self._context = mp.get_context("spawn")
        # At most one evaluation may wait behind the one currently running.
        # Newer checkpoints replace stale pending work rather than building an
        # unbounded queue that reports obsolete policies hours later.
        self._requests = self._context.Queue(maxsize=1)
        self._results = self._context.Queue()
        self._worker = self._context.Process(
            target=_evaluate_checkpoint_worker,
            args=(self._requests, self._results),
            name="pixelobs-evaluator",
            daemon=True,
        )
        self._worker.start()
        if self.verbose:
            print("async evaluator started: CPU-only standby process")

    def _on_step(self) -> bool:
        self._drain_results()
        if self.num_timesteps >= self._next_eval:
            self._enqueue_evaluation()
            self._next_eval += self.eval_freq
        return True

    def _enqueue_evaluation(self) -> None:
        step = int(self.num_timesteps)
        checkpoint = self.checkpoint_dir / f"{self.task}_pixel_sac_{step}_steps.zip"
        # A snapshot is required because the learner keeps changing while the
        # CPU worker evaluates. This is separate from the regular checkpoint
        # callback and is safe to overwrite only at a unique timestep.
        self.model.save(str(checkpoint))
        request = {
            "step": step,
            "checkpoint": str(checkpoint),
            "task": self.task,
            # Every checkpoint sees the same deterministic episode sequence.
            # This makes changes in eval/* attributable to the policy rather
            # than to a different random object/goal sample set.
            "seed": self.seed + 100_000,
            "episodes": self.n_eval_episodes,
            "image_size": self.image_size,
            "frame_stack": self.frame_stack,
            "max_episode_steps": self.max_episode_steps,
            "action_repeat": self.action_repeat,
            "camera_scale": self.camera_scale,
        }
        dropped = _put_latest(self._requests, request)
        if self.verbose:
            suffix = f" replaced={dropped}" if dropped else ""
            print(
                f"queued async evaluation: step={step} checkpoint={checkpoint}"
                f"{suffix}"
            )

    def _drain_results(self) -> None:
        if self._results is None:
            return
        while True:
            try:
                result = self._results.get_nowait()
            except queue.Empty:
                return
            if not result.get("ok", False):
                print(
                    f"async evaluation failed at step={result.get('step')}: "
                    f"{result.get('error')}"
                )
                continue
            self._record_result(result)

    def _record_result(self, result: Dict[str, Any]) -> None:
        step = int(result["step"])
        success = float(result["success_rate"])
        metrics = {
            "eval/success_rate": success,
            "eval/mean_reward": float(result["mean_reward"]),
            "eval/mean_ep_length": float(result["mean_ep_length"]),
            "eval/step": float(step),
        }
        if result.get("task", self.task) == "pick_place":
            metrics["eval/lift_rate"] = float(result["lift_rate"])
            metrics["eval/mean_final_stage"] = float(result["mean_final_stage"])
        if not write_tensorboard_scalars(self.logger, metrics, step):
            for name, value in metrics.items():
                self.logger.record(name, value)
        self._records.append(result)
        with (self.eval_dir / "results.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")
        np.savez(
            self.eval_dir / "evaluations.npz",
            timesteps=np.asarray([item["step"] for item in self._records], dtype=np.int64),
            success_rates=np.asarray(
                [item["success_rate"] for item in self._records], dtype=np.float32
            ),
            mean_rewards=np.asarray(
                [item["mean_reward"] for item in self._records], dtype=np.float32
            ),
            mean_ep_lengths=np.asarray(
                [item["mean_ep_length"] for item in self._records], dtype=np.float32
            ),
        )
        if success > self._best_success:
            self._best_success = success
            best_path = self.eval_dir.parent / "best_model" / "best_model.zip"
            best_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(result["checkpoint"], best_path)
            if self.verbose:
                print(f"new best async eval: step={step} success_rate={success:.3f}")

    def _on_training_end(self) -> None:
        self._drain_results()
        if self._requests is not None:
            _put_latest(self._requests, None)
        if self._worker is not None:
            self._worker.join(timeout=3.0)
            if self._worker.is_alive():
                self._worker.terminate()
                self._worker.join(timeout=1.0)
        self._drain_results()
        if self.verbose:
            print("async evaluator stopped")


__all__ = ["AsyncCpuEvalCallback", "_put_latest"]
