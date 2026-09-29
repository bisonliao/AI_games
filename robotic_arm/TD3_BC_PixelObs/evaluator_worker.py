"""Spawn-safe asynchronous checkpoint evaluator."""
from __future__ import annotations

from pathlib import Path

from .config import EnvConfig, setup
from .evaluate import evaluate_checkpoint


STOP = "__TD3_BC_STOP__"


def worker_main(job_queue, result_queue, config_dict: dict, device: str = "cpu", threads: int = 2) -> None:
    config = EnvConfig(**config_dict)
    setup(0, device, threads=threads)
    while True:
        job = job_queue.get()
        if job == STOP:
            return
        try:
            result = evaluate_checkpoint(
                Path(job["checkpoint"]), config, int(job["episodes"]),
                int(job["seed_start"]), device=device, randomize=bool(job.get("randomize", False)),
            )
            result.update({"sampled_transitions": job["sampled_transitions"], "ok": True})
        except Exception as exc:  # report failures to the learner process
            result = {
                "checkpoint": job["checkpoint"],
                "sampled_transitions": job["sampled_transitions"],
                "ok": False,
                "error": repr(exc),
            }
        result_queue.put(result)
