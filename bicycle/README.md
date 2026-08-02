# Bicycle RL

PyBullet bicycle-balancing environment and a distributed DQN
trainer. The agent controls a reaction wheel while the bicycle is driven at
approximately 2 m/s through seeded, smooth cross-wind gusts.

Install the local packages, inspect the environment, and run the physics gates:

```bash
python -m pip install -e '.[test]'
python -m env.demo --headless --episodes 1
python -m env.baseline --episodes 100
pytest -q
```

Start the default 4 actor x 4 environment CUDA run:

```bash
python -m dqn.train
tensorboard --logdir=runs --host=0.0.0.0 --port=6008
```

Each training process creates a sortable, unique directory such as
`runs/20260802-121530_distributed-dqn_pid12345/`. The timestamp contains year, month,
day, hour, minute, and second; the algorithm and process ID make concurrent
runs easy to distinguish. Pass `--run-dir PATH` only when an exact directory
is desired, or `--runs-root PATH` to change the history root.

This project installs a TensorBoard-compatible startup shim under both
`tensorboard` and `bicycle-tensorboard`. It bounds TensorBoard 2.21's blocking
localhost port probe, which can otherwise take more than a minute under some
WSL networking configurations. The full equivalent command is:

```bash
bicycle-tensorboard --logdir=runs \
  --host=0.0.0.0 --port=6008
```

Evaluate or resume a checkpoint:

```bash
RUN_DIR=runs/20260802-121530_distributed-dqn_pid12345
python -m dqn.evaluate --checkpoint "$RUN_DIR/checkpoints/best.pt" --episodes 100
python -m dqn.evaluate --checkpoint "$RUN_DIR/checkpoints/best.pt" --episodes 3 --display
python -m dqn.train --resume "$RUN_DIR/checkpoints/final.pt" \
  --total-env-steps 20000000
```

For a small CPU-only development run:

```bash
python -m dqn.train --device cpu --actors 1 --envs-per-actor 2 \
  --total-env-steps 10000 --warmup 1000 --batch-size 128 \
  --replay-capacity 20000 --no-save-replay --run-dir /tmp/bicycle_smoke
```

Production training defaults to CUDA. `--device cpu` is intended for smoke
tests and development only.

The business acceptance signal is
`business/eval_success_rate_100` in TensorBoard. The target is at least 0.95
over the fixed 100-seed evaluation set with the configured 5-15 N gusts.
