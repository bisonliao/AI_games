# Bicycle RL

Two PyBullet bicycle-balancing environments and a shared distributed
Double/Dueling DQN trainer. Environment 1 controls a reaction wheel;
environment 2 removes that wheel and balances by steering the front handlebar.
Both bicycles are automatically driven at approximately 2 m/s through seeded,
smooth cross-wind gusts.

## Installation and checks

Python 3.11 is expected. The environment already needs PyBullet, Gymnasium,
PyTorch, and TensorBoard; the editable install also registers the local
`env`/`dqn` packages and the TensorBoard startup shim.

```bash
python -m pip install -e '.[test]'
pytest -q
python -m env.baseline --episodes 100
python -m env2.baseline --episodes 100
```

On the current fixed 100-seed gust set, environment 1's reaction-wheel PD gate
requires at least 95% success and less than 10% no-op success. Environment 2's
steering PD baseline reaches 100% while its no-action policy reaches 0%.

## Environments implemented today

Both environments use one independent PyBullet client per instance, 240 Hz
physics, 20 Hz agent control, a `Discrete(3)` action space, and five normalized
observations. Both also use automatic 2 m/s rear-wheel drive, initial roll in
±5 degrees, initial roll rate in ±0.25 rad/s, 5-15 N seeded smooth gusts, and
fall detection at ±45 degrees or non-wheel ground contact.

### Environment 1: reaction wheel

`BicycleBalance-v0`, selected by `--env_id 1`, is the original task:

- `Discrete(3)` actions: negative, zero, or positive reaction-wheel torque;
- five normalized observations: roll sin/cos, local roll rate,
  reaction-wheel speed, and forward speed;
- 15 N·m reaction-wheel torque limit and 120 rad/s speed limit;
- steering is locked straight;
- success at 60 m, fall at ±45 degrees or non-wheel ground contact, and a
  40-second time limit;
- +0.01 reward per newly reached metre, +1 success bonus, and -1 fall penalty.

Inspect passive behavior in a GUI, or run it headlessly:

```bash
python -m env.demo --episodes 1
python -m env.demo --headless --episodes 1 --seed 0
```

### Environment 2: front steering only

`BicycleSteeringBalance-v0`, selected by `--env_id 2`, has no reaction wheel:

- actions are `0` no turn, `1` turn left, and `2` turn right;
- left/right request ±1.2 rad/s steering velocity, action 0 brakes steering
  velocity and holds the current angle, and steering is limited to ±0.35 rad;
- observations are roll sin/cos, local roll rate, normalized steering angle,
  and forward speed;
- heading and lateral drift are unrestricted;
- success means either reaching a 40 m Euclidean radius from the starting point
  or remaining upright for 30 seconds; both end with `terminated=True` and add
  the +1 success reward, while a fall on the final step still counts as failure;
- progress reward uses only a newly reached maximum radius, so circling or
  moving back and forth cannot collect duplicate reward;
- terminal bonuses and fall rules are the same as environment 1.

Inspect the scripted steering controller in the GUI or headlessly:

```bash
python -m env2.demo --episodes 1
python -m env2.demo --headless --episodes 1 --seed 0
```

## DQN implemented today

The training architecture is distributed DQN, not prioritized Ape-X replay:

- four CPU actors by default, each with four spawned `AsyncVectorEnv` workers;
- one learner is the only process that uses `cuda:0`;
- one independent CPU evaluator runs fixed-seed greedy evaluations;
- two-layer 256-unit Dueling Q-network and Double-DQN bootstrap targets;
- 3-step returns, gamma 0.99, and a standard uniform replay ring;
- replay capacity 1,000,000, warm-up 50,000, minibatch size 512;
- sampled/inserted replay ratio 4 and Adam learning rate 1e-4;
- hard target-network copy every 2,000 learner updates;
- CPU actor policy publication every 200 learner updates;
- four-actor epsilon defaults: 0.4, 0.047, 0.006, and 0.001.

Run `python -m dqn.train --help` for every runtime option and its unit.

## Training

Start the default 4 actor × 4 environment CUDA run:

```bash
python -m dqn.train
```

Environment 1 is the default. Select steering balance with:

```bash
python -m dqn.train --env_id 2
```

Fresh training defaults to 5,000,000 global environment steps. It evaluates 100
fixed-seed greedy episodes every 500,000 environment steps and writes a
lightweight checkpoint every 100,000 learner updates.

Every fresh training run creates a sortable history directory such as:

```text
runs/20260802-121530_distributed-dqn_env2_pid12345/
├── config.json
├── tensorboard/
└── checkpoints/
    ├── latest.pt   # periodic learner state, without replay
    ├── full.pt     # every tenth checkpoint interval, including replay
    ├── best.pt     # written after evaluation success exceeds the prior best
    └── final.pt    # always written on shutdown; includes replay by default
```

The timestamp contains year, month, day, hour, minute, and second. The algorithm,
environment ID, and PID distinguish concurrent runs. Use `--runs-root PATH` to
change the history root, or `--run-dir PATH` to request an exact output
directory. `config.json` and every new checkpoint also record `env_id`.

For a small CPU-only development run:

```bash
python -m dqn.train --env_id 2 --device cpu --actors 1 --envs-per-actor 2 \
  --total-env-steps 10000 --warmup 1000 --batch-size 128 \
  --replay-capacity 20000 --no-save-replay --run-dir /tmp/bicycle_smoke
```

Production training defaults to CUDA and fails early when CUDA is unavailable.
`--device cpu` is intended for tests and short development runs.

Resume into a new timestamped history directory:

```bash
RUN_DIR=runs/20260802-121530_distributed-dqn_env2_pid12345
python -m dqn.train --env_id 2 --resume "$RUN_DIR/checkpoints/final.pt" \
  --total-env-steps 10000000
```

Resume restores the checkpoint's network/replay hyperparameters, optimizer,
counters, RNG state, and replay when saved. Runtime topology, selected
environment, and the new total step target still come from the new command
line. Keep `--env_id` equal to the checkpoint's environment unless intentionally
performing transfer learning.

## Evaluation and GUI display

Evaluation is greedy and CPU-only. GUI rendering is disabled by default:

```bash
RUN_DIR=runs/20260802-121530_distributed-dqn_env2_pid12345
python -m dqn.evaluate --env_id 2 \
  --checkpoint "$RUN_DIR/checkpoints/final.pt" --episodes 100
python -m dqn.evaluate --env_id 2 --checkpoint "$RUN_DIR/checkpoints/final.pt" \
  --episodes 3 --display
```

`--display` opens the PyBullet GUI, follows the bicycle, and plays at the 20 Hz
control rate. Evaluation warns if the selected `--env_id` differs from the ID
stored in the checkpoint. Run `python -m dqn.evaluate --help` for all options.

## TensorBoard

Monitor the complete run history rather than one timestamped directory:

```bash
tensorboard --logdir=runs --host=0.0.0.0 --port=6008
```

The editable install provides the same startup shim as both `tensorboard` and
`bicycle-tensorboard`. It bounds TensorBoard 2.21's blocking localhost port
probe, which otherwise takes more than a minute in some WSL networking modes.

The primary acceptance curve is `business/eval_success_rate_100`; the target is
at least 0.95 over the fixed 100-seed evaluation set. TensorBoard also records
train success/fall/timeout rates, episode returns, roll and speed diagnostics,
wind/action distributions, DQN losses and Q-values, replay size, collection
throughput, queue depth, policy lag, and GPU memory.

## Source layout

- `src/env`: reaction-wheel Gymnasium environment, URDF, wind process, demo,
  and PD gate.
- `src/env2`: steering-only Gymnasium environment, reaction-wheel-free URDF,
  demo, and steering PD baseline; it reuses the seeded wind implementation.
- `src/dqn`: actor, learner, uniform replay, evaluation, checkpoints, training,
  and TensorBoard launcher.
- `tests/env` and `tests/dqn`: physics, API, algorithm, checkpoint, CLI, and
  logging tests.
