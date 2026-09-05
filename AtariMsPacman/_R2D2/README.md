# Self-contained R2D2 for Ms. Pac-Man

This directory contains an independent recurrent prioritized replay DQN
implementation. It imports only Python/third-party dependencies, `PacManEnv`,
and modules in `_R2D2`; it does not import `DQN`, `R2D2`, `_HRA`, or `hra`.

Run from the project root:

```bash
python -m _R2D2.train
```

Useful overrides include `--total-transitions`, `--num-actors`,
`--envs-per-actor`, `--learner-device`, and `--resume PATH`.

For example:

```bash
python -m _R2D2.train \
  --num-actors 2 \
  --envs-per-actor 2 \
  --total-transitions 10000000 \
  --learner-device cuda:0
```

TensorBoard events are written under the project-level `runs/` directory and
atomic checkpoints under the project-level `checkpoint/` directory; each run
directory includes a timestamp and process id. These output directories are
configured by `PROJECT_ROOT` in `_R2D2/config.py`.

```text
runs/YYYYMMDD-HHMMSS-pid<PID>/
checkpoint/YYYYMMDD-HHMMSS-pid<PID>/checkpoint_step_XXXXXXXXXXXX.pt
```

Resume from a checkpoint with:

```bash
python -m _R2D2.train --resume checkpoint/<run>/checkpoint_step_XXXXXXXXXXXX.pt
```

Run a checkpoint showcase with the same command-line interface as `DQN.play`:

```bash
python -m _R2D2.play \
  --checkpoint checkpoint/<run>/checkpoint_step_XXXXXXXXXXXX.pt \
  --episodes 10 \
  --gui \
  --fps 30
```

The online/target networks, optimizer, transition/update counters, and policy
version are restored. Replay is deliberately not checkpointed, so a resumed
run fills a fresh sequence replay before learning continues.

The implementation uses 4-frame grayscale observations, 5-step Double
Q-learning, value rescaling, 40-step burn-in, 40-step learning unrolls, stored
LSTM states, overlapping sequence replay, and sequence-level prioritized replay.

Training and evaluation consume the raw, unclipped `PacManEnv` reward and do
not treat a lost life as an episode boundary. The rollout `episode_return` and
`raw_score` therefore agree for this configuration but are logged separately
to keep the optimization return and business score explicit.
Training games are capped at 30,000 decisions (the same 108,000-emulator-frame
protocol as the paper); a cap is treated as a replay boundary, while evaluator
caps are reported separately as `capped_episode_count`.

Core TensorBoard metrics include rollout and learner throughput, replay size,
loss/Q/TD-error/gradient statistics, actor epsilon range and policy lag,
rollout raw score/return, and evaluation raw score/return (including score
quantiles and capped-episode counts). Each checkpoint schedules an asynchronous
CPU evaluation; normal completion waits for all scheduled evaluations.

The paper's published Atari results used 256 actors and billions of frames.
The defaults here are sized for a local workstation and can be scaled through
`R2D2Config` or the command-line overrides; they should not be interpreted as
reproducing the paper's compute budget.

Notable local defaults are eight actors, 40M decisions, a 25,000-sequence
replay (up to 1M learning transitions), a 50K-learning-transition warmup, and
checkpoints every 2M decisions. Replay occupies about 14.5 GiB for raw packed
frames and about 17.3 GiB with a conservative 20% object overhead. A full
50,000-sequence copy of the reference replay would require about 28.9 GiB
before overhead and is unsafe on a 34 GiB host. Core paper parameters remain
unchanged: `gamma=0.997`, 5-step targets, 40-step burn-in and learning unrolls,
512 LSTM units, PER exponents 0.9/0.6, and a 2,500-update target interval.

The learner follows the reference implementation's weighted MSE objective.
Importance-sampling weights are normalized by the minimum priority in each
sampled batch, and their min/mean/max are reported in TensorBoard.

Run the package tests from the project root with:

```bash
python -m pytest -q _R2D2/tests
```
