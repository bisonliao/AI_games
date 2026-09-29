# TD3+BC PixelObs

This package trains a deterministic visual TD3+BC policy from the existing
expert showcase without stepping a training environment. The evaluator is a
separate spawned process and is the only process that runs PyBullet episodes.

## Train

```bash
python -m TD3_BC_PixelObs.train \
  --dataset SAC_BC_PixelObs/runs/expert_showcase_v3 \
  --total-transitions 2000000 \
  --batch-size 128 \
  --alpha 2.5 \
  --augmentation-shift 4 \
  --checkpoint-transitions 500000 \
  --log-transitions 10000
```

The default profile is random initialization, batch size 128, eight learner
CPU threads, and one asynchronous evaluator process using two CPU threads.
This is sized for an 8GB GPU and 12GB system memory. Periodic checkpoints are
never more frequent than 500K sampled transitions; only the newest two
completed periodic checkpoints are retained. The final checkpoint is always
kept. `--bc-init --bc-model ...` opts into BC actor initialization.

`--total-transitions` is an upper bound. Training performs complete batches
and stops after the first update that reaches or crosses the bound, so the
actual sampled count can exceed it by less than one batch.

The learner uses raw proprioception, matching the validated PixelObs policy
input. Image augmentation is enabled by default with a four-pixel shift,
applied consistently across all three views and stacked history frames. The
target, critic-current, and actor-current paths each use one random shift
(`K=1`, `M=1`); evaluation never applies augmentation. Set
`--no-augmentation` (or `--augmentation-shift 0`) for the no-augmentation
ablation. `--alpha` supports
the paper's recommended sweep `{1, 2, 2.5, 3, 4}`.

The learner uses a spaces-only environment and will raise if reset or step is
called. It does not call SB3 `learn()`, `CheckpointCallback`, or replay-buffer
serialization, so SB3 cannot write frequent training snapshots.

Training TensorBoard includes critic/actor/BC losses, target and current Q
means/stddevs, TD error, Q1/Q2 gap, gradient norms, TD3+BC lambda, action
magnitude/saturation, stage sampling fractions, augmentation settings,
sampled-transition throughput, and asynchronous business metrics. Evaluation
additionally reports mean episode length.

Runs are stored under `TD3_BC_PixelObs/runs/td3_bc_*`; TensorBoard events are
stored under the repository-level `tb_logs/td3_bc_*` directory. Each periodic
checkpoint and `final_model.zip` is evaluated asynchronously.

## Evaluate

```bash
python -m TD3_BC_PixelObs.evaluate \
  --checkpoint TD3_BC_PixelObs/runs/td3_bc_*/checkpoints/final_model.zip \
  --episodes 100 \
  --device cpu
```

Use `--randomize` for the randomized scene distribution and `--gui` for a
human PyBullet window.
