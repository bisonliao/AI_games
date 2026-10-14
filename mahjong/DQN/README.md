# SB3 Masked Double-DQN

Requires `stable-baselines3>=2.9,<3` (the implementation does not depend on
`sb3-contrib`).

This implementation uses Stable-Baselines3 for rollout, replay storage,
optimizers, target-network synchronization, logging, and checkpoints. The
custom algorithm keeps the current environment interface and implements
mask-aware exploration plus Double-DQN targets.

```bash
python -m DQN.train --total-timesteps 5000000 --num-envs 8 \
  --batch-size 256 --buffer-size 300000 --learning-starts 20000 \
  --train-freq 4 --gradient-steps 1 --eval-freq 100000 \
  --eval-episodes 500 --log-interval-steps 10000 --device auto
```

The observation remains a Dict containing `vector[306]` and `action_mask[141]`.
The vector is the only neural-network feature input; the mask is used for
random exploration, greedy selection, and Double-DQN next-action selection.
The action space remains `Discrete(141)`, with automatic self-draw termination.

Checkpoints are SB3 zip files under `checkpoints/sb3_<run_id>/`; old `DDDQN/*.pt`
files are not compatible. Periodic checkpoints, `best/best_model.zip`, and
`sb3_last.zip` are all stored in this run directory. TensorBoard output is
written below `tb_logs/sb3_<run_id>/`. Set a different `--run-id` for each
experiment (the CLI default is `run`) to keep their models separate.
Console and TensorBoard dumps are emitted every `log_interval_steps` environment
steps (default `10000`); training updates still run at `train_freq`.
The default `exploration_fraction=0.4` linearly decays epsilon from `1.0` to
`0.05` during the first 40% of the configured training budget, then keeps it at
`0.05`. Override it with `--exploration-fraction` when comparing schedules.
Evaluation additionally records `eval/self_draw_rate`, `eval/draw_rate`,
`eval/loss_rate`, and `eval/win_rate` from each episode's terminal reason.
