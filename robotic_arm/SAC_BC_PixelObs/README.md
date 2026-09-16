# SAC_BC_PixelObs

This directory contains the isolated three-stage visual pick-place pipeline. Existing project directories are read-only dependencies.

## 1. Generate demonstrations

```bash
python -m SAC_BC_PixelObs.expert_showcase \
  --episodes 200 --frame-stack 2 \
  --out SAC_BC_PixelObs/runs/showcase
```

The default teacher is `SAC_VecObs/runs/pick_place_20260818_211439/checkpoints/pick_place_sac_1000000_steps.zip`. Its matching `pick_place_sac_vecnormalize_1000000_steps.pkl` is loaded automatically. The teacher is queried from the same PyBullet client as the pixel environment, so images, actions, rewards and stage transitions are time aligned. Each compressed episode stores image, proprio, action, reward, terminated/truncated flags, stage and result metadata.

## 2. BC pretraining

```bash
python -m SAC_BC_PixelObs.bc_pretrain \
  --data SAC_BC_PixelObs/runs/showcase \
  --epochs 50 --frame-stack 2 \
  --train-eval-episodes 5 --eval-every 5 \
  --eval-seed-start 10000 --eval-episodes 100 \
  --out SAC_BC_PixelObs/runs/bc_model.pt
```

Only successful episodes are used. Episodes, rather than individual frames, are split into train/validation sets. During training, only five business-evaluation episodes are run every five epochs. After the final epoch, the best checkpoint is evaluated once on all 100 unseen seeds. The output is a versioned Stable-Baselines3-compatible `Actor` checkpoint.

## 3. SAC fine-tuning

```bash
python -m SAC_BC_PixelObs.sac_finetune \
  --bc-model SAC_BC_PixelObs/runs/bc_model.pt \
  --expert-data SAC_BC_PixelObs/runs/showcase \
  --total-timesteps 1000000 --frame-stack 2 \
  --n-actors 8 \
  --utd-ratio 0.25 --prior-ratio 0.25
```

The actor starts from BC weights. Training uses an Ape-X style process layout:

- CPU actors only run `PixelTaskEnv` and send bounded transition chunks.
- One learner owns SAC, replay, the GPU, TensorBoard and checkpoints.
- `--n-actors` controls the number of CPU actors; the default is 8.
- `--utd-ratio 0.25` performs one SAC update per four learner-received online transitions.
- The online SB3 replay is augmented with a prior buffer; every batch uses the requested 25% prior / 75% online ratio when both sources have enough transitions.
- The learner's x-axis is `learner_transition_count`: transitions received from actors and written to online replay. Prior demonstrations do not increase this count.

The learner does not use `model.learn()` for rollout collection. It explicitly receives chunks, writes them to replay, spends the UTD update budget, synchronizes actor weights, records business metrics and saves checkpoints. Existing SAC reward and termination semantics are retained. The new directory does not patch `SAC_PixelObs`.

## 4. Evaluation

`evaluate.py` 区分两类 checkpoint，评测方式不同：

- `bc_pretrain.py` 生成的 `*.pt` 是 standalone visual actor，使用普通 Gymnasium 环境逐回合评测。
- `sac_finetune.py` 生成的 `*.zip` 是 Stable-Baselines3 SAC 模型，使用 `DummyVecEnv` 评测。
- `SAC_VecObs` 专家虽然也是 `*.zip`，但输入是 52 维向量而非图像字典，不能传给这个评估入口；程序会明确报错。

默认根据后缀自动判断，也可以显式指定 `--checkpoint-type bc` 或 `--checkpoint-type sac`。

```bash
# BC checkpoint
python -m SAC_BC_PixelObs.evaluate \
  --checkpoint SAC_BC_PixelObs/runs/bc_model.pt \
  --checkpoint-type bc \
  --episodes 100

# SAC fine-tuning checkpoint
python -m SAC_BC_PixelObs.evaluate \
  --checkpoint SAC_BC_PixelObs/runs/sac_finetune/final_model.zip \
  --checkpoint-type sac \
  --episodes 100
```

BC `.pt` checkpoints and SB3 SAC `.zip` checkpoints are both accepted. The default evaluation uses unseen seeds `10000..10099`; use `--seed-set train` only for a diagnostic evaluation on training seeds. Success, grasp, lift, approach-timeout and mean return are printed.

```bash
#经过了4M步的finetune，也就是sac_finetune.py执行4M步后，评测的成功率为70%，从tb曲线看，成功率难以进一步提升。（2026年9月15日的进展）
pybullet build time: Jan 29 2025 23:17:20
Evaluation seed set: unseen, seeds=10000..10019
TensorBoard logs: /home/bison/mygames/robots/SAC_BC_PixelObs/tb_logs/evaluate_20260915_074757_pid4672
{'mode': 'sac', 'episodes': 20, 'success_rate': 0.7, 'grasp_rate': 0.85, 'lift_rate': 0.85, 'approach_timeout_rate': 0.15, 'mean_return': 10.227298061626152}
```

The current retraining configuration uses `frame_stack=2`, the nominal PixelObs reset distribution without physical or camera jitter during SAC fine-tuning, episode-level train/validation splitting, and consistent four-pixel translation augmentation across all views and history frames. The old pixel checkpoints were removed because their input shape and training distribution are incompatible with this configuration.

DAgger is intentionally an optional follow-up. `dagger.py` is reserved for collecting on-policy observations and labeling them with the in-simulation vector teacher after diagnostics show unrecoverable covariate shift.

## TensorBoard 日志

`bc_pretrain.py`、`sac_finetune.py` 和 `evaluate.py` 的日志统一写入项目根目录
`tb_logs/`。每次运行自动创建目录，格式为
`<阶段名>_<YYYYMMDD>_<HHMMSS>_pid<PID>`，时间使用运行机器的本地时间，例如：

```text
tb_logs/
├── bc_pretrain_20260910_153012_pid12345/
├── sac_finetune_20260910_160530_pid12346/
└── evaluate_20260910_183045_pid12347/
```

事件文件直接保存在对应运行目录中。若同一进程在同一秒再次启动相同阶段，
目录末尾追加递增序号，避免日志混合。启动时会打印日志的绝对路径。
日志路径不再跟随 checkpoint、`--out` 或 `--output`；三个入口已移除
`--tensorboard-log` 参数，统一使用上述固定目录。

```bash
tensorboard --logdir tb_logs
```
