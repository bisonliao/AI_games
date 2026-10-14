# Masked Dueling Double DQN

Train the 306-dimensional vector observation and fixed 141-action space with
spawn-based environment workers:

```bash
python -m DDDQN.train \
  --total-env-steps 5000000 \
  --num-envs 8 \
  --batch-size 256 \
  --replay-capacity 300000 \
  --log-interval-steps 1000 \
  --checkpoint-interval-steps 100000 \
  --eval-episodes 200 \
  --device auto
```

TensorBoard logs are written to `tb_logs/<run_id>/`. By default, `run_id` is
`YYYYMMDD_HHMMSS_pidPID`; use `--run-id experiment_01` to choose a name:

```bash
tensorboard --logdir tb_logs
```

Each run retains `checkpoints/<run_id>/best.pt` and `checkpoints/<run_id>/last.pt`,
using the same run ID as its logs. Different run IDs do not overwrite each
other. `--checkpoint-dir` changes the parent directory. Replay data is
preallocated in RAM and is never serialized. Resume in the same directory with:

```bash
python -m DDDQN.train --run-id experiment_01 --resume checkpoints/experiment_01/last.pt
```

When resuming without `--run-id`, a new run ID is generated. A new output
directory ranks `best.pt` using evaluations from the resumed run; resuming in
the original directory preserves its historical best. Existing checkpoints
from the old flat directory layout can still be loaded via `--resume`.

`best.pt` is ranked by self-draw rate, then average return, then shorter
episode length. Every saved checkpoint contains metrics from a deterministic
evaluation completed immediately before the save.

To inspect a trained policy's decisions without opening the GUI, run the
evaluator in verbose mode (the default). It prints each observed hand, the
current response/play phase, and the selected action in tile order: red
centers, wan, bing, tiao, with the newly drawn tile shown separately.

```bash
python -m DDDQN.evaluate \
  --checkpoint checkpoints/experiment_01/last.pt \
  --episodes 1 \
  --num-envs 1 \
  --device auto
```

Use `--quiet` to print only the final evaluation summary.

## 出牌质量观察（2026-10-13）

当前模型的评测成功率已达到约 25%，但人工查看逐步出牌记录后，仍认为其出牌技术较差。
成功率的提升说明模型学到了有效策略，但不能据此认定其已具备稳定、合理的牌效判断能力；
还需要结合具体牌局检查弃牌选择。

以下是使用旧版路径 `checkpoints/best.pt` 进行文字评测时观察到的连续两个决策：

```text
手牌：2万 4万 4万 8万 2饼 6饼 6饼 7饼 8饼 3条 4条 5条 5条  | 摸到：9万
模型动作：打出 8万
```


