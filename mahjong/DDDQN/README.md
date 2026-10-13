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

TensorBoard logs are written to `tb_logs/YYYYMMDD_HHMMSS_pidPID/`:

```bash
tensorboard --logdir tb_logs
```

Only `checkpoints/best.pt` and `checkpoints/last.pt` are retained. Replay data
is preallocated in RAM and is never serialized. Resume with:

```bash
python -m DDDQN.train --resume checkpoints/last.pt
```

`best.pt` is ranked by self-draw rate, then average return, then shorter
episode length. Every saved checkpoint contains metrics from a deterministic
evaluation completed immediately before the save.

To inspect a trained policy's decisions without opening the GUI, run the
evaluator in verbose mode (the default). It prints each observed hand, the
current response/play phase, and the selected action in tile order: red
centers, wan, bing, tiao, with the newly drawn tile shown separately.

```bash
python -m DDDQN.evaluate \
  --checkpoint checkpoints/last.pt \
  --episodes 1 \
  --num-envs 1 \
  --device auto
```

Use `--quiet` to print only the final evaluation summary.

## 出牌质量观察（2026-10-13）

当前模型的评测成功率已达到约 25%，但人工查看逐步出牌记录后，仍认为其出牌技术较差。
成功率的提升说明模型学到了有效策略，但不能据此认定其已具备稳定、合理的牌效判断能力；
还需要结合具体牌局检查弃牌选择。

以下是使用 `checkpoints/best.pt` 进行文字评测时观察到的连续两个决策：

```text
手牌：2万 4万 4万 8万 2饼 6饼 6饼 7饼 8饼 3条 4条 5条 5条  | 摸到：9万
模型动作：打出 8万
```


