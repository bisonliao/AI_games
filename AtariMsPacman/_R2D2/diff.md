# `_R2D2` 与参考 `R2D2/` 实现的差异排查

## 结论

目前没有发现明确的 n-step 索引错误。R2D2 在 8M decisions 时表现较差，最主要的原因更可能是训练规模和若干训练细节与参考实现不同，而不是 learner update 数量不足。

## 1. 训练规模没有对齐

参考实现中约 6000 分对应的曲线大致使用了：

- 约 `1.1e8` emulator frames；
- 约 `27.5M` environment decisions；
- `100K` learner updates；
- 评测为 5 局。

当前 `_R2D2` 在 8M decisions 时约为：

- `32M` emulator frames；
- `24,310` learner updates；
- 约 2.54 小时；
- 30 局评测平均原始分约 `1411`。

因此当前只达到参考实现最终数据量的约 29%、learner update 数的约 24%，不能直接把 8M 的分数与参考实现约 6000 分横向对齐。

## 2. 最大代码级嫌疑：loss 与 IS weight

参考实现使用逐元素 MSE：

```python
is_weights * (Q - target) ** 2
```

当前 `_R2D2` 使用 Huber loss：

```python
is_weights * smooth_l1_loss(...)
```

两者的优先经验回放权重归一化也不同：

- 参考实现使用当前 batch 中的最小 priority 归一化 IS weight；
- 当前 `_R2D2` 使用整个 replay 中的最小 priority 归一化。

后者可能使整个 batch 的有效权重变小。8M 附近日志显示：

```text
td_error_abs_mean ≈ 1.48
loss ≈ 0.081
```

这说明实际训练信号可能被明显压缩。再结合 `Adam eps=1e-3`，这是当前最值得优先验证的实现差异。

## 3. `updates_per_sequence=1/8` 并不表示 update 太少

当前 8M 训练的有效 replay ratio 约为：

```text
24310 × 64 × 40 / 8M ≈ 7.78
```

参考实现最终大致为：

```text
100K × 64 × 40 / 27.5M ≈ 9.3
```

所以当前更新量与参考实现处于相近数量级，不建议仅因为 `updates_per_sequence` 数值看起来小就再扩大四倍。

## 4. 其他重要差异

### 输入和环境随机性

- 参考实现使用单帧输入，当前 `_R2D2` 使用 4-frame stack；
- 参考实现 `repeat_action_probability=0`，当前 `_R2D2` 使用 `0.25`。

使用同一个 8M checkpoint 做的 10 局额外评测：

```text
sticky=0.25：平均 raw score ≈ 1586
sticky=0.0 ：平均 raw score ≈ 1798
```

sticky action 约造成 13% 差距，但不是当前低分的唯一或主要原因。

### Replay 容量和 warmup

- 参考实现 replay 约 `50K sequences`；
- 当前 `_R2D2` replay 为 `7500 sequences`；
- 参考实现 warmup 约 `50K transitions`；
- 当前 `_R2D2` warmup 约 `20K transitions`。

当前 replay 容量和 warmup 都明显小于参考实现。

### Burn-in 反向传播

参考实现的 burn-in 计算路径仍处在同一套 recurrent 计算中；当前 `_R2D2` 在 burn-in 后显式 detach hidden state。当前做法符合常见的 R2D2 训练写法，但与该参考实现并不完全一致，值得做消融对比。

### 初始 priority

参考实现由 actor 根据行为策略产生的 TD 误差计算新 block/sequence 的初始 priority；当前 `_R2D2` 新 sequence 统一使用 replay 当前最大 priority。当前做法可以保证新数据被采样，但会暂时过度提高新 sequence 的采样概率。

## 5. 确定但不是主要性能原因的问题

- 当前训练 episode cap 为 `30,000 decisions`；`frame_skip=4` 时约为 `120,000 emulator frames`，而论文协议是 `108,000 frames`，参考实现配置为 `27,000 decisions`。
- 当前配置已改为 8 actors，但部分注释仍写着 4 actors，容易造成误解。

## 建议排查顺序

1. 对比 MSE 与 Huber loss；
2. 记录并比较 IS weight 的实际分布，确认 replay 最小 priority 归一化是否导致有效梯度过小；
3. 将 replay 扩展到更接近参考实现的规模；
4. 将 warmup 调整到约 `50K transitions`；
5. 对比 burn-in detach 与不 detach；
6. 最后再考虑 sticky action、episode cap 和其他评测协议差异。

本文件只记录差异分析，不代表已经实施上述实验或修改。
