# Courage Strict V1：Valid 训练与评测报告

运行完成时间：2026-08-19（Asia/Shanghai）
运行身份：`courage_strict_v1`
Qlib Recorder：`448d548c5c0641cd95ff7000cd8c6292`

## 边界

本报告只覆盖冻结的 Valid 区间 `[2026-03-02, 2026-04-01)`。它不是 April
Development replay，也不是任何最终/盲测结论。May 和 June 均未读取。

模型从随机初始化训练；未导入历史 checkpoint、optimizer、scaler 或预测。训练目标为
七个未来 VWAP1 收益 head 的 train-only median/IQR 标准化 Huber 等权损失；checkpoint
唯一依据是 Valid 七头等权标准化 Huber。ACC、balanced ACC、MCC、Rank IC 和
Top-Bottom 仅为诊断项，不参与模型选择。

## 训练与 checkpoint 选择

| Epoch | Train equal-head Huber | Valid equal-head Huber | 选择 |
|---:|---:|---:|---|
| 1 | 0.434269 | **0.627258** | best |
| 2 | 0.399584 | 0.661382 | 未改善 |
| 3 | 0.386085 | 0.671635 | 未改善，early stop |

训练 loss 连续下降、Valid loss 连续变差，因此最终使用 epoch 1，而不是最后一个 epoch。

## Valid 指标

| Horizon (min) | Rows | RMSE | MAE | Pearson | Rank IC | Top-Bottom | ACC | Majority | BAcc | MCC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 1,139,144 | 0.004084 | 0.002586 | -0.010695 | 0.014234 | -0.000075 | 50.43% | 54.28% | 49.77% | -0.004643 |
| 15 | 1,137,601 | 0.007705 | 0.004824 | 0.032358 | 0.021811 | 0.000237 | 50.94% | 55.46% | 50.29% | 0.005774 |
| 30 | 1,135,749 | 0.011190 | 0.007322 | 0.051393 | 0.021115 | 0.000495 | 51.31% | 56.03% | 50.57% | 0.011374 |
| 60 | 1,132,483 | 0.015892 | 0.010987 | 0.056803 | 0.026072 | 0.001022 | 51.69% | 56.02% | 50.96% | 0.019294 |
| 120 | 1,131,334 | 0.022584 | 0.016423 | 0.088315 | 0.039590 | 0.002255 | 52.78% | 56.85% | 51.97% | 0.039377 |
| 240 | 1,146,598 | 0.031921 | 0.023913 | 0.105790 | 0.046556 | 0.003279 | 54.62% | 59.28% | 53.49% | 0.069236 |
| 480 | 1,088,157 | 0.046058 | 0.034572 | 0.077429 | 0.056218 | 0.005731 | 53.75% | 59.63% | 52.72% | 0.053702 |

ACC 排除真实收益恰为零的行；预测零值对非零真实收益计错。Majority 是同一评测人口的多数类准确率，所以裸 ACC 不能解释为方向能力。480 分钟有效行较少，是逐 head maturity 规则的预期结果。

## 有限结论

- 5 分钟没有支持预测能力的证据：Pearson 为负、BAcc 低于 50%、Top-Bottom 为负；
- 15--60 分钟只有弱的正相关/排序诊断，方向诊断接近随机；
- 120--480 分钟出现正的 Pearson、Rank IC 和 Top-Bottom，但仍不能据此声称可交易 Alpha；
- 长 horizon 的裸 ACC 仍低于多数类基线，不能将其表述为“高准确率涨跌预测”；
- 本轮只证明了 V1 数据、Qlib Dataset、PatchTST、DDP 训练、Valid checkpoint selection 和指标输出链已完整跑通。任何接受、策略、回测或交易结论都留待后续冻结流程。

## 可复核身份

| 物料 | SHA-256 |
|---|---|
| best checkpoint（不提交） | `3440945b2a0138726a9acbaf22686cd281eb9925aa5583ea582f174637d24228` |
| train scaler（不提交） | `4429ef17458a4e3ff12db413b6850b55394ebbe2ce3d5a5fcb152b728334cfe0` |
| Valid metrics（不提交） | `d86a0e2d348a982c5fcaebf9f2426d67705c4d462a31362aab3ea7d5e2243e22` |
| provider catalog | `f82c5d5a3d6f71ba3ab510d179b685e7952e8dd28397b1f6150ce5bc15a0025a` |
| provider file set | `12a4411339c59dbbfb15481e8c272f91cbd8634db58583068636c8424dc05612` |

逐样本预测、checkpoint、scaler、Qlib bin 和原始数据均保持在本地忽略目录，不进入 Git。
