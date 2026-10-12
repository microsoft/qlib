# Courage Strict Continuous V1 三项 Sanity Check 报告

## 结论

首轮严格判定：`FAIL_ONE_OR_MORE_SANITY_CHECKS`。本实验只读取 Origin-1 Train/January Valid 与其原始分钟源，未读取四月及以后数据，也未晋升 checkpoint。

> 后续10-seed initialization-only配对与交易日 block bootstrap 已完成，最终判定为 `PASS_NO_REPEATABLE_SHUFFLED_LABEL_LEAKAGE_PATTERN`。首轮19/21结果及其严格FAIL作为历史证据保留，不做结果后改门。详见 [SANITY_NULL_DISTRIBUTION_REPORT.md](SANITY_NULL_DISTRIBUTION_REPORT.md)。

| 检查 | 判定 | 核心结果 |
|---|---|---|
| 原始 Bar Label 独立复算 | `PASS_LABEL_RECOMPUTATION` | 105/105 条 float32 bit-exact；最大误差 0 |
| 固定 512 条小样本过拟合 | `PASS_SMALL_SAMPLE_OVERFIT` | Huber 0.585392 → 9.09265e-05，下降 99.98% |
| 3-seed shuffled-label | `FAIL` | 完整固定 Valid；每 seed × 7 heads 均按随机界限审计 |

## 1. Label 独立复算

按每个 horizon 15 条、合计 105 条进行时间分层抽取。直接读取不可变分钟 Parquet，以 `amount/volume` 分别计算 entry/exit VWAP，再计算收益并转为 float32 与 provider bin 比较。RANSAC 未使用，因为这是确定性数据核对，不是鲁棒拟合问题。

逐条证据：`artifacts/courage_strict_continuous_v1/sanity_checks_v1/label_recomputation_rows.csv`。

## 2. 小样本过拟合

固定 512 条七头均成熟且符合训练密度的样本；dropout=0、weight_decay=0，最多 3000 steps。最终每头 Huber：

| horizon | standardized Huber |
|---:|---:|
| 120 | 7.8105282e-05 |
| 15 | 9.6902475e-05 |
| 240 | 0.0001216827 |
| 30 | 0.00014928226 |
| 480 | 4.6119674e-05 |
| 5 | 6.795552e-05 |
| 60 | 7.6437846e-05 |

曲线：![Small-sample overfit curve](../../artifacts/courage_strict_continuous_v1/sanity_checks_v1/SMALL_SAMPLE_OVERFIT_CURVE.png)

## 3. Shuffled-label 随机对照

Train 标签按 horizon 在各自有效训练样本内独立置换，保留输入、mask、训练流程和完整固定 Valid。随机界限为：|BAcc−0.5|≤0.02、|AUC−0.5|≤0.02、|MCC|≤0.04、|Rank IC|≤0.03。

| seed | horizon | BAcc | AUC | MCC | Rank IC | 随机界限 |
|---:|---:|---:|---:|---:|---:|---|
| 20260831 | 5 | 49.9764% | 50.4475% | -0.00214 | 0.01226 | PASS |
| 20260831 | 15 | 49.9662% | 49.6057% | -0.00371 | -0.01385 | PASS |
| 20260831 | 30 | 50.0012% | 50.3284% | 0.00203 | 0.01247 | PASS |
| 20260831 | 60 | 49.8643% | 49.2760% | -0.00656 | -0.00899 | PASS |
| 20260831 | 120 | 48.9871% | 48.8583% | -0.02077 | -0.00595 | PASS |
| 20260831 | 240 | 49.7386% | 50.2216% | -0.00828 | 0.01145 | PASS |
| 20260831 | 480 | 50.6922% | 53.1228% | 0.01947 | 0.03116 | FAIL |
| 20260832 | 5 | 49.9854% | 50.2363% | -0.00114 | 0.01149 | PASS |
| 20260832 | 15 | 49.9675% | 49.7612% | -0.00333 | -0.00618 | PASS |
| 20260832 | 30 | 49.9815% | 51.0490% | -0.00322 | 0.01839 | PASS |
| 20260832 | 60 | 50.0078% | 49.7233% | 0.00067 | -0.00842 | PASS |
| 20260832 | 120 | 50.0001% | 50.1561% | 0.00002 | -0.00541 | PASS |
| 20260832 | 240 | 50.0006% | 49.1880% | 0.00008 | -0.02279 | PASS |
| 20260832 | 480 | 49.5335% | 49.7717% | -0.01316 | 0.00105 | PASS |
| 20260833 | 5 | 50.0908% | 50.8093% | 0.00471 | 0.02941 | PASS |
| 20260833 | 15 | 49.9101% | 48.7967% | -0.00451 | -0.03580 | FAIL |
| 20260833 | 30 | 50.0640% | 49.9583% | 0.00707 | -0.00327 | PASS |
| 20260833 | 60 | 49.0265% | 48.9259% | -0.01950 | -0.02320 | PASS |
| 20260833 | 120 | 49.9810% | 50.5858% | -0.00824 | 0.02161 | PASS |
| 20260833 | 240 | 49.3596% | 48.6326% | -0.02964 | -0.01356 | PASS |
| 20260833 | 480 | 51.1828% | 51.1486% | 0.02433 | 0.01225 | PASS |

### Shuffled-label 结果解释

预注册的逐 seed 严格门结果为 **19/21** 个单元通过。越界单元为 seed 20260831/480m 与 seed 20260833/15m；没有任何 horizon 在两个或更多 seed 重复越界。因此只能记为 `FAIL_SHUFFLED_LABEL_RANDOM_CONTROL`，但当前没有形成跨 seed 可重复的泄漏模式。该一致性分析是结果后的解释，不修改原门。

还观察到若干置换模型的 prediction up-rate 接近 0，说明 500-step 随机标签训练容易输出近常数；因此本检查主要依据 BAcc/AUC/MCC/Rank IC，而不能把普通 ACC 当随机性证据。

## 解释边界

- Label复算与小样本过拟合已经排除被抽查Label的公式/落盘错误以及明显的模型不可训练问题。
- 首轮 shuffled-label 严格FAIL已通过独立预注册的10-seed null-distribution复核；未发现可重复泄漏模式，但首轮结果不被追溯改写为PASS。
- 这些结果只降低明显实现错误或稳定泄漏的优先级，不证明正式模型具有泛化能力。
- 本报告不改变原正式模型的基线门失败结论。
