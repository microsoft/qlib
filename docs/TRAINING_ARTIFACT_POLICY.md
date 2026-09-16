# 每次训练的 Train/Valid Loss 曲线强制规则

状态：`ACCEPTED_REPOSITORY_WIDE_TRAINING_RULE`。

本规则适用于此仓库之后的每一次正式模型训练，不限于某个 Courage 版本。参考图为 [Courage V1 step curve](courage_strict_v1_step_curve/TRAIN_VALID_LOSS_CURVE.png)。

## 必须记录

- 在配置冻结的每个 validation step 同时记录当前 Train window loss 与完整固定 Valid loss。
- Train loss 必须是刚结束窗口的损失，不能用从step 1开始的累计平均值冒充局部训练曲线。
- Valid population、Label mask、scaler和选择loss在一次运行中必须固定。
- `loss_curve.json` 必须保留原始逐点值、global step、学习率、样本覆盖和best checkpoint选择指标。

## 必须绘图

每次训练必须生成：

1. 运行目录中的 `TRAIN_VALID_LOSS_CURVE.png`；
2. 训练报告文档目录中的同名字节副本；
3. 图中使用global optimizer step作为横轴，绘制未平滑的Train与Valid loss，并标出best step；
4. 运行目录和训练报告目录中的 `PER_HEAD_TRAIN_VALID_LOSS_CURVES.png`，分别展示每个horizon的Train-window与完整固定Valid曲线；
5. 训练报告必须链接两张图，并写明验证间隔、总验证点数和best step。

gradient/LR曲线建议同时生成，但不能替代上述两张Train/Valid曲线。

## 最佳checkpoint分组诊断

日期、分钟位置、PIT行业和换手率分组不在每个validation step重复计算。训练完成并冻结best checkpoint后，必须对完整固定Valid自动执行一次，生成`BEST_CHECKPOINT_GROUPED_METRICS.csv`或内容等价且被manifest固定的产物。

- 分组至少覆盖`date`、`minute_slot`、`industry_id`、`turnover_band`；
- 必须覆盖所有已训练horizon，并保存样本数、RMSE、bias、ACC、BAcc、MCC、AUC、target/prediction up rate；
- 该步骤只允许读取训练时已经授权的固定Valid，不能用分组结果重新选择checkpoint；
- 若基线门要求先比较zero、Train mean/median、momentum/reversal和Ridge，则分组诊断应在同一post-training closure中完成。

## 完成门禁

以下任一情况都标记为 `TRAINING_ARTIFACTS_INCOMPLETE`：

- 缺少`loss_curve.json`或`TRAIN_VALID_LOSS_CURVE.png`；
- 缺少逐horizon Train/Valid曲线或best-checkpoint分组指标；
- 图与JSON的step或best checkpoint不一致；
- 文档副本缺失或与运行目录不是相同字节；
- 只有epoch末值，无法观察epoch内部变化；
- 用平滑值替换原始loss，或者Train和Valid使用不同选择口径。

该门禁只约束训练可观测性，不授权读取新数据、训练、refit、策略、回测或交易。
