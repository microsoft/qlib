# Courage 严格路线 C1 实施决策

版本：`courage_strict_c1_v1.0.0`  
日期：2026-08-18  
状态：`PASS_C1_ALL_FALSE_IMPLEMENTATION_DECISIONS / STOP_AFTER_C1_BEFORE_C2`

## 1. 边界

本文记录用户接受后的 ACOT 工程决定。除明确标记为 `COURAGE_EXPLICIT` 的方向外，具体参数均为
`ACOT_IMPLEMENTATION_DECISION`，不能重写为 Courage 原话。

C1 不授权读取数据、物化、runtime、训练、评价、holdout、refit、回测或交易。

## 2. 股票池

- 基础范围：待 C2 准入的沪深 A 股；
- 使用百分数单位的真实换手率字段；
- 每个 signal date 只使用截至 T-1 的 60 个完整交易日；
- 至少 50 个有效观测；
- 60 日算术均值位于闭区间 `[5.0%,15.0%]`；
- 成员资格每日更新；未入池期间的合法分钟历史仍可作为后续 lookback；
- ST、退市风险状态按有效日期 T-1 排除；
- 上市至少 120 个官方交易日；
- v1 不另设成交额硬阈值，换手率组承担基础流动性筛选；
- 当日停牌、涨跌停、无成交不用于提前改变成员资格，只影响相应 Feature/Label mask。

## 3. 时间范围和样本密度

- warmup/source scope：`[2025-04-01,2026-07-01)`；
- Train：`[2025-07-01,2026-03-02)`，即至 2026-02-27；
- Valid：`[2026-03-02,2026-04-01)`；
- Development Test：`[2026-04-01,2026-05-01)`；
- Reserved Confirm：`[2026-05-01,2026-07-01)`，C1 不授权读取；
- Train/Valid 对每天按时间排序后的合法 signal states 使用固定 1-in-5 网格；
- Development Test 冻结后使用全部合法 signal states；
- 不随机抽股票；每个被选 signal time 保留完整目标换手率组；
- split 边界使用一个官方交易日 embargo，并按 horizon 独立执行 maturity/purge；
- 不使用随机 K-fold；首版只使用一个连续时间切分。

由于这些日期曾被历史实验接触，Development Test 只能用于开发结论，不能声称为全新 blind holdout。

## 4. Label

- 输出是研究型 gross return，不扣手续费、滑点或冲击成本；
- signal time 只能使用已 ready 的完整分钟状态；
- entry 为 signal 后第一个完整一分钟成交窗口的 VWAP1；
- horizon `H` 的 exit 为沿官方交易分钟轴向后 H 个槽位对应完整一分钟窗口的 VWAP1；
- `gross_return_H = exit_vwap1 / entry_vwap1 - 1`；
- horizon 精确为 `5/15/30/60/120/240/480` 个官方分钟；
- 午休和隔夜通过官方分钟轴跨越，不按墙钟分钟计算；
- entry/exit 无完整、有限、正值成交价格时，仅该 horizon 无效；
- 一个 head 无效不得删除同一样本的其他有效 head；
- 模型预测与可成交净收益、仓位和订单严格分离。

## 5. Feature

### 5.1 分钟动态 12 项

1. `stock_ret_1`
2. `stock_ret_5`
3. `stock_ret_20`
4. `stock_vwap_bias_20`
5. `stock_realized_vol_20`
6. `stock_high_low_range_20`
7. `stock_volume_ratio_20`
8. `stock_amount_ratio_20`
9. `sector_ret_5_median`
10. `sector_ret_20_median`
11. `sector_positive_breadth_1`
12. `sector_amount_ratio_20_median`

行业状态必须先在有效日期行业的完整已准入沪深 A 股截面计算，再投影到换手率组。宽基市场指数不能
替代板块状态。`buy_amount`、Level-2、逐笔和盘口因本轮未完成独立数据准入，不进入 v1。

### 5.2 T-1 慢变量

- `daily_ret_5`
- `daily_ma20_bias`
- `daily_vol_20`
- `turnover_mean_60`
- `market_cap_log`
- effective-dated industry ID embedding

慢变量只使用上一完整交易日或更早的信息，并由独立 encoder 编码后与 PatchTST 表示融合。

Feature scaler 只在 Train 拟合：连续值使用 0.001/0.999 分位裁剪后 median/IQR；缺失值保留独立
availability/missing mask，禁止从未来回填。

## 6. 模型和 Loss

- 一个共享 PatchTST backbone；
- lookback=`1200` 个官方分钟；
- patch length=`30`、stride=`15`；
- `d_model=128`、3 encoder layers、4 attention heads、dropout=`0.1`；
- 慢变量使用独立 2-layer MLP/industry embedding encoder；
- fusion 后输出 7 个标量收益 head；
- v1 不输出独立上涨概率、最大上涨空间或 rank head；
- target scaler 按 Train、每 horizon 独立拟合 median/IQR，不裁剪 target；
- loss 为标准化 target 上的 Huber，`delta=1.0`；
- 有效 horizon 等权，按 head 独立 numerator/denominator；
- checkpoint 主指标为 Valid 七头等权标准化 Huber；
- optimizer、batch、precision、epoch budget 和 early stopping 数值在 C5/C6 执行配置中预注册，不能在
  看到 Development Test 后修改。

## 7. Baseline、报告与有限门禁

必须报告：

- zero prediction；
- Train mean；
- naive momentum/reversal；
- Ridge reference；
- raw-return MAE/RMSE/bias；
- Pearson、Rank IC、涨跌 ACC、balanced ACC、MCC、top-bottom spread 和按日稳定性。

仅 Valid 用于 checkpoint 和设计选择。Development Test 冻结后读取一次。首版继续的最低信息门为：

- 七头等权 raw RMSE 相对最强 no-information baseline 的 skill 严格大于 0；
- 至少 4/7 horizon 的 RMSE skill 严格大于 0；
- 所有主指标有限、key/mask/time identity 完整。

Ridge、Rank IC、普通 ACC 和 65% ACC 都不是 PatchTST 启动或 checkpoint 的硬门。

## 8. 当前停止条件

C1 决策已闭合，但所有执行 authority 仍为 false。下一阶段只能是 C2 输入审计与准入；C2 需要新的
精确配置、实现闭包和单独授权。
