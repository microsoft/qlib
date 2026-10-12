# Courage 严格路线 C1-v2 沪深主板范围修订

版本：`courage_strict_c1_mainboard_scope_amendment_v2.0.0`  
日期：2026-08-18  
状态：`PASS_C1_V2_MAINBOARD_SCOPE / STOP_BEFORE_C2_MAINBOARD_REACCEPTANCE`

## 决定

用户最新明确决定：本路线只研究沪深主板。

新的基础证券范围精确定义为：

- `exchange in {SSE, SZSE}`；
- `market == 主板`；
- 上交所科创板、深交所创业板和北交所全部排除；
- 上市/退市有效区间仍由 C2 证券主表控制；
- 后续 T-1 换手率分组、状态过滤、行业聚合、Feature、Label、训练与评价都只能在该范围及其合法历史上执行。

该决定是用户当前指令和 `ACOT_IMPLEMENTATION_DECISION`，不是 Courage 原话。

## 版本关系

原 C1 v1 的 Label、Feature、模型、时间切分、loss、评价和权限边界全部保持不变；只替换基础证券范围。
旧 C1/C2 物料作为不可变历史证据保留，不原地改写。C2 必须先证明沪深主板是已接受 5,220 股范围的
精确子集，并重验日频、状态、行业、分钟文件和公司行动的键闭包，之后才能恢复 C3。

## 权限边界

本修订不授权 Development Test、May--June reserved、训练、评价、refit、策略、回测、交易或远端推送。
