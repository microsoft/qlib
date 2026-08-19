# Courage Strict V1 实施状态

更新时间：2026-08-19（Asia/Shanghai）

## 唯一工程身份

- 工程根目录：`/data1/lxl/workspace/datasets/.tmp/qlib`
- 版本名称：`courage_strict_v1`
- Python 环境：`/data1/lxl/workspace/datasets/.tmp/qlib/.venv`
- 数据目录：`/data1/lxl/workspace/datasets/.tmp/qlib/data/courage_strict_v1`
- 实验产物：`/data1/lxl/workspace/datasets/.tmp/qlib/artifacts/courage_strict_v1`
- 训练、评测和后续开发均在当前 Qlib fork 内完成；运行代码不依赖其他工程目录。
- 未导入任何历史 checkpoint、optimizer、scaler、预测或模型选择结果。

## 已完成的数据阶段

1. 固定主板 PIT 基础范围、交易状态、行业、复权和日频换手率事实；
2. 按 T-1 可用信息生成 5%–15% 换手率 PIT membership；
3. 生成 1-in-5 合法 signal grid；
4. 从项目自己的分钟快照计算 VWAP1 七 horizon Label；
5. 生成分钟动态特征、PIT 行业状态和 T-1 慢变量；
6. 输出 Qlib 标准 `calendar / instruments / features/*.1min.bin` provider。

Qlib provider 终态：

- 决策：`PASS_V1_QLIB_MATERIALIZATION_AND_PARITY`；
- catalog SHA-256：`f82c5d5a3d6f71ba3ab510d179b685e7952e8dd28397b1f6150ce5bc15a0025a`；
- file-set SHA-256：`12a4411339c59dbbfb15481e8c272f91cbd8634db58583068636c8424dc05612`；
- 203,228 个 bin 文件，约 45 GiB；
- 94 个字段/股票，包含原始 OHLCVA、bar mask、PIT membership、动态/慢特征及七头 Label；
- Qlib `D.features` 读取和模型 Dataset 读取均通过。
- Qlib Recorder 使用项目内 SQLite tracking backend，artifact 固定写入
  `artifacts/courage_strict_v1/qlib_records/artifacts/`；基础设施 smoke 已通过。

当前 Train/Valid 数据合同：

- Train：`[2025-07-01, 2026-03-02)`；
- Valid：`[2026-03-02, 2026-04-01)`；
- horizon：`5 / 15 / 30 / 60 / 120 / 240 / 480` 分钟；
- lookback：1200 个官方交易分钟；
- 样本：10,269,984；
- PIT membership：213,958 个股票日；
- 入选股票：2,162；
- 动态特征：125,568,960 行；
- May、June 均未读取。

## Qlib 原生执行链

```text
Qlib 1min bin provider
  → CourageStrictV1Dataset（qlib.data.Dataset）
  → Train-only scaler
  → CourageStrictV1Model（qlib.model.Model）
  → shared PatchTST + 7 heads
  → Valid checkpoint selection
  → Qlib Recorder
  → Valid prediction and metric report
```

正式 loss 与选择规则：

- 每个 horizon 的 Train target 独立使用 median/IQR 标准化；
- Huber `delta=1.0`；
- 每个 batch 仅对合法 target 计损失；
- 七个有效 head 等权；
- checkpoint 仅按 Valid 七头等权标准化 Huber 选择；
- ACC、balanced ACC、MCC、Rank IC 属于诊断项，不参与 checkpoint 选择。

## 当前运行状态

- 2026-08-19 17:29（Asia/Shanghai）已通过 8×RTX PRO 5000 72GB GPU 可见性检查；
- BF16 与 checkpoint-resume 有界画像决策：`PASS_V1_CUDA_BF16_RUNTIME_PROFILE`；
- 画像 batch size 256、峰值显存 4,349,196,800 bytes、单卡约 1,764 samples/s；
- 正式 8 卡 DDP 训练已从随机初始化启动；Qlib Recorder：
  `467c3833ee0a448480b3cc634c6c819f`（Experiment 2）；
- 正式启动必须使用 `examples/courage_strict_v1/launch_training.sh`，确保各 rank 优先加载
  项目 `.venv/lib` 中与 SQLite/ICU 匹配的 C++ runtime；
- 当前全局 batch 为 2,048（256×8），May、June 仍未读取。

## 尚需完成

1. 等待正式训练完成并写出最佳 checkpoint；
2. 输出 Valid 指标、逐样本预测和评测报告；
3. 模型、配置和选择规则冻结后，单独生成并读取 April Development 数据；
4. May 保持 reserved，June 保持 sealed。

代码验收：36 项 `courage_strict_v1` 测试通过，Ruff import/syntax 检查通过。
