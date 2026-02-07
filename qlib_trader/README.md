# QLib Trader - 量化交易集成平台

基于微软 [Qlib](https://github.com/microsoft/qlib) 框架的一站式量化交易集成软件，将数据管理、模型训练、策略回测整合为简单易用的交互式工具。

## 功能特性

- **一键流水线**: 数据 → 模型训练 → 策略回测 → 绩效报告，一键完成
- **丰富的 AI 模型**: 内置 8 种模型（LightGBM、XGBoost、CatBoost、LSTM、GRU、Transformer 等）
- **多种交易策略**: TopK-Dropout、加权选股、保守策略等预设方案
- **交互式界面**: 中文菜单驱动，无需编写代码
- **命令行工具**: 支持命令行参数，便于脚本化和自动化
- **预设模板**: 5 种经典工作流模板，开箱即用

## 快速开始

### 1. 交互式模式（推荐新手）

```bash
python -m qlib_trader
```

启动后进入中文交互菜单，按提示操作即可。

### 2. 命令行模式

```bash
# 查看帮助
python -m qlib_trader --help

# 一键运行默认流水线（LightGBM + Alpha158 + TopK策略）
python -m qlib_trader pipeline

# 指定模型和策略
python -m qlib_trader pipeline --model xgboost --strategy conservative

# 使用 LSTM 深度学习模型
python -m qlib_trader pipeline --model lstm

# 查看数据状态
python -m qlib_trader data --status

# 下载中国市场数据
python -m qlib_trader data --download cn

# 训练模型
python -m qlib_trader train --model lightgbm

# 查看所有可用模型
python -m qlib_trader models
```

## 支持的 AI 模型

| 模型 | 类型 | 说明 |
|------|------|------|
| `lightgbm` | 树模型 | 微软梯度提升框架，训练快速，**推荐入门** |
| `xgboost` | 树模型 | 极端梯度提升，经典高性能 |
| `catboost` | 树模型 | Yandex提升树，自动处理类别特征 |
| `linear` | 线性 | 线性回归，适合基线对比 |
| `lstm` | 深度学习 | 长短期记忆网络，捕获时序依赖 |
| `gru` | 深度学习 | 门控循环单元，轻量级时序模型 |
| `transformer` | 深度学习 | Transformer，自注意力机制 |
| `alstm` | 深度学习 | 注意力LSTM，LSTM + 注意力增强 |

## 交易策略

| 策略 | 持仓数 | 换手率 | 说明 |
|------|--------|--------|------|
| `topk_dropout` | 50 | 中等 | 持有Top50，每期淘汰5只 |
| `topk_weighted` | 30 | 较高 | 集中持仓30只，每期换10只 |
| `conservative` | 80 | 低 | 分散持仓80只，每期换3只 |

## 工作流模板

1. **经典价值选股** — LightGBM + Alpha158 + TopK-Dropout
2. **深度学习选股** — LSTM + Alpha158
3. **保守稳健** — XGBoost + 大范围持仓
4. **Transformer 选股** — Transformer + TopK加权
5. **快速验证** — 线性模型 + 短期数据

## 项目结构

```
qlib_trader/
├── __init__.py          # 包初始化
├── __main__.py          # 入口点
├── app.py               # 主应用 & CLI 解析
├── config.py            # 配置管理 & 预设
├── data_manager.py      # 数据管理
├── model_manager.py     # 模型管理
├── backtest_engine.py   # 回测引擎
├── pipeline.py          # 一键流水线
├── utils.py             # 工具函数
└── README.md            # 说明文档
```

## 配置参数

### 市场

| 代码 | 市场 | 基准 | 选股范围 |
|------|------|------|----------|
| `cn` | 中国A股 | 沪深300 | CSI300 |
| `us` | 美股 | S&P500 | SP500 |

### 时间段

| 标识 | 名称 | 训练集 | 验证集 | 测试集 |
|------|------|--------|--------|--------|
| `default` | 标准时段 | 2008-2014 | 2015-2016 | 2017-2020 |
| `recent` | 近期时段 | 2015-2019 | 2020 | 2021-2023 |
| `short` | 短期验证 | 2016-2017 | 2018 | 2019-2020 |
| `long` | 长周期 | 2005-2015 | 2016-2017 | 2018-2023 |

## 前置要求

- Python 3.8+
- Qlib (`pip install pyqlib`)
- 网络连接（首次下载数据时需要）

## 注意事项

- 首次使用需要下载市场数据（约 200MB），程序会自动提示
- 深度学习模型（LSTM、Transformer等）训练时间较长，建议首次使用树模型
- 回测结果保存在 `mlruns/` 目录，可通过 `mlflow ui` 可视化查看
- 数据来源为 Yahoo Finance，仅供研究使用
