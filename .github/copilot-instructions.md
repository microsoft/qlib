# Qlib AI 编程助手指南

## 项目概述
Qlib 是微软面向AI的量化投资平台，支持基于机器学习的交易策略研究。该平台提供从数据处理到模型训练、回测和部署的完整ML流水线。

## 核心架构

### 数据层 (`qlib/data/`)
- **提供者**: LocalProvider、ClientProvider 处理数据访问模式
- **数据集**: Alpha158（158个技术指标）、Alpha360（360个特征）是主要数据集
- **表达式**: 使用金融领域专用语言，如 `$close`、`Ref($close, 1)`、`Mean($close, 5)`
- **处理器**: `DataHandlerLP` 使用可配置的处理器处理原始数据（标准化、过滤）

### 模型层 (`qlib/model/`, `qlib/contrib/model/`)
- 模型继承自 `Model` 基类，具有 `.fit()` 和 `.predict()` 方法
- 贡献模型包括 LightGBM、神经网络（LSTM、GRU、Transformer变体）
- 模型配置指定 `class`、`module_path` 和初始化的 `kwargs`

### 工作流层 (`qlib/workflow/`)
- **记录器 (R)**: 使用MLflow后端的全局实验跟踪系统
- **QlibRecorder**: 通过 `R.start()`、`R.log_params()`、`R.log_metrics()` 管理实验
- **配置驱动工作流**: YAML配置定义整个ML流水线

### 策略与执行 (`qlib/strategy/`, `qlib/backtest/`)
- **TopkDropoutStrategy**: 选择前k只股票，丢弃后n只以减少换手
- **BacktestTracker**: 包含交易成本、滑点的投资组合模拟
- **NestedExecutor**: 多层策略优化（投资组合+订单执行）

## 开发工作流程

### 运行模型
```bash
# 单一模型执行
cd examples && qrun benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml

# 调试模式
python -m pdb qlib/cli/run.py examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml

# 批量模型比较
python examples/run_all_model.py run 3 lightgbm Alpha158
```

### 测试模式
```bash
# 快速测试（排除慢速标记）
cd tests && python -m pytest . -m "not slow"

# 特定测试类别
python -m pytest tests/model/ -v
python -m pytest tests/data/ -k "test_alpha" 
```

### 数据管理
```bash
# 下载公开数据
python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn

# 健康检查
python scripts/check_data_health.py check_data --qlib_dir ~/.qlib/qlib_data/cn_data
```

## 关键约定

### 配置模式
- **嵌套配置**: 使用 `<MODEL>`, `<DATASET>` 占位符进行交叉引用
- **市场区域**: `REG_CN`（中国）、`REG_US`（美国）影响数据路径和交易规则
- **时间分段**: `segments: {train: [start, end], valid: [...], test: [...]}`

### 数据处理器处理
```python
# 学习vs推理的默认处理器
_DEFAULT_LEARN_PROCESSORS = [
    {"class": "DropnaLabel"},
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
]
_DEFAULT_INFER_PROCESSORS = [
    {"class": "ProcessInf", "kwargs": {}},
    {"class": "ZScoreNorm", "kwargs": {}},
    {"class": "Fillna", "kwargs": {}},
]
```

### 表达式语法
- `$close`, `$volume`, `$high`, `$low` 用于OHLCV数据
- `Ref($close, 1)` 用于回望（昨日收盘价）
- `Mean($close, 5)` 用于滚动窗口
- `Greater($close, Ref($close, 1))` 用于条件判断

## 开发指南

### 添加新模型
1. 在 `examples/benchmarks/ModelName/` 中创建文件夹
2. 包含 `requirements.txt`、`README.md`、`workflow_config_modelname_Alpha158.yaml`
3. 在 `qlib/contrib/model/` 中按照现有模式实现
4. 模型类需要 `.fit(dataset)` 和 `.predict(dataset)` 方法

### 内存与性能
- 使用 `NUM_USABLE_CPU = max(multiprocessing.cpu_count() - 2, 1)` 进行并行处理
- 缓存设置：`expression_cache`、`dataset_cache` 用于性能优化
- 高频数据需要 `"maxtasksperchild": 1` 以避免内存泄漏

### 错误处理
- 使用 `qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")` 初始化Qlib
- 实验前检查数据健康状况
- 使用 `R.start()` 上下文管理器正确处理实验生命周期

### 测试集成 
- 模型应同时支持Alpha158和Alpha360数据集
- 在适用时验证多个市场区域（cn/us）
- 用不同时间段测试以确保时间稳健性

## 常见问题
- 运行 `qrun` 前务必 `cd examples` 以避免导入冲突
- macOS上的LightGBM需要 `brew install libomp` 
- Windows/macOS使用不同的多进程方法 - 检查平台兼容性
- 表达式缓存依赖Redis进行分布式设置
- Pandas版本兼容性：在groupby操作中设置 `group_key=False`