# Qlib Core - 精简核心版本

这是 Qlib 的精简核心版本，移除了所有外围模块，只保留了最核心的量化研究框架。

## 项目说明

此版本专注于 Qlib 的核心功能，适合：
- 深入研究 Qlib 核心架构
- 基于核心模块进行二次开发
- 构建自定义的量化研究系统

## 保留的核心模块

### 数据层 (qlib/data/)
- **base.py** - 数据表达式基类
- **data.py** - 数据提供器和加载器
- **cache.py** - 多级缓存系统
- **ops.py** - 基础数学运算
- **filter.py** - 数据过滤器
- **dataset/** - 数据集处理
- **storage/** - 存储接口
- **_libs/** - Cython 高性能扩展

### 模型层 (qlib/model/)
- **base.py** - 模型基类定义
- **trainer.py** - 训练器基类
- **utils.py** - 模型工具函数

### 策略层 (qlib/strategy/)
- **base.py** - 策略基类

### 回测层 (qlib/backtest/)
- **backtest.py** - 回测引擎
- **exchange.py** - 市场模拟
- **executor.py** - 订单执行
- **account.py** - 账户管理
- **position.py** - 持仓管理
- **report.py** - 报告生成

### 工作流层 (qlib/workflow/)
- **recorder.py** - 实验记录
- **exp.py** - 实验管理
- **utils.py** - 工具函数

### 工具层 (qlib/utils/)
- 时间处理、数据处理等基础工具

## 已移除的模块

- ✗ examples/ - 所有示例代码
- ✗ scripts/ - 数据下载脚本
- ✗ tests/ - 测试文件
- ✗ docs/ - 文档
- ✗ .github/ - CI/CD 工作流
- ✗ qlib/contrib/ - 贡献的模型和策略（30+ 模型）
- ✗ qlib/rl/ - 强化学习模块
- ✗ qlib/cli/ - 命令行接口
- ✗ 各种配置文件（Dockerfile, Makefile 等）

## 安装

### 1. 安装依赖

```bash
pip install -e .
```

### 2. 编译 Cython 扩展

```bash
python setup.py build_ext --inplace
```

## 使用指南

### 初始化 Qlib

```python
import qlib
from qlib.config import REG_CN

# 初始化
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
```

### 数据处理

```python
from qlib.data import D
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

# 获取数据
df = D.features(["SH600000"], ["$close", "$volume"], start_time="2020-01-01", end_time="2021-01-01")
```

### 自定义模型

```python
from qlib.model.base import Model

class MyModel(Model):
    def fit(self, dataset):
        # 实现训练逻辑
        pass
    
    def predict(self, dataset):
        # 实现预测逻辑
        pass
```

### 回测

```python
from qlib.backtest import backtest
from qlib.strategy.base import BaseStrategy

class MyStrategy(BaseStrategy):
    def generate_order_list(self, score_series, current, trade_exchange):
        # 实现策略逻辑
        pass

# 运行回测
report = backtest(pred, strategy=MyStrategy(), executor=executor)
```

## 核心依赖

- numpy - 数值计算
- pandas>=1.1 - 数据处理
- pyyaml - 配置文件
- filelock - 文件锁
- tqdm - 进度条
- joblib - 并行处理
- loguru - 日志系统
- pyarrow - 数据序列化

## 开发建议

1. **扩展模型**：继承 `qlib.model.base.Model` 实现自定义模型
2. **自定义策略**：继承 `qlib.strategy.base.BaseStrategy` 实现交易策略
3. **数据处理**：使用 `qlib.data.ops` 中的运算符进行特征工程
4. **实验管理**：使用 `qlib.workflow.recorder` 记录实验

## 注意事项

1. 本版本移除了所有预实现的模型，需要自行实现
2. 没有数据下载脚本，需要自行准备数据
3. Cython 扩展需要编译后才能使用
4. 建议基于此核心版本构建自己的量化研究系统

## 相关文件

- `CORE_MODULES.md` - 详细的模块保留/移除清单
- `项目结构说明.md` - 原始项目的完整结构说明
- `CLAUDE.md` - Claude AI 使用指南

## License

MIT License (保留原项目协议)