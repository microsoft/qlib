# A股/港股/ETF 自动化交易系统

## 📖 简介

这是一个基于 Qlib 构建的完整自动化量化交易系统，支持 A股、港股、ETF 的每日自动化交易流程。

## ✨ 核心功能

- **多市场数据更新**: 自动更新 A股、港股、ETF 数据
- **智能信号生成**: 滚动训练 + 多模型集成预测
- **组合优化**: 基于风险约束的凸优化，生成目标权重和订单
- **回测评估**: 模拟真实交易环境，评估策略表现
- **自动化报告**: 生成 HTML/Excel 格式的交易建议和风险分析报告

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 初始化数据

```bash
# 下载 A股 数据
python scripts/data_collector/yahoo/collector.py download_data \
    --source_dir ~/.qlib/stock_data/source/cn \
    --region CN \
    --start 2010-01-01 \
    --end 2024-12-31

# 转换为 Qlib 格式
python scripts/data_collector/yahoo/collector.py dump_bin \
    --csv_path ~/.qlib/stock_data/source/cn \
    --qlib_dir ~/.qlib/qlib_data/cn_data \
    --include_fields open,close,high,low,volume
```

### 3. 配置系统

编辑 `config.yaml`：

```yaml
data_dir: ~/.qlib/qlib_data
market: cn
benchmark: SH000300

strategy:
  topk: 30
  rebalance_freq: daily

risk_control:
  max_turnover: 0.3
  max_position: 0.10
```

### 4. 运行自动化流程

```bash
# 运行今天的流程
python scripts/auto_trading/main_controller.py --config config.yaml

# 指定日期运行
python scripts/auto_trading/main_controller.py --config config.yaml --date 2024-11-14
```

### 5. 配置定时任务

```bash
# 配置 Crontab（每个交易日下午4点运行）
bash scripts/auto_trading/setup_cron.sh
```

## 📁 文件结构

```
scripts/auto_trading/
├── __init__.py                  # 包初始化
├── README.md                    # 本文档
├── IMPLEMENTATION_PLAN.md       # 详细实现方案
├── config.yaml                  # 系统配置文件
├── data_updater.py             # 数据更新模块
├── signal_generator.py         # 信号生成模块
├── portfolio_optimizer.py      # 组合优化模块
├── backtester.py               # 回测评估模块
├── report_generator.py         # 报告生成模块
├── main_controller.py          # 主控制器
├── setup_cron.sh               # Cron配置脚本
└── utils/                      # 工具模块
    ├── position_manager.py     # 持仓管理
    ├── benchmark_loader.py     # 基准加载
    └── notification.py         # 通知服务
```

## 🔧 模块说明

### MultiMarketDataUpdater
负责多市场数据的自动更新和质量检查。

```python
from scripts.auto_trading import MultiMarketDataUpdater

updater = MultiMarketDataUpdater(base_dir='~/.qlib/qlib_data')
updater.update_all_markets()  # 更新所有市场
```

### SignalGenerator
生成交易信号，支持滚动训练和多模型集成。

```python
from scripts.auto_trading import SignalGenerator

generator = SignalGenerator(market='cn')
signals, quality = generator.generate_signals(date='2024-11-14')
```

### PortfolioOptimizer
基于信号和当前持仓生成优化的目标权重和订单。

```python
from scripts.auto_trading import PortfolioOptimizer

optimizer = PortfolioOptimizer(market='cn')
plan = optimizer.generate_rebalance_plan(
    signals=signals,
    current_positions=current_positions,
    benchmark_weights=benchmark_weights
)
```

### BacktestEvaluator
回测验证策略表现。

```python
from scripts.auto_trading import BacktestEvaluator

evaluator = BacktestEvaluator(market='cn')
results = evaluator.run_backtest(
    signals=signals,
    start_date='2024-01-01',
    end_date='2024-11-14'
)
```

### ReportGenerator
生成 HTML/Excel 报告。

```python
from scripts.auto_trading import ReportGenerator

reporter = ReportGenerator(output_dir='./reports')
files = reporter.generate_daily_report(
    date='2024-11-14',
    signals=signals,
    rebalance_plan=plan,
    backtest_results=results,
    risk_analysis=risk_analysis
)
```

## 📊 输出报告

运行完成后会生成以下文件：

- `reports/report_YYYYMMDD.html` - 可视化日报
- `reports/orders_YYYYMMDD.xlsx` - 交易订单表
- `logs/auto_trading_YYYYMMDD.log` - 运行日志

## ⚙️ 配置参数

### 策略参数
- `topk`: 持仓股票数量（默认30）
- `rebalance_freq`: 再平衡频率（默认daily）

### 风险控制
- `max_turnover`: 最大换手率（默认0.3）
- `max_position`: 单只股票最大权重（默认0.10）
- `tracking_error`: 跟踪误差限制（默认0.05）

### 交易成本
- `open_cost`: 开仓成本（默认0.0005）
- `close_cost`: 平仓成本（默认0.0015）
- `min_cost`: 最小手续费（默认5元）

## 🧪 测试

```bash
# 运行单元测试
pytest tests/auto_trading/

# 测试数据更新
pytest tests/auto_trading/test_data_updater.py

# 测试完整流程
pytest tests/auto_trading/test_integration.py
```

## 📚 文档

详细的实现方案请参考：[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

## 📧 联系方式

如有问题，请联系: your_email@example.com
