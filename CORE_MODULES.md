# Qlib Core Modules Plan

## 保留的核心模块 (Core - KEEP)

### 根目录文件
- pyproject.toml (需要简化)
- setup.py (需要简化)
- README.md (需要更新)
- LICENSE
- .gitignore

### qlib/ 核心模块
```
qlib/
├── __init__.py                 # 初始化和核心API
├── config.py                   # 配置管理
├── constant.py                 # 常量定义
├── log.py                      # 日志系统
├── typehint.py                # 类型提示
├── data/                       # 数据层（核心）
│   ├── __init__.py
│   ├── base.py                # 基础数据类
│   ├── data.py                # 数据提供器
│   ├── cache.py               # 缓存系统
│   ├── ops.py                 # 基础运算
│   ├── filter.py              # 数据过滤
│   ├── dataset/               # 数据集处理
│   └── storage/               # 存储接口
├── model/                      # 模型层（仅基础）
│   ├── __init__.py
│   ├── base.py                # 模型基类
│   └── trainer.py             # 训练器基类
├── strategy/                   # 策略层（仅基础）
│   ├── __init__.py
│   └── base.py                # 策略基类
├── backtest/                   # 回测引擎（核心）
│   ├── __init__.py
│   ├── backtest.py
│   ├── exchange.py
│   ├── executor.py
│   ├── account.py
│   ├── position.py
│   └── report.py
├── workflow/                   # 工作流（简化）
│   ├── __init__.py
│   ├── recorder.py            # 实验记录
│   └── exp.py                 # 实验管理
└── utils/                      # 工具函数
    ├── __init__.py
    ├── time.py
    └── data.py
```

## 移除的外围模块 (Peripheral - REMOVE)

### 完全移除的目录
- examples/                     # 所有示例
- scripts/                      # 数据脚本
- tests/                        # 测试文件
- docs/                         # 文档
- .github/                      # GitHub Actions

### qlib/ 中移除的模块
- qlib/contrib/                 # 所有贡献模型
- qlib/rl/                      # 强化学习
- qlib/cli/                     # 命令行接口
- qlib/tests/                   # 内部测试

### qlib/data/ 中简化
- qlib/data/_libs/*.pyx         # Cython源文件（保留.so）

### qlib/model/ 中移除
- qlib/model/ens/               # 集成方法
- qlib/model/meta/              # 元学习
- qlib/model/riskmodel/         # 风险模型
- qlib/model/interpret/         # 模型解释

### qlib/workflow/ 中简化
- qlib/workflow/task/           # 任务管理
- qlib/workflow/online/         # 在线服务
- qlib/workflow/cli.py
- qlib/workflow/expm.py

### 根目录移除的文件
- Makefile
- Dockerfile
- build_docker_image.sh
- .dockerignore
- .pylintrc
- .mypy.ini
- .pre-commit-config.yaml
- .readthedocs.yaml
- .commitlintrc.js
- .deepsource.toml
- MANIFEST.in
- CHANGES.rst
- CHANGELOG.md
- CODE_OF_CONDUCT.md
- SECURITY.md

## 简化后的依赖
保留最小依赖集：
- numpy
- pandas>=1.1
- pyyaml
- filelock
- tqdm
- joblib
- loguru

移除的依赖：
- mlflow
- lightgbm
- torch
- tianshou
- cvxpy
- pymongo
- redis
- 等其他依赖