"""
QLib Trader 入口点
用法: python -m qlib_trader [命令]

示例:
    python -m qlib_trader              # 启动交互式菜单
    python -m qlib_trader --help       # 查看帮助
    python -m qlib_trader pipeline     # 运行一键流水线
    python -m qlib_trader data         # 数据管理
    python -m qlib_trader train        # 模型训练
    python -m qlib_trader backtest     # 回测
"""

from qlib_trader.app import main

if __name__ == "__main__":
    main()
