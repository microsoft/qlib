import inspect
import importlib
from typing import Dict, Callable

def load_operators() -> Dict[str, Callable]:
    """
    动态加载 factor_engine.operators 模块中所有以 'op_' 开头的算子函数。

    Returns:
        一个将算子名称 (e.g., 'op_add') 映射到其可调用函数对象的字典。
    """
    operators_map = {}
    module_name = 'factor_engine.operators'
    try:
        operators_module = importlib.import_module(module_name)
        for name, func in inspect.getmembers(operators_module, inspect.isfunction):
            if name.startswith('op_'):
                operators_map[name] = func
    except ImportError as e:
        print(f"无法加载模块 '{module_name}': {e}")
        # 或者根据需要进行更严格的错误处理
        raise
        
    return operators_map

if __name__ == '__main__':
    # 一个简单的测试，展示如何使用该函数
    # 需要在项目根目录下并设置 PYTHONPATH=. 才能运行
    # (e.g., python -m factor_engine.engine.utils)
    try:
        ops = load_operators()
        print("成功加载的算子:")
        for name, func in ops.items():
            print(f"- {name}: {func.__doc__.splitlines()[0] if func.__doc__ else 'No docstring'}")
    except ImportError:
        print("\n请确保从项目根目录运行此脚本，或已设置 PYTHONPATH。") 