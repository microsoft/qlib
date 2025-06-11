import sys
from pathlib import Path

# 将项目根目录 (factor_engine) 添加到 Python 搜索路径
# 这能确保测试在运行时可以找到 'data_layer' 等模块
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root)) 