#!/usr/bin/env python3

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.qlib_service import qlib_service

# 测试QLIBService
print("测试QLIBService...")

# 初始化QLIB
print("初始化QLIB...")
provider_uri = os.path.expanduser("~/.qlib/qlib_data/cn_data")
result = qlib_service.init_qlib(provider_uri=provider_uri, force=True)
print(f"初始化结果: {result}")

# 获取股票代码
print("获取股票代码...")
instruments = qlib_service.get_instruments()
print(f"获取到的股票代码类型: {type(instruments)}")
print(f"获取到的股票代码内容: {instruments}")
print(f"获取到的股票代码数量: {len(instruments)}")

# 转换为列表
if hasattr(instruments, 'keys'):
    instruments_list = list(instruments.keys())
elif hasattr(instruments, '__iter__'):
    instruments_list = list(instruments)
else:
    instruments_list = [instruments]

print(f"转换后的股票代码列表: {instruments_list}")

# 测试获取股票数据
if instruments:
    code = instruments[0]
    print(f"测试获取股票 {code} 的数据...")
    stock_data = qlib_service.get_stock_data(code, "2025-11-01", "2025-11-28")
    print(f"获取到股票 {code} 的数据，共 {len(stock_data)} 个键值对")
    print(f"前5个键值对: {list(stock_data.items())[:5]}")
else:
    print("没有获取到股票代码，无法测试获取股票数据")

print("测试完成")
