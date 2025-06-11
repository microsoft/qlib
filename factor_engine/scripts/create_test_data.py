import pandas as pd
from pathlib import Path

def create_test_parquet_data():
    """
    生成用于单元测试的模拟 Parquet 数据。
    - 数据包含 date, code, close, volume 字段。
    - 按 /test_data/{YYYY}/{YYYYMMDD}.parquet 的结构存储。
    """
    # 模拟数据
    dates = pd.to_datetime(['2023-01-03', '2023-01-03', '2023-01-04', '2023-01-04', '2023-01-05', '2023-01-05'])
    codes = ['SH600000', 'SZ000001', 'SH600000', 'SZ000001', 'SH600000', 'SZ000001']
    closes = [10.1, 20.1, 10.2, 20.2, 10.3, 20.3]
    volumes = [1000, 2000, 1100, 2100, 1200, 2200]
    
    df = pd.DataFrame({
        'date': dates,
        'code': codes,
        'close': closes,
        'volume': volumes
    })
    
    # 定义输出目录
    output_dir = Path(__file__).parent.parent / 'tests' / 'test_data'
    
    # 按日期分组并写入 Parquet 文件
    for date_val, group in df.groupby('date'):
        year = date_val.year
        yyyymmdd = date_val.strftime('%Y%m%d')
        
        # 创建年月目录
        file_dir = output_dir / str(year)
        file_dir.mkdir(parents=True, exist_ok=True)
        
        # 定义文件路径
        file_path = file_dir / f"{yyyymmdd}.parquet"
        
        # 写入数据
        group.to_parquet(file_path)
        print(f"写入测试数据到: {file_path}")

if __name__ == '__main__':
    create_test_parquet_data() 