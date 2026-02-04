"""
清理 instruments 文件，移除不存在的股票

这个脚本会检查 csi300.txt 等 instrument 文件中列出的股票，
如果对应的 bin 文件不存在，则从列表中移除。
"""
from pathlib import Path
import sys

def clean_instruments(
    qlib_dir="D:/Quant-qlib-official/data/qlib_data",
    instrument_name="csi300"
):
    """
    清理 instruments 文件

    Parameters
    ----------
    qlib_dir : str
        qlib 数据目录
    instrument_name : str
        instrument 文件名（不含 .txt）
    """
    qlib_dir = Path(qlib_dir).expanduser()
    instruments_file = qlib_dir / "instruments" / f"{instrument_name}.txt"
    features_dir = qlib_dir / "features"

    if not instruments_file.exists():
        print(f"错误: instruments 文件不存在: {instruments_file}")
        return

    if not features_dir.exists():
        print(f"错误: features 目录不存在: {features_dir}")
        return

    # 读取 instruments 文件
    with open(instruments_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"原始股票数量: {len(lines)}")

    # 检查每只股票是否有对应的 bin 文件目录
    valid_lines = []
    removed_stocks = []

    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) < 1:
            continue

        stock_code = parts[0]
        stock_dir = features_dir / stock_code.lower()
        factor_file = stock_dir / "factor.day.bin"

        # 检查 factor.day.bin 文件是否存在
        if factor_file.exists():
            valid_lines.append(line)
        else:
            removed_stocks.append(stock_code)
            print(f"  移除: {stock_code} (目录或 factor.day.bin 不存在)")

    print(f"\n移除的股票数量: {len(removed_stocks)}")
    print(f"保留的股票数量: {len(valid_lines)}")

    if removed_stocks:
        # 备份原文件
        backup_file = instruments_file.with_suffix('.txt.bak')
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"\n原文件已备份到: {backup_file}")

        # 写入清理后的文件
        with open(instruments_file, 'w', encoding='utf-8') as f:
            f.writelines(valid_lines)
        print(f"已更新: {instruments_file}")

        print(f"\n移除的股票列表:")
        for stock in removed_stocks:
            print(f"  - {stock}")
    else:
        print(f"\n没有需要移除的股票，文件完好。")

if __name__ == "__main__":
    import fire
    fire.Fire(clean_instruments)
