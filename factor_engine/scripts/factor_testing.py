# factor_engine/scripts/factor_testing.py
import pandas as pd
from pathlib import Path

from factor_engine.data_layer.loader import load_daily_bundle
from factor_engine.data_layer.containers import PanelContainer
from factor_engine.registry import op_registry

# CRITICAL FIX: Import the operators module to execute all @op_registry.register decorators
import factor_engine.operators

def to_panel(df: pd.DataFrame, field: str) -> PanelContainer:
    """
    Converts a long-format DataFrame to a wide-format PanelContainer.
    """
    panel_df = df[field].unstack(level='instrument')
    panel_df.index = pd.to_datetime(panel_df.index)
    return PanelContainer(panel_df)

def main():
    """
    Main script to load data and test factor calculations.
    """
    print("--- 因子测试脚本开始 ---")

    # 1. 加载数据
    # ---------------------------------------------
    data_path = Path(__file__).parent.parent.parent / "database"
    print(f"从 {data_path} 加载数据...")
    
    # 为了演示，我们只加载少量股票和较短时间范围的数据
    instruments = ["000001.SZ", "000002.SZ", "000004.SZ", "000005.SZ", "000006.SZ"]
    df_bundle = load_daily_bundle(
        data_path=data_path,
        start_time="2024-01-01",
        end_time="2024-01-31",
        instruments=instruments
    )

    if df_bundle.empty:
        print("错误：未能加载数据，请检查路径和数据是否存在。")
        return

    print(f"成功加载 {len(df_bundle)} 条数据。")
    print("数据样本:")
    print(df_bundle.head())

    # 2. 预处理
    # ---------------------------------------------
    print("\n--- 数据预处理 ---")
    
    # 计算日收益率 ret_1d
    # 我们按股票分组计算，以避免受到其他股票数据的影响
    df_bundle['ret_1d'] = df_bundle.groupby(level='instrument')['close'].pct_change()
    
    # 将所需字段转换为 PanelContainer
    close_panel = to_panel(df_bundle, 'close')
    volume_panel = to_panel(df_bundle, 'volume')
    ret_1d_panel = to_panel(df_bundle, 'ret_1d')
    
    print("已将 'close', 'volume', 'ret_1d' 转换为宽格式面板数据。")

    # 3. 因子计算
    # ---------------------------------------------
    print("\n--- 因子计算 ---")

    # --- 因子1: 动量类 MA(5) - MA(10) ---
    print("\n计算因子: MA(5) - MA(10)")
    try:
        ma5_op = op_registry.get('ts_mean', window=5)
        ma10_op = op_registry.get('ts_mean', window=10)
        sub_op = op_registry.get('subtract')

        ma5 = ma5_op(close_panel)
        ma10 = ma10_op(close_panel)
        factor_momentum = sub_op(ma5, ma10)
        
        print("因子计算成功。因子值尾部数据：")
        print(factor_momentum.get_data().tail())
        print("\n因子描述性统计:")
        print(factor_momentum.get_data().describe())

    except Exception as e:
        print(f"计算因子时出错: {e}")

    # --- 因子2: 波动率类 ts_std(ret_1d, 3) ---
    print("\n计算因子: ts_std(ret_1d, 3)")
    try:
        std3_op = op_registry.get('ts_std', window=3)
        factor_volatility = std3_op(ret_1d_panel)

        print("因子计算成功。因子值尾部数据：")
        print(factor_volatility.get_data().tail())
        print("\n因子描述性统计:")
        print(factor_volatility.get_data().describe())

    except Exception as e:
        print(f"计算因子时出错: {e}")
        
    # --- 因子3: 量价背离类 corr(ret_1d, volume, 5) ---
    print("\n计算因子: corr(ret_1d, volume, 5)")
    try:
        corr5_op = op_registry.get('ts_corr', window=5)
        factor_corr = corr5_op(ret_1d_panel, volume_panel)
        
        print("因子计算成功。因子值尾部数据：")
        print(factor_corr.get_data().tail())
        print("\n因子描述性统计:")
        print(factor_corr.get_data().describe())
        
    except Exception as e:
        print(f"计算因子时出错: {e}")

    print("\n--- 因子测试脚本结束 ---")

if __name__ == "__main__":
    main() 