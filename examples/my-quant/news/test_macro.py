import akshare as ak
import pandas as pd
import os
from datetime import datetime

def get_and_save_macro_news():
    print("正在尝试获取【东方财富-全球财经直播】数据...")
    
    try:
        # 1. 获取数据
        df = ak.stock_info_global_em()
        
        if df is None or df.empty:
            print("❌ 数据为空，请检查网络或接口状态。")
            return None

        # 2. 准备保存路径
        # 在当前目录下创建 data/macro_news 文件夹
        save_dir = os.path.join("data", "macro_news")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"📂 已创建文件夹: {save_dir}")

        # 3. 生成文件名 (带时间戳，例如: news_20231027_153022.csv)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"news_{timestamp}.csv"
        file_path = os.path.join(save_dir, filename)

        # 4. 数据清洗与保存
        # 东方财富这个接口返回的列可能很多，为了阅读方便，我们把关键列放到前面
        # 尝试寻找常见的列名
        priority_cols = ['发布时间', '时间', '新闻标题', '内容', '消息内容']
        existing_cols = [c for c in priority_cols if c in df.columns]
        other_cols = [c for c in df.columns if c not in priority_cols]
        
        # 重新排列列顺序
        df = df[existing_cols + other_cols]

        # 保存为 CSV (utf-8-sig 可以在 Excel 中正常显示中文)
        df.to_csv(file_path, index=False, encoding="utf-8-sig")
        
        print(f"✅ 成功! 获取到 {len(df)} 条新闻。")
        print(f"💾 文件已保存至: {file_path}")
        
        # 打印前3条看看
        print("\n--- 数据预览 (Top 3) ---")
        print(df[existing_cols].head(3).to_markdown(index=False))
        
        return df

    except Exception as e:
        print(f"❌ 发生错误: {e}")
        # 如果是 akshare 版本问题，提示升级
        if "has no attribute" in str(e):
            print("💡 提示: 可能是 akshare 版本过旧，请运行: pip install --upgrade akshare")
        return None

if __name__ == "__main__":
    get_and_save_macro_news()