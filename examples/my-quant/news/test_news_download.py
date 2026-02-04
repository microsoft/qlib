import akshare as ak
import pandas as pd

def get_demo_news(symbol="600519"):
    """
    下载指定股票的新闻数据 (演示用)
    :param symbol: 股票代码 (例如 '600519' 贵州茅台)
    """
    print(f"正在下载股票 {symbol} 的新闻...")
    
    try:
        # 调用东方财富个股新闻接口
        # 注意：akshare 接口可能会更新，如果报错请检查 akshare 版本
        news_df = ak.stock_news_em(symbol=symbol)
        
        # 简单清洗：只保留我们关心的列
        # 原始列名通常包含: '关键词', '新闻标题', '新闻内容', '发布时间', '文章来源' 等
        if not news_df.empty:
            # 打印列名看看都有什么
            print(f"\n获取到的原始列名: {news_df.columns.tolist()}")
            
            # 选取核心字段
            # 注意：实际列名可能略有不同，这里做个防御性选择
            target_cols = ['发布时间', '新闻标题', '新闻内容', '文章来源']
            available_cols = [c for c in target_cols if c in news_df.columns]
            
            clean_df = news_df[available_cols]
            
            # 展示前 3 条数据
            print("\n--- 数据预览 (Top 3) ---")
            print(clean_df.head(3).to_markdown(index=False))
            
            # 保存个样本文件方便查看
            filename = f"news_sample_{symbol}.csv"
            clean_df.to_csv(filename, index=False, encoding="utf-8-sig")
            print(f"\n已保存样本文件到: {filename}")
            
            return clean_df
        else:
            print("未获取到数据，可能是股票代码错误或该股近期无新闻。")
            return None

    except Exception as e:
        print(f"下载出错: {e}")
        return None

if __name__ == "__main__":
    # 测试一下贵州茅台 (600519)
    df = get_demo_news("600519")