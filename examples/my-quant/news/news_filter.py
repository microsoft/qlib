import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ================= 🛡️ 路径与环境配置区 =================
PROJECT_ROOT = r"D:\Quant-qlib-official"
MODEL_CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "LLM-models")

# 强制创建目录
if not os.path.exists(MODEL_CACHE_DIR):
    try:
        os.makedirs(MODEL_CACHE_DIR)
    except Exception:
        pass

os.environ["HF_HOME"] = MODEL_CACHE_DIR
os.environ["SENTENCE_TRANSFORMERS_HOME"] = MODEL_CACHE_DIR
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# =======================================================

from sentence_transformers import SentenceTransformer

class NewsRobuster:
    def __init__(self):
        print(f"正在加载 Embedding 模型 (用于去重)...")
        try:
            self.model = SentenceTransformer(
                'paraphrase-multilingual-MiniLM-L12-v2', 
                cache_folder=MODEL_CACHE_DIR
            )
            print("✅ 模型加载成功！")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise e
            
        # 存向量
        self.history_embeddings = []
        # [新增] 存对应的标题，用来回溯是跟谁重复了
        self.history_titles = []

    def check_duplicate(self, text, threshold=0.85):
        """
        返回: (是否重复, 相似度, 原文标题)
        """
        if not self.history_embeddings:
            return False, 0.0, None
            
        current_emb = self.model.encode([text])
        
        # 计算与历史所有向量的相似度
        similarities = cosine_similarity(current_emb, np.vstack(self.history_embeddings))
        
        # 找到相似度最大的那个索引
        max_sim_idx = np.argmax(similarities)
        max_sim = similarities[0][max_sim_idx]
        
        if max_sim > threshold:
            # 找到对应的历史标题
            source_title = self.history_titles[max_sim_idx]
            return True, max_sim, source_title
        
        return False, max_sim, None

    def process_batch(self, df):
        clean_news = []
        print(f"\n开始处理 {len(df)} 条新闻 (无关键词过滤模式)...")
        
        # 1. 按时间正序排序 (确保最早的消息被保留，后面的被视为重复)
        # 尝试寻找时间列
        time_col = None
        for col in ['发布时间', '时间', 'datetime', 'time']:
            if col in df.columns:
                time_col = col
                break
        
        if time_col:
            df = df.sort_values(by=time_col, ascending=True)
            print(f"已按时间列 '{time_col}' 正序排列")
        
        for index, row in df.iterrows():
            title = str(row.get('新闻标题', row.get('标题', ''))) 
            content = str(row.get('新闻内容', row.get('内容', row.get('摘要', ''))))
            full_text = title + " " + content
            
            # --- 核心去重逻辑 ---
            is_dup, sim_score, source_title = self.check_duplicate(full_text)
            
            if is_dup:
                print("-" * 60)
                print(f"♻️ [发现重复] 相似度: {sim_score:.4f}")
                print(f"   当前新闻: {title[:30]}...")
                print(f"   重复来源: {source_title[:30]}...") # 打印出跟谁重复了
                print("-" * 60)
                continue
            
            # --- 入库 ---
            current_emb = self.model.encode([full_text])[0]
            self.history_embeddings.append(current_emb)
            self.history_titles.append(title) # 同时存入标题
            
            # 保持窗口大小
            if len(self.history_embeddings) > 2000:
                self.history_embeddings.pop(0)
                self.history_titles.pop(0)
            
            clean_news.append(row)
            # print(f"✅ [保留] {title[:20]}")

        return pd.DataFrame(clean_news)

if __name__ == "__main__":
    # ⚠️ 修改这里的文件名
    input_csv = "data/macro_news/news_20260203_185837.csv" 
    
    # 路径处理
    base_dir = os.path.join(PROJECT_ROOT, "examples", "my-quant")
    full_input_path = os.path.join(base_dir, input_csv)
    if not os.path.exists(full_input_path):
        full_input_path = input_csv

    if os.path.exists(full_input_path):
        robuster = NewsRobuster()
        df = pd.read_csv(full_input_path)
        
        df_clean = robuster.process_batch(df)
        
        output_path = full_input_path.replace("news_", "clean_news_")
        df_clean.to_csv(output_path, index=False, encoding="utf-8-sig")
        
        print(f"\n🎉 完成！原始: {len(df)} -> 剩余: {len(df_clean)}")
        print(f"文件: {output_path}")
    else:
        print(f"⚠️ 文件不存在: {full_input_path}")