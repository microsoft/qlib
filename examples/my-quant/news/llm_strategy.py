import pandas as pd
import json
import os
import math
from openai import OpenAI
from tqdm import tqdm

# ================= ⚙️ 配置区 =================
# Qwen
# API_KEY = "sk-62b3731cd79a4ae2841b952e43d491fc" 
# BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"    

# MODEL_NAME = "qwen3-max-2026-01-23"
# BATCH_SIZE = 10

# Gemini
API_KEY = "ut_e697d4dc724e4e39b54a8fc2" 
BASE_URL = "https://hk1.augmunt.com"    

MODEL_NAME = "gemini-3-pro-preview"
BATCH_SIZE = 10
# ============================================

class BalancedAnalyst:
    def __init__(self):
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    def analyze_batch(self, news_text):
        """
        V3 Prompt: 理性客观，区分“情绪炒作”与“实质利好”
        """
        prompt = f"""
        你是一位身经百战的宏观基金经理。你的风格是：**不见兔子不撒鹰**。
        你既不是死多头，也不是死空头。你只做**确定性最高**的交易。

        【待分析新闻】：
        {news_text}

        【决策逻辑 - 请学习这些案例】：
        - Case 1 (利好兑现 -> SHORT): "某股今日涨停，散户疯狂涌入" -> 情绪过热，短期顶部，做空。
        - Case 2 (实质利好 -> LONG): "央行意外降息/国家发布万亿级产业规划" -> 基本面改善，趋势刚开始，做多。
        - Case 3 (蹭热点 -> SHORT): "某养猪企业宣布进军芯片" -> 纯粹忽悠，做空。
        - Case 4 (技术突破 -> LONG): "华为/智元发布颠覆性技术产品" -> 产业链受益，做多。

        【输出格式】：
        输出 JSON 对象，包含 "signals" 列表。
        每个信号：
        - "title": 标题
        - "direction": "LONG" 或 "SHORT"
        - "sector": 板块
        - "reason": 理性分析（为什么这次不一样？）
        - "score": 信心分数 (1-10)。**只输出分数 >= 7 的高确定性机会！**

        只输出 JSON。
        """

        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a rational, data-driven hedge fund manager."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1, 
                response_format={ "type": "json_object" }
            )
            return response.choices[0].message.content
        except Exception as e:
            print(e)
            return None

    def run(self, csv_path):
        print(f"⚖️ 启动'理性派'分析师 V3，读取: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # 按时间排序
        if '发布时间' in df.columns: df = df.sort_values(by='发布时间', ascending=True)
        
        all_signals = []
        
        # 批处理
        for i in tqdm(range(0, len(df), BATCH_SIZE)):
            batch_df = df.iloc[i : i + BATCH_SIZE]
            batch_text = ""
            for _, row in batch_df.iterrows():
                t = str(row.get('发布时间', ''))[-8:]
                title = str(row.get('新闻标题', row.get('标题', '')))
                batch_text += f"- [{t}] {title}\n"
            
            json_str = self.analyze_batch(batch_text)
            
            if json_str:
                try:
                    data = json.loads(json_str)
                    for sig in data.get("signals", []):
                        score = int(sig.get('score', 0))
                        if score >= 7: # 只看高分
                            all_signals.append(sig)
                            
                            # 打印
                            icon = "🟢 LONG " if sig['direction'] == "LONG" else "🔴 SHORT"
                            print(f"\n{icon} [{sig['sector']}] (信心:{score})")
                            print(f"   新闻: {sig['title']}")
                            print(f"   逻辑: {sig['reason']}")
                except:
                    pass

        # 保存
        output_file = csv_path.replace(".csv", "_signals_balanced.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_signals, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # 使用你的文件
    raw_csv = "data/macro_news/news_20260203_185837.csv" 
    
    # 路径兼容
    base_dir = r"D:\Quant-qlib-official\examples\my-quant"
    full_path = os.path.join(base_dir, raw_csv)
    if not os.path.exists(full_path): full_path = raw_csv

    if os.path.exists(full_path):
        analyst = BalancedAnalyst()
        analyst.run(full_path)