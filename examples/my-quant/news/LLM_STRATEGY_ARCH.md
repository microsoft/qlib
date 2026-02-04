
# LLM-Driven Event Strategy Architecture (M.A.R.S.)

**Project:** Qlib Custom Quant Extension
**Status:** In Development
**Objective:** 构建一个高鲁棒性、低信噪比的事件驱动（Event-Driven）量化交易系统。
**Core Philosophy:** **Filter First, Reason Later.** (先过滤，后推理。用数学手段降噪，用 AI 手段推理，用回测手段验证。)

---

## 1. System Architecture Overview

我们将系统划分为四个独立的层级（Layers），数据像漏斗一样流过，逐层提纯。

```mermaid
graph TD
    %% Data Source
    Raw[Raw Data Stream<br/>(Akshare/EastMoney Global)] --> Layer1

    %% Layer 1: The Sieve
    subgraph "Layer 1: The Sieve (降噪层)"
        Layer1[Cleaning & Deduplication]
        L1_Rule[Rule Engine<br/>(Regex Blacklist)]
        L1_Vec[Vector DB<br/>(Embedding Similarity)]
        
        Raw --> L1_Rule
        L1_Rule --> L1_Vec
    end

    L1_Vec -- Clean Stream --> Layer2

    %% Layer 2: The Graph
    subgraph "Layer 2: The Structure (结构化层)"
        Layer2[Information Extractor]
        L2_LLM[Small LLM / Extractor]
        L2_JSON[Structured JSON<br/>(Event, Entities, Sentiment)]
        
        Layer2 --> L2_LLM
        L2_LLM --> L2_JSON
    end

    L2_JSON -- Structured Data --> Layer3

    %% Layer 3: The Brain
    subgraph "Layer 3: Adversarial Reasoning (对抗推理层)"
        Layer3[Multi-Agent Debate]
        Agent_A[Bull Agent<br/>寻找做多机会]
        Agent_B[Risk Agent<br/>寻找利空/证伪]
        Agent_C[Decision Maker<br/>加权打分]
        
        L2_JSON --> Agent_A
        L2_JSON --> Agent_B
        Agent_A --> Agent_C
        Agent_B --> Agent_C
    end

    Agent_C -- Trade Signal --> Layer4

    %% Layer 4: Validation
    subgraph "Layer 4: Execution (验证与执行层)"
        Layer4[Qlib Quant Engine]
        Q_Data[Technical Indicators]
        Q_Risk[Risk Control]
        
        Layer4 --> Q_Data
        Q_Data --> Q_Risk
        Q_Risk --> Final[Final Order]
    end

```

---

## 2. Layer Detail Specification

### Layer 1: The Sieve (物理降噪)

**目标**：在调用昂贵的 LLM API 之前，用低成本手段过滤 90% 的无效信息。

* **输入**：原始新闻流 (CSV/Stream)。
* **组件**：
* **Rule Engine**: 基于关键词的黑名单（如：`招聘`, `食堂`, `累计回购`, `大宗交易`）。
* **Embedding Filter**: 使用 `sentence-transformers` 将文本向量化。计算与过去 24 小时新闻的 Cosine Similarity。若相似度 > 0.85，视为重复/后续报道，丢弃。

* **技术栈**: Python, Pandas, ChromaDB (Vector Store), Sentence-Transformers.

### Layer 2: The Structure (信息结构化)

**目标**：将非结构化文本转化为机器可读的 JSON。

* **输入**：清洗后的单条新闻。
* **任务**：
* 实体提取 (NER): 涉及哪些公司、国家、商品？
* 事件分类: 宏观(Macro)、行业(Industry)、个股(Stock)、噪音(Noise)。

* **输出示例**:

```json
{
  "event_type": "MACRO_POLICY",
  "entities": ["央行", "MLF"],
  "action": "净投放",
  "value": "7000亿",
  "sentiment": 0.8
}

```

### Layer 3: The Brain (对抗性推理)

**目标**：模拟投研团队的辩论，解决 LLM 的“幻觉”和“盲从”问题。

* **Agent A (Opportunity Hunter)**: 激进风格。寻找一切可能的利好关联（如：中东局势 -> 利好黄金）。
* **Agent B (Risk Manager)**: 保守风格。专门“泼冷水”（如：利好已兑现？是否是捕风捉影？）。
* **Agent C (Judge)**: 综合 A 和 B 的发言，结合**短期记忆上下文**（Rolling Context），输出最终决策。
* **Prompt 策略**: Chain of Thought (CoT), Few-Shot Learning.

### Layer 4: Execution (量化验证)

**目标**：将逻辑信号转化为交易指令，并进行风控。

* **逻辑**:
* `If Signal(Gold) == Buy`: Check Qlib 数据。
* 如果 黄金板块 RSI > 80 (超买): **放弃买入** 或 **减少仓位**。
* 如果 黄金板块 处于均线上方: **执行买入**。

* **产物**: Qlib 可识别的 `signal.bin` 或实时交易指令。

---

## 3. Development Roadmap (开发路线)

### Phase 1: Infrastructure & Cleaning (当前阶段)

* [ ] **数据源**: 编写 `get_macro_news.py`，稳定获取东方财富全球直播数据。
* [ ] **向量库**: 引入 `ChromaDB` 或使用内存级向量缓存。
* [ ] **清洗器**: 编写 `NewsRobuster` 类，实现基于规则和向量的去重。
* [ ] **目标**: 能够从 100 条原始新闻中，精准筛选出 10 条核心新闻。

### Phase 2: The Reasoning Engine (智能体开发)

* [ ] **LLM 接入**: 封装 OpenAI/DeepSeek API 调用接口。
* [ ] **Prompt 工程**: 设计 Agent A/B/C 的 Prompt 模板。
* [ ] **知识库映射**: 建立 "板块 - 股票代码" 映射表 (Mapping Table)，让 LLM 知道 "光通信" 对应哪些股票。

### Phase 3: Integration & Testing (集成测试)

* [ ] **Pipeline 串联**: 实现 `run_pipeline()`，从下载数据到输出信号全自动运行。
* [ ] **历史回测**: 抓取过去 1 年的新闻，跑一遍系统，看生成的信号在 Qlib 回测中表现如何。

---

## 4. Key Configurations (配置备忘)

**Embedding Model:**
`paraphrase-multilingual-MiniLM-L12-v2` (轻量、支持中文、速度快)

**Thresholds:**

* Duplicate Similarity: `0.85`
* Spam Keyword List: `["招聘", "辞职", "监管函", "龙虎榜", "大宗交易", "会议", "调研"]`

**LLM Settings:**

* Temperature: `0.1` (保持推理稳定性，不要太发散)
* Model: `DeepSeek-V3` or `GPT-4o` (推理能力必须强)

---

## 5. Next Actions (下一步行动)

1. 创建 `data/vector_store` 目录。
2. 完善 `news_filter.py`，跑通 **Phase 1** 的去重逻辑。
3. 检查清洗后的数据质量，是否还需要调整黑名单关键词。
