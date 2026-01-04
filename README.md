# 🚀 ScholarRAG: Graph-Enhanced Research Assistant

**ScholarRAG** 是一个面向科研场景的智能 RAG (Retrieval-Augmented Generation) Agent。它不仅仅是一个简单的文献搜索工具，而是一个集成了 **OpenAlex 广度知识发现**、**知识图谱概念导航**、**本地向量库深度记忆** 以及 **BGE 重排序** 的全链路科研助手。

本项目旨在解决传统 LLM 在科研领域存在的“幻觉”与“引用不实”问题，通过双阶段检索漏斗与思维链 (CoT) 生成，提供可溯源、高可信度的学术回答。

---

## 🌟 核心特性 (Key Features)

1.  **双阶段检索漏斗 (Dual-Stage Retrieval Funnel)**:
    * **Stage 1 (Wide & Rerank)**: 利用 OpenAlex API 快速召回 Top-N 元数据，并引入 **BAAI/bge-reranker-v2-m3** 模型对候选文献进行语义级重排序 (Reranking)，确保相关性。
    * **Stage 2 (Deep & Vector)**: 自动下载全文 (PDF/ArXiv)，解析清洗后存入本地 **ChromaDB** 向量库，进行切片级语义检索。

2.  **图谱增强导航 (Graph-Enhanced Navigation)**:
    * 利用 OpenAlex 庞大的 Concept Ontology（概念本体），通过 LLM 将自然语言 Query 映射为精确的学术实体 ID。
    * 实现“指哪打哪”的精准检索，有效解决术语歧义问题。

3.  **工程化解耦 (Engineered Decoupling)**:
    * **Prompt 解耦**: 所有 System Prompt (Review/Explain/Expansion) 均抽离为 JSON 配置文件，便于微调且不污染业务代码。
    * **UI/Logic 分离**: 前端 (Streamlit) 与后端 (RAG Core) 逻辑彻底分离，确保代码的可维护性。

4.  **增量式知识库 (Incremental Knowledge Base)**:
    * 自动维护 `data/vector_store`，对已下载的论文进行哈希去重。一次下载，永久复用，越用越快。

---

## 📂 项目架构 (Project Structure)

代码结构：

```text
scholar_rag/
├── data/                       # [数据持久化层] (Git Ignore)
│   ├── cache/pdfs/             # PDF 原文缓存 (命名: ID_Title.pdf)
│   ├── vector_store/           # ChromaDB 本地向量索引
│   ├── scholar_ui.db           # SQLite 数据库 (UI 用户数据/历史记录)
│   └── papers_debug.txt        # 检索/下载过程的详细 Debug 日志
├── src/                        # [核心源码层]
│   ├── config.py               # 全局配置加载器 (读取 .env)
│   ├── core/                   # >> 大脑层 (生成与决策)
│   │   ├── llm.py              # LLM 服务封装 (支持 OpenAI/SiliconFlow/OpenRouter)
│   │   ├── generator.py        # RAG 生成器 (上下文组装 + CoT 执行)
│   │   └── prompts.json        # 📝 [配置] 生成任务的 Prompt 模板库
│   ├── graph/                  # >> 导航层 (意图理解)
│   │   ├── expansion.py        # Query -> Concept ID 的映射逻辑 (含打分算法)
│   │   └── expansion_prompts.json # 📝 [配置] 图谱扩展专用的 Prompt
│   ├── retrieval/              # >> 记忆与感知层 (检索执行)
│   │   ├── base.py             # 抽象基类
│   │   ├── openalex.py         # ✅ 广度检索 + BGE Reranker 重排序
│   │   └── vector_store.py     # ✅ 深度检索 (PDF下载/清洗/向量化/召回)
│   └── utils/                  # >> 基础设施
│       └── logger.py           # 统一日志模块
├── ui/                         # [交互层] (Web UI)
│   ├── app.py                  # 🚀 Streamlit 主入口 (路由与页面渲染)
│   ├── logic.py                # 🧠 UI 业务逻辑 (会话管理/RAG调用封装)
│   ├── db.py                   # 💾 SQLite ORM (用户/广场数据管理)
│   └── style.css               # 🎨 自定义样式表
├── .env                        # 🔐 核心环境配置 (API Keys)
├── main.py                     # 🚀 CLI 命令行启动入口
└── requirements.txt            # 依赖列表
```

## ⚙ 环境配置 Windows + Conda
```bash
conda create -n scholar_rag python=3.10

conda activate scholar_rag

pip install -r requirements.txt #在项目根目录下运行
```

## 配置文件
设置 API Key (.env)

在项目根目录创建一个名为 .env 的文件（无后缀），填入您的 LLM 密钥：
```bash
OPENAI_API_KEY=sk-
OPENAI_BASE_URL=https://api.siliconflow.cn/v1/
LLM_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
EMBEDDING_MODEL_NAME=BAAI/bge-m3
LLM_TEMPERATURE=0.3
# LLM 上下文限制 (Qwen2.5 支持 32k，不要设 2000 那么小)
LLM_MAX_TOKENS=8000

# --- OpenAlex 配置 ---
# 【必须】填入真实邮箱，否则接口会被限流
OPENALEX_EMAIL=""

# --- RAG 检索配置 ---
# 切片大小 (建议 500-1000)
RAG_CHUNK_SIZE=1000
# 切片之间的重叠字符数 (建议 10-20%，防止上下文断裂)
RAG_CHUNK_OVERLAP=100

# 2. Embedding API 策略 (用于 SiliconFlowEmbeddingFunction)
# 每次请求发送给硅基流动的最大片段数 (防止 Error 413，建议 32-64)
EMBEDDING_BATCH_SIZE=32

# --- RAG 漏斗配置 (关键修改) ---
# 阶段1: OpenAlex 广度搜索下载的数量 (Wide)
RAG_DOWNLOAD_K=20
# 阶段2: 向量库深度检索召回的数量 (Deep)
RAG_RETRIEVAL_K=15
```
设置.gitignore文件

在项目根目录创建一个名为.gitignore的文件（无后缀），防止.env中敏感内容上传，防止过大的向量库数据上传
```bash
# --- 安全与私密配置 (绝对不要上传) ---
.env
.env.local
config/secrets.yaml

# --- 数据与缓存 (太大了，不要上传) ---
data/
data/cache/
data/vector_store/
*.parquet
*.csv
*.jsonl

# --- Python 编译文件与环境 ---
__pycache__/
*.py[cod]
*$py.class
venv/
.venv/
env/
.env/

# --- IDE 配置 (个人设置，不要上传) ---
.idea/
.vscode/
*.swp
.DS_Store

# --- 日志文件 ---
*.log
logs/

# --- Jupyter Notebook ---
.ipynb_checkpoints/
```

## 启动命令
本项目通过命令行接口 (CLI) 运行。

注意：首次运行 Reranker 需要较大内存及网络带宽。

1. 基础模式 (关键词检索)
直接使用关键词在 OpenAlex 中检索并生成综述：
```bash
python main.py --query "Large Language Models in Healthcare" --mode review

# 例子：询问为什么深度学习泛化能力强
python main.py --query "Why does Deep Learning generalize well despite overparameterization?" --mode explain

python main.py --query "Optimization algorithms inspired by biological evolution" --mode inspire
```
2. 图增强模式 (推荐)
启用知识图谱扩展功能。系统会自动定位该 Query 所属的 OpenAlex 概念节点 (Concept Node)，并在此概念子图中进行精准检索，排除同名异义词的干扰：
```bash
python main.py --query "Machine Learning" --mode review --use_graph
```

## 🖥️ Web UI 界面 (Streamlit)

项目包含一个基于 Streamlit 的可视化界面，模拟 Gemini 交互风格，支持多轮对话与社区互动。启动方式:
```bash
streamlit run ui/app.py
```

## 🧩 模块功能详解

1. 导航层 src/graph/
    - expansion.py: 系统的“罗盘”。

        - Concept Mapping: 调用 LLM 将模糊的用户 Query 转换为 OpenAlex 标准学术概念（例如：将 "AI drawing" 映射为 "Generative Art"）。

        - Scoring Algorithm: 引入打分机制（名称匹配度 + 引用热度 - 层级惩罚），从候选概念中选出最优 Concept ID。

        - Config Loader: 动态加载 expansion_prompts.json，实现 Prompt 与代码分离。

2. 检索层 src/retrieval/
    - openalex.py (Stage 1: Broad Search):

        - Search Execution: 接收 Query 或 Concept ID，调用 OpenAlex API 获取 Top-K 元数据。

        - BGE Reranking: 关键特性。懒加载 BAAI/bge-reranker-v2-m3 模型，对召回的初步结果进行语义重排序，大幅提升 Top-N 的准确率。

        - Fallback Logic: 若 Concept 检索为空，自动降级为文本关键词检索。

    - vector_store.py (Stage 2: Deep Search):

        - Anti-Crawler Fetcher: 内置伪造 User-Agent 和异步下载器 (aiohttp)，针对 IEEE/ScienceDirect/ArXiv 优化下载成功率。

        - Vectorization: 使用 SiliconFlow 或 OpenAI Embedding API，并实现了 自动 Batching 机制，防止 API 报 413 Payload Too Large 错误。

        - Traceability: 存入向量库的不仅是 Chunk 文本，还包含 Title, URL, Year 等元数据，确保生成时可精准引用。

3. 核心层 src/core/
    - generator.py:

        - Context Formatting: 将检索到的 Chunks 格式化为 [Source ID] 结构。

        - Dynamic Prompting: 根据 prompts.json 中的任务类型 (Review/Explain/Inspire)，动态切换思维链 (CoT) 策略。

4. 交互层 ui/
    - logic.py:
        - State Management: 桥接 Streamlit 的 st.session_state 与 RAG 后端。

        - Recursive Summarization: 实现了多轮对话的摘要算法，防止 Context 长度爆炸。

    - db.py:

        - 使用 SQLite 管理用户系统和“灵感广场”数据，支持轻量级的用户隔离。


## 📝 开发计划 (Roadmap) & 融合架构升级

本项目正在从单纯的 RAG 工具向 **"数据挖掘层 Agent"** 演进。我们正在引入图算法与推荐系统，打造“千人千面”的科研助手。

### ✅ 已完成特性 (Phase 1: Baseline)
- [x] **MVP 全链路**: OpenAlex -> ChromaDB -> LLM (RAG 闭环)。
- [x] **Deep Retrieval**: 支持 ArXiv 自动补全与 PDF 全文下载/向量化。
- [x] **Stability**: 解决 Embedding API 批处理限制 (Batching) 与并发问题。
- [x] **Agentic Core**: 引入多模式 (Review/Explain/Inspire) 与思维链 (CoT) 机制。
- [x] **Web UI**: 基于 Streamlit 的可视化交互界面。
- [x] **图挖掘增强 RAG**: Citation Graph Re-ranking (引文重排序), Structured Review (结构化综述)
- [x] **灵感广场推荐系统**: Personalized RAG (个性化检索), Community Recommendation (灵感推荐)

### 🚀 正在进行 (Phase 2: Graph & Mining) - 当前开发重点

```txt
scholar_rag/
├── src/
│   ├── evaluation/             # ✨ [NEW] 自动化评测层 (Track 2)
│       ├── __init__.py
│       └── ragas_checker.py    # Ragas 评测脚本, 测试集生成
```
#### Track 2: RAG 性能自动化评测 (Automated Evaluation)

## 📊 评测体系 (Evaluation Framework)

本项目建立了完整的评测体系，从多个维度验证系统性能。

### 一、图检索功能A/B测试

**评测目标**：验证知识图谱增强检索的有效性

**评测方法**：
- **控制变量实验**：使用20个歧义query（如"transformer"），分别在`use_graph=True`和`use_graph=False`下运行
- **评估指标**：
  - **平均相关性分数**：多维度评分（concept匹配度、文本相关性、引用数、年份）
  - **NDCG（排序质量）**：考虑排序位置的归一化累积增益
  - **胜率**：图谱组优于对照组的query比例

**技术原理**：
- 通过OpenAlex Concept ID映射实现领域聚焦，避免歧义干扰（如"transformer"能准确区分深度学习模型和电气设备）
- 图谱扩展理解query的学术语义，检索结果更符合学术意图
- 多维度评分：concept匹配、文本相关性、引用数、年份综合评估

**运行方式**：
```bash
python -m evaluation.evaluate_graph_v2
```

---

### 二、三种模式对比测试

**评测目标**：验证Review、Explain、Inspire三种模式的差异化特征

**评测方法**：
- **对比评测**：对同一query使用三种模式生成回答，对比差异
- **统一维度**：使用三个统一维度（全面性、新颖性、深度）进行公平对比
- **评估指标**：
  - **Comprehensiveness（全面性）**：内容长度、引用数量、结构化程度
  - **Novelty（新颖性）**：创新性词汇、跨领域连接、未来方向
  - **Depth（深度特性）**：理论深度、逻辑推理、机制解释

**差异化原因**：

1. **检索策略的专业差异**：
   - **Review模式**：广度优先检索（top_k=20）+ Louvain社区发现算法 → 全面性
   - **Explain模式**：深度优先检索（高引用优先）+ 理论匹配 → 深度性
   - **Inspire模式**：跨领域检索 + 知识图谱扩展 → 新颖性

2. **上下文组织方式的差异**：
   - **Review模式**：结构化聚类组织（流派 → 代表论文 → 证据）
   - **Explain模式**：理论-现象链式组织（理论 → 机制 → 现象）
   - **Inspire模式**：跨领域类比组织（领域 → 方法 → 迁移）

3. **Prompt模板的差异**：
   - **Review模式**：分类-对比-总结机制
   - **Explain模式**：理论-现象推理机制
   - **Inspire模式**：抽象-连接-迁移机制

**运行方式**：
```bash
python -m evaluation.evaluate_mode_differences
python -m evaluation.analyze_results  # 生成可视化报告
```

## ⚠️ 常见问题 (FAQ)
### Q1: 第一次运行时卡在 Loading Reranker 很久？

A: 是的。src/retrieval/openalex.py 会首次下载 BAAI/bge-reranker-v2-m3 模型 (约 1-2GB)。请确保网络通畅。下载完成后，模型会缓存到本地，后续启动将瞬间完成。

### Q2: 为什么有些 PDF 下载失败？

A: 许多出版商 (如 IEEE, Springer, Nature) 具有严格的反爬虫机制或付费墙。vector_store.py 已内置了 ArXiv 优先策略和 Header 伪装，但仍无法保证 100% 下载。下载失败的论文将仅使用其 Abstract 参与生成。

### Q3: 如何清除缓存？

A: 删除 data/vector_store 文件夹可重置向量库；删除 data/cache/pdfs 可清空已下载的 PDF 文件。

### Q4: 代码修改了 prompts.json 不生效？

A: 请确保你修改的是 src/core/prompts.json (用于生成) 或 src/graph/expansion_prompts.json (用于图谱扩展)，且 JSON 格式合法。