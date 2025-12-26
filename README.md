# 🚀ScholarRAG：基于知识图谱增强的科研综述助手

ScholarRAG 是一个针对学术科研场景设计的智能 RAG (检索增强生成) Agent。它不仅仅是一个简单的搜索工具，而是一个具备双阶段检索 (Two-Stage Retrieval)、全链路溯源和多模态认知能力的科研助手。

它利用 OpenAlex 进行广度知识发现，通过 ArXiv 补全全文，构建本地增量式向量知识库 (ChromaDB)，并根据用户的意图（综述、解释、启发）动态调整生成策略，最终输出带有精确引用的高质量回答。

## 🌟核心特性

双阶段检索漏斗：

- 广度发现 (Wide)：利用 OpenAlex 和 Knowledge Graph 快速锁定 Top-20+ 篇核心文献。

- 深度挖掘 (Deep)：自动下载全文 (支持 ArXiv/DOI)，解析并切片入库，进行语义级 Top-15+ 精准召回。

全链路溯源：拒绝幻觉。生成的每一句话都能溯源到具体的论文标题、年份和 URL 链接。

增量式知识库：一次下载，永久复用。自动构建本地向量库，越用越快，越用越懂你。

多模式 Agent：

- review: 撰写结构化文献综述。

- explain: 基于理论解释反直觉现象。

- inspire: 寻找跨学科（如物理->AI）的灵感迁移。

智能兜底：OpenAlex 没有 PDF？系统自动检测并暴力解析 ArXiv 链接，确保全文获取率。

## 项目结构：
```txt
scholar_rag/
├── data/                   # [数据持久化] (自动生成，Git Ignore)
│   ├── cache/              # PDF 文件缓存
│   │   └── pdfs/           # 下载的论文 (文件名: ID_Title.pdf)
│   ├── vector_store/       # ChromaDB 本地向量数据库
│   └── papers_debug.txt    # 检索过程的中间元数据日志
├── src/                    # [核心源码]
│   ├── config.py           # 配置加载器 (读取 .env)
│   ├── core/               # >> 大脑层 (Agent & Generation)
│   │   ├── llm.py          # LLM 接口 (支持 SiliconFlow/OpenAI)
│   │   ├── generator.py    # 动态 Prompt 组装与上下文格式化
│   │   └── prompts.json    # 📝 核心 Prompt 模板库 (Review/Explain/Inspire)
│   ├── retrieval/          # >> 记忆与感知层
│   │   ├── base.py         # 接口定义
│   │   ├── openalex.py     # ✅ 广度检索器 (OpenAlex API + ArXiv 补全)
│   │   └── vector_store.py # ✅ 深度检索器 (PDF下载 + 解析 + 向量化 + 语义召回)
│   ├── graph/              # >> 导航层
│   │   ├── expansion.py    # 基于 Concept Ontology 的查询扩展
│   │   └── ranking.py      # (预留) 引文网络重排序
│   └── utils/              # >> 基础设施
│       └── logger.py       # 日志模块
├── .env                    # 🔐 核心配置文件 (API Key, TopK, BatchSize)
├── main.py                 # 🚀 CLI 启动入口 (串联整个 Pipeline)
├── requirements.txt        # 依赖列表
└── README.md               # 项目文档
```
在 scholar_rag/ 下新增 ui/ 目录，保持与 src/ 核心逻辑的隔离。
```txt
scholar_rag/
├── config/                 # [配置]
├── data/                   # [数据]
├── src/                    # [RAG 核心]
├── .env                    # [环境变量]
├── main.py                 # [CLI 入口]
├── requirements.txt        # [依赖]
├── README.md               # [文档]
└── ui/                     # [🆕 UI 前端模块]
    ├── __init__.py
    ├── app.py              # 🚀 Streamlit 主入口
    ├── db.py               # 💾 SQLite 数据库管理 (用户/灵感广场)
    ├── logic.py            # 🧠 RAG 逻辑封装 (缓存/多轮对话处理)
    └── style.css           # 🎨 自定义 CSS (Gemini 风格)
```

## 环境配置 Windows + Conda
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
LLM_TEMPERATURE=0.3
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

1. src/retrieval/openalex.py (广度发现)

    功能：负责“大海捞针”。从 2.5 亿篇论文中锁定最相关的 Top-K 元数据。

    核心逻辑：

    - 智能回退：优先获取 OpenAlex 官方 PDF 链接；如果没有，自动解析 ids 字段，暴力构造 ArXiv PDF 链接。

    - 倒排索引还原：将 OpenAlex 的 Inverted Index 重构为可读摘要。

2. src/retrieval/vector_store.py (深度记忆)

    功能：负责“细嚼慢咽”。构建本地专属的向量知识库。

    核心逻辑：

    - 增量更新：自动检测论文是否已存在，跳过重复下载，越用越快。

    - 自动分批：实现了 SiliconFlowEmbeddingFunction 的自动 Batching，解决 API 长度限制 (Error 413)。

    - 全字段存储：存入 ChromaDB 的不仅是向量，还包含 Title, URL, Year，确保生成时可引用。

3. src/core/generator.py (大脑中枢)

    功能：负责“思考与表达”。

    核心逻辑：

    - 动态 Prompt：加载 prompts.json，根据用户选择的 Mode (review/explain/inspire) 切换不同的 思维链 (CoT) 模板。

    - 精准引用：将检索到的上下文格式化为 [Source X] Title... URL... 格式，强制 LLM 进行基于事实的回答。

4. ui

- app.py: 入口,前端主程序。负责页面路由（对话/广场/个人）、状态管理（SessionState）、侧边栏逻辑以及组件渲染。

- logic.py: 逻辑,后端业务层。封装了 RAG 核心链路（OpenAlex检索 -> 向量入库 -> LLM生成）。核心亮点是实现了 recursive_summarize（递归摘要算法），自动维护长对话的记忆。

- db.py: 存储,数据持久层 (SQLite)。管理用户账户、私人对话历史（CRUD）、灵感广场帖子及点赞数据。

- style.css: 样式,全局样式表。强制开启深色模式，实现了“彩虹动态标题”和仿 Gemini 的侧边栏交互（鼠标悬停显示删除按钮）。

## 📝 开发计划 (Roadmap)
[x] MVP: 全链路跑通 (OpenAlex -> ChromaDB -> LLM)。

[x] Enhancement: 支持 ArXiv 自动补全与 PDF 下载。

[x] Stability: 解决 Embedding API 批处理限制 (Batching)。

[x] Agentic: 引入多模式 (Review/Explain/Inspire) 与思维链。

[ ] Citation Graph: 引入 PageRank 算法，根据引用影响力重排序检索结果。

[ ] Web UI: 基于 Streamlit/Gradio 的可视化交互界面。

## ⚠️ 常见问题

下载失败 (403/418): IEEE/Springer 等出版商有强力反爬。系统会自动跳过这些论文，优先处理 ArXiv 和 Open Access 论文。日志中会有 ⚠️ Download Failed 提示。

Embedding API Error: 如果遇到 Batch size 错误，请尝试在 .env 中调小 EMBEDDING_BATCH_SIZE。

速度慢: 第一次运行需要下载 PDF，速度取决于网络。第二次运行相同或相似主题时，会直接利用本地向量库，速度极快。