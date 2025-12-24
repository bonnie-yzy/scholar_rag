# ScholarRAG：基于知识图谱增强的科研综述助手

ScholarRAG 是一个针对学术科研场景设计的 RAG (检索增强生成) 工具。它不同于传统的基于文本切片的 RAG，而是利用 OpenAlex 的倒排索引重构摘要，并结合 学术概念树 (Concept Graph) 进行查询扩展，最终利用 LLM 生成高质量的文献综述。


## 项目结构：
```txt
scholar_rag/
├── config/                 # [配置] 系统参数与提示词
│   └── prompts.yaml        # (预留) RAG 的 Prompt 模板管理
├── data/                   # [数据] 数据持久化 (Git Ignore)
│   ├── cache/              # API 请求缓存
│   └── vector_store/       # (规划中) 向量数据库文件
├── src/                    # [源码] 核心代码逻辑
│   ├── core/               # >> LLM 交互核心
│   │   ├── llm.py          # OpenAI/DeepSeek 接口封装
│   │   └── generator.py    # 综述生成逻辑与 Prompt 组装
│   ├── retrieval/          # >> 检索层 (Strategy Pattern)
│   │   ├── base.py         # 检索器抽象基类
│   │   ├── acemap.py       # (可选) Acemap 接口实现
│   │   └── openalex.py     # ✅ OpenAlex API 封装与摘要重构
│   ├── graph/              # >> 图算法层
│   │   ├── expansion.py    # 基于 Concept 树的查询扩展
│   │   └── ranking.py      # (规划中) 基于引文网络的重排序
│   └── utils/              # >> 工具函数
│       └── logger.py       # 日志模块
├── .env                    # 环境变量 (API Key) - 不要上传到 GitHub
├── main.py                 # CLI 启动入口
├── requirements.txt        # Python 依赖列表
└── README.md               # 项目说明文档
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
LLM_MAX_TOKENS=2000

# --- OpenAlex 配置 ---
# 【必须】填入真实邮箱，否则接口会被限流
OPENALEX_EMAIL=""

# --- RAG 检索配置 ---
RAG_TOP_K=5
RAG_CHUNK_SIZE=1000
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
python main.py --query "Large Language Models in Healthcare"
```
2. 图增强模式 (推荐)
启用知识图谱扩展功能。系统会自动定位该 Query 所属的 OpenAlex 概念节点 (Concept Node)，并在此概念子图中进行精准检索，排除同名异义词的干扰：
```bash
python main.py --query "Machine Learning" --use_graph
```

## 模块职责详细解释

1. src/retrieval/ (检索层)
这是数据的来源。

openalex.py: 本项目的核心引擎。

摘要重构: OpenAlex 为了版权不直接存储摘要文本，而是存储 Inverted Index (倒排索引)。该模块负责将倒排索引还原为可读的摘要文本，这是 RAG 能否进行的关键。

动态检索: 实时调用 API，支持按 Relevance (相关性) 或 Cited_by_count (引用量) 排序。

2. src/graph/ (图增强层)
这是项目的“大脑”扩展部分。

expansion.py: 利用 OpenAlex 的 Ontology (本体论) 结构。

例如输入 "Transformer"，它能识别这是 "Deep Learning" 下的一个架构，而不是电气工程里的“变压器”。

它返回 Concept ID，强制检索层只在该 ID 的子图中搜索。

3. src/core/ (生成层)
这是 RAG 的输出端。

generator.py: 负责 Prompt Engineering。它将检索到的 Top-K 论文的元数据（标题、年份、摘要）组装成 Context，并指示 LLM 生成综述。

llm.py: 统一的 LLM 调用接口，兼容 OpenAI 格式的 API (包括 DeepSeek, Moonshot 等)。

## 📝 开发计划 (Roadmap)
[x] MVP: 基础检索 + 摘要还原 + 简单综述生成

[x] Graph: 基于 Concept 的 Query Expansion

[ ] Vector Store: 引入 ChromaDB/LanceDB，支持本地向量化存储，避免重复 API 调用

[ ] Citation Graph: 基于引文网络（PageRank）对检索结果进行二次重排序

[ ] Web UI: 基于 Streamlit 的可视化界面

## ⚠️ 注意事项
OpenAlex 速度: 如果不配置邮箱，请求可能会有延迟或被限流。

LLM 成本: 生成一篇综述大约消耗 2k-4k Token，请关注 API 余额。

网络问题: 如果在中国大陆地区使用 OpenAI，请确保终端配置了适当的代理环境。