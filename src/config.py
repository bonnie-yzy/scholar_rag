import os
# [优化] 采用队友的稳健导入方式，允许在未安装 python-dotenv 的环境(如CI/CD)中运行
try:
    from dotenv import load_dotenv  # type: ignore
except ModuleNotFoundError:
    # Mock load_dotenv if not installed
    def load_dotenv(*args, **kwargs): 
        return None

# 加载 .env 文件
load_dotenv()

class Settings:
    # =========================================================
    # 1. OpenAlex 配置 (基础)
    # =========================================================
    OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL")

    # =========================================================
    # 2. LLM 配置 (基础)
    # =========================================================
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
    
    # 类型转换: 环境变量读出来默认是字符串，这里转为 float/int
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "8000"))
    
    # =========================================================
    # 3. RAG 配置 (基础)
    # =========================================================
    RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
    RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    RAG_DOWNLOAD_K = int(os.getenv("RAG_DOWNLOAD_K", "20"))
    RAG_RETRIEVAL_K = int(os.getenv("RAG_RETRIEVAL_K", "15"))

    # =========================================================
    # 4. Graph Mining 配置 (Track 1: Citation Graph Re-ranking) [新增]
    # =========================================================
    # 是否启用引文图重排序
    ENABLE_CITATION_RERANK = os.getenv("ENABLE_CITATION_RERANK", "true").lower() in {"1", "true", "yes", "y"}
    
    # 混合分数权重: Hybrid = alpha * BGE_Score + beta * PageRank
    CITATION_RERANK_ALPHA = float(os.getenv("CITATION_RERANK_ALPHA", "0.8"))  # 语义分权重
    CITATION_RERANK_BETA = float(os.getenv("CITATION_RERANK_BETA", "0.2"))   # 图结构分权重
    
    # PageRank 超参
    PAGERANK_DAMPING = float(os.getenv("PAGERANK_DAMPING", "0.85"))
    PAGERANK_MAX_ITER = int(os.getenv("PAGERANK_MAX_ITER", "100"))
    PAGERANK_TOL = float(os.getenv("PAGERANK_TOL", "1e-6"))

    # =========================================================
    # 5. Graph Mining 配置 (Track 1: Structured Review) [新增]
    # =========================================================
    # 是否启用基于社区发现的结构化综述
    ENABLE_STRUCTURED_REVIEW = os.getenv("ENABLE_STRUCTURED_REVIEW", "true").lower() in {"1", "true", "yes", "y"}
    
    # Louvain 社区发现超参
    LOUVAIN_RESOLUTION = float(os.getenv("LOUVAIN_RESOLUTION", "1.0"))
    LOUVAIN_MAX_LEVELS = int(os.getenv("LOUVAIN_MAX_LEVELS", "10"))
    LOUVAIN_MAX_ITER = int(os.getenv("LOUVAIN_MAX_ITER", "50"))
    LOUVAIN_MIN_EDGES = int(os.getenv("LOUVAIN_MIN_EDGES", "2"))
    
    # 综述生成控制
    STRUCTURED_REVIEW_TOP_PAPERS_PER_CLUSTER = int(os.getenv("STRUCTURED_REVIEW_TOP_PAPERS_PER_CLUSTER", "3"))
    STRUCTURED_REVIEW_TOP_CHUNKS_PER_PAPER = int(os.getenv("STRUCTURED_REVIEW_TOP_CHUNKS_PER_PAPER", "2"))
    STRUCTURED_REVIEW_MAX_CHUNK_CHARS = int(os.getenv("STRUCTURED_REVIEW_MAX_CHUNK_CHARS", "450"))

    # =========================================================
    # 验证逻辑
    # =========================================================
    @classmethod
    def validate(cls):
        if not cls.OPENALEX_EMAIL:
            raise ValueError("❌ 缺少配置: 请在 .env 中设置 OPENALEX_EMAIL (用于 API 身份验证)")
        if not cls.OPENAI_API_KEY:
            raise ValueError("❌ 缺少配置: 请在 .env 中设置 OPENAI_API_KEY")

# 实例化并验证，确保一启动程序就能发现配置错误
try:
    Settings.validate()
except ValueError as e:
    print(e)
    exit(1)

settings = Settings()