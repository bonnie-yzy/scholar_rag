import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

class Settings:
    # 1. OpenAlex 配置
    OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL")

    # 2. LLM 配置
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
    
    # 类型转换: 环境变量读出来默认是字符串，这里转为 float/int
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))

    # 3. RAG 配置
    RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
    RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))

    # 验证逻辑
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