import json
import re
from typing import Any, Dict, Optional, List

import openai

from src.config import settings
from src.utils.logger import setup_logger

def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    [新增工具] 从 LLM 输出中提取第一个 JSON 对象
    用于支持 OpenAlexRetriever 中的 Concept 映射功能
    """
    if not text:
        return None

    # 1. 优先处理 Markdown 代码块 ```json ... ```
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

    # 2. 扫描第一个匹配 {} 的片段
    start = text.find("{")
    if start < 0:
        return None
    for end in range(len(text), start, -1):
        if text[end - 1] != "}":
            continue
        candidate = text[start:end]
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except Exception:
            continue
    return None


class LLMService:
    """
    [Merged LLMService]
    保留了你的接口 chat()，增加了 chat_json() 和 OpenRouter 支持。
    """

    def __init__(self):
        self.logger = setup_logger("LLMService")
        self.model = settings.LLM_MODEL_NAME
        self.embedding_model = getattr(settings, "EMBEDDING_MODEL_NAME", "text-embedding-3-small")

        # 初始化 Client (保持你原有的逻辑)
        openrouter_key = getattr(settings, "OPENROUTER_API_KEY", None)
        openrouter_base = getattr(settings, "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        
        if openrouter_key:
            self.logger.info(f"Initializing LLM via OpenRouter")
            self.client = openai.OpenAI(
                api_key=openrouter_key,
                base_url=openrouter_base,
                default_headers={"X-Title": "ScholarRAG"}
            )
        else:
            self.logger.info(f"Initializing LLM via OpenAI-compatible")
            self.client = openai.OpenAI(
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL
            )

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """
        标准对话接口 (保持与你原有接口完全一致)
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
            )
            
            # [增强] 错误检查
            choices = getattr(response, "choices", None) or []
            if not choices:
                msg = f"LLM Call Failed: empty choices (model={self.model})"
                self.logger.error(msg)
                return msg

            content = choices[0].message.content
            if content:
                return content
            
            return "LLM Call Failed: empty content"

        except Exception as e:
            error_msg = f"LLM Call Failed: {str(e)}"
            self.logger.error(error_msg)
            return error_msg

    def chat_json(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        """
        [新增] 结构化输出接口 (OpenAlexRetriever 需要用到)
        """
        try:
            text = self.chat(system_prompt=system_prompt, user_prompt=user_prompt)
            return _extract_first_json_object(text)
        except Exception as e:
            self.logger.error(f"LLM JSON Call Failed: {str(e)}")
            return None
        
    def chat_stream(self, system_prompt: str, user_prompt: str):
        """
        [新增] 流式对话接口
        返回一个生成器 (Generator)，逐个 token 产出
        """
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
                stream=True,  # <--- 保留这个
                # stream_options={"include_usage": False}  <--- 🔴 [删除这一行] 这一行导致了 400 错误
            )
            
            for chunk in stream:
                if hasattr(chunk, 'choices') and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        yield delta.content

        except Exception as e:
            error_msg = f"\n[LLM Stream Failed: {str(e)}]"
            self.logger.error(error_msg)
            yield error_msg

    def get_embedding(self, text: str) -> List[float]:
        """
        [新增] 获取文本的向量表示 (Embedding)
        """
        try:
            # 移除换行符，避免影响 embedding 质量
            text = text.replace("\n", " ")
            
            response = self.client.embeddings.create(
                input=[text],
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"Embedding Failed: {e}")
            # 失败时返回零向量的替代方案或空列表，视下游处理而定
            # 这里返回空列表让调用者处理
            return []