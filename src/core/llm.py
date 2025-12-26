import json
import re
from typing import Any, Dict, Optional

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

        # [新增] 优先检查 OpenRouter 配置 (队友逻辑)
        # 使用 getattr 防止你的 settings.py 里没有定义这些变量而报错
        openrouter_key = getattr(settings, "OPENROUTER_API_KEY", None)
        openrouter_base = getattr(settings, "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        
        if openrouter_key:
            self.logger.info(f"Initializing LLM via OpenRouter: {self.model}")
            self.client = openai.OpenAI(
                api_key=openrouter_key,
                base_url=openrouter_base,
                default_headers={"X-Title": "ScholarRAG"}
            )
        else:
            # [原有] 默认 OpenAI/SiliconFlow 配置
            self.logger.info(f"Initializing LLM via OpenAI-compatible: {self.model}")
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