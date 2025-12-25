import json
import re
from typing import Any, Dict, Optional

import openai

from src.config import settings
from src.utils.logger import setup_logger


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    标题: 从 LLM 输出中提取第一个 JSON 对象

    Input:
      - 参数:
          - text (str, required): LLM 原始输出文本，可能包含解释性文字与代码块
      - 上下文:
          - 无
      - 依赖:
          - Python 内置 json / re

    Output:
      - 返回:
          - Optional[Dict[str, Any]]: 解析成功返回 dict，否则返回 None
      - 副作用:
          - 无
      - 错误:
          - 无（内部吞掉 JSONDecodeError，返回 None）

    Why:
      - LLM 常输出“解释 + JSON”，需要一个稳健的提取器，避免后续逻辑因格式飘逸失败。

    References:
      - PEP 8 / Python typing

    Calls:
      - json.loads (python stdlib): 解析 JSON 字符串
      - re (python stdlib): 定位候选 JSON 片段

    Example:
      - 输入: 'Here is result: {"a":1}'
      - 输出: {"a": 1}
    """
    if not text:
        return None

    # 优先处理 ```json ... ``` 代码块
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

    # 退化：扫描第一个“看起来像 JSON object”的片段
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
    标题: LLMService - 统一的 OpenAI-compatible Chat 客户端（支持 OpenRouter）

    Input:
      - 参数:
          - 无（从 `src/config.py#Settings` 读取配置）
      - 上下文:
          - 环境变量: OPENROUTER_API_KEY / OPENAI_API_KEY / OPENAI_BASE_URL / LLM_MODEL_NAME 等
      - 依赖:
          - openai (python package): OpenAI-compatible 客户端

    Output:
      - 返回:
          - chat(...): str
          - chat_json(...): Optional[Dict[str, Any]]
      - 副作用:
          - 网络调用（调用 LLM 服务）
          - 日志输出
      - 错误:
          - 异常会被捕获并返回错误字符串（chat）或 None（chat_json）

    Why:
      - 项目需要统一 LLM 调用入口，便于替换底层提供商（OpenAI/OpenRouter）并集中处理超参、日志与结构化输出。

    References:
      - OpenAI Python SDK (OpenAI-compatible base_url)
      - OpenRouter Docs (OpenAI-compatible endpoint)

    Calls:
      - openai.OpenAI (openai): 创建客户端
      - client.chat.completions.create (openai): 发送对话补全请求
      - _extract_first_json_object (src/core/llm.py): 提取结构化 JSON

    Example:
      - 输入: system_prompt='Return JSON', user_prompt='...'
      - 输出: chat_json(...) -> {"phrases": ["protein structure prediction"]}
    """

    def __init__(self):
        self.logger = setup_logger("LLMService")
        self.model = settings.LLM_MODEL_NAME

        # 优先使用 OpenRouter（你只有 OpenRouter key 的场景）
        if settings.OPENROUTER_API_KEY:
            api_key = settings.OPENROUTER_API_KEY
            base_url = settings.OPENROUTER_BASE_URL
            default_headers = {"X-Title": settings.OPENROUTER_APP_NAME}
            self.logger.info(f"Initializing LLM via OpenRouter: {self.model}")
        else:
            api_key = settings.OPENAI_API_KEY
            base_url = settings.OPENAI_BASE_URL
            default_headers = None
            self.logger.info(f"Initializing LLM via OpenAI-compatible: {self.model}")

        # openai SDK 在 base_url=None 时也可用默认；这里保持兼容
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        if default_headers:
            kwargs["default_headers"] = default_headers

        self.client = openai.OpenAI(**kwargs)

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """
        标题: 发送一次 ChatCompletion 并返回纯文本

        Input:
          - 参数:
              - system_prompt (str, required): 系统提示词
              - user_prompt (str, required): 用户提示词
          - 上下文:
              - settings.LLM_TEMPERATURE / settings.LLM_MAX_TOKENS
          - 依赖:
              - openai client

        Output:
          - 返回:
              - str: 模型输出文本（可能包含 JSON/解释）
          - 副作用:
              - 网络调用 + 日志
          - 错误:
              - 异常时返回以 "LLM Call Failed:" 开头的错误字符串

        Why:
          - 作为最薄封装，供上层做概念映射、总结、生成等任务复用。

        References:
          - OpenAI Chat Completions API

        Calls:
          - self.client.chat.completions.create (openai): 请求补全

        Example:
          - 输入: system_prompt='...', user_prompt='...'
          - 输出: '...'
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
            # OpenAI-compatible providers sometimes return tool-calls or empty content.
            choices = getattr(response, "choices", None) or []
            if not choices:
                msg = f"LLM Call Failed: empty choices (model={self.model})"
                self.logger.error(msg)
                return msg

            choice0 = choices[0]
            finish_reason = getattr(choice0, "finish_reason", None)
            message = getattr(choice0, "message", None)
            content = getattr(message, "content", None) if message is not None else None

            if isinstance(content, str) and content.strip():
                return content

            tool_calls = getattr(message, "tool_calls", None) if message is not None else None
            msg = (
                "LLM Call Failed: empty content "
                f"(model={self.model}, finish_reason={finish_reason}, tool_calls={'yes' if tool_calls else 'no'})"
            )
            self.logger.error(msg)
            return msg
        except Exception as e:
            error_msg = f"LLM Call Failed: {str(e)}"
            self.logger.error(error_msg)
            return error_msg

    def chat_json(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        """
        标题: 发送一次 ChatCompletion 并尽力解析首个 JSON 对象

        Input:
          - 参数:
              - system_prompt (str, required): 系统提示词（应要求只输出 JSON）
              - user_prompt (str, required): 用户提示词
          - 上下文:
              - 无
          - 依赖:
              - _extract_first_json_object

        Output:
          - 返回:
              - Optional[Dict[str, Any]]: 成功解析的 JSON dict，否则 None
          - 副作用:
              - 网络调用 + 日志
          - 错误:
              - 异常被捕获并返回 None

        Why:
          - 概念映射需要结构化输出；相比纯文本解析，JSON 更可控、更易审计。

        References:
          - JSON Schema / Structured Outputs best practices

        Calls:
          - self.chat (src/core/llm.py#LLMService.chat): 获取文本
          - _extract_first_json_object (src/core/llm.py): 提取 JSON

        Example:
          - 输入: 要求模型输出 {"phrases":["..."]}
          - 输出: {"phrases": ["protein structure prediction"]}
        """
        try:
            text = self.chat(system_prompt=system_prompt, user_prompt=user_prompt)
            return _extract_first_json_object(text)
        except Exception as e:
            self.logger.error(f"LLM JSON Call Failed: {str(e)}")
            return None