import json
import os
from typing import List, Union, Dict
from src.core.llm import LLMService
from src.utils.logger import setup_logger

class ReviewGenerator:
    def __init__(self):
        self.llm = LLMService()
        self.logger = setup_logger("Generator")
        
        # 加载 Prompt 模板
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts.json")
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                self.prompts = json.load(f)
        except FileNotFoundError:
            self.logger.error(f"prompts.json not found at {prompt_path}")
            raise

    def _format_context(self, context_data: Union[List[str], List[Dict]]) -> str:
        """
        处理上下文格式，兼容旧版(OpenAlex Metadata)和新版(Vector Chunks)
        """
        formatted_text = ""
        
        if not context_data:
            return "No valid context data provided."

        # 情况 A: 输入是纯字符串列表 (旧版兼容)
        if isinstance(context_data[0], str):
            for i, chunk in enumerate(context_data, 1):
                formatted_text += f"[Source {i}]: {chunk}\n\n"
                
        # 情况 B: 输入是字典列表 (这是我们现在主要用的)
        # 兼容 OpenAlex Metadata (有 abstract) 和 Vector Store Chunks (有 content, url)
        elif isinstance(context_data[0], dict):
            for i, item in enumerate(context_data, 1):
                # 1. 提取标题
                title = item.get('title', 'Unknown Title')
                
                # 2. 提取年份
                year = item.get('year', 'N/A')
                
                # 3. 智能提取链接 (优先用 DOI/URL，没有则用 PDF 链接，最后用 ID)
                url = item.get('url') or item.get('pdf_url') or item.get('id', 'N/A')
                
                # 4. 智能提取正文 (优先用 content-切片，没有则用 abstract-摘要)
                # 注意：vector_store.py 返回的是 'content'
                # openalex.py 返回的是 'abstract'
                text_content = item.get('content') or item.get('abstract', '')
                
                # 5. 组装格式化字符串 (这是 LLM 看到的最终样子)
                formatted_text += (
                    f"[Source {i}]\n"
                    f"Title: {title}\n"
                    f"Year: {year}\n"
                    f"URL: {url}\n"
                    f"Content: {text_content}\n"
                    f"{'-'*40}\n"
                )
        
        return formatted_text

    def generate(self, user_query: str, context_data: Union[List[str], List[Dict]], task_type: str = "review") -> str:
        """
        生成回复
        :param user_query: 用户问题
        :param context_data: 上下文数据 (chunk 列表或 paper 字典列表)
        :param task_type: 任务类型 (review/explain/inspire)
        """
        self.logger.info(f"Generating response. Mode: {task_type}")
        
        # 即使数据为空，也尝试让 LLM 回答（可能会利用其自身知识），或者返回提示
        if not context_data:
            return "未找到相关背景知识，无法生成回答。"

        # 1. 获取对应的 Prompt 配置
        task_config = self.prompts["tasks"].get(task_type, self.prompts["tasks"]["review"])
        
        # 2. 格式化上下文
        context_text = self._format_context(context_data)

        # 3. 组装最终 Prompt
        try:
            full_prompt = self.prompts["template"].format(
                base_system=self.prompts["system_base"],
                instruction=task_config["instruction"],
                cot=task_config["cot"],
                context=context_text,
                query=user_query
            )
        except KeyError as e:
            self.logger.error(f"Prompt template missing key: {e}")
            return "Error constructing prompt."

        # 4. 调用 LLM
        return self.llm.chat("You are a helpful research assistant.", full_prompt)