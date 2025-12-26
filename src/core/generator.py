import json
import os
from typing import List, Union, Dict
from src.core.llm import LLMService
from src.utils.logger import setup_logger

class ReviewGenerator:
    """
    [Merged Generator]
    架构：保留你的 Prompt 模板加载机制 (灵活性高)。
    内容：增强了 Context 格式化逻辑，吸收了队友的 Metadata 提取 (Cited By, Concepts)。
    """
    def __init__(self):
        self.llm = LLMService()
        self.logger = setup_logger("Generator")
        
        # 加载 Prompt 模板
        try:
            # 兼容不同运行路径
            base_dir = os.path.dirname(os.path.abspath(__file__))
            prompt_path = os.path.join(base_dir, "prompts.json")
            
            if os.path.exists(prompt_path):
                with open(prompt_path, "r", encoding="utf-8") as f:
                    self.prompts = json.load(f)
            else:
                # 兜底：如果找不到文件，使用默认模板 (防止报错)
                self.logger.warning("prompts.json not found, using default fallback.")
                self.prompts = {
                    "system_base": "You are a helpful academic assistant.",
                    "template": "{base_system}\n\nContext:\n{context}\n\nQuery: {query}",
                    "tasks": {"review": {"instruction": "Answer the query.", "cot": ""}}
                }
        except Exception as e:
            self.logger.error(f"Error loading prompts: {e}")
            raise

    def _format_context(self, context_data: Union[List[str], List[Dict]]) -> str:
        """
        [增强版] 格式化上下文
        """
        formatted_text = ""
        
        if not context_data:
            return "No valid context data provided."

        # 情况 A: 纯字符串列表 (Chunk 模式 fallback)
        if isinstance(context_data[0], str):
            for i, chunk in enumerate(context_data, 1):
                formatted_text += f"[Source {i}]: {chunk}\n\n"
                
        # 情况 B: 字典列表 (OpenAlex Paper 或 Vector Chunk)
        elif isinstance(context_data[0], dict):
            for i, item in enumerate(context_data, 1):
                # 基础信息
                title = item.get('title', 'Unknown Title')
                year = item.get('year', 'N/A')
                
                # [队友增强] 引用数和概念
                cited = item.get('cited_by', 'N/A')
                concepts = item.get('concepts', [])
                if isinstance(concepts, list):
                    concepts_str = ", ".join(concepts[:3]) # 只取前3个概念
                else:
                    concepts_str = str(concepts)

                # 链接处理
                url = item.get('url') or item.get('pdf_url') or item.get('id', 'N/A')
                
                # 正文内容 (优先 content，其次 abstract)
                text_content = item.get('content') or item.get('abstract', 'No content available.')
                
                # 组装 (融合了队友的详细字段)
                formatted_text += (
                    f"[Paper {i}]\n"
                    f"Title: {title}\n"
                    f"Year: {year} | Cited By: {cited}\n"
                    f"Concepts: {concepts_str}\n"
                    f"URL: {url}\n"
                    f"Content/Abstract: {text_content}\n"
                    f"{'-'*40}\n"
                )
        
        return formatted_text

    def generate(self, user_query: str, context_data: Union[List[str], List[Dict]], task_type: str = "review") -> str:
        """
        生成回复 (接口保持不变)
        """
        self.logger.info(f"Generating response. Mode: {task_type}")
        
        if not context_data:
            return "未找到相关背景知识，无法生成回答。"

        # 1. 获取 Prompt 配置
        task_config = self.prompts["tasks"].get(task_type, self.prompts["tasks"]["review"])
        
        # 2. 格式化上下文
        context_text = self._format_context(context_data)

        # 3. 组装 Prompt
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
            return f"Error constructing prompt: {e}"

        # 4. 调用 LLM
        # 注意：这里使用 self.prompts["system_base"] 作为 system prompt，这比队友硬编码的更灵活
        return self.llm.chat(self.prompts["system_base"], full_prompt)