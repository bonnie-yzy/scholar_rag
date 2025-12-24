import openai
from src.utils.logger import setup_logger
from src.config import settings  # <--- 引入新配置

class LLMService:
    def __init__(self):
        self.logger = setup_logger("LLMService")
        
        self.model = settings.LLM_MODEL_NAME
        self.logger.info(f"Initializing LLM: {self.model}")

        self.client = openai.OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL
        )

    def chat(self, system_prompt, user_prompt):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                # 使用配置文件中的参数
                temperature=settings.LLM_TEMPERATURE, 
                max_tokens=settings.LLM_MAX_TOKENS
            )
            return response.choices[0].message.content
        except Exception as e:
            error_msg = f"LLM Call Failed: {str(e)}"
            self.logger.error(error_msg)
            return error_msg