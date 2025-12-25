from src.core.llm import LLMService
from src.utils.logger import setup_logger

class ReviewGenerator:
    def __init__(self):
        self.llm = LLMService()
        self.logger = setup_logger("Generator")

    def generate(self, user_query, papers):
        self.logger.info("Constructing prompt for RAG...")
        
        if not papers:
            return "未找到相关论文，无法生成综述。"

        # 1. 构建 Context (这是 RAG 的关键)
        context_text = ""
        for i, p in enumerate(papers, 1):
            context_text += f"""
            [Paper {i}]
            Title: {p['title']}
            Year: {p['year']}
            Cited By: {p['cited_by']}
            Concepts: {', '.join(p['concepts'])}
            Abstract: {p['abstract']}
            --------------------------------
            """

        # 2. 构建 Prompt
        system_prompt = """
        You are an expert academic researcher. 
        Your goal is to write a comprehensive literature review based ONLY on the provided papers.
        
        Rules:
        1. Synthesis: Do not just list papers. Group them by themes or methodologies.
        2. Citation: Use strict [Paper X] format when referencing ideas.
        3. Critical Thinking: Point out common trends and contradictions.
        4. Language: Answer in the same language as the user's query (likely Simplified Chinese).
        """
        
        user_prompt = f"""
        Research Question: {user_query}
        
        Here are the retrieved papers from the database:
        {context_text}
        
        Please write the literature review now:
        """

        # 3. 生成
        self.logger.info("Sending to LLM...")
        review = self.llm.chat(system_prompt, user_prompt)
        if not (review or "").strip():
            return "LLM 返回空内容，无法生成综述（请检查模型名/配额/提供商状态）。"
        return review