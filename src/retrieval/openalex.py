import pyalex
from pyalex import Works
from src.retrieval.base import BaseRetriever  # 注意：原来的 base.py 引用路径是否正确
from src.utils.logger import setup_logger
from src.config import settings  # <--- 引入新配置模块

# 直接从 settings 对象配置 pyalex
pyalex.config.email = settings.OPENALEX_EMAIL

class OpenAlexRetriever(BaseRetriever):
    def __init__(self):
        self.logger = setup_logger("OpenAlex")

    def _invert_abstract(self, inverted_index):
        """将 OpenAlex 的倒排索引重组为可读文本"""
        if not inverted_index:
            return None
        max_len = 0
        word_map = {}
        for word, positions in inverted_index.items():
            for pos in positions:
                word_map[pos] = word
                if pos > max_len:
                    max_len = pos
        tokens = [word_map.get(i, "") for i in range(max_len + 1)]
        return " ".join(tokens)

    def search(self, query: str, top_k: int = None, concept_ids: list = None) -> list:
        # 如果调用时没指定 top_k，就使用全局配置
        if top_k is None:
            top_k = settings.RAG_TOP_K
            
        self.logger.info(f"Searching OpenAlex for: '{query}' (Limit: {top_k})")
        
        search_query = Works().search(query).filter(has_abstract=True)
        
        if concept_ids:
            search_query = search_query.filter(concepts={"id": "|".join(concept_ids)})
            self.logger.info(f"Filtering by Concepts: {concept_ids}")

        search_query = search_query.filter(from_publication_date="2019-01-01")
        
        # 获取结果
        try:
            results = search_query.sort(cited_by_count="desc").get(per_page=top_k)
        except Exception as e:
            self.logger.error(f"API Request Failed: {e}")
            return []

        papers = []
        for work in results:
            abstract = self._invert_abstract(work.get("abstract_inverted_index"))
            if not abstract or len(abstract) < 50: 
                continue

            papers.append({
                "id": work["id"],
                "title": work["display_name"],
                "year": work["publication_year"],
                "cited_by": work["cited_by_count"],
                "abstract": abstract,
                "url": work.get("doi") or work.get("id"),
                "concepts": [c["display_name"] for c in work.get("concepts", [])[:3]]
            })
            
        self.logger.info(f"Found {len(papers)} valid papers.")
        return papers