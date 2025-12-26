import pyalex
from pyalex import Works
from src.retrieval.base import BaseRetriever  # 注意：原来的 base.py 引用路径是否正确
from src.utils.logger import setup_logger
from src.config import settings  # <--- 引入新配置模块
import os

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

    # def search(self, query: str, top_k: int = None, concept_ids: list = None) -> list:
    #     # 如果调用时没指定 top_k，就使用全局配置
    #     if top_k is None:
    #         top_k = settings.RAG_TOP_K
            
    #     self.logger.info(f"Searching OpenAlex for: '{query}' (Limit: {top_k})")
        
    #     search_query = Works().search(query).filter(has_abstract=True)
        
    #     if concept_ids:
    #         search_query = search_query.filter(concepts={"id": "|".join(concept_ids)})
    #         self.logger.info(f"Filtering by Concepts: {concept_ids}")

    #     search_query = search_query.filter(from_publication_date="2019-01-01")
        
    #     # 获取结果
    #     try:
    #         results = search_query.sort(cited_by_count="desc").get(per_page=top_k)
    #     except Exception as e:
    #         self.logger.error(f"API Request Failed: {e}")
    #         return []

    #     papers = []
    #     for work in results:
    #         abstract = self._invert_abstract(work.get("abstract_inverted_index"))
    #         if not abstract or len(abstract) < 50: 
    #             continue

    #     # --- [新增] 获取 PDF 链接 ---
    #     pdf_url = None
    #     best_oa = work.get("best_oa_location")
    #     if best_oa:
    #         pdf_url = best_oa.get("pdf_url")
        
    #     # 有些时候 pdf_url 在 primary_location 里
    #     if not pdf_url and work.get("primary_location"):
    #         pdf_url = work.get("primary_location").get("pdf_url")

    #     papers.append({
    #         "id": work["id"],
    #         "title": work["display_name"],
    #         "year": work["publication_year"],
    #         "cited_by": work["cited_by_count"],
    #         "abstract": abstract,
    #         "url": work.get("doi") or work.get("id"),
    #         "pdf_url": pdf_url,
    #         "authors": [a["author"]["display_name"] for a in work.get("authors", [])],
    #         "concepts": [c["display_name"] for c in work.get("concepts", [])[:3]]
    #     })
            
    #     self.logger.info(f"Found {len(papers)} valid papers.")

    #     # --- [新增] 调试输出到 data/papers_debug.txt ---
    #     debug_path = os.path.join("data", "papers_debug.txt")
    #     # 确保 data 目录存在
    #     os.makedirs("data", exist_ok=True) 
    #     with open(debug_path, "w", encoding="utf-8") as f:
    #         for p in papers:
    #             f.write(f"ID: {p['id']}\nTitle: {p['title']}\nPDF: {p['pdf_url']}\n{'-'*30}\n")
    #     self.logger.info(f"Saved metadata to {debug_path}")

    #     return papers

    def search(self, query: str, top_k: int = None, concept_ids: list = None) -> list:
        if top_k is None:
            top_k = settings.RAG_DOWNLOAD_K
            
        self.logger.info(f"Searching OpenAlex for: '{query}' (Limit: {top_k})")
        
        # 1. 基础搜索
        # Works().search(query) 默认就是按 relevance_score 排序的，这正是我们想要的！
        search_query = Works().search(query).filter(has_abstract=True)
        
        # 2. 概念过滤 (如果有)
        if concept_ids:
            search_query = search_query.filter(concepts={"id": "|".join(concept_ids)})
            self.logger.info(f"Filtering by Concepts: {concept_ids}")

        # 3. 年份过滤 (保留最近几年的)
        search_query = search_query.filter(from_publication_date="2020-01-01")
        
        try:
            # [关键修改] 移除 .sort(cited_by_count="desc")
            # 只有当没有 query (纯浏览模式) 时，才需要按引用排序。
            # 这里我们有 query，所以相信 OpenAlex 的 BM25 相关性排序。
            results = search_query.get(per_page=top_k)
        except Exception as e:
            self.logger.error(f"API Request Failed: {e}")
            return []

        papers = []
        for work in results:
            abstract = self._invert_abstract(work.get("abstract_inverted_index"))
            if not abstract or len(abstract) < 50: 
                continue

            # --- PDF 链接获取策略 (保持你之前的增强版逻辑) ---
            pdf_url = None
            
            # 策略 1: Best OA
            best_oa = work.get("best_oa_location")
            if best_oa:
                pdf_url = best_oa.get("pdf_url")
            
            # 策略 2: Primary Location
            if not pdf_url and work.get("primary_location"):
                pdf_url = work.get("primary_location").get("pdf_url")

            # 策略 3: ArXiv Fallback
            if not pdf_url:
                ids = work.get("ids", {})
                arxiv_url = ids.get("arxiv")
                if arxiv_url:
                    pdf_url = arxiv_url.replace("/abs/", "/pdf/") + ".pdf"
                    self.logger.info(f"🔗 Recovered ArXiv PDF for {work['id']}: {pdf_url}")

            papers.append({
                "id": work["id"],
                "title": work["display_name"],
                "year": work["publication_year"],
                "cited_by": work["cited_by_count"],
                "abstract": abstract,
                "url": work.get("doi") or work.get("ids", {}).get("openalex") or work.get("id"),
                "pdf_url": pdf_url,
                "authors": [a["author"]["display_name"] for a in work.get("authors", [])],
                "concepts": [c["display_name"] for c in work.get("concepts", [])[:3]]
            })
            
        self.logger.info(f"Found {len(papers)} valid papers.")

        # Debug 输出
        debug_path = os.path.join("data", "papers_debug.txt")
        os.makedirs("data", exist_ok=True) 
        with open(debug_path, "w", encoding="utf-8") as f:
            for p in papers:
                status = "✅ Has PDF" if p['pdf_url'] else "❌ No PDF"
                f.write(f"[{status}] {p['id']} | {p['title']}\nURL: {p['url']}\nPDF: {p['pdf_url']}\n{'-'*30}\n")
        self.logger.info(f"Saved metadata to {debug_path}")

        return papers