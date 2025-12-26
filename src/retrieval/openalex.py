import time
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import pyalex
from pyalex import Works
# from sentence_transformers import CrossEncoder # Lazy load

from src.config import settings
from src.retrieval.base import BaseRetriever
from src.utils.logger import setup_logger

# 配置 OpenAlex
pyalex.config.email = settings.OPENALEX_EMAIL

class OpenAlexRetriever(BaseRetriever):
    """
    [Clean Version] OpenAlexRetriever
    职责单一化：只负责根据传入的指令（Query 或 Concept IDs）执行搜索和清洗。
    不再包含任何 LLM 思考逻辑。
    """

    def __init__(self):
        self.logger = setup_logger("OpenAlex")
        # 移除了 self.llm 和 self.prompts，因为检索器不需要思考，只需要执行
        self._abstract_cache: Dict[str, str] = {}
        self._bge_reranker = None

    def _decode_abstract_inverted_index(self, inverted_index: Optional[Dict[str, List[int]]]) -> Optional[str]:
        """将倒排索引还原为文本"""
        if not inverted_index: return None
        try:
            max_pos = -1
            for positions in inverted_index.values():
                if not positions: continue
                pmax = max(positions)
                if pmax > max_pos: max_pos = pmax
            if max_pos < 0: return None

            tokens = [""] * (max_pos + 1)
            for word, positions in inverted_index.items():
                if not word or not positions: continue
                for pos in positions:
                    if 0 <= pos <= max_pos and not tokens[pos]:
                        tokens[pos] = word
            text = " ".join(tokens).strip()
            return text if text else None
        except Exception:
            return None

    def _rank_papers(self, query: str, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """使用 BGE-Reranker 进行重排"""
        if not papers: return []
        
        if self._bge_reranker is None:
            try:
                from sentence_transformers import CrossEncoder
                self.logger.info("Loading Reranker: BAAI/bge-reranker-v2-m3 ...")
                self._bge_reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
            except Exception as e:
                self.logger.error(f"Reranker init failed: {e}")
                return papers

        pairs = []
        for p in papers:
            doc = f"{p.get('title', '')}\n{p.get('abstract', '')}"
            pairs.append((query, doc))

        try:
            scores = self._bge_reranker.predict(pairs, batch_size=16)
            scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            
            ranked_papers = []
            for idx, score in scored:
                p = papers[idx].copy()
                p["_rerank_score"] = float(score)
                ranked_papers.append(p)
            
            self.logger.info(f"Reranked {len(papers)} papers.")
            return ranked_papers
        except Exception as e:
            self.logger.error(f"Rerank prediction failed: {e}")
            return papers

    def search(self, query: str, top_k: int = None, concept_ids: list = None, **kwargs) -> list:
        """
        执行搜索
        :param query: 用户原始查询
        :param top_k: 下载数量
        :param concept_ids: [可选] 由 expansion.py 提供的 Concept ID 列表
        """
        if top_k is None:
            top_k = getattr(settings, "RAG_DOWNLOAD_K", 10)
            
        # 默认只查最近 5 年
        since_years = kwargs.get("since_years", 5)
        from_date = f"{datetime.utcnow().year - since_years}-01-01"
        
        # 基础过滤器：有摘要 + 5年内
        base_query = Works().filter(has_abstract=True).filter(from_publication_date=from_date)
        
        # --- 分支逻辑：是否使用 Concept ---
        if concept_ids:
            self.logger.info(f"Searching with Concept IDs: {concept_ids}")
            # 使用 Concept ID 过滤，按引用量降序
            works_q = base_query.filter(concepts={"id": "|".join(concept_ids)}).sort(cited_by_count="desc")
        else:
            self.logger.info(f"Searching by Text Match: {query}")
            # 纯文本搜索
            works_q = base_query.search(query)

        # 执行 API 请求 (获取 3倍 候选供 Rerank)
        candidate_k = min(100, top_k * 3)
        results = []
        
        try:
            results = works_q.get(per_page=candidate_k)
        except Exception as e:
            self.logger.warning(f"OpenAlex API Error: {e}, retrying...")
            time.sleep(1)
            try:
                results = works_q.get(per_page=candidate_k)
            except Exception:
                pass

        # 如果 Concept 搜索为空，自动降级为文本搜索兜底 (保持鲁棒性)
        if not results and concept_ids:
            self.logger.warning("Concept search returned 0 results. Fallback to text search.")
            results = base_query.search(query).get(per_page=candidate_k)

        # --- 数据清洗与 PDF URL 提取 ---
        papers = []
        for work in results or []:
            abstract = self._decode_abstract_inverted_index(work.get("abstract_inverted_index"))
            if not abstract or len(abstract) < 50: continue

            # PDF URL 获取策略
            pdf_url = None
            if work.get("best_oa_location"):
                pdf_url = work.get("best_oa_location").get("pdf_url")
            if not pdf_url and work.get("primary_location"):
                pdf_url = work.get("primary_location").get("pdf_url")
            if not pdf_url:
                ids = work.get("ids", {})
                arxiv_url = ids.get("arxiv")
                if arxiv_url:
                    pdf_url = arxiv_url.replace("/abs/", "/pdf/")
                    if not pdf_url.endswith(".pdf"): pdf_url += ".pdf"

            authors = [a.get("author", {}).get("display_name") for a in work.get("authorships", [])]
            
            papers.append({
                "id": work["id"],
                "title": work["display_name"],
                "year": work["publication_year"],
                "cited_by": work.get("cited_by_count", 0),
                "abstract": abstract,
                "url": work.get("doi") or work.get("id"),
                "pdf_url": pdf_url,
                "authors": authors[:5],
                "concepts": [c["display_name"] for c in work.get("concepts", [])[:3]]
            })

        # Rerank
        self.logger.info(f"Reranking {len(papers)} candidates...")
        ranked_papers = self._rank_papers(query, papers)
        final_papers = ranked_papers[:top_k]

        # Debug 日志 (写入文件)
        try:
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            debug_path = os.path.join(root_dir, "data", "papers_debug.txt")
            os.makedirs(os.path.dirname(debug_path), exist_ok=True)
            
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(f"Query: {query}\nTime: {datetime.now()}\nStrategy: {'Concept Graph' if concept_ids else 'Text Match'}\n{'='*40}\n")
                for p in final_papers:
                    status = "✅ PDF" if p.get('pdf_url') else "❌ No PDF"
                    score = p.get("_rerank_score", 0)
                    f.write(f"[{status}] [Score:{score:.4f}] {p['title']}\nID: {p['id']}\nPDF: {p['pdf_url']}\n{'-'*30}\n")
        except Exception:
            pass

        return final_papers