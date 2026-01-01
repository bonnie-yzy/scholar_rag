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

# [新增] 导入 PageRank 计算函数 (请确保 src/mining/graph_ranking.py 存在)
try:
    from src.mining.graph_ranking import calculate_pagerank
except ImportError:
    # 兜底：如果文件没拷过来，定义一个空函数防止报错，但功能会失效
    def calculate_pagerank(**kwargs): return {}

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
        """使用 BGE-Reranker 对候选论文进行语义重排序"""
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

    # [新增方法] 引用图重排序逻辑
    def _apply_citation_graph_rerank(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        基于局部引文子图的 PageRank 融合重排序 (Citation Graph Re-ranking)
        融合公式: Hybrid = alpha * BGE_Score + beta * PageRank_Score
        """
        if not papers:
            return []
        # 检查配置开关，默认为 True
        if not getattr(settings, "ENABLE_CITATION_RERANK", True):
            return papers

        try:
            alpha = float(getattr(settings, "CITATION_RERANK_ALPHA", 0.8))
            beta = float(getattr(settings, "CITATION_RERANK_BETA", 0.2))
            
            # 构建节点 ID 列表
            node_ids = [p.get("id") for p in papers if p.get("id")]
            node_set = set(node_ids)

            # 构建局部引文图的边 (只保留 Top-N 内部的引用关系)
            edges = []
            edge_cnt = 0
            for p in papers:
                src = p.get("id")
                if not src or src not in node_set:
                    continue
                refs = p.get("referenced_works") or []
                for dst in refs:
                    if dst in node_set and dst != src:
                        edges.append((src, dst))
                        edge_cnt += 1

            # 如果边太少，PageRank 没意义，直接返回原列表
            if edge_cnt < 2:
                return papers

            # 计算 PageRank
            pr = calculate_pagerank(
                nodes=node_ids,
                edges=edges,
                damping=float(getattr(settings, "PAGERANK_DAMPING", 0.85)),
                max_iter=int(getattr(settings, "PAGERANK_MAX_ITER", 100)),
                tol=float(getattr(settings, "PAGERANK_TOL", 1e-6)),
            )

            # 归一化 BGE 分数 (Min-Max Normalization)
            rerank_scores = [float(p.get("_rerank_score", 0.0)) for p in papers]
            r_min = min(rerank_scores) if rerank_scores else 0.0
            r_max = max(rerank_scores) if rerank_scores else 0.0
            r_denom = (r_max - r_min) if (r_max - r_min) > 1e-12 else 1.0

            # 归一化 PageRank 分数 (除以最大值)
            pr_scores = [float(pr.get(p.get("id", ""), 0.0)) for p in papers]
            pr_max = max(pr_scores) if pr_scores else 0.0
            pr_denom = pr_max if pr_max > 1e-12 else 1.0

            enriched = []
            for p in papers:
                pid = p.get("id", "")
                rerank = float(p.get("_rerank_score", 0.0))
                pr_v = float(pr.get(pid, 0.0))
                
                # 计算归一化分数
                rerank_norm = (rerank - r_min) / r_denom
                pr_norm = pr_v / pr_denom

                p2 = p.copy()
                p2["_pagerank_score"] = pr_v
                p2["_rerank_score_norm"] = rerank_norm
                p2["_pagerank_score_norm"] = pr_norm
                # 计算最终混合分数
                p2["_hybrid_score"] = alpha * rerank_norm + beta * pr_norm
                enriched.append(p2)

            # 按混合分数重新排序
            enriched.sort(key=lambda x: x.get("_hybrid_score", 0.0), reverse=True)
            self.logger.info(f"Citation rerank applied (nodes={len(node_ids)}, edges={edge_cnt}).")
            return enriched
        except Exception as e:
            self.logger.warning(f"Citation rerank failed, fallback to rerank only. err={e}")
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
                "concepts": [c["display_name"] for c in work.get("concepts", [])[:3]],
                # [新增] 提取引用列表，用于构建局部引文图
                "referenced_works": work.get("referenced_works") or [],
            })

        # 1. 语义重排 (BGE Rerank)
        self.logger.info(f"Reranking {len(papers)} candidates...")
        ranked_papers = self._rank_papers(query, papers)

        # 2. [新增] 引用图重排 (PageRank 融合)
        # 允许通过 kwargs 动态覆盖配置，否则读取 settings
        enable_citation_rerank = kwargs.get("enable_citation_rerank", None)
        if enable_citation_rerank is not None:
            if bool(enable_citation_rerank):
                final_ranked = self._apply_citation_graph_rerank(ranked_papers)
            else:
                final_ranked = ranked_papers
        else:
            final_ranked = self._apply_citation_graph_rerank(ranked_papers)

        # 截取 Top-K
        final_papers = final_ranked[:top_k]

        # Debug 日志 (写入文件) - [更新] 包含 Hybrid 分数
        try:
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            debug_path = os.path.join(root_dir, "data", "papers_debug.txt")
            os.makedirs(os.path.dirname(debug_path), exist_ok=True)
            
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(f"Query: {query}\nTime: {datetime.now()}\nStrategy: {'Concept Graph' if concept_ids else 'Text Match'}\n{'='*40}\n")
                for p in final_papers:
                    status = "✅ PDF" if p.get('pdf_url') else "❌ No PDF"
                    score = p.get("_rerank_score", 0)
                    # [新增] 打印高级分数详情
                    pr_s = p.get("_pagerank_score", None)
                    hyb = p.get("_hybrid_score", None)
                    extra = ""
                    if pr_s is not None or hyb is not None:
                        extra = f" [PR:{float(pr_s or 0):.6f}] [Hybrid:{float(hyb or 0):.6f}]"
                    
                    f.write(f"[{status}] [Score:{score:.4f}]{extra} {p['title']}\nID: {p['id']}\nPDF: {p['pdf_url']}\n{'-'*30}\n")
        except Exception:
            pass

        return final_papers