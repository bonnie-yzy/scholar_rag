import math
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pyalex
from pyalex import Concepts, Works

from src.config import settings
from src.core.llm import LLMService
from src.retrieval.base import BaseRetriever  # 注意：原来的 base.py 引用路径是否正确
from src.utils.logger import setup_logger

# 直接从 settings 对象配置 pyalex
pyalex.config.email = settings.OPENALEX_EMAIL


def _normalize_terms(phrases: List[str]) -> List[str]:
    """
    标题: 将短语列表归一化为轻量检索词（用于重排）

    Example:
      - 输入: ["Protein Structure Prediction", "machine learning for proteins"]
      - 输出: ["protein", "structure", "prediction", "machine", "learning", "proteins"]
    """
    out: List[str] = []
    seen = set()
    for p in phrases or []:
        for w in (p or "").lower().replace("-", " ").split():
            w = w.strip(".,;:()[]{}\"'")
            if len(w) < 3:
                continue
            if w in seen:
                continue
            seen.add(w)
            out.append(w)
    return out


class OpenAlexRetriever(BaseRetriever):
    """
    标题: OpenAlexRetriever - 基于 Concept 过滤的论文召回与摘要还原（避免 works.search 噪音）

    Input:
      - 参数:
          - search(query, top_k=None, concept_ids=None, **kwargs)
              - query (str, required): 用户自然语言 query（可为中文）
              - top_k (int, optional, default=None): 返回条数；None 时使用 settings.RAG_TOP_K
              - concept_ids (list, optional, default=None): 外部已指定的 OpenAlex Concept IDs；为空则内部用 LLM 辅助映射
              - kwargs:
                  - since_years (int, optional, default=5): 近 N 年召回窗口（默认 5 年）
      - 上下文:
          - settings.OPENALEX_EMAIL / settings.RAG_TOP_K

    Output:
      - 返回:
          - List[Dict]: 标准化论文列表（包含 id/title/abstract/year/authors...）

    Why:
      - OpenAlex works.search 是 BM25F 全文关键词匹配，容易“太杂/太旧”；改为先定位 Concept，再用 filter 精准召回。

    References:
      - OpenAlex API: Works filter / Concepts endpoint
      - BM25F: keyword-based full-text retrieval limitations

    Calls:
      - Concepts().search (pyalex): 概念候选查询
      - Works().filter/sort/get (pyalex): 论文召回
      - LLMService.chat_json (src/core/llm.py): query→英文短语

    Example:
      - 输入: query="如何用机器学习做蛋白质预测", top_k=10
      - 输出: [{"title":"...", "abstract":"...", "year":2023, ...}, ...]
    """

    def __init__(self):
        self.logger = setup_logger("OpenAlex")
        self.llm = LLMService()
        self._concept_cache: Dict[str, List[str]] = {}
        self._abstract_cache: Dict[str, str] = {}

    def _decode_abstract_inverted_index(self, inverted_index: Optional[Dict[str, List[int]]]) -> Optional[str]:
        """
        标题: 将 OpenAlex 的 abstract_inverted_index 还原为可读摘要（高性能版）

        Input:
          - 参数:
              - inverted_index (dict, optional): {"word":[pos...], ...}
          - 上下文:
              - self._abstract_cache（work_id 级缓存）
          - 依赖:
              - 无

        Output:
          - 返回:
              - Optional[str]: 还原后的摘要文本；缺失时返回 None
          - 副作用:
              - 无
          - 错误:
              - 无（输入异常时返回 None）

        Why:
          - OpenAlex 不直接提供摘要文本；RAG 需要可读摘要作为基础语料。

        References:
          - OpenAlex: abstract_inverted_index schema

        Calls:
          - 无

        Example:
          - 输入: {"word":[1,5], "is":[2]}
          - 输出: " word is  word"
        """
        if not inverted_index:
            return None

        try:
            max_pos = -1
            for positions in inverted_index.values():
                if not positions:
                    continue
                pmax = max(positions)
                if pmax > max_pos:
                    max_pos = pmax
            if max_pos < 0:
                return None

            tokens = [""] * (max_pos + 1)
            for word, positions in inverted_index.items():
                if not word or not positions:
                    continue
                for pos in positions:
                    if 0 <= pos <= max_pos and not tokens[pos]:
                        tokens[pos] = word
            text = " ".join(tokens).strip()
            return text if text else None
        except Exception:
            return None

    def _llm_phrases_for_query(self, query: str) -> List[str]:
        """
        标题: 将用户 query 转为英文“学术概念短语”候选（用于 Concepts 检索）

        Input:
          - 参数:
              - query (str, required): 用户自然语言 query（可为中文/英文）
          - 上下文:
              - self.llm: OpenRouter/OpenAI-compatible LLM 客户端
          - 依赖:
              - LLMService.chat_json

        Output:
          - 返回:
              - List[str]: 2~5 个英文短语（失败则回退为 [query]）
          - 副作用:
              - 网络调用（LLM）
          - 错误:
              - 异常时回退为 [query]

        Why:
          - 不让 LLM “背 concept 列表”，只让其把意图翻译成规范英文短语，再用 OpenAlex Concepts 端点做真实候选召回。

        References:
          - OpenAlex Concepts endpoint supports search by display_name

        Calls:
          - LLMService.chat_json (src/core/llm.py): 结构化短语生成

        Example:
          - 输入: "如何用机器学习做蛋白质预测"
          - 输出: ["protein structure prediction", "machine learning for proteins", ...]
        """
        system_prompt = (
            "You are The Search Engineer for academic retrieval.\n"
            "Return ONLY valid JSON object.\n"
            "Task: Convert the user query into 3-5 concise English academic concept phrases.\n"
            'Output schema: {"phrases": ["...", "..."]}\n'
            "Rules: phrases should be noun phrases; avoid long sentences; no extra keys."
        )
        user_prompt = f"User query: {query}"
        obj = self.llm.chat_json(system_prompt=system_prompt, user_prompt=user_prompt)
        phrases = None
        if isinstance(obj, dict):
            phrases = obj.get("phrases")

        cleaned: List[str] = []
        if isinstance(phrases, list):
            for p in phrases:
                if isinstance(p, str) and p.strip():
                    cleaned.append(p.strip())
            if cleaned:
                return cleaned[:5]

        # Fallback: JSON 解析失败时，尝试让模型输出“每行一个短语”，再做简单解析（避免中文 query 直接拿去 Concepts.search 命中为 0）
        system_prompt_fallback = (
            "You are The Search Engineer for academic retrieval.\n"
            "Task: Convert the user query into 3-5 concise English academic concept phrases.\n"
            "Output rules:\n"
            "- Output ONLY the phrases\n"
            "- One phrase per line\n"
            "- No numbering, no bullets, no extra text"
        )
        text = self.llm.chat(system_prompt=system_prompt_fallback, user_prompt=user_prompt) or ""
        for line in text.splitlines():
            s = line.strip().lstrip("-").strip()
            if not s:
                continue
            # 去掉可能的编号
            s = s.split(".", 1)[-1].strip() if s[:2].isdigit() and "." in s[:4] else s
            if len(s) >= 3:
                cleaned.append(s)
            if len(cleaned) >= 5:
                break
        return cleaned[:5] if cleaned else [query]

    def _map_query_to_concept_ids(self, query: str, max_ids: int = 3, phrases: Optional[List[str]] = None) -> List[str]:
        """
        标题: query → OpenAlex Concept IDs（LLM 短语 + Concepts 搜索）

        Input:
          - 参数:
              - query (str, required): 用户 query
              - max_ids (int, optional, default=3): 最多返回多少个 concept id
              - phrases (Optional[List[str]], optional, default=None): 预先生成的英文短语；提供时将跳过 LLM 调用以减少重复成本
          - 上下文:
              - self._concept_cache: 避免重复映射
          - 依赖:
              - pyalex.Concepts
              - self._llm_phrases_for_query

        Output:
          - 返回:
              - List[str]: Concept IDs（形如 "https://openalex.org/Cxxxx"）
          - 副作用:
              - 网络调用（LLM + OpenAlex）
          - 错误:
              - 失败时返回 []

        Why:
          - 放弃 works.search 的 BM25F 全文检索噪音，改为用 Concepts 本体图谱做语义对齐，再用 works.filter 精准召回。

        References:
          - OpenAlex Concepts: ~65k concepts with display_name/level/ancestors

        Calls:
          - Concepts().search (pyalex): 根据短语召回概念候选

        Example:
          - 输入: "protein structure prediction"
          - 输出: ["https://openalex.org/C1234567890"]
        """
        qkey = (query or "").strip().lower()
        if not qkey:
            return []
        if qkey in self._concept_cache:
            return self._concept_cache[qkey]

        phrases = phrases if (isinstance(phrases, list) and phrases) else self._llm_phrases_for_query(query)
        candidates: List[Tuple[int, int, str]] = []
        # tuple: (level, works_count, concept_id)
        try:
            for ph in phrases:
                results = Concepts().search(ph).get(per_page=5)
                for c in results or []:
                    cid = c.get("id")
                    if not cid:
                        continue
                    level = int(c.get("level") or 0)
                    works_count = int(c.get("works_count") or 0)
                    candidates.append((level, works_count, cid))
        except Exception as e:
            self.logger.error(f"Concept lookup failed: {e}")
            self._concept_cache[qkey] = []
            return []

        # 更偏向更“具体”的概念：level 越高越具体；同 level 下选 works_count 更大的更稳
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        seen = set()
        picked: List[str] = []
        for _, __, cid in candidates:
            if cid in seen:
                continue
            seen.add(cid)
            picked.append(cid)
            if len(picked) >= max_ids:
                break

        self._concept_cache[qkey] = picked
        if picked:
            self.logger.info(f"Mapped query -> Concept IDs: {picked}")
        else:
            self.logger.warning("No concept IDs mapped; will fallback to looser retrieval.")
        return picked

    def _rank_papers(self, query_terms: List[str], papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        标题: 对候选论文做轻量重排（去旧、去噪、提相关）

        Input:
          - 参数:
              - query_terms (List[str], required): 归一化后的关键词（通常来自 LLM 英文短语/概念名）
              - papers (List[Dict], required): 候选论文（必须含 title/abstract/year/cited_by）
          - 上下文:
              - 无
          - 依赖:
              - math.log1p（引用量压缩）

        Output:
          - 返回:
              - List[Dict]: 按 score 降序排序后的 papers（每项附带内部 score）
          - 副作用:
              - 无
          - 错误:
              - 无

        Why:
          - API 侧排序难以同时兼顾“新近度+影响力+相关性”，在客户端做可解释的重排更稳。

        References:
          - log1p scaling for heavy-tailed counts

        Calls:
          - math.log1p (python stdlib): 压缩 cited_by_count

        Example:
          - 输入: query_terms=["protein","prediction"], papers=[...]
          - 输出: 排序后的 papers（含 score）
        """
        if not papers:
            return []

        current_year = datetime.utcnow().year
        max_cited = max(int(p.get("cited_by") or 0) for p in papers) or 1
        max_cited_log = math.log1p(max_cited)

        def text_match_score(text: str) -> float:
            if not text or not query_terms:
                return 0.0
            t = text.lower()
            hits = 0
            for term in query_terms:
                if term in t:
                    hits += 1
            return hits / max(1, len(query_terms))

        ranked: List[Dict[str, Any]] = []
        for p in papers:
            year = int(p.get("year") or 0)
            cited = int(p.get("cited_by") or 0)
            title = p.get("title") or ""
            abstract = p.get("abstract") or ""

            recency = 0.0
            if year > 0:
                # 近 10 年内线性归一化（即使 future 扩大 since_years 也能工作）
                recency = max(0.0, min(1.0, (year - (current_year - 10)) / 10.0))

            influence = math.log1p(cited) / max(1e-9, max_cited_log)
            match = 0.6 * text_match_score(title) + 0.4 * text_match_score(abstract)

            score = 0.50 * match + 0.30 * recency + 0.20 * influence
            p2 = dict(p)
            p2["_score"] = round(float(score), 6)
            ranked.append(p2)

        ranked.sort(key=lambda x: x.get("_score", 0.0), reverse=True)
        return ranked

    def search(self, query: str, top_k: int = None, concept_ids: list = None, **kwargs) -> list:
        """
        标题: 输入自然语言 query，返回按优先级排序的论文摘要（含标题信息）

        Input:
          - 参数:
              - query (str, required): 用户 query
              - top_k (int, optional, default=None): 返回条数；None 时使用 settings.RAG_TOP_K
              - concept_ids (list, optional, default=None): 外部指定 concept ids；为空则内部映射
              - kwargs:
                  - since_years (int, optional, default=5): 近 N 年过滤窗口（仅对主题检索生效）
          - 上下文:
              - settings.RAG_TOP_K
          - 依赖:
              - OpenAlex Works/Concepts API

        Output:
          - 返回:
              - list[dict]: 论文列表（含 id/title/abstract/year/cited_by/url/concepts/authors）
          - 副作用:
              - 网络请求 + 日志
          - 错误:
              - 请求失败返回 []

        Why:
          - 兼容原有接口，同时引入“Concept→filter→重排”的检索主链路以降低噪音。

        References:
          - OpenAlex Works filter: from_publication_date, concepts.id, has_abstract

        Calls:
          - _map_query_to_concept_ids: query→concept ids
          - Works().filter/sort/get: 召回 works
          - _decode_abstract_inverted_index: 摘要还原
          - _rank_papers: 轻量重排

        Example:
          - 输入: query="Machine Learning for protein prediction", top_k=10
          - 输出: 前 10 条论文摘要列表
        """
        if top_k is None:
            top_k = settings.RAG_TOP_K
            
        # 默认近 5 年
        since_years = int(kwargs.get("since_years", 10) or 10)
        from_year = datetime.utcnow().year - since_years
        from_date = f"{from_year}-01-01"

        self.logger.info(f"Retrieving from OpenAlex via filter (top_k={top_k}, since_years={since_years})")

        # 只生成一次 LLM phrases，用于：Concept 映射 + 重排 query_terms（避免重复调用 LLM）
        phrases = self._llm_phrases_for_query(query)

        mapped_concepts: List[str] = []
        # 若外部未指定 concept_ids，则内部映射（让“单独用 retriever”也能走 Concept-filter 主链路）
        if concept_ids:
            mapped_concepts = list(concept_ids)
        else:
            mapped_concepts = self._map_query_to_concept_ids(query, phrases=phrases)

        works_q = Works().filter(has_abstract=True).filter(from_publication_date=from_date)
        if mapped_concepts:
            works_q = works_q.filter(concepts={"id": "|".join(mapped_concepts)})
            self.logger.info(f"Filtering by Concepts: {mapped_concepts}")
        else:
            # 如果 Concept 映射失败，不要直接“全库最新论文”——那会非常杂。
            # 兜底：用 LLM 生成的英文短语做 Works().search，但仍保留 has_abstract + 近 5 年过滤，尽量压噪。
            fallback_q = phrases[0] if phrases else query
            self.logger.warning(f"No concept IDs mapped; fallback to Works.search with phrase='{fallback_q}'")
            works_q = Works().search(fallback_q).filter(has_abstract=True).filter(from_publication_date=from_date)

        # 拉更大的候选集用于重排（但避免过大响应体导致连接中断）
        per_page = min(100, max(30, int(top_k) * 4))

        def _fetch_with_retries(fetch_q, *, label: str) -> Optional[list]:
            """
            标题: 带退避重试的 OpenAlex 请求包装（修复 Response ended prematurely 等偶发失败）

            Input:
              - 参数:
                  - fetch_q: pyalex query object（已设置 filter/sort）
                  - label (str, required): 日志标记
            Output:
              - return (Optional[list]): 成功返回 list，否则 None

            Why:
              - OpenAlex/网络在高峰或响应较大时会出现连接提前断开；重试可显著提升成功率。
            """
            last_err = None
            for attempt in range(3):
                try:
                    return fetch_q.get(per_page=per_page)
                except Exception as e:
                    last_err = e
                    sleep_s = 0.8 * (2 ** attempt)
                    self.logger.warning(f"{label} failed (attempt {attempt+1}/3): {e}; retry in {sleep_s:.1f}s")
                    time.sleep(sleep_s)
            self.logger.error(f"{label} failed after retries: {last_err}")
            return None

        # filter 模式：优先用新近度做 API 侧粗排，避免“老神文”淹没
        results = _fetch_with_retries(works_q.sort(publication_date="desc"), label="OpenAlex works(filter)")
        if results is None and mapped_concepts:
            # 回退：concept 过滤导致太脆弱/过窄时，去掉 concept 再试一次（仍保持近 5 年 + has_abstract）
            works_fallback = Works().filter(has_abstract=True).filter(from_publication_date=from_date)
            results = _fetch_with_retries(works_fallback.sort(publication_date="desc"), label="OpenAlex works(fallback no-concept)")
        if results is None:
            return []

        # 用 LLM phrases + concept display_name 生成 query_terms（用于轻量重排）
        query_terms = _normalize_terms(phrases)

        papers: List[Dict[str, Any]] = []
        current_year = datetime.utcnow().year
        for work in results or []:
            wid = work.get("id")
            if wid and wid in self._abstract_cache:
                abstract = self._abstract_cache[wid]
            else:
                abstract = self._decode_abstract_inverted_index(work.get("abstract_inverted_index"))
                if wid and abstract:
                    self._abstract_cache[wid] = abstract

            if not abstract or len(abstract) < 50: 
                continue

            authors = []
            for a in work.get("authorships", [])[:5]:
                au = a.get("author", {}) or {}
                name = au.get("display_name")
                if name:
                    authors.append(name)

            # 基础 sanity check：过滤掉明显的未来年份（OpenAlex 偶发脏数据）
            year = work.get("publication_year")
            try:
                year_i = int(year) if year is not None else 0
            except Exception:
                year_i = 0
            if year_i > current_year + 1:
                continue

            papers.append(
                {
                    "id": wid,
                    "title": work.get("display_name"),
                    "year": year_i if year_i else year,
                    "cited_by": work.get("cited_by_count", 0),
                    "abstract": abstract,
                    "url": work.get("doi") or wid,
                    "concepts": [c.get("display_name") for c in (work.get("concepts") or [])[:5] if c.get("display_name")],
                    "authors": authors,
                }
            )

        # 重排 + 截断
        ranked = self._rank_papers(query_terms=query_terms, papers=papers)
        final = ranked[: int(top_k)]
        self.logger.info(f"Retrieved {len(final)} papers (from {len(papers)} candidates).")
        return final