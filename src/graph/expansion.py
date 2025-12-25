import re
import time
from pyalex import Concepts
from src.utils.logger import setup_logger
from src.core.llm import LLMService

class ConceptExpander:
    def __init__(self):
        self.logger = setup_logger("GraphExpander")
        self.llm = LLMService()

    def expand_query(self, query: str) -> dict:
        """
        1. 搜索 query 对应的 Concept 节点
        2. 获取该节点的父节点 (Ancestors) 和 关联节点 (Related)
        返回: {"primary_concept_id": str, "concept_name": str}
        """
        self.logger.info(f"Expanding query via Knowledge Graph: {query}")

        system_prompt = (
            "You are The Search Engineer for OpenAlex Concepts.\n"
            "Return ONLY valid JSON.\n"
            'Output schema: {"phrases": ["..."]}\n'
            "Task: Convert the query into 2-4 concise English academic concept phrases."
        )
        obj = self.llm.chat_json(system_prompt=system_prompt, user_prompt=f"Query: {query}")
        phrases = []
        if isinstance(obj, dict) and isinstance(obj.get("phrases"), list):
            phrases = [p.strip() for p in obj["phrases"] if isinstance(p, str) and p.strip()]

        # JSON 失败时回退：让模型输出“每行一个短语”
        if not phrases:
            system_prompt_fallback = (
                "You are The Search Engineer for OpenAlex Concepts.\n"
                "Task: Convert the query into 2-4 concise English academic concept phrases.\n"
                "Output rules:\n"
                "- Output ONLY the phrases\n"
                "- One phrase per line\n"
                "- No numbering, no bullets, no extra text"
            )
            text = self.llm.chat(system_prompt_fallback, f"Query: {query}") or ""
            for line in text.splitlines():
                s = line.strip().lstrip("-").strip()
                if not s:
                    continue
                s = s.split(".", 1)[-1].strip() if s[:2].isdigit() and "." in s[:4] else s
                if len(s) >= 3:
                    phrases.append(s)
                if len(phrases) >= 4:
                    break

        if not phrases:
            phrases = [query]

        def _keyword_phrase(q: str) -> str:
            """
            标题: 将问句/长句压缩为更适合 Concepts.search 的关键词短语

            Input:
              - 参数:
                  - q (str, required): 原始 query（可能包含 how/what/to 等问句成分）
            Output:
              - return (str): 关键词短语（可能为空字符串）

            Why:
              - OpenAlex Concepts 搜索更偏向“概念名/名词短语”；对完整问句直接 search 往往召回为空。
              - 通过去掉常见功能词，把 query 压缩为 2-6 个核心 token，可显著提高命中率。
            """
            s = (q or "").strip()
            if not s:
                return ""
            # 只对英文做简单 stopwords 过滤；中文直接返回空让其他短语路径处理
            if re.search(r"[\u4e00-\u9fff]", s):
                return ""
            stop = {
                "how", "to", "use", "using", "what", "why", "which", "when", "where", "who",
                "is", "are", "was", "were", "do", "does", "did", "can", "could", "should", "would",
                "a", "an", "the", "of", "and", "or", "for", "in", "on", "with", "by", "from", "as",
                "about", "into", "over", "under", "between", "within", "without",
            }
            toks = []
            for w in re.split(r"[\s\-_/]+", s.lower()):
                w = w.strip(".,;:()[]{}\"'")
                if len(w) < 3:
                    continue
                if w in stop:
                    continue
                toks.append(w)
            if not toks:
                return ""
            # 只取前几个核心词，避免过长
            return " ".join(toks[:6]).strip()

        # 追加一个“关键词短语”作为 fallback，提升问句命中率（不改变原结构，只增强召回）
        kw = _keyword_phrase(query)
        if kw and kw not in phrases:
            phrases = [kw] + phrases
        if query and query not in phrases:
            phrases.append(query)

        def _tokenize(s: str) -> set:
            return {w.strip(".,;:()[]{}\"'").lower() for w in (s or "").replace("-", " ").split() if len(w) >= 3}

        def _score_candidate(ph: str, c: dict) -> float:
            """
            标题: 为 Concepts 搜索候选打分（避免误选无关的高 level 概念）

            Input:
              - 参数:
                  - ph (str, required): 当前用于 Concepts().search 的短语
                  - c (dict, required): OpenAlex concept 候选对象
            Output:
              - return (float): 越大越好

            Why:
              - 之前用“level 越高越具体”会把宽泛 query（如 Machine Learning）错映射到无关的细分概念（如 Wake-sleep algorithm）。
              - 现在以“词面匹配 + works_count 稳定性”为主，让 Concepts 搜索的相关性排序起主要作用。
            """
            name = (c.get("display_name") or "").strip()
            if not name:
                return -1e9
            ph_l = ph.lower()
            name_l = name.lower()
            exact = 1.0 if name_l == ph_l else 0.0
            overlap = len(_tokenize(ph) & _tokenize(name)) / max(1, len(_tokenize(ph)))
            works = float(c.get("works_count") or 0.0)
            # log 压缩，防止 works_count 一票否决
            works_score = 0.0
            if works > 0:
                import math
                works_score = math.log1p(works) / 10.0
            # 轻微惩罚过深层级（通常更细分更容易偏题）
            level = float(c.get("level") or 0.0)
            level_penalty = 0.03 * level
            return 1.5 * exact + 1.2 * overlap + works_score - level_penalty

        # 在 Concept 图谱中搜索实体：对多个短语取候选，按“词面相关 + works_count”选最优
        best_match = None
        best_score = -1e9
        for ph in phrases[:4]:
            candidates = None
            last_err = None
            for attempt in range(3):
                try:
                    candidates = Concepts().search(ph).get(per_page=10)
                    break
                except Exception as e:
                    last_err = e
                    sleep_s = 0.6 * (2 ** attempt)
                    self.logger.warning(f"Concepts.search failed (attempt {attempt+1}/3): {e}; retry in {sleep_s:.1f}s")
                    time.sleep(sleep_s)
            if candidates is None:
                self.logger.warning(f"Concepts.search gave no result for phrase='{ph}' (err={last_err})")
                continue
            for c in candidates or []:
                s = _score_candidate(ph, c)
                if s > best_score:
                    best_score = s
                    best_match = c
        
        if not best_match:
            self.logger.warning("No matching concept found in KG.")
            return None

        name = best_match['display_name']
        cid = best_match['id']
        level = best_match['level']
        
        self.logger.info(f"Mapped '{query}' -> Concept Node: {name} (Level {level})")
        
        # 这里你可以做更复杂的操作，比如获取它的子节点
        # 目前 MVP 我们只返回 ID，用于精准过滤
        return {
            "id": cid,
            "name": name,
            "level": level
        }