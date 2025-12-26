import time
import math
import re
import json
import os
from typing import List, Dict, Optional, Set
from pyalex import Concepts
from src.utils.logger import setup_logger
from src.core.llm import LLMService

class ConceptExpander:
    def __init__(self):
        self.logger = setup_logger("GraphExpander")
        self.llm = LLMService()
        # [新增] 加载外部 Prompt 配置
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict:
        """加载同目录下的 expansion_prompts.json"""
        try:
            # 获取当前脚本所在目录 (src/graph)
            base_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(base_dir, "expansion_prompts.json")
            
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    self.logger.info(f"Loaded prompts from {json_path}")
                    return json.load(f)
            else:
                self.logger.warning(f"Prompt file not found at {json_path}, using hardcoded defaults.")
                return {}
        except Exception as e:
            self.logger.error(f"Failed to load prompts: {e}")
            return {}

    def _tokenize(self, s: str) -> Set[str]:
        """[Helper] 分词并清洗"""
        return {w.strip(".,;:()[]{}\"'").lower() for w in (s or "").replace("-", " ").split() if len(w) >= 3}

    def _keyword_phrase(self, q: str) -> str:
        """[Helper] 兜底策略：提取核心关键词"""
        s = (q or "").strip()
        if not s: return ""
        # 过滤中文
        if re.search(r"[\u4e00-\u9fff]", s): return ""
        
        stop = {
            "how", "to", "use", "using", "what", "why", "which", "when", "where", "who",
            "is", "are", "was", "were", "do", "does", "did", "can", "could", "should", "would",
            "a", "an", "the", "of", "and", "or", "for", "in", "on", "with", "by", "from", "as",
            "about", "into", "over", "under", "between", "within", "without",
        }
        toks = []
        for w in re.split(r"[\s\-_/]+", s.lower()):
            w = w.strip(".,;:()[]{}\"'")
            if len(w) < 3 or w in stop: continue
            toks.append(w)
        return " ".join(toks[:6]).strip()

    def _get_search_phrases(self, query: str) -> List[str]:
        """[Core] 利用 LLM 将 Query 转换为学术 Concept 短语"""
        
        # 1. 获取 Prompt (优先从 JSON 读取，否则使用默认值)
        default_system = (
            "You are The Search Engineer for OpenAlex Concepts.\n"
            "Return ONLY valid JSON.\n"
            'Output schema: {"phrases": ["..."]}\n'
            "Task: Convert the query into 2-4 concise English academic concept phrases."
        )
        
        # 安全读取配置
        cfg = self.prompts.get("concept_extraction", {})
        system_prompt = cfg.get("system", default_system)
        user_tmpl = cfg.get("user_template", "Query: {query}")

        # 2. 尝试 LLM JSON 模式
        try:
            obj = self.llm.chat_json(
                system_prompt=system_prompt, 
                user_prompt=user_tmpl.format(query=query)
            )
            if obj and isinstance(obj.get("phrases"), list):
                return [p.strip() for p in obj["phrases"] if isinstance(p, str) and p.strip()]
        except Exception:
            pass

        # 3. 兜底策略 (如果 LLM 失败或 JSON 解析失败)
        kw = self._keyword_phrase(query)
        phrases = [kw] if kw else []
        if query and query not in phrases:
            phrases.append(query)
        return phrases

    def _score_candidate(self, phrase: str, concept: Dict) -> float:
        """[Core] 打分算法：名称匹配 + 引用热度 - 层级惩罚"""
        name = (concept.get("display_name") or "").strip()
        if not name: return -1e9
        
        ph_l = phrase.lower()
        name_l = name.lower()
        
        exact = 1.0 if name_l == ph_l else 0.0
        
        ph_toks = self._tokenize(phrase)
        name_toks = self._tokenize(name)
        overlap = len(ph_toks & name_toks) / max(1, len(ph_toks))
        
        works = float(concept.get("works_count") or 0.0)
        works_score = math.log1p(works) / 10.0 if works > 0 else 0.0
        
        level = float(concept.get("level") or 0.0)
        level_penalty = 0.03 * level
        
        return 1.5 * exact + 1.2 * overlap + works_score - level_penalty

    def expand_query(self, query: str) -> Optional[Dict]:
        """
        执行图谱扩展流程
        """
        self.logger.info(f"Expanding query via Knowledge Graph: {query}")
        
        phrases = self._get_search_phrases(query)
        if not phrases:
            return None

        best_match = None
        best_score = -1e9
        
        # 遍历短语，尝试搜索
        for ph in phrases[:4]:
            candidates = None
            for attempt in range(2): # Retry 机制
                try:
                    candidates = Concepts().search(ph).get(per_page=5)
                    break
                except Exception:
                    time.sleep(0.5)
            
            if not candidates: continue

            for c in candidates:
                score = self._score_candidate(ph, c)
                if score > best_score:
                    best_score = score
                    best_match = c

        if not best_match:
            self.logger.warning("No matching concept found in KG.")
            return None

        name = best_match['display_name']
        level = best_match['level']
        self.logger.info(f"Mapped '{query}' -> Concept: {name} (L{level}) | Score: {best_score:.2f}")
        
        return {
            "id": best_match['id'],
            "name": name,
            "level": level
        }