from pyalex import Concepts
from src.utils.logger import setup_logger

class ConceptExpander:
    def __init__(self):
        self.logger = setup_logger("GraphExpander")

    def expand_query(self, query: str) -> dict:
        """
        1. 搜索 query 对应的 Concept 节点
        2. 获取该节点的父节点 (Ancestors) 和 关联节点 (Related)
        返回: {"primary_concept_id": str, "concept_name": str}
        """
        self.logger.info(f"Expanding query via Knowledge Graph: {query}")
        
        # 在 Concept 图谱中搜索实体
        candidates = Concepts().search(query).get(per_page=1)
        
        if not candidates:
            self.logger.warning("No matching concept found in KG.")
            return None

        best_match = candidates[0]
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