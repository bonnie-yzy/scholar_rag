from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseRetriever(ABC):
    @abstractmethod
    def search(self, query: str, top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        输入 query，返回标准化的论文列表。
        标准化格式必须包含: id, title, abstract, year, authors
        """
        pass