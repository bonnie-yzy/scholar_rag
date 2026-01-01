import json
import os
from typing import Any, Dict, List, Optional, Union
from src.core.llm import LLMService
from src.utils.logger import setup_logger
from src.config import settings

# [新增] 尝试导入社区发现算法，用于结构化综述
try:
    from src.mining.graph_ranking import detect_communities_louvain
except ImportError:
    detect_communities_louvain = None

class ReviewGenerator:
    """
    [Merged Generator]
    架构：保留你的 Prompt 模板加载机制 (灵活性高)。
    内容：增强了 Context 格式化逻辑，吸收了队友的 Metadata 提取 (Cited By, Concepts)。
    功能：同时支持 流式输出 (Stream) 和 结构化综述 (Structured Review)。
    """
    def __init__(self):
        self.llm = LLMService()
        self.logger = setup_logger("Generator")
        
        # 加载 Prompt 模板
        try:
            # 兼容不同运行路径
            base_dir = os.path.dirname(os.path.abspath(__file__))
            prompt_path = os.path.join(base_dir, "prompts.json")
            
            if os.path.exists(prompt_path):
                with open(prompt_path, "r", encoding="utf-8") as f:
                    self.prompts = json.load(f)
            else:
                # 兜底：如果找不到文件，使用默认模板 (防止报错)
                self.logger.warning("prompts.json not found, using default fallback.")
                self.prompts = {
                    "system_base": "You are a helpful academic assistant.",
                    "template": "{base_system}\n\nContext:\n{context}\n\nQuery: {query}",
                    "tasks": {"review": {"instruction": "Answer the query.", "cot": ""}}
                }
        except Exception as e:
            self.logger.error(f"Error loading prompts: {e}")
            raise

    def _format_context(self, context_data: Union[List[str], List[Dict]]) -> str:
        """
        [基础版] 格式化上下文：将检索到的 chunk 或 paper 列表转为字符串
        """
        formatted_text = ""
        
        if not context_data:
            return "No valid context data provided."

        # 情况 A: 纯字符串列表 (Chunk 模式 fallback)
        if isinstance(context_data[0], str):
            for i, chunk in enumerate(context_data, 1):
                formatted_text += f"[Source {i}]: {chunk}\n\n"
                
        # 情况 B: 字典列表 (OpenAlex Paper 或 Vector Chunk)
        elif isinstance(context_data[0], dict):
            for i, item in enumerate(context_data, 1):
                # 基础信息
                title = item.get('title', 'Unknown Title')
                year = item.get('year', 'N/A')
                paper_id = item.get("paper_id") or item.get("id") or "N/A"
                
                # [增强] 引用数和概念
                cited = item.get('cited_by', 'N/A')
                concepts = item.get('concepts', [])
                if isinstance(concepts, list):
                    concepts_str = ", ".join(concepts[:3]) # 只取前3个概念
                else:
                    concepts_str = str(concepts)

                # 链接处理
                url = item.get('url') or item.get('pdf_url') or item.get('id', 'N/A')
                
                # 正文内容 (优先 content，其次 abstract)
                text_content = item.get('content') or item.get('abstract', 'No content available.')
                
                # 组装
                formatted_text += (
                    f"[Paper {i}]\n"
                    f"Paper ID: {paper_id}\n"
                    f"Title: {title}\n"
                    f"Year: {year} | Cited By: {cited}\n"
                    f"Concepts: {concepts_str}\n"
                    f"URL: {url}\n"
                    f"Content/Abstract: {text_content}\n"
                    f"{'-'*40}\n"
                )
        
        return formatted_text

    def _format_structured_review_context(
        self,
        papers_metadata: List[Dict[str, Any]],
        chunks: Union[List[str], List[Dict[str, Any]]],
    ) -> str:
        """
        [新增] 结构化综述上下文格式化
        利用 Louvain 算法将论文分簇，生成“流派->代表作->证据”的结构。
        """
        # 如果缺少依赖或配置关闭，降级为普通格式化
        if not detect_communities_louvain or not papers_metadata:
            return self._format_context(chunks)

        enable = getattr(settings, "ENABLE_STRUCTURED_REVIEW", True)
        if not enable:
            return self._format_context(chunks)

        try:
            # --- 1. 读取配置 (带默认值) ---
            resolution = float(getattr(settings, "LOUVAIN_RESOLUTION", 1.0))
            max_levels = int(getattr(settings, "LOUVAIN_MAX_LEVELS", 10))
            max_iter = int(getattr(settings, "LOUVAIN_MAX_ITER", 50))
            min_edges = int(getattr(settings, "LOUVAIN_MIN_EDGES", 2))
            top_papers_per_cluster = int(getattr(settings, "STRUCTURED_REVIEW_TOP_PAPERS_PER_CLUSTER", 3))
            top_chunks_per_paper = int(getattr(settings, "STRUCTURED_REVIEW_TOP_CHUNKS_PER_PAPER", 2))
            max_chunk_chars = int(getattr(settings, "STRUCTURED_REVIEW_MAX_CHUNK_CHARS", 450))

            # --- 2. 构建图 (Nodes & Edges) ---
            nodes = [p.get("id") for p in papers_metadata if p.get("id")]
            node_set = set(nodes)
            edges = []
            for p in papers_metadata:
                src = p.get("id")
                if not src or src not in node_set: continue
                # 仅保留在 Top-N 范围内的引用边
                for dst in (p.get("referenced_works") or []):
                    if dst in node_set and dst != src:
                        edges.append((src, dst))

            # --- 3. 执行 Louvain 社区发现 ---
            partition = detect_communities_louvain(
                nodes=nodes,
                edges=edges,
                resolution=resolution,
                max_levels=max_levels,
                max_iter=max_iter,
                min_edges=min_edges,
            )

            # 按社区 ID 分组
            comm_to_papers = {}
            for p in papers_metadata:
                pid = p.get("id")
                if not pid: continue
                cid = int(partition.get(pid, 0)) # 默认为社区 0
                comm_to_papers.setdefault(cid, []).append(p)

            # --- 4. 辅助数据准备 ---
            # 排序辅助函数：优先 Hybrid 分数，其次 Rerank 分数，最后引用数
            def _paper_rank_key(p):
                for k in ("_hybrid_score", "_rerank_score", "cited_by"):
                    v = p.get(k)
                    try: return float(v)
                    except: continue
                return 0.0

            # 将 chunk 按 paper_id 索引，方便查找证据
            chunks_by_pid = {}
            if chunks and isinstance(chunks, list) and isinstance(chunks[0], dict):
                for c in chunks:
                    pid = c.get("paper_id")
                    if pid: chunks_by_pid.setdefault(pid, []).append(c)

            # --- 5. 构建输出文本 ---
            out = []
            out.append("=== Structured Review Clusters (Community Detection) ===\n")

            # 按簇大小降序排列
            sorted_cids = sorted(comm_to_papers.keys(), key=lambda c: len(comm_to_papers[c]), reverse=True)

            for idx, cid in enumerate(sorted_cids, 1):
                papers = comm_to_papers[cid]

                # 提取 Top Concepts 作为簇的“弱标签”
                concept_freq = {}
                for p in papers:
                    cs = p.get("concepts") or []
                    if isinstance(cs, list):
                        for x in cs[:5]: concept_freq[x] = concept_freq.get(x, 0) + 1
                top_concepts = sorted(concept_freq.items(), key=lambda x: x[1], reverse=True)[:3]
                top_concept_str = ", ".join([c for c, _ in top_concepts]) if top_concepts else "General"

                out.append(f"[Cluster {idx}] size={len(papers)} | Keywords: {top_concept_str}\n")

                # 选出代表性论文
                papers_sorted = sorted(papers, key=_paper_rank_key, reverse=True)
                reps = papers_sorted[:max(1, top_papers_per_cluster)]
                
                out.append("Representative Papers & Evidence:\n")
                for rp in reps:
                    pid = rp.get("id", "N/A")
                    title = rp.get("title", "Unknown")
                    year = rp.get("year", "N/A")
                    score = rp.get("_hybrid_score", rp.get("_rerank_score", ""))
                    
                    out.append(f"- [PaperID: {pid}] {title} ({year}) | Score: {score}\n")

                    # 查找证据 (Chunks)
                    evidences = chunks_by_pid.get(pid, [])[:max(0, top_chunks_per_paper)]
                    if evidences:
                        for ev in evidences:
                            txt = (ev.get("content") or "").strip().replace("\n", " ")
                            if len(txt) > max_chunk_chars: txt = txt[:max_chunk_chars] + "..."
                            out.append(f"  - Evidence: {txt}\n")
                    else:
                        # Fallback 到 Abstract
                        abs_txt = (rp.get("abstract") or "").strip().replace("\n", " ")
                        if len(abs_txt) > max_chunk_chars: abs_txt = abs_txt[:max_chunk_chars] + "..."
                        out.append(f"  - Abstract: {abs_txt}\n")
                out.append("\n")

            # 保留原始 Chunks 以供 Reference Grounding
            out.append("=== Raw Retrieved Chunks (For Citation) ===\n")
            out.append(self._format_context(chunks))
            
            return "".join(out)

        except Exception as e:
            self.logger.warning(f"Structured review context failed, fallback. Error: {e}")
            return self._format_context(chunks)

    def generate(
        self, 
        user_query: str, 
        context_data: Union[List[str], List[Dict]], 
        task_type: str = "review", 
        language: str = "中文",
        papers_metadata: Optional[List[Dict[str, Any]]] = None # [新增参数]
    ) -> str:
        """
        生成回复 (普通模式)
        """
        self.logger.info(f"Generating response. Mode: {task_type}")
        
        if not context_data:
            return "未找到相关背景知识，无法生成回答。"

        # 1. 获取 Prompt 配置
        task_config = self.prompts["tasks"].get(task_type, self.prompts["tasks"]["review"])
        
        # 2. [合并逻辑] 格式化上下文
        # 如果是 review 模式且提供了 metadata，则使用结构化综述
        if task_type == "review" and papers_metadata:
            context_text = self._format_structured_review_context(papers_metadata, context_data)
        else:
            context_text = self._format_context(context_data)

        lang_instruction = f"\nIMPORTANT: You must output the final response in {language} language."

        # 3. 组装 Prompt
        try:
            full_prompt = self.prompts["template"].format(
                base_system=self.prompts["system_base"] + lang_instruction,
                instruction=task_config["instruction"],
                cot=task_config["cot"],
                context=context_text,
                query=user_query
            )
        except KeyError as e:
            self.logger.error(f"Prompt template missing key: {e}")
            return f"Error constructing prompt: {e}"

        # 4. 调用 LLM
        return self.llm.chat(self.prompts["system_base"], full_prompt)
    
    def generate_stream(
        self, 
        user_query: str, 
        context_data: Union[List[str], List[Dict]], 
        task_type: str = "review", 
        language: str = "中文",
        papers_metadata: Optional[List[Dict[str, Any]]] = None # [新增参数]
    ):
        """
        流式生成方法 (支持结构化综述)
        """
        self.logger.info(f"Generating stream response. Mode: {task_type}")
        
        if not context_data:
            yield "未找到相关背景知识，无法生成回答。"
            return

        # 1. 准备 Prompt (逻辑同 generate)
        task_config = self.prompts["tasks"].get(task_type, self.prompts["tasks"]["review"])
        
        # [合并逻辑] 上下文格式化
        if task_type == "review" and papers_metadata:
            context_text = self._format_structured_review_context(papers_metadata, context_data)
        else:
            context_text = self._format_context(context_data)
            
        lang_instruction = f"\nIMPORTANT: You must output the final response in {language} language."

        try:
            full_prompt = self.prompts["template"].format(
                base_system=self.prompts["system_base"] + lang_instruction,
                instruction=task_config["instruction"],
                cot=task_config["cot"],
                context=context_text,
                query=user_query
            )
        except KeyError as e:
            yield f"Error constructing prompt: {e}"
            return

        # 2. 调用 LLM 流式接口
        yield from self.llm.chat_stream(self.prompts["system_base"], full_prompt)