# ui/logic.py
import sys
import os
import streamlit as st

# 确保能导入 src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retrieval.openalex import OpenAlexRetriever
from src.retrieval.vector_store import LocalVectorStore
from src.graph.expansion import ConceptExpander
from src.core.generator import ReviewGenerator
from src.config import settings

# 使用 Streamlit 缓存机制，避免每次刷新都重新初始化模型
@st.cache_resource
def get_engine():
    retriever = OpenAlexRetriever()
    local_store = LocalVectorStore()
    expander = ConceptExpander()
    generator = ReviewGenerator()
    return retriever, local_store, expander, generator

def recursive_summarize(generator, current_summary, new_messages):
    """
    输入:
    - current_summary: 之前的摘要 (String)
    - new_messages: 尚未被总结的新对话 (List[Dict])
    
    输出:
    - new_summary: 更新后的摘要
    """
    if not new_messages:
        return current_summary

    # 将新对话格式化为文本
    new_dialogue = "\n".join([f"{m['role']}: {m['content']}" for m in new_messages])
    
    # 构造 Prompt：基于旧摘要 + 新增量 -> 更新摘要
    if current_summary:
        prompt = f"""
        You are a memory manager for a research assistant.
        
        Current Knowledge Summary:
        "{current_summary}"
        
        New Interaction to Integrate:
        {new_dialogue}
        
        Task: Update the summary to include key insights from the new interaction. 
        Keep it concise. Do not lose important previous context. No more than 200 words.
        
        Updated Summary:
        """
    else:
        # 如果是第一轮，没有旧摘要
        prompt = f"""
        Summarize the key research questions and findings from this conversation concisely:
        {new_dialogue}
        """

    # 调用 LLM 生成 (使用简单的 chat 接口，不带 RAG)
    try:
        new_summary = generator.llm.chat("You are a helpful summarizer.", prompt)
        return new_summary
    except Exception as e:
        print(f"Summary failed: {e}")
        return current_summary # 失败则返回旧的，防止丢失


def process_query(query, mode, use_graph, history_context_str):
    """
    注意：现在的 process_query 不再负责维护历史，它只负责回答当前问题。
    历史维护逻辑上移到 app.py 中。
    
    query: 当前问题
    history_context_str: 已经被 app.py 处理好的、包含摘要的上下文字符串
    """
    retriever, local_store, expander, generator = get_engine()
    
    # 1. 广度搜索
    concept_ids = None
    if use_graph:
        info = expander.expand_query(query)
        if info:
            concept_ids = [info['id']]
    
    # 2. OpenAlex
    papers = retriever.search(query, top_k=settings.RAG_DOWNLOAD_K, concept_ids=concept_ids)
    if not papers:
        # 即使没有新论文，也可以基于历史回答，所以不要直接 return
        pass 

    # 3. 入库
    if papers:
        local_store.add_papers(papers)
    
    # 4. 深度召回
    chunks = local_store.search(query, top_k=settings.RAG_RETRIEVAL_K)
    
    # 5. 拼接 Context
    # Prompt = 历史摘要及上下文 + 检索到的切片 + 当前问题
    augmented_query = f"""
    [Conversation History Context]:
    {history_context_str}
    
    [Current User Question]: 
    {query}
    """
    
    response = generator.generate(augmented_query, chunks, task_type=mode)
    
    return response, papers