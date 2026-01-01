# ui/logic.py
import sys
import os
import streamlit as st
import json # 确保导入 json

# 确保能导入 src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retrieval.openalex import OpenAlexRetriever
from src.retrieval.vector_store import LocalVectorStore
from src.graph.expansion import ConceptExpander
# 确保导入合并后的 Generator (支持 papers_metadata 参数)
from src.core.generator import ReviewGenerator
from src.config import settings

# 使用 Streamlit 缓存机制，避免每次刷新都重新初始化模型
@st.cache_resource(show_spinner=False)
def get_engine():
    retriever = OpenAlexRetriever()
    local_store = LocalVectorStore()
    expander = ConceptExpander()
    generator = ReviewGenerator()
    return retriever, local_store, expander, generator

def recursive_summarize(generator, current_summary, new_messages):
    """
    递归摘要更新逻辑 (保持不变)
    """
    if not new_messages:
        return current_summary

    new_dialogue = "\n".join([f"{m['role']}: {m['content']}" for m in new_messages])
    
    if current_summary:
        prompt = f"""
        You are a memory manager for a research assistant.
        Current Knowledge Summary: "{current_summary}"
        New Interaction to Integrate: {new_dialogue}
        Task: Update the summary to include key insights from the new interaction. 
        Keep it concise. Do not lose important previous context. No more than 200 words.
        Updated Summary:
        """
    else:
        prompt = f"""
        Summarize the key research questions and findings from this conversation concisely:
        {new_dialogue}
        """

    try:
        new_summary = generator.llm.chat("You are a helpful summarizer.", prompt)
        return new_summary
    except Exception as e:
        print(f"Summary failed: {e}")
        return current_summary

def generate_viral_copy(conversation_text):
    """
    生成社交媒体文案 (保持不变)
    """
    _, _, _, generator = get_engine()
    
    prompt = f"""
    你是一个不仅懂学术，还深谙社交媒体传播规律的“学术博主”。
    请将以下对话内容总结为一段适合发布在“灵感广场”的短文案。
    
    [对话内容]
    {conversation_text}
    
    [要求]
    1. **极简主义**：总字数严格控制在 150 字以内。
    2. **金句化**：第一句必须是抓人眼球的 Insight 或反直觉结论。
    3. **结构化**：采用 "💡核心观点 + 📌三个关键点" 的格式。
    4. **风格**：使用 Emoji 增加可读性，语气轻松但有深度。
    5. **输出**：直接输出文案内容，不要包含"好的"、"文案如下"等废话。
    """
    
    try:
        viral_copy = generator.llm.chat("You are a creative social media editor.", prompt)
        return viral_copy
    except Exception as e:
        return "💡 灵感摘要生成失败，请手动编辑。"

def perform_retrieval(query, use_graph, history_context_str):
    """
    [阶段一] 执行检索与知识库构建
    返回: (context_chunks, source_papers, log_messages)
    """
    retriever, local_store, expander, generator = get_engine()
    logs = [] 
    
    # 1. 广度搜索 (Graph Expansion)
    concept_ids = None
    if use_graph:
        logs.append("正在构建概念图谱 (Graph Expansion)...")
        info = expander.expand_query(query)
        if info:
            concept_ids = [info['id']]
            logs.append(f"识别核心概念: {info['name']}")
    
    # 2. OpenAlex 检索
    logs.append("正在 OpenAlex 检索权威文献...")
    # 注意：这里 retrieve 的 papers 已经包含了 teammate 的 PageRank 逻辑
    papers = retriever.search(query, top_k=settings.RAG_DOWNLOAD_K, concept_ids=concept_ids)
    
    if papers:
        logs.append(f"检索到 {len(papers)} 篇相关文献，准备阅读...")
        # 3. 入库 (耗时操作)
        local_store.add_papers(papers)
    else:
        logs.append("未检索到新文献，将基于现有知识库回答。")

    # 4. 深度召回
    logs.append("正在进行向量重排与上下文构建...")
    chunks = local_store.search(query, top_k=settings.RAG_RETRIEVAL_K)
    
    return chunks, papers, logs

# [关键修改] 增加了 papers_metadata 参数
def get_response_stream(query, mode, history_context_str, chunks, language="Chinese", papers_metadata=None):
    """
    [阶段二] 生成流式回答
    返回: generator (yield string)
    """
    _, _, _, generator = get_engine()
    
    # 5. 拼接 Context
    augmented_query = f"""
    [Conversation History Context]:
    {history_context_str}
    
    [Current User Question]: 
    {query}
    """
    
    # 6. 调用 Generator 的流式接口
    # [关键] 只有在 review 模式下才传入 papers_metadata，用于结构化综述
    meta_to_pass = papers_metadata if mode == "review" else None
    
    return generator.generate_stream(
        augmented_query, 
        chunks, 
        task_type=mode, 
        language=language,
        papers_metadata=meta_to_pass # 透传给 generator
    )

def generate_follow_up_questions(history_context_str):
    """
    生成后续追问 (保持不变)
    """
    _, _, _, generator = get_engine()
    
    prompt = f"""
    You are a helpful research assistant. Based on the conversation history below, suggest 3 short, relevant academic follow-up questions that the user might want to ask next.
    
    [Conversation History]
    {history_context_str}
    
    [Requirements]
    1. Output strictly a JSON list of strings. Example: ["Question 1?", "Question 2?", "Question 3?"]
    2. Questions should be concise (under 20 words).
    3. Language: Match the language of the conversation (Chinese/English).
    4. Focus on digging deeper, clarifying concepts, or exploring related fields.
    """
    
    try:
        response = generator.llm.chat("You are a follow-up question generator.", prompt)
        clean_text = response.replace("```json", "").replace("```", "").strip()
        questions = json.loads(clean_text)
        
        if isinstance(questions, list):
            return questions[:3]
        return []
    except Exception as e:
        print(f"Follow-up generation failed: {e}")
        return []