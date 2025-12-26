import argparse
import os
import sys # 新增
from src.retrieval.openalex import OpenAlexRetriever
from src.graph.expansion import ConceptExpander
from src.retrieval.vector_store import LocalVectorStore
from src.core.generator import ReviewGenerator
from src.utils.logger import setup_logger
from src.config import settings

def main():
    # 0. 参数解析
    parser = argparse.ArgumentParser(description="ScholarRAG: Graph-enhanced Research Assistant")
    parser.add_argument("--query", type=str, required=True, help="Research topic/question")
    parser.add_argument("--use_graph", action="store_true", help="Enable Knowledge Graph expansion")
    parser.add_argument("--mode", type=str, default="review", choices=["review", "explain", "inspire"], 
                        help="Mode: review(综述), explain(解释), inspire(启发)")
    args = parser.parse_args()

    logger = setup_logger("Main")
    
    # [队友功能] 更健壮的 Key 检查 (支持 OpenRouter)
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")):
        logger.error("❌ Missing API Key: Please set OPENAI_API_KEY or OPENROUTER_API_KEY in .env")
        sys.exit(1)

    # 1. 初始化模块
    retriever = OpenAlexRetriever()
    local_store = LocalVectorStore() # 你的核心优势：本地向量库
    expander = ConceptExpander()
    generator = ReviewGenerator()

    # ------------------------------------------------------------------
    # Stage 1: Broad Search (广度检索)
    # ------------------------------------------------------------------
    logger.info(">>> Stage 1: Broad Search (OpenAlex)")
    
    # 1.1 图扩展逻辑
    concept_ids = None
    if args.use_graph:
        concept_info = expander.expand_query(args.query)
        if concept_info:
            logger.info(f"Graph Expansion applied. Focused on Concept ID: {concept_info['id']}")
            concept_ids = [concept_info['id']]

    # 1.2 OpenAlex 检索 (使用 .env 配置)
    logger.info(f"Fetching top {settings.RAG_DOWNLOAD_K} papers metadata...")
    papers_metadata = retriever.search(
        args.query, 
        top_k=settings.RAG_DOWNLOAD_K, 
        concept_ids=concept_ids
    )
    
    if not papers_metadata:
        logger.warning("No papers found. Exiting.")
        return

    # [队友功能] 打印预览列表 (提升用户体验)
    print(f"\n[Info] Found {len(papers_metadata)} papers. Top 5 previews:")
    for p in papers_metadata[:5]:
        print(f"  - [{p['year']}] {p['title'][:60]}... (Cited: {p['cited_by']})")
    print("-" * 50)

    # ------------------------------------------------------------------
    # Stage 2: Deep Processing (深度处理与入库)
    # ------------------------------------------------------------------
    logger.info(">>> Stage 2: Deep Processing (Downloading & Vectorizing)")
    
    # 下载并存入 ChromaDB (自动去重)
    local_store.add_papers(papers_metadata)

    # ------------------------------------------------------------------
    # Stage 3: Semantic Retrieval (语义召回)
    # ------------------------------------------------------------------
    logger.info(">>> Stage 3: Deep Retrieval from Vector Store")
    
    # 从全文中召回最相关的片段
    detail_chunks = local_store.search(args.query, top_k=settings.RAG_RETRIEVAL_K)
    
    if not detail_chunks:
        logger.warning("No detailed chunks retrieved from vector store. Falling back to abstracts.")
        # 兜底：如果向量检索失败，直接使用 abstract (队友的逻辑)
        detail_chunks = papers_metadata[:10]

    logger.info(f"Retrieved {len(detail_chunks)} semantic chunks for generation.")

    # ------------------------------------------------------------------
    # Stage 4: Agentic Generation (生成)
    # ------------------------------------------------------------------
    logger.info(f">>> Stage 4: Generation (Mode: {args.mode})")
    
    print("\n" + "="*50)
    print("Generating Response... Please wait.")
    print("="*50 + "\n")

    response = generator.generate(
        user_query=args.query, 
        context_data=detail_chunks, 
        task_type=args.mode
    )

    # 5. 输出
    print("\n" + "#"*20 + " FINAL RESPONSE " + "#"*20)
    print(response)
    print("#"*60)

if __name__ == "__main__":
    main()