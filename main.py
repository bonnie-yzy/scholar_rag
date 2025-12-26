import argparse
import os
from src.retrieval.openalex import OpenAlexRetriever
from src.graph.expansion import ConceptExpander
from src.retrieval.vector_store import LocalVectorStore
from src.core.generator import ReviewGenerator
from src.utils.logger import setup_logger
from src.config import settings  # <--- 引入配置模块读取 .env

def main():
    # 0. 参数解析
    parser = argparse.ArgumentParser(description="ScholarRAG: Graph-enhanced Research Assistant")
    parser.add_argument("--query", type=str, required=True, help="Research topic/question")
    parser.add_argument("--use_graph", action="store_true", help="Enable Knowledge Graph expansion")
    parser.add_argument("--mode", type=str, default="review", choices=["review", "explain", "inspire"], 
                        help="Mode: review(综述), explain(解释现象), inspire(跨界启发)")
    args = parser.parse_args()

    logger = setup_logger("Main")
    
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("Please set OPENAI_API_KEY in .env file")
        return

    # 1. 初始化模块
    retriever = OpenAlexRetriever()
    local_store = LocalVectorStore()
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

    # 1.2 OpenAlex 检索
    # [关键修改] 使用 .env 中的 RAG_DOWNLOAD_K (例如 20)，扩大下载范围
    logger.info(f"Fetching top {settings.RAG_DOWNLOAD_K} papers metadata...")
    papers_metadata = retriever.search(
        args.query, 
        top_k=settings.RAG_DOWNLOAD_K, 
        concept_ids=concept_ids
    )
    
    if not papers_metadata:
        logger.warning("No papers found. Exiting.")
        return

    print(f"\n[Info] Found {len(papers_metadata)} papers, preparing to download/vectorize...\n")

    # ------------------------------------------------------------------
    # Stage 2: Deep Processing (深度处理与入库)
    # ------------------------------------------------------------------
    logger.info(">>> Stage 2: Deep Processing (Downloading & Vectorizing)")
    
    # 将这 Top-N 篇论文下载并存入 ChromaDB
    # 注意：vector_store 内部有去重逻辑，已经存在的不会重复下载
    local_store.add_papers(papers_metadata)

    # ------------------------------------------------------------------
    # Stage 3: Semantic Retrieval (语义召回)
    # ------------------------------------------------------------------
    logger.info(">>> Stage 3: Deep Retrieval from Vector Store")
    
    # [关键修改] 使用 .env 中的 RAG_RETRIEVAL_K (例如 15)
    # 这一步是从所有已下载的全文中，找出最符合 query 的 15 个具体片段
    detail_chunks = local_store.search(args.query, top_k=settings.RAG_RETRIEVAL_K)
    
    logger.info(f"Retrieved {len(detail_chunks)} semantic chunks from local knowledge base.")

    # ------------------------------------------------------------------
    # Stage 4: Agentic Generation (生成)
    # ------------------------------------------------------------------
    logger.info(f">>> Stage 4: Generation (Mode: {args.mode})")
    
    print("\n" + "="*50)
    print("Generating Response... Please wait.")
    print("="*50 + "\n")

    # generator 会根据 args.mode 自动选择 prompts.json 中的模板
    response = generator.generate(
        user_query=args.query, 
        context_data=detail_chunks, # 传入的是切片列表
        task_type=args.mode
    )

    # 5. 输出
    print("\n" + "#"*20 + " FINAL RESPONSE " + "#"*20)
    print(response)
    print("#"*60)

if __name__ == "__main__":
    main()