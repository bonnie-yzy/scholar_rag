import argparse
import os
from src.retrieval.openalex import OpenAlexRetriever
from src.graph.expansion import ConceptExpander
from src.core.generator import ReviewGenerator
from src.utils.logger import setup_logger

def main():
    # 0. 参数解析
    parser = argparse.ArgumentParser(description="ScholarRAG: Graph-enhanced Research Assistant")
    parser.add_argument("--query", type=str, required=True, help="Research topic/question")
    parser.add_argument("--use_graph", action="store_true", help="Enable Knowledge Graph expansion")
    parser.add_argument("--top_k", type=int, default=5, help="Top-K papers to retrieve (default: use RAG_TOP_K from .env)")
    args = parser.parse_args()

    logger = setup_logger("Main")
    
    # 检查 API Key
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")):
        logger.error("Please set OPENAI_API_KEY or OPENROUTER_API_KEY in .env file")
        return

    # 1. 初始化模块
    retriever = OpenAlexRetriever()
    expander = ConceptExpander()
    generator = ReviewGenerator()

    # 2. 图扩展 (Query Expansion)
    concept_ids = None
    if args.use_graph:
        concept_info = expander.expand_query(args.query)
        if concept_info:
            logger.info(f"Graph Expansion applied. Focused on Concept ID: {concept_info['id']}")
            concept_ids = [concept_info['id']] # 可以在这里加入 related concepts

    # 3. 检索 (Retrieval)
    papers = retriever.search(
        args.query,
        top_k=args.top_k,
        concept_ids=concept_ids,
    )
    
    if not papers:
        logger.warning("No papers found. Exiting.")
        return

    # 打印一下检索到的标题供用户确认
    print("\n" + "="*50)
    print(f"Retrieved {len(papers)} Papers:")
    for p in papers:
        print(f"- [{p['year']}] {p['title']} (Cited: {p['cited_by']})")
    print("="*50 + "\n")

    # 4. 生成 (Generation)
    print("Generating Review... Please wait.\n")
    review = generator.generate(args.query, papers)

    # 5. 输出
    print("\n" + "#"*20 + " RESEARCH REVIEW " + "#"*20)
    print(review)
    print("#"*60)

if __name__ == "__main__":
    main()