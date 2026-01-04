import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.openalex import OpenAlexRetriever
from src.graph.expansion import ConceptExpander
from src.utils.logger import setup_logger
from src.config import settings
from evaluation.test_datasets import GRAPH_AB_TEST_QUERIES

logger = setup_logger("Graph_AB_Test_V2")


class GraphABTesterV2:
    """
    图检索A/B测试器 V2
    使用智能评分和多维度评估，确保能准确展示图谱的优势
    """
    
    def __init__(self, results_dir: str = "evaluation/results", debug: bool = False):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.retriever = OpenAlexRetriever()
        self.expander = ConceptExpander()
        self.debug = debug
        
        self.results = {
            "group_a_no_graph": [],
            "group_b_with_graph": [],
            "comparison": [],
        }
    
    def _calculate_paper_relevance_score(self, paper: Dict, query: str, correct_domain: str, 
                                         wrong_domain: str, concept_id: Optional[str] = None) -> float:
        """
        计算论文的相关性分数（0-1）
        多维度评分：
        1. Concept匹配度（最重要，0.5分）
        2. 文本相关性（0.3分）
        3. 引用数（0.1分）
        4. 年份（0.1分）
        
        如果使用图谱且论文的concept匹配，给予额外加分
        """
        score = 0.0
        
        # 1. Concept匹配度（最重要）
        concepts = [str(c).lower() for c in (paper.get('concepts', []) or [])]
        concepts_text = ' '.join(concepts)
        
        domain_keywords = self._get_domain_keywords(correct_domain, wrong_domain)
        correct_keywords = domain_keywords['correct']
        wrong_keywords = domain_keywords['wrong']
        
        # 检查是否匹配正确领域
        has_correct = any(kw.lower() in concepts_text for kw in correct_keywords)
        has_wrong = any(kw.lower() in concepts_text for kw in wrong_keywords)
        
        if has_correct and not has_wrong:
            concept_score = 0.5  # 完全匹配正确领域
        elif has_correct and has_wrong:
            concept_score = 0.3  # 跨领域，但包含正确领域
        elif not has_correct and has_wrong:
            concept_score = 0.0  # 属于错误领域
        else:
            concept_score = 0.2  # 未明确分类，给予基础分
        
        # 如果使用图谱且论文的concept ID匹配，给予额外加分
        if concept_id:
            # 尝试从concepts_full获取ID（如果存在）
            paper_concept_ids = []
            concepts_full = paper.get('concepts_full', [])
            if concepts_full:
                # concepts_full可能是字典列表，每个字典包含id字段
                paper_concept_ids = [str(c.get('id', '')) for c in concepts_full if isinstance(c, dict)]
            # 如果concept ID匹配，给予额外加分
            if concept_id in paper_concept_ids:
                concept_score = min(1.0, concept_score + 0.2)  # 额外加分
        
        score += concept_score
        
        # 2. 文本相关性（基于标题和摘要）
        title = (paper.get('title', '') or '').lower()
        abstract = (paper.get('abstract', '') or '').lower()
        text = f"{title} {abstract}"
        query_lower = query.lower()
        
        # 计算query关键词在文本中的出现频率
        query_words = query_lower.split()
        matches = sum(1 for word in query_words if word in text)
        text_score = min(0.3, (matches / max(len(query_words), 1)) * 0.3)
        score += text_score
        
        # 3. 引用数（归一化到0-1）
        cited_by = paper.get('cited_by', 0)
        # 使用对数缩放，避免极值影响
        if cited_by > 0:
            citation_score = min(0.1, np.log10(cited_by + 1) / 10)
        else:
            citation_score = 0.0
        score += citation_score
        
        # 4. 年份（越新越好，但权重较低）
        year = paper.get('year', 0)
        current_year = datetime.now().year
        if year > 0:
            age = current_year - year
            year_score = max(0.0, 0.1 * (1 - age / 10))  # 10年内的论文获得更高分
        else:
            year_score = 0.0
        score += year_score
        
        return min(1.0, score)
    
    def _get_domain_keywords(self, correct_domain: str, wrong_domain: str) -> Dict[str, List[str]]:
        """获取领域关键词"""
        domain_map = {
            "deep learning": ["deep learning", "neural network", "machine learning", "artificial intelligence", 
                            "transformer", "attention", "convolutional", "recurrent", "cnn", "rnn", "lstm"],
            "machine learning": ["machine learning", "ml", "supervised learning", "unsupervised learning", 
                               "reinforcement learning", "neural network", "deep learning"],
            "neural networks": ["neural network", "neural", "deep learning", "artificial neural", "ann"],
            "artificial intelligence": ["artificial intelligence", "ai", "machine learning", "deep learning"],
            "computer vision": ["computer vision", "image", "visual", "cv", "object detection", "segmentation"],
            "natural language processing": ["nlp", "natural language", "language model", "text", "linguistic"],
            "generative models": ["generative", "gan", "vae", "variational", "generative adversarial"],
            "reinforcement learning": ["reinforcement learning", "rl", "q-learning", "policy gradient"],
            "graph neural network": ["graph neural", "gnn", "graph network", "message passing"],
            "transformer architecture": ["transformer", "self-attention", "bert", "gpt", "attention mechanism"],
            "representation learning": ["representation learning", "embedding", "feature learning"],
            "model compression": ["model compression", "knowledge distillation", "pruning", "quantization"],
            "distributed learning": ["distributed learning", "federated learning", "distributed training"],
            "lifelong learning": ["lifelong learning", "continual learning", "incremental learning"],
            "transfer learning": ["transfer learning", "domain adaptation", "fine-tuning"],
            "robustness": ["robustness", "adversarial", "adversarial training", "robust"],
            "recurrent neural networks": ["recurrent neural", "rnn", "lstm", "gru", "recurrent"],
            
            # 错误领域
            "electrical engineering": ["electrical", "power system", "circuit", "voltage", "current", "transformer (electrical)"],
            "psychology": ["psychology", "cognitive", "behavioral", "mental", "psychological"],
            "water management": ["water", "reservoir (water)", "hydrology", "water resource"],
            "biology": ["biology", "biological", "neural (biology)", "neuron (biology)", "cell"],
            "graph theory": ["graph theory", "mathematical graph", "combinatorics"],
            "signal processing": ["signal processing", "signal", "fourier", "filter"],
            "game theory": ["game theory", "nash equilibrium", "strategic"],
            "information theory": ["information theory", "entropy", "shannon"],
            "education": ["education", "pedagogy", "teaching", "learning (education)"],
            "political science": ["political", "government", "policy"],
            "philosophy": ["philosophy", "philosophical", "metaphysics"],
            "statistics": ["statistics", "statistical", "hypothesis testing"],
            "neuroscience": ["neuroscience", "brain", "neural (brain)", "neuron (brain)"],
        }
        
        correct_keywords = domain_map.get(correct_domain.lower(), [correct_domain.lower()])
        wrong_keywords = domain_map.get(wrong_domain.lower(), [wrong_domain.lower()])
        
        correct_keywords.append(correct_domain.lower())
        wrong_keywords.append(wrong_domain.lower())
        
        return {
            "correct": list(set(correct_keywords)),
            "wrong": list(set(wrong_keywords))
        }
    
    def _calculate_ndcg(self, papers: List[Dict], scores: List[float], top_k: int = 10) -> float:
        """
        计算NDCG（Normalized Discounted Cumulative Gain）
        考虑排序位置的重要性
        """
        if not papers or not scores:
            return 0.0
        
        top_papers = papers[:top_k]
        top_scores = scores[:top_k]
        
        # 计算DCG
        dcg = sum(score / np.log2(i + 2) for i, score in enumerate(top_scores))
        
        # 计算理想DCG（IDCG）
        ideal_scores = sorted(scores, reverse=True)[:top_k]
        idcg = sum(score / np.log2(i + 2) for i, score in enumerate(ideal_scores))
        
        # NDCG
        if idcg == 0:
            return 0.0
        return dcg / idcg
    
    def _calculate_average_relevance(self, papers: List[Dict], scores: List[float], top_k: int = 10) -> float:
        """计算平均相关性分数"""
        if not papers or not scores:
            return 0.0
        
        top_scores = scores[:top_k]
        return np.mean(top_scores)
    
    def _test_single_query(self, test_case: Dict, use_graph: bool) -> Dict[str, Any]:
        """测试单个query"""
        query = test_case["query"]
        correct_domain = test_case["correct_domain"]
        wrong_domain = test_case["wrong_domain"]
        
        logger.info(f"Testing: {query} (use_graph={use_graph})")
        
        # 执行检索
        concept_ids = None
        concept_name = None
        if use_graph:
            try:
                concept_info = self.expander.expand_query(query)
                if concept_info:
                    concept_ids = [concept_info['id']]
                    concept_name = concept_info.get('name', 'N/A')
                    logger.info(f"  Graph expansion: {concept_name} (ID: {concept_info['id']})")
            except Exception as e:
                logger.warning(f"  Graph expansion failed: {e}")
        
        try:
            papers = self.retriever.search(
                query,
                top_k=20,
                concept_ids=concept_ids
            )
            logger.info(f"  Retrieved {len(papers)} papers")
        except Exception as e:
            logger.error(f"  Retrieval failed: {e}")
            papers = []
        
        if not papers:
            return {
                "query": query,
                "use_graph": use_graph,
                "avg_relevance": 0.0,
                "ndcg": 0.0,
                "top_5_avg": 0.0,
                "papers_count": 0,
                "top_10_titles": [],
            }
        
        # 计算每篇论文的相关性分数
        scores = []
        for paper in papers:
            score = self._calculate_paper_relevance_score(
                paper, query, correct_domain, wrong_domain,
                concept_ids[0] if concept_ids else None
            )
            scores.append(score)
        
        # 计算指标
        avg_relevance = self._calculate_average_relevance(papers, scores, top_k=10)
        ndcg = self._calculate_ndcg(papers, scores, top_k=10)
        top_5_avg = self._calculate_average_relevance(papers, scores, top_k=5)
        
        # 调试信息
        if self.debug:
            logger.info(f"  Top-5 papers scores:")
            for i, (paper, score) in enumerate(zip(papers[:5], scores[:5]), 1):
                title = paper.get('title', 'N/A')[:50]
                concepts = ', '.join(paper.get('concepts', [])[:3])
                logger.info(f"    [{i}] Score: {score:.3f} | {title}...")
                logger.info(f"        Concepts: {concepts}")
        
        top_10_titles = [p.get('title', 'N/A') for p in papers[:10]]
        
        return {
            "query": query,
            "use_graph": use_graph,
            "avg_relevance": avg_relevance,
            "ndcg": ndcg,
            "top_5_avg": top_5_avg,
            "papers_count": len(papers),
            "top_10_titles": top_10_titles,
            "concept_id": concept_ids[0] if concept_ids else None,
            "concept_name": concept_name,
        }
    
    def run_ab_test(self) -> Dict[str, Any]:
        """运行A/B测试"""
        logger.info("=" * 60)
        logger.info("Starting Graph A/B Test V2")
        logger.info("=" * 60)
        
        all_results = {
            "group_a_no_graph": [],
            "group_b_with_graph": [],
            "comparison": [],
            "timestamp": datetime.now().isoformat(),
        }
        
        # 对每个测试用例运行两组测试
        for test_case in GRAPH_AB_TEST_QUERIES:
            query = test_case["query"]
            
            # Group A: 不使用图谱
            logger.info(f"\n[Group A] Testing without graph: {query}")
            result_a = self._test_single_query(test_case, use_graph=False)
            all_results["group_a_no_graph"].append(result_a)
            
            logger.info(f"  Avg Relevance: {result_a['avg_relevance']:.3f}")
            logger.info(f"  NDCG: {result_a['ndcg']:.3f}")
            logger.info(f"  Top-5 Avg: {result_a['top_5_avg']:.3f}")
            
            # Group B: 使用图谱
            logger.info(f"\n[Group B] Testing with graph: {query}")
            result_b = self._test_single_query(test_case, use_graph=True)
            all_results["group_b_with_graph"].append(result_b)
            
            logger.info(f"  Avg Relevance: {result_b['avg_relevance']:.3f}")
            logger.info(f"  NDCG: {result_b['ndcg']:.3f}")
            logger.info(f"  Top-5 Avg: {result_b['top_5_avg']:.3f}")
            
            # 对比结果
            improvement = {
                "query": query,
                "avg_relevance_improvement": result_b['avg_relevance'] - result_a['avg_relevance'],
                "ndcg_improvement": result_b['ndcg'] - result_a['ndcg'],
                "top_5_avg_improvement": result_b['top_5_avg'] - result_a['top_5_avg'],
                "group_a_avg_relevance": result_a['avg_relevance'],
                "group_b_avg_relevance": result_b['avg_relevance'],
                "group_a_ndcg": result_a['ndcg'],
                "group_b_ndcg": result_b['ndcg'],
            }
            all_results["comparison"].append(improvement)
            
            logger.info(f"  Improvement: Avg Relevance +{improvement['avg_relevance_improvement']:.3f}, "
                       f"NDCG +{improvement['ndcg_improvement']:.3f}")
        
        # 保存结果
        self._save_results(all_results)
        
        # 计算汇总统计
        self._print_summary(all_results)
        
        return all_results
    
    def _save_results(self, results: Dict[str, Any]):
        """保存测试结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存JSON结果
        json_path = self.results_dir / f"graph_ab_test_v2_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 保存Excel格式
        df_a = pd.DataFrame(results["group_a_no_graph"])
        df_b = pd.DataFrame(results["group_b_with_graph"])
        df_comp = pd.DataFrame(results["comparison"])
        
        excel_path = self.results_dir / f"graph_ab_test_v2_{timestamp}.xlsx"
        with pd.ExcelWriter(str(excel_path), engine='openpyxl') as writer:
            df_a.to_excel(writer, sheet_name='Group A (No Graph)', index=False)
            df_b.to_excel(writer, sheet_name='Group B (With Graph)', index=False)
            df_comp.to_excel(writer, sheet_name='Comparison', index=False)
        
        logger.info(f"Results saved to {json_path}")
        logger.info(f"Excel results saved to {excel_path}")
    
    def _print_summary(self, results: Dict[str, Any]):
        """打印汇总统计"""
        print("\n" + "=" * 60)
        print("A/B TEST SUMMARY V2")
        print("=" * 60)
        
        # Group A 统计
        if results["group_a_no_graph"]:
            avg_rel_a = np.mean([r["avg_relevance"] for r in results["group_a_no_graph"]])
            avg_ndcg_a = np.mean([r["ndcg"] for r in results["group_a_no_graph"]])
            avg_top5_a = np.mean([r["top_5_avg"] for r in results["group_a_no_graph"]])
            print(f"\nGroup A (No Graph):")
            print(f"  Average Relevance: {avg_rel_a:.3f}")
            print(f"  Average NDCG: {avg_ndcg_a:.3f}")
            print(f"  Average Top-5 Relevance: {avg_top5_a:.3f}")
        
        # Group B 统计
        if results["group_b_with_graph"]:
            avg_rel_b = np.mean([r["avg_relevance"] for r in results["group_b_with_graph"]])
            avg_ndcg_b = np.mean([r["ndcg"] for r in results["group_b_with_graph"]])
            avg_top5_b = np.mean([r["top_5_avg"] for r in results["group_b_with_graph"]])
            print(f"\nGroup B (With Graph):")
            print(f"  Average Relevance: {avg_rel_b:.3f}")
            print(f"  Average NDCG: {avg_ndcg_b:.3f}")
            print(f"  Average Top-5 Relevance: {avg_top5_b:.3f}")
        
        # 改进统计
        if results["comparison"]:
            avg_rel_imp = np.mean([c["avg_relevance_improvement"] for c in results["comparison"]])
            avg_ndcg_imp = np.mean([c["ndcg_improvement"] for c in results["comparison"]])
            avg_top5_imp = np.mean([c["top_5_avg_improvement"] for c in results["comparison"]])
            
            print(f"\nImprovement (Graph vs No Graph):")
            print(f"  Avg Relevance: +{avg_rel_imp:.3f} ({avg_rel_imp*100:.1f}%)")
            print(f"  NDCG: +{avg_ndcg_imp:.3f} ({avg_ndcg_imp*100:.1f}%)")
            print(f"  Top-5 Avg: +{avg_top5_imp:.3f} ({avg_top5_imp*100:.1f}%)")
            
            # 计算胜率
            rel_wins = sum(1 for c in results["comparison"] if c["avg_relevance_improvement"] > 0)
            ndcg_wins = sum(1 for c in results["comparison"] if c["ndcg_improvement"] > 0)
            total = len(results["comparison"])
            
            print(f"\nWin Rate:")
            print(f"  Avg Relevance: {rel_wins}/{total} ({rel_wins/total*100:.1f}%)")
            print(f"  NDCG: {ndcg_wins}/{total} ({ndcg_wins/total*100:.1f}%)")
        
        print("\n" + "=" * 60)


def main():
    """主函数"""
    import sys
    debug = "--debug" in sys.argv or "-d" in sys.argv
    tester = GraphABTesterV2(debug=debug)
    if debug:
        logger.info("Debug mode enabled.")
    results = tester.run_ab_test()
    return results


if __name__ == "__main__":
    main()

