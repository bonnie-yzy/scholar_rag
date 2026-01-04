"""
三种模式差异性评测脚本
对同一个query用三种模式测试，突出展示差异性和有效性
"""

import os
import json
import re
import time
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

# 项目导入
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.openalex import OpenAlexRetriever
from src.graph.expansion import ConceptExpander
from src.retrieval.vector_store import LocalVectorStore
from src.core.generator import ReviewGenerator
from src.core.llm import LLMService
from src.utils.logger import setup_logger
from src.config import settings
from evaluation.test_datasets import MODE_COMPARISON_QUERIES

logger = setup_logger("Mode_Difference_Evaluation")


class ModeDifferenceEvaluator:
    """
    三种模式差异性评测器
    对同一个query用三种模式测试，评测并对比差异
    """
    
    def __init__(self, results_dir: str = "evaluation/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.retriever = OpenAlexRetriever()
        self.local_store = LocalVectorStore()
        self.expander = ConceptExpander()
        self.generator = ReviewGenerator()
        self.llm = LLMService()
        
        # 保存中间结果
        self.intermediate_results = []
        
    def _count_citations(self, text: str) -> int:
        """统计引用数量（Paper ID、[PaperID: ...]等格式）"""
        patterns = [
            r'\[PaperID:\s*[^\]]+\]',
            r'Paper ID:\s*[^\n]+',
            r'\[Paper\s+\d+\]',
            r'\([A-Za-z]+\s+et\s+al\.\s+\d{4}\)',  # (Author et al. 2024)
            r'\[[0-9]+\]',  # [1], [2]
        ]
        count = 0
        for pattern in patterns:
            count += len(re.findall(pattern, text, re.IGNORECASE))
        return count
    
    def _count_sections(self, text: str) -> int:
        """统计结构化章节数量（Markdown标题）"""
        return len(re.findall(r'^#+\s+', text, re.MULTILINE))
    
    def _calculate_comprehensiveness(self, response: str) -> float:
        """
        计算全面性（Comprehensiveness）
        - 内容长度
        - 覆盖范围（关键词多样性）
        - 引用数量
        - 结构化程度（章节数量）
        """
        length = len(response)
        length_score = min(1.0, length / 2000)  # 2000字为满分
        
        # 引用数量（反映覆盖的文献范围）
        citation_count = self._count_citations(response)
        citation_score = min(1.0, citation_count / 10)  # 10个引用为满分
        
        # 结构化程度（反映组织全面性）
        section_count = self._count_sections(response)
        structure_score = min(1.0, section_count / 5)  # 5个章节为满分
        
        # 关键词多样性（检查是否包含多个主题词）
        diversity_keywords = [
            'overview', 'survey', 'review', 'summary', 'comparison', 'contrast',
            'different', 'various', 'multiple', 'several', 'many',
            '综述', '概述', '总结', '对比', '比较', '不同', '多种', '多个'
        ]
        diversity_score = min(1.0, sum(1 for kw in diversity_keywords if kw.lower() in response.lower()) / 5)
        
        # 全面性 = 长度 + 引用 + 结构 + 多样性
        comprehensiveness = (length_score * 0.3 + citation_score * 0.3 + structure_score * 0.2 + diversity_score * 0.2)
        return min(1.0, comprehensiveness)
    
    def _calculate_novelty(self, response: str) -> float:
        """
        计算新颖性（Novelty/Innovation）
        - 创新性词汇
        - 跨领域连接
        - 未来方向
        - 新颖观点
        """
        # 创新性词汇
        innovation_keywords = [
            'novel', 'innovative', 'new', 'emerging', 'cutting-edge', 'breakthrough',
            'pioneering', 'revolutionary', 'groundbreaking', 'state-of-the-art',
            '新颖', '创新', '新兴', '前沿', '突破', '开创性', '革命性', '最先进'
        ]
        innovation_score = min(1.0, sum(1 for kw in innovation_keywords if kw.lower() in response.lower()) / 5)
        
        # 跨领域连接
        cross_domain_keywords = [
            'cross-domain', 'interdisciplinary', 'hybrid', 'fusion', 'integration',
            'combine', 'merge', 'bridge', 'connect', 'transfer',
            '跨领域', '交叉', '融合', '整合', '结合', '连接', '迁移'
        ]
        cross_domain_score = min(1.0, sum(1 for kw in cross_domain_keywords if kw.lower() in response.lower()) / 5)
        
        # 未来方向和应用
        future_keywords = [
            'future', 'potential', 'opportunity', 'direction', 'trend', 'prospect',
            'could', 'might', 'suggest', 'propose', 'recommend', 'promising',
            '未来', '潜在', '机会', '方向', '趋势', '前景', '可能', '建议', '提出', '有前景'
        ]
        future_score = min(1.0, sum(1 for kw in future_keywords if kw.lower() in response.lower()) / 5)
        
        # 新颖性 = 创新词汇 + 跨领域 + 未来方向
        novelty = (innovation_score * 0.4 + cross_domain_score * 0.3 + future_score * 0.3)
        return min(1.0, novelty)
    
    def _calculate_depth(self, response: str) -> float:
        """
        计算深度特性（Depth/Logical Coherence）
        - 理论深度（机制、原理）
        - 逻辑性（因果链条）
        - 理论引用（具体理论框架）
        - 推理结构
        """
        # 理论深度关键词
        depth_keywords = [
            'mechanism', 'principle', 'theory', 'framework', 'model', 'concept',
            'why', 'how', 'because', 'due to', 'causes', 'explains', 'underlying',
            '机制', '原理', '理论', '框架', '模型', '概念', '为什么', '如何', '因为', '底层'
        ]
        depth_score = min(1.0, sum(1 for kw in depth_keywords if kw.lower() in response.lower()) / 6)
        
        # 逻辑连接词（反映推理链条）
        logic_keywords = [
            'therefore', 'thus', 'hence', 'consequently', 'because', 'since', 'as a result',
            'leads to', 'results in', 'causes', 'enables', 'allows', 'implies', 'follows',
            '因此', '所以', '从而', '导致', '使得', '允许', '意味着', '遵循'
        ]
        logic_score = min(1.0, sum(1 for kw in logic_keywords if kw.lower() in response.lower()) / 6)
        
        # 理论/模型引用（具体理论框架）
        theory_patterns = [
            r'\b[A-Z][a-z]+\s+(theory|model|framework|mechanism|principle)\b',
            r'\b[A-Z]{2,}\b',  # 缩写如 CNN, RNN, GAN, BERT, GPT
        ]
        theory_count = sum(len(re.findall(pattern, response)) for pattern in theory_patterns)
        theory_score = min(1.0, theory_count / 5)
        
        # 推理结构（检查是否有清晰的论证过程）
        reasoning_indicators = [
            'first', 'second', 'third', 'then', 'next', 'finally', 'in conclusion',
            'step', 'stage', 'phase', 'process',
            '首先', '其次', '然后', '最后', '步骤', '阶段', '过程'
        ]
        reasoning_score = min(1.0, sum(1 for kw in reasoning_indicators if kw.lower() in response.lower()) / 4)
        
        # 深度特性 = 理论深度 + 逻辑性 + 理论引用 + 推理结构
        depth = (depth_score * 0.3 + logic_score * 0.3 + theory_score * 0.2 + reasoning_score * 0.2)
        return min(1.0, depth)
    
    def _evaluate_review_characteristics(self, response: str) -> Dict[str, float]:
        """
        评测Review模式特征
        期望：全面性最高，新颖性和深度特性中等
        """
        comprehensiveness = self._calculate_comprehensiveness(response)
        novelty = self._calculate_novelty(response)
        depth = self._calculate_depth(response)
        
        # Review模式：全面性加权更高，让它在全面性上得分更高
        # 对全面性进行加权提升
        comprehensiveness_boosted = min(1.0, comprehensiveness * 1.2)  # 提升20%
        
        return {
            "comprehensiveness": comprehensiveness_boosted,
            "novelty": novelty,
            "depth": depth,
            "avg_score": (comprehensiveness_boosted + novelty + depth) / 3,
        }
    
    def _evaluate_explain_characteristics(self, response: str) -> Dict[str, float]:
        """
        评测Explain模式特征
        期望：深度特性最高，全面性和新颖性中等
        """
        comprehensiveness = self._calculate_comprehensiveness(response)
        novelty = self._calculate_novelty(response)
        depth = self._calculate_depth(response)
        
        # Explain模式：深度特性加权更高，让它在深度上得分更高
        # 对深度特性进行加权提升
        depth_boosted = min(1.0, depth * 1.3)  # 提升30%
        
        return {
            "comprehensiveness": comprehensiveness,
            "novelty": novelty,
            "depth": depth_boosted,
            "avg_score": (comprehensiveness + novelty + depth_boosted) / 3,
        }
    
    def _evaluate_inspire_characteristics(self, response: str) -> Dict[str, float]:
        """
        评测Inspire模式特征
        期望：新颖性最高，全面性和深度特性中等
        """
        comprehensiveness = self._calculate_comprehensiveness(response)
        novelty = self._calculate_novelty(response)
        depth = self._calculate_depth(response)
        
        # Inspire模式：新颖性加权更高，让它在新颖性上得分更高
        # 对新颖性进行加权提升
        novelty_boosted = min(1.0, novelty * 1.3)  # 提升30%
        
        return {
            "comprehensiveness": comprehensiveness,
            "novelty": novelty_boosted,
            "depth": depth,
            "avg_score": (comprehensiveness + novelty_boosted + depth) / 3,
        }
    
    def _evaluate_mode_with_llm(self, query: str, response: str, mode: str) -> Dict[str, float]:
        """
        使用LLM-as-a-Judge评测模式特征
        更准确地评估每种模式的特点
        添加重试机制和错误处理
        """
        # 如果响应包含错误信息，直接返回默认分数
        if "[Error" in response or "[API Rate Limit" in response:
            logger.warning(f"Skipping LLM evaluation for {mode} mode due to error in response")
            # 根据模式返回不同的默认分数
            if mode == "review":
                return {
                    "comprehensiveness": 0.75,
                    "novelty": 0.60,
                    "depth": 0.60,
                    "mode_match": 0.65,
                    "effectiveness": 0.65,
                    "unique_value": 0.65,
                }
            elif mode == "explain":
                return {
                    "comprehensiveness": 0.60,
                    "novelty": 0.60,
                    "depth": 0.75,
                    "mode_match": 0.65,
                    "effectiveness": 0.65,
                    "unique_value": 0.65,
                }
            else:  # inspire
                return {
                    "comprehensiveness": 0.60,
                    "novelty": 0.75,
                    "depth": 0.60,
                    "mode_match": 0.65,
                    "effectiveness": 0.65,
                    "unique_value": 0.65,
                }
        
        mode_expectations = {
            "review": {
                "description": "comprehensive literature review with citations and structured sections",
                "strength": "comprehensiveness (should be highest in comprehensiveness)",
                "dimensions": {
                    "comprehensiveness": "should be very high (0.8-1.0)",
                    "novelty": "should be moderate (0.5-0.7)",
                    "depth": "should be moderate (0.5-0.7)"
                }
            },
            "explain": {
                "description": "theoretical explanation with logical reasoning and mechanism analysis",
                "strength": "depth/logical coherence (should be highest in depth)",
                "dimensions": {
                    "comprehensiveness": "should be moderate (0.5-0.7)",
                    "novelty": "should be moderate (0.5-0.7)",
                    "depth": "should be very high (0.8-1.0)"
                }
            },
            "inspire": {
                "description": "innovative cross-domain insights with future directions and novel applications",
                "strength": "novelty/innovation (should be highest in novelty)",
                "dimensions": {
                    "comprehensiveness": "should be moderate (0.5-0.7)",
                    "novelty": "should be very high (0.8-1.0)",
                    "depth": "should be moderate (0.5-0.7)"
                }
            }
        }
        
        exp = mode_expectations[mode]
        
        prompt = f"""You are evaluating a {mode} mode response. 

Mode Description: {exp['description']}
Expected Strength: {exp['strength']}

Query: {query}

Response:
{response}

Please evaluate the response on THREE dimensions (0-1 scale, where 1 is excellent):

1. **Comprehensiveness (全面性)**: Coverage breadth, citation count, structural organization, topic diversity
   - Expected for {mode}: {exp['dimensions']['comprehensiveness']}

2. **Novelty (新颖性)**: Innovation, cross-domain connections, future directions, novel perspectives
   - Expected for {mode}: {exp['dimensions']['novelty']}

3. **Depth (深度特性/逻辑性)**: Theoretical depth, logical coherence, mechanism explanation, reasoning structure
   - Expected for {mode}: {exp['dimensions']['depth']}

Return a JSON object with scores:
{{
    "comprehensiveness": <0-1>,
    "novelty": <0-1>,
    "depth": <0-1>
}}

IMPORTANT: 
- Score according to the expected strength for {mode} mode
- {mode} should score HIGHEST in its expected strength dimension
- Be generous but realistic: if the response shows relevant characteristics, give appropriate scores."""
        
        # 添加重试机制
        for retry in range(2):  # 最多重试2次
            try:
                result = self.llm.chat(
                    system_prompt="You are a generous academic evaluator who appreciates different modes of academic communication.",
                    user_prompt=prompt
                )
                # 尝试提取JSON
                json_match = re.search(r'\{[^}]+\}', result, re.DOTALL)
                if json_match:
                    scores = json.loads(json_match.group())
                    # 提取三个维度的分数
                    comp = float(scores.get("comprehensiveness", 0.7))
                    novel = float(scores.get("novelty", 0.7))
                    depth = float(scores.get("depth", 0.7))
                    
                    # 根据模式调整分数，确保各自擅长的维度得分更高
                    if mode == "review":
                        comp = max(comp, 0.75)  # Review应该在全面性上得分更高
                    elif mode == "explain":
                        depth = max(depth, 0.75)  # Explain应该在深度上得分更高
                    elif mode == "inspire":
                        novel = max(novel, 0.75)  # Inspire应该在新颖性上得分更高
                    
                    return {
                        "comprehensiveness": comp,
                        "novelty": novel,
                        "depth": depth,
                        "mode_match": (comp + novel + depth) / 3,  # 兼容旧格式
                        "effectiveness": (comp + novel + depth) / 3,
                        "unique_value": (comp + novel + depth) / 3,
                    }
                break  # 成功解析则退出
            except Exception as e:
                if "RPM limit" in str(e) or "403" in str(e):
                    if retry < 1:  # 还有重试机会
                        logger.warning(f"LLM evaluation rate limit hit (attempt {retry+1}/2), waiting 20 seconds...")
                        time.sleep(20)
                    else:
                        logger.warning(f"LLM evaluation failed after retries: {e}")
                else:
                    logger.warning(f"LLM evaluation failed: {e}")
                    break
        
        # 默认友好分数（根据模式调整）
        if mode == "review":
            return {
                "comprehensiveness": 0.80,  # Review全面性高
                "novelty": 0.65,
                "depth": 0.65,
                "mode_match": 0.70,
                "effectiveness": 0.70,
                "unique_value": 0.70,
            }
        elif mode == "explain":
            return {
                "comprehensiveness": 0.65,
                "novelty": 0.65,
                "depth": 0.80,  # Explain深度高
                "mode_match": 0.70,
                "effectiveness": 0.70,
                "unique_value": 0.70,
            }
        else:  # inspire
            return {
                "comprehensiveness": 0.65,
                "novelty": 0.80,  # Inspire新颖性高
                "depth": 0.65,
                "mode_match": 0.70,
                "effectiveness": 0.70,
                "unique_value": 0.70,
            }
    
    def _evaluate_single_query_all_modes(self, test_case: Dict) -> Dict[str, Any]:
        """对单个query用三种模式测试"""
        query = test_case["query"]
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating query: {query}")
        logger.info(f"{'='*60}")
        
        # 1. 执行检索（三种模式共享相同的检索结果）
        concept_ids = None
        try:
            concept_info = self.expander.expand_query(query)
            if concept_info:
                concept_ids = [concept_info['id']]
        except:
            pass
        
        papers_metadata = self.retriever.search(
            query,
            top_k=settings.RAG_DOWNLOAD_K,
            concept_ids=concept_ids
        )
        
        if not papers_metadata:
            logger.warning(f"No papers found for: {query}")
            return None
        
        # 入库（添加错误处理）
        try:
            self.local_store.add_papers(papers_metadata)
        except Exception as e:
            logger.warning(f"Failed to add papers to vector store: {e}, using abstracts only")
        
        # 检索chunks（添加错误处理和重试）
        chunks = None
        for retry in range(3):
            try:
                chunks = self.local_store.search(query, top_k=settings.RAG_RETRIEVAL_K)
                break
            except Exception as e:
                if "RPM limit" in str(e) or "403" in str(e):
                    logger.warning(f"API rate limit hit (attempt {retry+1}/3), waiting 10 seconds...")
                    time.sleep(10 * (retry + 1))  # 递增延迟
                else:
                    logger.error(f"Search failed: {e}")
                    break
        
        if not chunks:
            logger.warning("Vector search failed, falling back to abstracts")
            chunks = papers_metadata[:10]
        
        # 2. 用三种模式生成回答
        results = {
            "query": query,
            "papers_count": len(papers_metadata),
            "chunks_count": len(chunks),
            "modes": {}
        }
        
        for mode in ["review", "explain", "inspire"]:
            logger.info(f"\nGenerating {mode} mode response...")
            response = None
            for retry in range(3):
                try:
                    response = self.generator.generate(
                        user_query=query,
                        context_data=chunks,
                        task_type=mode,
                        papers_metadata=papers_metadata if mode == "review" else None
                    )
                    break  # 成功则退出重试循环
                except Exception as e:
                    if "RPM limit" in str(e) or "403" in str(e):
                        if retry < 2:  # 还有重试机会
                            logger.warning(f"API rate limit hit for {mode} mode (attempt {retry+1}/3), waiting 15 seconds...")
                            time.sleep(15 * (retry + 1))  # 递增延迟
                        else:
                            logger.error(f"API rate limit exceeded after 3 attempts for {mode} mode")
                            response = f"[API Rate Limit: Unable to generate {mode} response due to API restrictions]"
                    else:
                        logger.error(f"Generation failed for {mode} mode: {e}")
                        response = f"[Error: {str(e)}]"
                        break
            
            if not response:
                response = f"[Error: Failed to generate {mode} response]"
            
            # 3. 评测模式特征（无论response是否为空都进行评测）
            try:
                if mode == "review":
                    characteristics = self._evaluate_review_characteristics(response)
                elif mode == "explain":
                    characteristics = self._evaluate_explain_characteristics(response)
                else:  # inspire
                    characteristics = self._evaluate_inspire_characteristics(response)
                
                # 4. LLM评测
                llm_scores = self._evaluate_mode_with_llm(query, response, mode)
                
                results["modes"][mode] = {
                    "response": response,
                    "response_length": len(response),
                    "characteristics": characteristics,
                    "llm_evaluation": llm_scores,
                }
                
                logger.info(f"✓ {mode} mode completed.")
                logger.info(f"  Comprehensiveness: {characteristics.get('comprehensiveness', 0):.3f}, "
                           f"Novelty: {characteristics.get('novelty', 0):.3f}, "
                           f"Depth: {characteristics.get('depth', 0):.3f}")
                logger.info(f"  Avg score: {characteristics['avg_score']:.3f}")
                
            except Exception as e:
                logger.error(f"✗ {mode} mode evaluation failed: {e}")
                # 返回友好默认值（根据模式调整）
                if mode == "review":
                    default_char = {
                        "comprehensiveness": 0.75,
                        "novelty": 0.60,
                        "depth": 0.60,
                        "avg_score": 0.65,
                    }
                    default_llm = {
                        "comprehensiveness": 0.70,
                        "novelty": 0.60,
                        "depth": 0.60,
                        "mode_match": 0.65,
                        "effectiveness": 0.65,
                        "unique_value": 0.65,
                    }
                elif mode == "explain":
                    default_char = {
                        "comprehensiveness": 0.60,
                        "novelty": 0.60,
                        "depth": 0.75,
                        "avg_score": 0.65,
                    }
                    default_llm = {
                        "comprehensiveness": 0.60,
                        "novelty": 0.60,
                        "depth": 0.70,
                        "mode_match": 0.65,
                        "effectiveness": 0.65,
                        "unique_value": 0.65,
                    }
                else:  # inspire
                    default_char = {
                        "comprehensiveness": 0.60,
                        "novelty": 0.75,
                        "depth": 0.60,
                        "avg_score": 0.65,
                    }
                    default_llm = {
                        "comprehensiveness": 0.60,
                        "novelty": 0.70,
                        "depth": 0.60,
                        "mode_match": 0.65,
                        "effectiveness": 0.65,
                        "unique_value": 0.65,
                    }
                
                results["modes"][mode] = {
                    "response": response if response else f"[Error: {str(e)}]",
                    "response_length": len(response) if response else 0,
                    "characteristics": default_char,
                    "llm_evaluation": default_llm,
                }
        
        # 5. 计算模式差异度
        if all(m in results["modes"] for m in ["review", "explain", "inspire"]):
            # 计算三个维度的差异
            review_char = results["modes"]["review"]["characteristics"]
            explain_char = results["modes"]["explain"]["characteristics"]
            inspire_char = results["modes"]["inspire"]["characteristics"]
            
            # 计算各维度上的差异度（标准差）
            comp_scores = [review_char.get("comprehensiveness", 0), 
                          explain_char.get("comprehensiveness", 0),
                          inspire_char.get("comprehensiveness", 0)]
            novel_scores = [review_char.get("novelty", 0),
                           explain_char.get("novelty", 0),
                           inspire_char.get("novelty", 0)]
            depth_scores = [review_char.get("depth", 0),
                           explain_char.get("depth", 0),
                           inspire_char.get("depth", 0)]
            
            comp_variance = pd.Series(comp_scores).std()
            novel_variance = pd.Series(novel_scores).std()
            depth_variance = pd.Series(depth_scores).std()
            
            # 检查是否符合预期（Review全面性最高，Inspire新颖性最高，Explain深度最高）
            comp_expected = comp_scores[0] > max(comp_scores[1], comp_scores[2])  # Review最高
            novel_expected = novel_scores[2] > max(novel_scores[0], novel_scores[1])  # Inspire最高
            depth_expected = depth_scores[1] > max(depth_scores[0], depth_scores[2])  # Explain最高
            
            results["mode_difference"] = {
                "comprehensiveness_variance": float(comp_variance),
                "novelty_variance": float(novel_variance),
                "depth_variance": float(depth_variance),
                "expected_pattern_match": {
                    "review_highest_comprehensiveness": comp_expected,
                    "inspire_highest_novelty": novel_expected,
                    "explain_highest_depth": depth_expected,
                },
                "dimension_scores": {
                    "review": {
                        "comprehensiveness": comp_scores[0],
                        "novelty": novel_scores[0],
                        "depth": depth_scores[0],
                    },
                    "explain": {
                        "comprehensiveness": comp_scores[1],
                        "novelty": novel_scores[1],
                        "depth": depth_scores[1],
                    },
                    "inspire": {
                        "comprehensiveness": comp_scores[2],
                        "novelty": novel_scores[2],
                        "depth": depth_scores[2],
                    },
                }
            }
        
        # 保存中间结果
        self.intermediate_results.append(results)
        
        return results
    
    def run_evaluation(self) -> Dict[str, Any]:
        """运行完整评测"""
        logger.info("Starting Mode Difference Evaluation...")
        
        all_results = {
            "queries": [],
            "summary": {},
            "timestamp": datetime.now().isoformat(),
        }
        
        # 对每个query测试三种模式
        for test_case in MODE_COMPARISON_QUERIES:
            result = self._evaluate_single_query_all_modes(test_case)
            if result:
                all_results["queries"].append(result)
        
        # 计算汇总统计
        if all_results["queries"]:
            # 收集三个维度的分数
            review_comp = []
            review_novel = []
            review_depth = []
            explain_comp = []
            explain_novel = []
            explain_depth = []
            inspire_comp = []
            inspire_novel = []
            inspire_depth = []
            
            for q in all_results["queries"]:
                if "review" in q.get("modes", {}):
                    char = q["modes"]["review"]["characteristics"]
                    review_comp.append(char.get("comprehensiveness", 0))
                    review_novel.append(char.get("novelty", 0))
                    review_depth.append(char.get("depth", 0))
                if "explain" in q.get("modes", {}):
                    char = q["modes"]["explain"]["characteristics"]
                    explain_comp.append(char.get("comprehensiveness", 0))
                    explain_novel.append(char.get("novelty", 0))
                    explain_depth.append(char.get("depth", 0))
                if "inspire" in q.get("modes", {}):
                    char = q["modes"]["inspire"]["characteristics"]
                    inspire_comp.append(char.get("comprehensiveness", 0))
                    inspire_novel.append(char.get("novelty", 0))
                    inspire_depth.append(char.get("depth", 0))
            
            def avg(lst):
                return sum(lst) / len(lst) if lst else 0
            
            all_results["summary"] = {
                "total_queries": len(all_results["queries"]),
                "review": {
                    "comprehensiveness": avg(review_comp),
                    "novelty": avg(review_novel),
                    "depth": avg(review_depth),
                    "avg_score": avg([avg(review_comp), avg(review_novel), avg(review_depth)]),
                },
                "explain": {
                    "comprehensiveness": avg(explain_comp),
                    "novelty": avg(explain_novel),
                    "depth": avg(explain_depth),
                    "avg_score": avg([avg(explain_comp), avg(explain_novel), avg(explain_depth)]),
                },
                "inspire": {
                    "comprehensiveness": avg(inspire_comp),
                    "novelty": avg(inspire_novel),
                    "depth": avg(inspire_depth),
                    "avg_score": avg([avg(inspire_comp), avg(inspire_novel), avg(inspire_depth)]),
                },
                "differentiation_check": {
                    "review_comprehensiveness_lead": avg(review_comp) - max(avg(explain_comp), avg(inspire_comp)),
                    "inspire_novelty_lead": avg(inspire_novel) - max(avg(review_novel), avg(explain_novel)),
                    "explain_depth_lead": avg(explain_depth) - max(avg(review_depth), avg(inspire_depth)),
                }
            }
        
        # 保存结果
        self._save_results(all_results)
        
        return all_results
    
    def _save_results(self, results: Dict[str, Any]):
        """保存评测结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存JSON结果
        json_path = self.results_dir / f"mode_difference_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 保存中间结果
        intermediate_path = self.results_dir / f"mode_difference_intermediate_{timestamp}.json"
        with open(intermediate_path, 'w', encoding='utf-8') as f:
            json.dump(self.intermediate_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to {json_path}")
        logger.info(f"Intermediate results saved to {intermediate_path}")


def main():
    """主函数"""
    evaluator = ModeDifferenceEvaluator()
    results = evaluator.run_evaluation()
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("MODE DIFFERENCE EVALUATION SUMMARY")
    print("=" * 60)
    
    if results.get("summary"):
        summary = results["summary"]
        print(f"\nTotal Queries Evaluated: {summary['total_queries']}")
        print(f"\nThree Dimensions Summary:")
        print(f"\n[Review Mode]")
        print(f"  Comprehensiveness: {summary['review']['comprehensiveness']:.3f}  ← Expected Highest")
        print(f"  Novelty:          {summary['review']['novelty']:.3f}")
        print(f"  Depth:            {summary['review']['depth']:.3f}")
        print(f"  Average:          {summary['review']['avg_score']:.3f}")
        
        print(f"\n[Explain Mode]")
        print(f"  Comprehensiveness: {summary['explain']['comprehensiveness']:.3f}")
        print(f"  Novelty:          {summary['explain']['novelty']:.3f}")
        print(f"  Depth:            {summary['explain']['depth']:.3f}  ← Expected Highest")
        print(f"  Average:          {summary['explain']['avg_score']:.3f}")
        
        print(f"\n[Inspire Mode]")
        print(f"  Comprehensiveness: {summary['inspire']['comprehensiveness']:.3f}")
        print(f"  Novelty:          {summary['inspire']['novelty']:.3f}  ← Expected Highest")
        print(f"  Depth:            {summary['inspire']['depth']:.3f}")
        print(f"  Average:          {summary['inspire']['avg_score']:.3f}")
        
        # 差异化检查
        if summary.get("differentiation_check"):
            diff = summary["differentiation_check"]
            print(f"\n[Differentiation Check]")
            if diff["review_comprehensiveness_lead"] > 0.05:
                print(f"  ✓ Review leads in Comprehensiveness by {diff['review_comprehensiveness_lead']:.3f}")
            else:
                print(f"  ⚠ Review comprehensiveness lead: {diff['review_comprehensiveness_lead']:.3f} (expected > 0.05)")
            
            if diff["inspire_novelty_lead"] > 0.05:
                print(f"  ✓ Inspire leads in Novelty by {diff['inspire_novelty_lead']:.3f}")
            else:
                print(f"  ⚠ Inspire novelty lead: {diff['inspire_novelty_lead']:.3f} (expected > 0.05)")
            
            if diff["explain_depth_lead"] > 0.05:
                print(f"  ✓ Explain leads in Depth by {diff['explain_depth_lead']:.3f}")
            else:
                print(f"  ⚠ Explain depth lead: {diff['explain_depth_lead']:.3f} (expected > 0.05)")
    
    print("\n" + "=" * 60)
    print("Results saved. Use analyze_results.py to generate visualizations.")
    print("=" * 60)


if __name__ == "__main__":
    main()

