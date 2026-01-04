

"""
结果分析和可视化脚本
生成美观的评测报告
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置样式
sns.set_style("whitegrid")
sns.set_palette("husl")


class ResultAnalyzer:
    """结果分析器"""
    
    def __init__(self, results_dir: str = "evaluation/results"):
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_rag_results(self, json_file: str = None):
        """分析RAG评测结果"""
        # 查找最新的结果文件
        if json_file is None:
            json_files = list(self.results_dir.glob("rag_evaluation_*.json"))
            if not json_files:
                print("No RAG evaluation results found!")
                return
            json_file = max(json_files, key=lambda p: p.stat().st_mtime)
        
        with open(json_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        print("\n" + "=" * 60)
        print("RAG EVALUATION ANALYSIS")
        print("=" * 60)
        
        # 分析Review模式
        if results.get("review"):
            self._analyze_review_mode(results["review"])
        
        # 分析Explain模式
        if results.get("explain"):
            self._analyze_explain_mode(results["explain"])
        
        # 分析Inspire模式
        if results.get("inspire"):
            self._analyze_inspire_mode(results["inspire"])
        
        # 生成可视化
        self._visualize_rag_results(results)
    
    def _analyze_review_mode(self, review_results: List[Dict]):
        """分析Review模式结果"""
        print("\n[Review Mode Analysis]")
        print("-" * 60)
        
        df = pd.DataFrame(review_results)
        
        avg_faith = df['faithfulness'].mean()
        avg_prec = df['context_precision'].mean()
        
        print(f"Average Faithfulness: {avg_faith:.3f} ({avg_faith*100:.1f}%)")
        print(f"Average Context Precision: {avg_prec:.3f} ({avg_prec*100:.1f}%)")
        print(f"\nBest Performance:")
        print(f"  Highest Faithfulness: {df.loc[df['faithfulness'].idxmax(), 'query']} ({df['faithfulness'].max():.3f})")
        print(f"  Highest Precision: {df.loc[df['context_precision'].idxmax(), 'query']} ({df['context_precision'].max():.3f})")
    
    def _analyze_explain_mode(self, explain_results: List[Dict]):
        """分析Explain模式结果"""
        print("\n[Explain Mode Analysis]")
        print("-" * 60)
        
        df = pd.DataFrame(explain_results)
        
        avg_corr = df['answer_correctness'].mean()
        avg_recall = df['context_recall'].mean()
        
        print(f"Average Answer Correctness: {avg_corr:.3f} ({avg_corr*100:.1f}%)")
        print(f"Average Context Recall: {avg_recall:.3f} ({avg_recall*100:.1f}%)")
        print(f"\nBest Performance:")
        print(f"  Highest Correctness: {df.loc[df['answer_correctness'].idxmax(), 'query']} ({df['answer_correctness'].max():.3f})")
        print(f"  Highest Recall: {df.loc[df['context_recall'].idxmax(), 'query']} ({df['context_recall'].max():.3f})")
    
    def _analyze_inspire_mode(self, inspire_results: List[Dict]):
        """分析Inspire模式结果"""
        print("\n[Inspire Mode Analysis]")
        print("-" * 60)
        
        df = pd.DataFrame(inspire_results)
        
        avg_rel = df['answer_relevance'].mean()
        avg_innov = df['innovation_score'].mean()
        
        print(f"Average Answer Relevance: {avg_rel:.3f} ({avg_rel*100:.1f}%)")
        print(f"Average Innovation Score: {avg_innov:.3f} ({avg_innov*100:.1f}%)")
        print(f"\nBest Performance:")
        print(f"  Highest Relevance: {df.loc[df['answer_relevance'].idxmax(), 'query']} ({df['answer_relevance'].max():.3f})")
        print(f"  Highest Innovation: {df.loc[df['innovation_score'].idxmax(), 'query']} ({df['innovation_score'].max():.3f})")
    
    def _visualize_rag_results(self, results: Dict):
        """可视化RAG结果"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('RAG Evaluation Results Summary', fontsize=16, fontweight='bold')
        
        # 1. Review模式
        if results.get("review"):
            ax1 = axes[0, 0]
            df_review = pd.DataFrame(results["review"])
            metrics = ['faithfulness', 'context_precision']
            df_review[metrics].mean().plot(kind='bar', ax=ax1, color=['#3498db', '#2ecc71'])
            ax1.set_title('Review Mode Metrics', fontweight='bold')
            ax1.set_ylabel('Score')
            ax1.set_ylim([0, 1])
            ax1.set_xticklabels(['Faithfulness', 'Context Precision'], rotation=0)
            ax1.grid(axis='y', alpha=0.3)
            for i, v in enumerate(df_review[metrics].mean()):
                ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
        # 2. Explain模式
        if results.get("explain"):
            ax2 = axes[0, 1]
            df_explain = pd.DataFrame(results["explain"])
            metrics = ['answer_correctness', 'context_recall']
            df_explain[metrics].mean().plot(kind='bar', ax=ax2, color=['#e74c3c', '#9b59b6'])
            ax2.set_title('Explain Mode Metrics', fontweight='bold')
            ax2.set_ylabel('Score')
            ax2.set_ylim([0, 1])
            ax2.set_xticklabels(['Answer Correctness', 'Context Recall'], rotation=0)
            ax2.grid(axis='y', alpha=0.3)
            for i, v in enumerate(df_explain[metrics].mean()):
                ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
        # 3. Inspire模式
        if results.get("inspire"):
            ax3 = axes[1, 0]
            df_inspire = pd.DataFrame(results["inspire"])
            metrics = ['answer_relevance', 'innovation_score']
            df_inspire[metrics].mean().plot(kind='bar', ax=ax3, color=['#f39c12', '#1abc9c'])
            ax3.set_title('Inspire Mode Metrics', fontweight='bold')
            ax3.set_ylabel('Score')
            ax3.set_ylim([0, 1])
            ax3.set_xticklabels(['Answer Relevance', 'Innovation Score'], rotation=0)
            ax3.grid(axis='y', alpha=0.3)
            for i, v in enumerate(df_inspire[metrics].mean()):
                ax3.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
        # 4. 综合对比
        ax4 = axes[1, 1]
        all_metrics = []
        all_values = []
        
        if results.get("review"):
            df_review = pd.DataFrame(results["review"])
            all_metrics.extend(['Review\nFaithfulness', 'Review\nPrecision'])
            all_values.extend([
                df_review['faithfulness'].mean(),
                df_review['context_precision'].mean()
            ])
        
        if results.get("explain"):
            df_explain = pd.DataFrame(results["explain"])
            all_metrics.extend(['Explain\nCorrectness', 'Explain\nRecall'])
            all_values.extend([
                df_explain['answer_correctness'].mean(),
                df_explain['context_recall'].mean()
            ])
        
        if results.get("inspire"):
            df_inspire = pd.DataFrame(results["inspire"])
            all_metrics.extend(['Inspire\nRelevance', 'Inspire\nInnovation'])
            all_values.extend([
                df_inspire['answer_relevance'].mean(),
                df_inspire['innovation_score'].mean()
            ])
        
        ax4.barh(all_metrics, all_values, color=sns.color_palette("husl", len(all_metrics)))
        ax4.set_title('Overall Performance', fontweight='bold')
        ax4.set_xlabel('Score')
        ax4.set_xlim([0, 1])
        ax4.grid(axis='x', alpha=0.3)
        for i, v in enumerate(all_values):
            ax4.text(v + 0.01, i, f'{v:.3f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"rag_evaluation_{timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_path}")
        plt.close()
    
    def analyze_graph_ab_results(self, json_file: str = None):
        """分析图检索A/B测试结果（支持V1和V2版本）"""
        # 查找最新的结果文件（优先V2版本）
        if json_file is None:
            json_files_v2 = list(self.results_dir.glob("graph_ab_test_v2_*.json"))
            json_files_v1 = list(self.results_dir.glob("graph_ab_test_*.json"))
            # 排除V2文件
            json_files_v1 = [f for f in json_files_v1 if 'v2' not in f.name]
            
            if json_files_v2:
                json_file = max(json_files_v2, key=lambda p: p.stat().st_mtime)
                is_v2 = True
            elif json_files_v1:
                json_file = max(json_files_v1, key=lambda p: p.stat().st_mtime)
                is_v2 = False
            else:
                print("⚠️  No graph A/B test results found!")
                print("   Please run: python -m evaluation.evaluate_graph_v2")
                return
        else:
            # 根据文件名判断版本
            is_v2 = 'v2' in str(json_file)
        
        print(f"\n📊 Reading AB test results from: {json_file.name}")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
        except Exception as e:
            print(f"❌ Error reading results file: {e}")
            return
        
        print("\n" + "=" * 60)
        print(f"GRAPH A/B TEST ANALYSIS {'V2' if is_v2 else 'V1'}")
        print("=" * 60)
        
        # 分析对比结果
        if results.get("comparison"):
            if is_v2:
                self._analyze_comparison_v2(results["comparison"])
            else:
                self._analyze_comparison(results["comparison"])
        else:
            print("⚠️  No comparison data found in results")
        
        # 生成可视化
        if is_v2:
            self._visualize_graph_results_v2(results)
        else:
            self._visualize_graph_results(results)
    
    def _analyze_comparison(self, comparison_results: List[Dict]):
        """分析对比结果"""
        print("\n[Comparison Analysis]")
        print("-" * 60)
        
        df = pd.DataFrame(comparison_results)
        
        avg_hit_improvement = df['hit_rate_improvement'].mean()
        avg_noise_reduction = df['noise_rate_reduction'].mean()
        
        print(f"Average Hit Rate Improvement: +{avg_hit_improvement:.3f} ({avg_hit_improvement*100:.1f}%)")
        print(f"Average Noise Rate Reduction: -{avg_noise_reduction:.3f} ({avg_noise_reduction*100:.1f}%)")
        
        # 统计胜率
        hit_wins = (df['hit_rate_improvement'] > 0).sum()
        noise_wins = (df['noise_rate_reduction'] > 0).sum()
        total = len(df)
        
        print(f"\nWin Rate:")
        print(f"  Hit Rate: {hit_wins}/{total} ({hit_wins/total*100:.1f}%)")
        print(f"  Noise Rate: {noise_wins}/{total} ({noise_wins/total*100:.1f}%)")
        
        print(f"\nBest Improvements:")
        best_hit = df.loc[df['hit_rate_improvement'].idxmax()]
        best_noise = df.loc[df['noise_rate_reduction'].idxmax()]
        print(f"  Best Hit Rate: {best_hit['query']} (+{best_hit['hit_rate_improvement']:.3f})")
        print(f"  Best Noise Reduction: {best_noise['query']} (-{best_noise['noise_rate_reduction']:.3f})")
    
    def _visualize_graph_results(self, results: Dict):
        """可视化图检索A/B测试结果"""
        if not results.get("group_a_no_graph") or not results.get("group_b_with_graph"):
            print("⚠️  No data to visualize for graph A/B test")
            return
        
        try:
            df_a = pd.DataFrame(results["group_a_no_graph"])
            df_b = pd.DataFrame(results["group_b_with_graph"])
            df_comp = pd.DataFrame(results.get("comparison", []))
            
            if df_a.empty or df_b.empty:
                print("⚠️  Empty dataframes, skipping visualization")
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(18, 14))
            fig.suptitle('Graph Retrieval A/B Test Results', fontsize=18, fontweight='bold', y=0.995)
            
            num_queries = len(df_a)
            x = range(num_queries)
            width = 0.35
            
            # 获取query名称（截断过长的）
            queries = [q[:20] + '...' if len(q) > 20 else q for q in df_a['query'].tolist()]
            
            # 1. Hit Rate对比
            ax1 = axes[0, 0]
            bars1a = ax1.bar([i - width/2 for i in x], df_a['hit_rate'], width, 
                            label='No Graph (对照组)', color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)
            bars1b = ax1.bar([i + width/2 for i in x], df_b['hit_rate'], width, 
                            label='With Graph (实验组)', color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=0.5)
            ax1.set_title('Hit Rate Comparison (命中率对比)', fontweight='bold', fontsize=14)
            ax1.set_ylabel('Hit Rate', fontsize=12)
            ax1.set_xlabel('Test Query', fontsize=12)
            ax1.set_xticks(x)
            ax1.set_xticklabels([f"Q{i+1}" for i in x], rotation=45, ha='right', fontsize=9)
            ax1.set_ylim([0, 1.1])
            ax1.legend(loc='upper right', fontsize=10)
            ax1.grid(axis='y', alpha=0.3, linestyle='--')
            ax1.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
            
            # 添加数值标签
            for i, (bar_a, bar_b) in enumerate(zip(bars1a, bars1b)):
                height_a = bar_a.get_height()
                height_b = bar_b.get_height()
                ax1.text(bar_a.get_x() + bar_a.get_width()/2., height_a + 0.02,
                        f'{height_a:.2f}', ha='center', va='bottom', fontsize=8)
                ax1.text(bar_b.get_x() + bar_b.get_width()/2., height_b + 0.02,
                        f'{height_b:.2f}', ha='center', va='bottom', fontsize=8)
            
            # 2. Noise Rate对比
            ax2 = axes[0, 1]
            bars2a = ax2.bar([i - width/2 for i in x], df_a['noise_rate'], width, 
                            label='No Graph (对照组)', color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)
            bars2b = ax2.bar([i + width/2 for i in x], df_b['noise_rate'], width, 
                            label='With Graph (实验组)', color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=0.5)
            ax2.set_title('Noise Rate Comparison (噪声率对比)', fontweight='bold', fontsize=14)
            ax2.set_ylabel('Noise Rate', fontsize=12)
            ax2.set_xlabel('Test Query', fontsize=12)
            ax2.set_xticks(x)
            ax2.set_xticklabels([f"Q{i+1}" for i in x], rotation=45, ha='right', fontsize=9)
            ax2.set_ylim([0, 1.1])
            ax2.legend(loc='upper right', fontsize=10)
            ax2.grid(axis='y', alpha=0.3, linestyle='--')
            ax2.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
            
            # 添加数值标签
            for i, (bar_a, bar_b) in enumerate(zip(bars2a, bars2b)):
                height_a = bar_a.get_height()
                height_b = bar_b.get_height()
                ax2.text(bar_a.get_x() + bar_a.get_width()/2., height_a + 0.02,
                        f'{height_a:.2f}', ha='center', va='bottom', fontsize=8)
                ax2.text(bar_b.get_x() + bar_b.get_width()/2., height_b + 0.02,
                        f'{height_b:.2f}', ha='center', va='bottom', fontsize=8)
            
            # 3. 改进幅度（Hit Rate）
            ax3 = axes[1, 0]
            if not df_comp.empty and 'hit_rate_improvement' in df_comp.columns:
                colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in df_comp['hit_rate_improvement']]
                bars3 = ax3.barh(range(len(df_comp)), df_comp['hit_rate_improvement'], 
                                color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
                ax3.set_title('Hit Rate Improvement (命中率提升)', fontweight='bold', fontsize=14)
                ax3.set_xlabel('Improvement (提升幅度)', fontsize=12)
                ax3.set_ylabel('Test Query', fontsize=12)
                ax3.set_yticks(range(len(df_comp)))
                ax3.set_yticklabels([f"Q{i+1}" for i in range(len(df_comp))], fontsize=9)
                ax3.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
                ax3.grid(axis='x', alpha=0.3, linestyle='--')
                
                # 添加数值标签
                for i, (bar, val) in enumerate(zip(bars3, df_comp['hit_rate_improvement'])):
                    width_bar = bar.get_width()
                    ax3.text(width_bar + 0.01 if width_bar > 0 else width_bar - 0.01, bar.get_y() + bar.get_height()/2,
                            f'{val:+.3f}', ha='left' if width_bar > 0 else 'right', va='center', fontsize=8)
            
            # 4. 噪声减少（Noise Rate）
            ax4 = axes[1, 1]
            if not df_comp.empty and 'noise_rate_reduction' in df_comp.columns:
                colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in df_comp['noise_rate_reduction']]
                bars4 = ax4.barh(range(len(df_comp)), df_comp['noise_rate_reduction'], 
                                color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
                ax4.set_title('Noise Rate Reduction (噪声率降低)', fontweight='bold', fontsize=14)
                ax4.set_xlabel('Reduction (降低幅度)', fontsize=12)
                ax4.set_ylabel('Test Query', fontsize=12)
                ax4.set_yticks(range(len(df_comp)))
                ax4.set_yticklabels([f"Q{i+1}" for i in range(len(df_comp))], fontsize=9)
                ax4.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
                ax4.grid(axis='x', alpha=0.3, linestyle='--')
                
                # 添加数值标签
                for i, (bar, val) in enumerate(zip(bars4, df_comp['noise_rate_reduction'])):
                    width_bar = bar.get_width()
                    ax4.text(width_bar + 0.01 if width_bar > 0 else width_bar - 0.01, bar.get_y() + bar.get_height()/2,
                            f'{val:+.3f}', ha='left' if width_bar > 0 else 'right', va='center', fontsize=8)
            
            plt.tight_layout(rect=[0, 0, 1, 0.98])
            
            # 保存图片
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"graph_ab_test_{timestamp}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"\n✅ Visualization saved to: {output_path}")
            plt.close()
            
        except Exception as e:
            print(f"❌ Error generating visualization: {e}")
            import traceback
            traceback.print_exc()
    
    def _analyze_comparison_v2(self, comparison_results: List[Dict]):
        """分析V2版本的对比结果"""
        print("\n[Comparison Analysis V2]")
        print("-" * 60)
        
        df = pd.DataFrame(comparison_results)
        
        avg_rel_improvement = df['avg_relevance_improvement'].mean()
        avg_ndcg_improvement = df['ndcg_improvement'].mean()
        avg_top5_improvement = df['top_5_avg_improvement'].mean()
        
        print(f"Average Relevance Improvement: +{avg_rel_improvement:.3f} ({avg_rel_improvement*100:.1f}%)")
        print(f"Average NDCG Improvement: +{avg_ndcg_improvement:.3f} ({avg_ndcg_improvement*100:.1f}%)")
        print(f"Average Top-5 Relevance Improvement: +{avg_top5_improvement:.3f} ({avg_top5_improvement*100:.1f}%)")
        
        # 统计胜率
        rel_wins = (df['avg_relevance_improvement'] > 0).sum()
        ndcg_wins = (df['ndcg_improvement'] > 0).sum()
        top5_wins = (df['top_5_avg_improvement'] > 0).sum()
        total = len(df)
        
        print(f"\nWin Rate:")
        print(f"  Avg Relevance: {rel_wins}/{total} ({rel_wins/total*100:.1f}%)")
        print(f"  NDCG: {ndcg_wins}/{total} ({ndcg_wins/total*100:.1f}%)")
        print(f"  Top-5 Avg: {top5_wins}/{total} ({top5_wins/total*100:.1f}%)")
        
        print(f"\nBest Improvements:")
        best_rel = df.loc[df['avg_relevance_improvement'].idxmax()]
        best_ndcg = df.loc[df['ndcg_improvement'].idxmax()]
        print(f"  Best Relevance: {best_rel['query']} (+{best_rel['avg_relevance_improvement']:.3f})")
        print(f"  Best NDCG: {best_ndcg['query']} (+{best_ndcg['ndcg_improvement']:.3f})")
    
    def _visualize_graph_results_v2(self, results: Dict):
        """可视化V2版本的图检索A/B测试结果"""
        if not results.get("group_a_no_graph") or not results.get("group_b_with_graph"):
            print("⚠️  No data to visualize for graph A/B test")
            return
        
        try:
            df_a = pd.DataFrame(results["group_a_no_graph"])
            df_b = pd.DataFrame(results["group_b_with_graph"])
            df_comp = pd.DataFrame(results.get("comparison", []))
            
            if df_a.empty or df_b.empty:
                print("⚠️  Empty dataframes, skipping visualization")
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(18, 14))
            fig.suptitle('Graph Retrieval A/B Test Results V2', fontsize=18, fontweight='bold', y=0.995)
            
            num_queries = len(df_a)
            x = range(num_queries)
            width = 0.35
            
            # 1. 平均相关性对比
            ax1 = axes[0, 0]
            bars1a = ax1.bar([i - width/2 for i in x], df_a['avg_relevance'], width, 
                            label='No Graph (对照组)', color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)
            bars1b = ax1.bar([i + width/2 for i in x], df_b['avg_relevance'], width, 
                            label='With Graph (实验组)', color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=0.5)
            ax1.set_title('Average Relevance Comparison (平均相关性对比)', fontweight='bold', fontsize=14)
            ax1.set_ylabel('Average Relevance', fontsize=12)
            ax1.set_xlabel('Test Query', fontsize=12)
            ax1.set_xticks(x)
            ax1.set_xticklabels([f"Q{i+1}" for i in x], rotation=45, ha='right', fontsize=9)
            ax1.set_ylim([0, 1.1])
            ax1.legend(loc='upper right', fontsize=10)
            ax1.grid(axis='y', alpha=0.3, linestyle='--')
            
            # 添加数值标签
            for i, (bar_a, bar_b) in enumerate(zip(bars1a, bars1b)):
                height_a = bar_a.get_height()
                height_b = bar_b.get_height()
                ax1.text(bar_a.get_x() + bar_a.get_width()/2., height_a + 0.02,
                        f'{height_a:.2f}', ha='center', va='bottom', fontsize=8)
                ax1.text(bar_b.get_x() + bar_b.get_width()/2., height_b + 0.02,
                        f'{height_b:.2f}', ha='center', va='bottom', fontsize=8)
            
            # 2. NDCG对比
            ax2 = axes[0, 1]
            bars2a = ax2.bar([i - width/2 for i in x], df_a['ndcg'], width, 
                            label='No Graph (对照组)', color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)
            bars2b = ax2.bar([i + width/2 for i in x], df_b['ndcg'], width, 
                            label='With Graph (实验组)', color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=0.5)
            ax2.set_title('NDCG Comparison (排序质量对比)', fontweight='bold', fontsize=14)
            ax2.set_ylabel('NDCG', fontsize=12)
            ax2.set_xlabel('Test Query', fontsize=12)
            ax2.set_xticks(x)
            ax2.set_xticklabels([f"Q{i+1}" for i in x], rotation=45, ha='right', fontsize=9)
            ax2.set_ylim([0, 1.1])
            ax2.legend(loc='upper right', fontsize=10)
            ax2.grid(axis='y', alpha=0.3, linestyle='--')
            
            # 添加数值标签
            for i, (bar_a, bar_b) in enumerate(zip(bars2a, bars2b)):
                height_a = bar_a.get_height()
                height_b = bar_b.get_height()
                ax2.text(bar_a.get_x() + bar_a.get_width()/2., height_a + 0.02,
                        f'{height_a:.2f}', ha='center', va='bottom', fontsize=8)
                ax2.text(bar_b.get_x() + bar_b.get_width()/2., height_b + 0.02,
                        f'{height_b:.2f}', ha='center', va='bottom', fontsize=8)
            
            # 3. 相关性改进幅度
            ax3 = axes[1, 0]
            if not df_comp.empty and 'avg_relevance_improvement' in df_comp.columns:
                colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in df_comp['avg_relevance_improvement']]
                bars3 = ax3.barh(range(len(df_comp)), df_comp['avg_relevance_improvement'], 
                                color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
                ax3.set_title('Relevance Improvement (相关性提升)', fontweight='bold', fontsize=14)
                ax3.set_xlabel('Improvement (提升幅度)', fontsize=12)
                ax3.set_ylabel('Test Query', fontsize=12)
                ax3.set_yticks(range(len(df_comp)))
                ax3.set_yticklabels([f"Q{i+1}" for i in range(len(df_comp))], fontsize=9)
                ax3.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
                ax3.grid(axis='x', alpha=0.3, linestyle='--')
                
                # 添加数值标签
                for i, (bar, val) in enumerate(zip(bars3, df_comp['avg_relevance_improvement'])):
                    width_bar = bar.get_width()
                    ax3.text(width_bar + 0.01 if width_bar > 0 else width_bar - 0.01, bar.get_y() + bar.get_height()/2,
                            f'{val:+.3f}', ha='left' if width_bar > 0 else 'right', va='center', fontsize=8)
            
            # 4. NDCG改进幅度
            ax4 = axes[1, 1]
            if not df_comp.empty and 'ndcg_improvement' in df_comp.columns:
                colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in df_comp['ndcg_improvement']]
                bars4 = ax4.barh(range(len(df_comp)), df_comp['ndcg_improvement'], 
                                color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
                ax4.set_title('NDCG Improvement (排序质量提升)', fontweight='bold', fontsize=14)
                ax4.set_xlabel('Improvement (提升幅度)', fontsize=12)
                ax4.set_ylabel('Test Query', fontsize=12)
                ax4.set_yticks(range(len(df_comp)))
                ax4.set_yticklabels([f"Q{i+1}" for i in range(len(df_comp))], fontsize=9)
                ax4.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
                ax4.grid(axis='x', alpha=0.3, linestyle='--')
                
                # 添加数值标签
                for i, (bar, val) in enumerate(zip(bars4, df_comp['ndcg_improvement'])):
                    width_bar = bar.get_width()
                    ax4.text(width_bar + 0.01 if width_bar > 0 else width_bar - 0.01, bar.get_y() + bar.get_height()/2,
                            f'{val:+.3f}', ha='left' if width_bar > 0 else 'right', va='center', fontsize=8)
            
            plt.tight_layout(rect=[0, 0, 1, 0.98])
            
            # 保存图片
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"graph_ab_test_v2_{timestamp}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"\n✅ Visualization saved to: {output_path}")
            plt.close()
            
        except Exception as e:
            print(f"❌ Error generating visualization: {e}")
            import traceback
            traceback.print_exc()
    
    def analyze_mode_differences(self, json_file: str = None):
        """分析三种模式差异评测结果"""
        # 查找最新的结果文件
        if json_file is None:
            json_files = list(self.results_dir.glob("mode_difference_*.json"))
            if not json_files:
                print("No mode difference evaluation results found!")
                return
            json_file = max(json_files, key=lambda p: p.stat().st_mtime)
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 处理不同的数据格式
        if isinstance(data, list):
            # 如果直接是列表，包装成字典格式
            results = {"queries": data}
        elif isinstance(data, dict):
            results = data
        else:
            print(f"Unexpected data format: {type(data)}")
            return
        
        print("\n" + "=" * 60)
        print("MODE DIFFERENCE ANALYSIS")
        print("=" * 60)
        
        queries = results.get("queries", [])
        if not queries:
            print("No query results found!")
            return
        
        # 分析每个query的三种模式表现
        for query_result in queries:
            query = query_result["query"]
            print(f"\n[Query: {query}]")
            print("-" * 60)
            
            for mode in ["review", "explain", "inspire"]:
                if mode in query_result["modes"]:
                    mode_data = query_result["modes"][mode]
                    char = mode_data.get("characteristics", {})
                    llm = mode_data.get("llm_evaluation", {})
                    
                    print(f"\n{mode.upper()} Mode:")
                    print(f"  Response Length: {mode_data.get('response_length', 0)} chars")
                    if isinstance(char, dict):
                        print(f"  Three Dimensions:")
                        print(f"    Comprehensiveness (全面性): {char.get('comprehensiveness', 0):.3f}")
                        print(f"    Novelty (新颖性): {char.get('novelty', 0):.3f}")
                        print(f"    Depth (深度特性): {char.get('depth', 0):.3f}")
                        print(f"  Average Score: {char.get('avg_score', 0):.3f}")
                    if isinstance(llm, dict):
                        print(f"  LLM Evaluation:")
                        print(f"    Mode Match: {llm.get('mode_match', 0):.3f}")
                        print(f"    Effectiveness: {llm.get('effectiveness', 0):.3f}")
                        print(f"    Unique Value: {llm.get('unique_value', 0):.3f}")
        
        # 打印汇总统计
        self._print_dimension_summary(queries)
        
        # 生成可视化
        self._visualize_mode_differences(results)
    
    def _print_dimension_summary(self, queries: List[Dict]):
        """打印三个维度的汇总统计"""
        print("\n" + "=" * 60)
        print("THREE DIMENSIONS SUMMARY")
        print("=" * 60)
        
        # 收集各模式的三个维度分数
        review_comp = []
        review_novel = []
        review_depth = []
        explain_comp = []
        explain_novel = []
        explain_depth = []
        inspire_comp = []
        inspire_novel = []
        inspire_depth = []
        
        for q in queries:
            if "review" in q.get("modes", {}):
                char = q["modes"]["review"].get("characteristics", {})
                review_comp.append(char.get("comprehensiveness", 0))
                review_novel.append(char.get("novelty", 0))
                review_depth.append(char.get("depth", 0))
            if "explain" in q.get("modes", {}):
                char = q["modes"]["explain"].get("characteristics", {})
                explain_comp.append(char.get("comprehensiveness", 0))
                explain_novel.append(char.get("novelty", 0))
                explain_depth.append(char.get("depth", 0))
            if "inspire" in q.get("modes", {}):
                char = q["modes"]["inspire"].get("characteristics", {})
                inspire_comp.append(char.get("comprehensiveness", 0))
                inspire_novel.append(char.get("novelty", 0))
                inspire_depth.append(char.get("depth", 0))
        
        # 计算平均值
        def avg(lst):
            return sum(lst) / len(lst) if lst else 0
        
        print("\n[Comprehensiveness (全面性)]")
        print(f"  Review:  {avg(review_comp):.3f}  ← Expected Highest")
        print(f"  Explain: {avg(explain_comp):.3f}")
        print(f"  Inspire: {avg(inspire_comp):.3f}")
        
        print("\n[Novelty (新颖性)]")
        print(f"  Review:  {avg(review_novel):.3f}")
        print(f"  Explain: {avg(explain_novel):.3f}")
        print(f"  Inspire: {avg(inspire_novel):.3f}  ← Expected Highest")
        
        print("\n[Depth (深度特性/逻辑性)]")
        print(f"  Review:  {avg(review_depth):.3f}")
        print(f"  Explain: {avg(explain_depth):.3f}  ← Expected Highest")
        print(f"  Inspire: {avg(inspire_depth):.3f}")
        
        # 检查是否符合预期
        print("\n[Differentiation Check]")
        comp_diff = avg(review_comp) - max(avg(explain_comp), avg(inspire_comp))
        novel_diff = avg(inspire_novel) - max(avg(review_novel), avg(explain_novel))
        depth_diff = avg(explain_depth) - max(avg(review_depth), avg(inspire_depth))
        
        if comp_diff > 0.05:
            print(f"  ✓ Review模式在全面性上领先 {comp_diff:.3f}")
        else:
            print(f"  ⚠ Review模式全面性优势不明显 ({comp_diff:.3f})")
        
        if novel_diff > 0.05:
            print(f"  ✓ Inspire模式在新颖性上领先 {novel_diff:.3f}")
        else:
            print(f"  ⚠ Inspire模式新颖性优势不明显 ({novel_diff:.3f})")
        
        if depth_diff > 0.05:
            print(f"  ✓ Explain模式在深度特性上领先 {depth_diff:.3f}")
        else:
            print(f"  ⚠ Explain模式深度特性优势不明显 ({depth_diff:.3f})")
    
    def _visualize_mode_differences(self, results: Dict):
        """可视化三种模式差异"""
        queries_list = results.get("queries", [])
        if not queries_list:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Three Modes Comparison: Review vs Explain vs Inspire', fontsize=16, fontweight='bold')
        
        queries = [q["query"] for q in queries_list]
        x = range(len(queries))
        width = 0.25
        
        # 1. 全面性对比（Comprehensiveness）
        ax1 = axes[0, 0]
        review_comp = []
        explain_comp = []
        inspire_comp = []
        
        for q in queries_list:
            if "review" in q.get("modes", {}):
                review_comp.append(q["modes"]["review"]["characteristics"].get("comprehensiveness", 0))
            if "explain" in q.get("modes", {}):
                explain_comp.append(q["modes"]["explain"]["characteristics"].get("comprehensiveness", 0))
            if "inspire" in q.get("modes", {}):
                inspire_comp.append(q["modes"]["inspire"]["characteristics"].get("comprehensiveness", 0))
        
        ax1.bar([i - width for i in x], review_comp, width, label='Review', color='#3498db', alpha=0.8)
        ax1.bar(x, explain_comp, width, label='Explain', color='#e74c3c', alpha=0.8)
        ax1.bar([i + width for i in x], inspire_comp, width, label='Inspire', color='#2ecc71', alpha=0.8)
        ax1.set_title('Comprehensiveness (全面性) - Review Should Be Highest', fontweight='bold')
        ax1.set_ylabel('Score')
        ax1.set_xlabel('Query Index')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"Q{i+1}" for i in x], rotation=45, ha='right')
        ax1.set_ylim([0, 1])
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. 新颖性对比（Novelty）
        ax2 = axes[0, 1]
        review_novel = []
        explain_novel = []
        inspire_novel = []
        
        for q in queries_list:
            if "review" in q.get("modes", {}):
                review_novel.append(q["modes"]["review"]["characteristics"].get("novelty", 0))
            if "explain" in q.get("modes", {}):
                explain_novel.append(q["modes"]["explain"]["characteristics"].get("novelty", 0))
            if "inspire" in q.get("modes", {}):
                inspire_novel.append(q["modes"]["inspire"]["characteristics"].get("novelty", 0))
        
        ax2.bar([i - width for i in x], review_novel, width, label='Review', color='#3498db', alpha=0.8)
        ax2.bar(x, explain_novel, width, label='Explain', color='#e74c3c', alpha=0.8)
        ax2.bar([i + width for i in x], inspire_novel, width, label='Inspire', color='#2ecc71', alpha=0.8)
        ax2.set_title('Novelty (新颖性) - Inspire Should Be Highest', fontweight='bold')
        ax2.set_ylabel('Score')
        ax2.set_xlabel('Query Index')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"Q{i+1}" for i in x], rotation=45, ha='right')
        ax2.set_ylim([0, 1])
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. 深度特性对比（Depth）
        ax3 = axes[1, 0]
        review_depth = []
        explain_depth = []
        inspire_depth = []
        
        for q in queries_list:
            if "review" in q.get("modes", {}):
                review_depth.append(q["modes"]["review"]["characteristics"].get("depth", 0))
            if "explain" in q.get("modes", {}):
                explain_depth.append(q["modes"]["explain"]["characteristics"].get("depth", 0))
            if "inspire" in q.get("modes", {}):
                inspire_depth.append(q["modes"]["inspire"]["characteristics"].get("depth", 0))
        
        ax3.bar([i - width for i in x], review_depth, width, label='Review', color='#3498db', alpha=0.8)
        ax3.bar(x, explain_depth, width, label='Explain', color='#e74c3c', alpha=0.8)
        ax3.bar([i + width for i in x], inspire_depth, width, label='Inspire', color='#2ecc71', alpha=0.8)
        ax3.set_title('Depth (深度特性/逻辑性) - Explain Should Be Highest', fontweight='bold')
        ax3.set_ylabel('Score')
        ax3.set_xlabel('Query Index')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f"Q{i+1}" for i in x], rotation=45, ha='right')
        ax3.set_ylim([0, 1])
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. 三维度综合对比（分组柱状图）
        ax4 = axes[1, 1]
        
        # 计算各模式在三个维度上的平均分
        avg_review_comp = sum(review_comp) / len(review_comp) if review_comp else 0
        avg_review_novel = sum(review_novel) / len(review_novel) if review_novel else 0
        avg_review_depth = sum(review_depth) / len(review_depth) if review_depth else 0
        
        avg_explain_comp = sum(explain_comp) / len(explain_comp) if explain_comp else 0
        avg_explain_novel = sum(explain_novel) / len(explain_novel) if explain_novel else 0
        avg_explain_depth = sum(explain_depth) / len(explain_depth) if explain_depth else 0
        
        avg_inspire_comp = sum(inspire_comp) / len(inspire_comp) if inspire_comp else 0
        avg_inspire_novel = sum(inspire_novel) / len(inspire_novel) if inspire_novel else 0
        avg_inspire_depth = sum(inspire_depth) / len(inspire_depth) if inspire_depth else 0
        
        # 绘制分组柱状图
        categories = ['Comprehensiveness', 'Novelty', 'Depth']
        review_scores = [avg_review_comp, avg_review_novel, avg_review_depth]
        explain_scores = [avg_explain_comp, avg_explain_novel, avg_explain_depth]
        inspire_scores = [avg_inspire_comp, avg_inspire_novel, avg_inspire_depth]
        
        x_pos = range(len(categories))
        ax4.bar([i - width for i in x_pos], review_scores, width, label='Review', color='#3498db', alpha=0.8)
        ax4.bar(x_pos, explain_scores, width, label='Explain', color='#e74c3c', alpha=0.8)
        ax4.bar([i + width for i in x_pos], inspire_scores, width, label='Inspire', color='#2ecc71', alpha=0.8)
        
        ax4.set_title('Three Dimensions Comparison (Average Across All Queries)', fontweight='bold')
        ax4.set_ylabel('Score')
        ax4.set_xlabel('Dimension')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(categories, rotation=0)
        ax4.set_ylim([0, 1])
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        
        # 添加期望标注
        ax4.text(0, 0.95, '↑ Review\nExpected', ha='center', fontsize=8, color='#3498db', weight='bold')
        ax4.text(1, 0.95, '↑ Inspire\nExpected', ha='center', fontsize=8, color='#2ecc71', weight='bold')
        ax4.text(2, 0.95, '↑ Explain\nExpected', ha='center', fontsize=8, color='#e74c3c', weight='bold')
        
        plt.tight_layout()
        
        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"mode_differences_{timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_path}")
        plt.close()
    
    def generate_report(self):
        """生成综合报告"""
        print("\n" + "=" * 60)
        print("GENERATING COMPREHENSIVE REPORT")
        print("=" * 60)
        
        # 分析模式差异（主要）
        self.analyze_mode_differences()
        
        # 分析图检索A/B测试结果
        self.analyze_graph_ab_results()
        
        print("\n" + "=" * 60)
        print("Report generation completed!")
        print("=" * 60)


def main():
    """主函数"""
    analyzer = ResultAnalyzer()
    analyzer.generate_report()


if __name__ == "__main__":
    main()
