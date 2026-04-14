"""Metrics calculator for benchmark results.

This module computes summary statistics from BenchmarkResult objects,
including mean/median scores, latency, token usage, and retrieval counts.
"""

import statistics
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from .judge import JudgeResult
from ..data.models import BenchmarkResult


@dataclass
class ArchitectureMetrics:
    """Summary metrics for a single architecture/pipeline.
    
    Attributes:
        pipeline_name: Name of the pipeline (e.g., 'llm-wiki-agent', 'rag')
        num_samples: Number of benchmark results
        mean_score: Mean LLM-as-Judge score (1-5)
        median_score: Median LLM-as-Judge score
        std_score: Standard deviation of scores
        min_score: Minimum score
        max_score: Maximum score
        mean_latency_seconds: Mean latency in seconds
        median_latency_seconds: Median latency in seconds
        total_token_usage: Total tokens used across all samples
        mean_token_usage: Mean tokens per sample
        mean_retrieval_count: Mean number of retrieved documents/pages
        score_distribution: Count of each score (1-5)
    """
    pipeline_name: str
    num_samples: int = 0
    mean_score: Optional[float] = None
    median_score: Optional[float] = None
    std_score: Optional[float] = None
    min_score: Optional[float] = None
    max_score: Optional[float] = None
    mean_latency_seconds: Optional[float] = None
    median_latency_seconds: Optional[float] = None
    total_token_usage: int = 0
    mean_token_usage: Optional[float] = None
    mean_retrieval_count: Optional[float] = None
    score_distribution: Dict[int, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a dictionary for DataFrame creation."""
        return {
            "pipeline_name": self.pipeline_name,
            "num_samples": self.num_samples,
            "mean_score": self.mean_score,
            "median_score": self.median_score,
            "std_score": self.std_score,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "mean_latency_seconds": self.mean_latency_seconds,
            "median_latency_seconds": self.median_latency_seconds,
            "total_token_usage": self.total_token_usage,
            "mean_token_usage": self.mean_token_usage,
            "mean_retrieval_count": self.mean_retrieval_count,
            "score_1_count": self.score_distribution.get(1, 0),
            "score_2_count": self.score_distribution.get(2, 0),
            "score_3_count": self.score_distribution.get(3, 0),
            "score_4_count": self.score_distribution.get(4, 0),
            "score_5_count": self.score_distribution.get(5, 0),
        }


class MetricsCalculator:
    """Calculates summary statistics from BenchmarkResult objects.
    
    Computes metrics for:
    - LLM-as-Judge scores (mean, median, std, min, max, distribution)
    - Latency (mean, median)
    - Token usage (total, mean)
    - Retrieval count (mean)
    """
    
    def __init__(self):
        """Initialize the metrics calculator."""
        pass
    
    def _calculate_score_metrics(self, scores: List[float]) -> Dict[str, float]:
        """Calculate statistical metrics for a list of scores.
        
        Args:
            scores: List of score values
            
        Returns:
            Dictionary with mean, median, std, min, max
        """
        if not scores:
            return {
                "mean": None,
                "median": None,
                "std": None,
                "min": None,
                "max": None
            }
        
        # Filter out None values
        valid_scores = [s for s in scores if s is not None]
        
        if not valid_scores:
            return {
                "mean": None,
                "median": None,
                "std": None,
                "min": None,
                "max": None
            }
        
        mean_val = statistics.mean(valid_scores)
        median_val = statistics.median(valid_scores)
        std_val = statistics.stdev(valid_scores) if len(valid_scores) > 1 else 0.0
        min_val = min(valid_scores)
        max_val = max(valid_scores)
        
        return {
            "mean": mean_val,
            "median": median_val,
            "std": std_val,
            "min": min_val,
            "max": max_val
        }
    
    def _calculate_score_distribution(self, scores: List[float]) -> Dict[int, int]:
        """Count occurrences of each score value (rounded to nearest integer).
        
        Args:
            scores: List of score values
            
        Returns:
            Dictionary mapping score (1-5) to count
        """
        distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        
        for score in scores:
            if score is not None:
                # Round to nearest integer for distribution
                rounded = round(score)
                # Clamp to valid range
                rounded = max(1, min(5, rounded))
                distribution[rounded] += 1
        
        return distribution
    
    def calculate_architecture_metrics(
        self, 
        results: List[BenchmarkResult],
        pipeline_name: Optional[str] = None
    ) -> ArchitectureMetrics:
        """Calculate summary metrics for a list of benchmark results.
        
        Args:
            results: List of BenchmarkResult objects for one architecture
            pipeline_name: Optional pipeline name override (uses first result's name if not provided)
            
        Returns:
            ArchitectureMetrics with all computed statistics
        """
        if not results:
            return ArchitectureMetrics(
                pipeline_name=pipeline_name or "unknown",
                num_samples=0
            )
        
        # Extract values from results
        scores = [r.score for r in results if r.score is not None]
        latencies = [r.latency_seconds for r in results if r.latency_seconds is not None]
        token_usages = [r.token_usage for r in results if r.token_usage is not None]
        retrieval_counts = [r.retrieval_count for r in results if r.retrieval_count is not None]
        
        # Calculate score metrics
        score_metrics = self._calculate_score_metrics(scores)
        score_distribution = self._calculate_score_distribution(scores)
        
        # Calculate latency metrics
        latency_metrics = self._calculate_score_metrics(latencies)
        
        # Calculate token usage
        total_tokens = sum(token_usages) if token_usages else 0
        mean_tokens = statistics.mean(token_usages) if token_usages else None
        
        # Calculate retrieval count
        mean_retrieval = statistics.mean(retrieval_counts) if retrieval_counts else None
        
        # Determine pipeline name
        actual_pipeline_name = pipeline_name or results[0].pipeline_name
        
        return ArchitectureMetrics(
            pipeline_name=actual_pipeline_name,
            num_samples=len(results),
            mean_score=score_metrics["mean"],
            median_score=score_metrics["median"],
            std_score=score_metrics["std"],
            min_score=score_metrics["min"],
            max_score=score_metrics["max"],
            mean_latency_seconds=latency_metrics["mean"],
            median_latency_seconds=latency_metrics["median"],
            total_token_usage=total_tokens,
            mean_token_usage=mean_tokens,
            mean_retrieval_count=mean_retrieval,
            score_distribution=score_distribution
        )
    
    def calculate_all_metrics(
        self,
        llm_wiki_results: List[BenchmarkResult],
        rag_results: List[BenchmarkResult]
    ) -> tuple[ArchitectureMetrics, ArchitectureMetrics]:
        """Calculate metrics for both architectures.
        
        Args:
            llm_wiki_results: Results from LLM-Wiki-Agent pipeline
            rag_results: Results from RAG pipeline
            
        Returns:
            Tuple of (llm_wiki_metrics, rag_metrics)
        """
        llm_wiki_metrics = self.calculate_architecture_metrics(
            llm_wiki_results, 
            "llm-wiki-agent"
        )
        rag_metrics = self.calculate_architecture_metrics(
            rag_results,
            "rag"
        )
        
        return llm_wiki_metrics, rag_metrics
