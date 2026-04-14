"""Evaluation module for llm-vs-rag-bench.

This module provides:
1. LLM-as-Judge evaluator to score answers against ground truth
2. Metrics calculator for summary statistics
3. Report generator for comparative analysis
"""

from .judge import LLMJudge, JudgeResult
from .metrics import MetricsCalculator, ArchitectureMetrics
from .report import ReportGenerator

__all__ = [
    "LLMJudge",
    "JudgeResult", 
    "MetricsCalculator",
    "ArchitectureMetrics",
    "ReportGenerator",
]