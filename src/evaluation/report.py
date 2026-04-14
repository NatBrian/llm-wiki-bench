"""Report generator for comparative benchmark analysis.

This module takes metrics from both architectures, produces a comparative
Pandas DataFrame, saves to CSV, and pretty-prints to console.
"""

import csv
from pathlib import Path
from typing import List, Optional, Dict, Any

from .metrics import ArchitectureMetrics


class ReportGenerator:
    """Generates comparative reports from architecture metrics.
    
    Features:
    - Creates Pandas DataFrame comparing both architectures
    - Saves results to CSV in the results directory
    - Pretty-prints formatted comparison to console
    """
    
    def __init__(self, results_dir: Optional[Path] = None):
        """Initialize the report generator.
        
        Args:
            results_dir: Directory to save CSV reports. If None, uses project root/results/
        """
        if results_dir is None:
            # Default to project root / results
            from ..config import Config, get_config
            config = get_config()
            self.results_dir = config.PROJECT_ROOT / "results"
        else:
            self.results_dir = results_dir
        
        # Ensure results directory exists
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def create_comparison_dataframe(
        self,
        llm_wiki_metrics: ArchitectureMetrics,
        rag_metrics: ArchitectureMetrics
    ) -> Dict[str, Any]:
        """Create a comparative data structure (DataFrame-like) from metrics.
        
        Args:
            llm_wiki_metrics: Metrics from LLM-Wiki-Agent pipeline
            rag_metrics: Metrics from RAG pipeline
            
        Returns:
            Dictionary representing the comparison table data
        """
        # Convert both to dicts
        llm_dict = llm_wiki_metrics.to_dict()
        rag_dict = rag_metrics.to_dict()
        
        # Create comparison structure
        comparison = {
            "Metric": [
                "num_samples",
                "mean_score",
                "median_score", 
                "std_score",
                "min_score",
                "max_score",
                "mean_latency_seconds",
                "median_latency_seconds",
                "total_token_usage",
                "mean_token_usage",
                "mean_retrieval_count",
                "score_1_count",
                "score_2_count",
                "score_3_count",
                "score_4_count",
                "score_5_count",
            ],
            "llm-wiki-agent": [
                llm_dict.get("num_samples", 0),
                llm_dict.get("mean_score"),
                llm_dict.get("median_score"),
                llm_dict.get("std_score"),
                llm_dict.get("min_score"),
                llm_dict.get("max_score"),
                llm_dict.get("mean_latency_seconds"),
                llm_dict.get("median_latency_seconds"),
                llm_dict.get("total_token_usage", 0),
                llm_dict.get("mean_token_usage"),
                llm_dict.get("mean_retrieval_count"),
                llm_dict.get("score_1_count", 0),
                llm_dict.get("score_2_count", 0),
                llm_dict.get("score_3_count", 0),
                llm_dict.get("score_4_count", 0),
                llm_dict.get("score_5_count", 0),
            ],
            "rag": [
                rag_dict.get("num_samples", 0),
                rag_dict.get("mean_score"),
                rag_dict.get("median_score"),
                rag_dict.get("std_score"),
                rag_dict.get("min_score"),
                rag_dict.get("max_score"),
                rag_dict.get("mean_latency_seconds"),
                rag_dict.get("median_latency_seconds"),
                rag_dict.get("total_token_usage", 0),
                rag_dict.get("mean_token_usage"),
                rag_dict.get("mean_retrieval_count"),
                rag_dict.get("score_1_count", 0),
                rag_dict.get("score_2_count", 0),
                rag_dict.get("score_3_count", 0),
                rag_dict.get("score_4_count", 0),
                rag_dict.get("score_5_count", 0),
            ]
        }
        
        return comparison
    
    def save_to_csv(
        self,
        comparison_data: Dict[str, Any],
        filename: str = "benchmark_comparison.csv"
    ) -> Path:
        """Save comparison data to CSV file.
        
        Args:
            comparison_data: Dictionary from create_comparison_dataframe
            filename: Name of the CSV file
            
        Returns:
            Path to the saved CSV file
        """
        filepath = self.results_dir / filename
        
        # Get headers and data rows
        metrics = comparison_data["Metric"]
        llm_values = comparison_data["llm-wiki-agent"]
        rag_values = comparison_data["rag"]
        
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(["Metric", "llm-wiki-agent", "rag"])
            
            # Write data rows
            for i, metric in enumerate(metrics):
                writer.writerow([metric, llm_values[i], rag_values[i]])
        
        return filepath
    
    def print_comparison(
        self,
        comparison_data: Dict[str, Any],
        title: str = "Benchmark Comparison Report"
    ) -> str:
        """Pretty-print the comparison to console and return as string.
        
        Args:
            comparison_data: Dictionary from create_comparison_dataframe
            title: Title for the report
            
        Returns:
            Formatted string representation of the comparison
        """
        lines = []
        
        # Title
        lines.append("=" * 70)
        lines.append(f"{title:^70}")
        lines.append("=" * 70)
        lines.append("")
        
        # Header
        lines.append(f"{'Metric':<30} {'llm-wiki-agent':>18} {'rag':>18}")
        lines.append("-" * 70)
        
        # Data rows
        metrics = comparison_data["Metric"]
        llm_values = comparison_data["llm-wiki-agent"]
        rag_values = comparison_data["rag"]
        
        for i, metric in enumerate(metrics):
            llm_val = self._format_value(llm_values[i])
            rag_val = self._format_value(rag_values[i])
            lines.append(f"{metric:<30} {llm_val:>18} {rag_val:>18}")
        
        lines.append("-" * 70)
        lines.append("")
        
        # Summary section
        lines.append("SUMMARY:")
        lines.append("")
        
        # Compare mean scores
        llm_mean = comparison_data["llm-wiki-agent"][1]  # mean_score index
        rag_mean = comparison_data["rag"][1]
        
        if llm_mean is not None and rag_mean is not None:
            diff = llm_mean - rag_mean
            winner = "llm-wiki-agent" if diff > 0 else "rag" if diff < 0 else "tie"
            lines.append(f"  Mean Score Winner: {winner}")
            lines.append(f"    llm-wiki-agent: {llm_mean:.3f}")
            lines.append(f"    rag: {rag_mean:.3f}")
            lines.append(f"    Difference: {diff:+.3f}")
            lines.append("")
        
        # Compare latency
        llm_latency = comparison_data["llm-wiki-agent"][6]  # mean_latency index
        rag_latency = comparison_data["rag"][6]
        
        if llm_latency is not None and rag_latency is not None:
            faster = "llm-wiki-agent" if llm_latency < rag_latency else "rag"
            lines.append(f"  Faster Pipeline: {faster}")
            lines.append(f"    llm-wiki-agent: {llm_latency:.2f}s")
            lines.append(f"    rag: {rag_latency:.2f}s")
            lines.append("")
        
        # Compare token usage
        llm_tokens = comparison_data["llm-wiki-agent"][9]  # mean_token_usage index
        rag_tokens = comparison_data["rag"][9]
        
        if llm_tokens is not None and rag_tokens is not None:
            efficient = "llm-wiki-agent" if llm_tokens < rag_tokens else "rag"
            lines.append(f"  More Token-Efficient: {efficient}")
            lines.append(f"    llm-wiki-agent: {llm_tokens:.1f} tokens/sample")
            lines.append(f"    rag: {rag_tokens:.1f} tokens/sample")
            lines.append("")
        
        lines.append("=" * 70)
        
        output = "\n".join(lines)
        print(output)
        return output
    
    def _format_value(self, value: Any) -> str:
        """Format a value for display.
        
        Args:
            value: Value to format
            
        Returns:
            Formatted string
        """
        if value is None:
            return "N/A"
        elif isinstance(value, float):
            # Format floats with appropriate precision
            if value == int(value):
                return str(int(value))
            else:
                return f"{value:.3f}"
        else:
            return str(value)
    
    def generate_full_report(
        self,
        llm_wiki_metrics: ArchitectureMetrics,
        rag_metrics: ArchitectureMetrics,
        filename: str = "benchmark_comparison.csv",
        title: str = "LLM-vs-RAG Benchmark Comparison"
    ) -> tuple[Dict[str, Any], Path, str]:
        """Generate complete report: DataFrame, CSV, and console output.
        
        Args:
            llm_wiki_metrics: Metrics from LLM-Wiki-Agent
            rag_metrics: Metrics from RAG pipeline
            filename: CSV filename
            title: Report title
            
        Returns:
            Tuple of (comparison_data, csv_path, console_output)
        """
        # Create comparison data
        comparison_data = self.create_comparison_dataframe(
            llm_wiki_metrics,
            rag_metrics
        )
        
        # Save to CSV
        csv_path = self.save_to_csv(comparison_data, filename)
        
        # Print to console
        console_output = self.print_comparison(comparison_data, title)
        
        return comparison_data, csv_path, console_output
