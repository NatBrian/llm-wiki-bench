"""Test script for the evaluation module.

This script creates mock BenchmarkResults, runs the evaluator and metrics 
calculator, and verifies the output.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.models import BenchmarkResult
from src.evaluation.judge import LLMJudge, JudgeResult
from src.evaluation.metrics import MetricsCalculator, ArchitectureMetrics
from src.evaluation.report import ReportGenerator


def create_mock_benchmark_results() -> tuple[list[BenchmarkResult], list[BenchmarkResult]]:
    """Create mock benchmark results for testing.
    
    Returns:
        Tuple of (llm_wiki_results, rag_results)
    """
    # Mock results for LLM-Wiki-Agent
    llm_wiki_results = [
        BenchmarkResult(
            pipeline_name="llm-wiki-agent",
            question_id="q001",
            predicted_answer="The patient has hypertension and diabetes.",
            latency_seconds=2.5,
            token_usage=150,
            retrieval_count=3,
            score=4.5,
            trajectory={"steps": ["search", "retrieve", "answer"]}
        ),
        BenchmarkResult(
            pipeline_name="llm-wiki-agent",
            question_id="q002",
            predicted_answer="The treatment involves medication and lifestyle changes.",
            latency_seconds=3.2,
            token_usage=200,
            retrieval_count=5,
            score=4.0,
        ),
        BenchmarkResult(
            pipeline_name="llm-wiki-agent",
            question_id="q003",
            predicted_answer="The diagnosis is confirmed through blood tests.",
            latency_seconds=1.8,
            token_usage=120,
            retrieval_count=2,
            score=5.0,
        ),
        BenchmarkResult(
            pipeline_name="llm-wiki-agent",
            question_id="q004",
            predicted_answer="Side effects include nausea and fatigue.",
            latency_seconds=2.1,
            token_usage=100,
            retrieval_count=2,
            score=3.5,
        ),
        BenchmarkResult(
            pipeline_name="llm-wiki-agent",
            question_id="q005",
            predicted_answer="The prognosis is generally positive with treatment.",
            latency_seconds=2.8,
            token_usage=180,
            retrieval_count=4,
            score=4.0,
        ),
    ]
    
    # Mock results for RAG
    rag_results = [
        BenchmarkResult(
            pipeline_name="rag",
            question_id="q001",
            predicted_answer="Patient has high blood pressure.",
            latency_seconds=1.2,
            token_usage=80,
            retrieval_count=2,
            score=3.0,
        ),
        BenchmarkResult(
            pipeline_name="rag",
            question_id="q002",
            predicted_answer="Treatment includes drugs.",
            latency_seconds=0.9,
            token_usage=60,
            retrieval_count=1,
            score=2.5,
        ),
        BenchmarkResult(
            pipeline_name="rag",
            question_id="q003",
            predicted_answer="Diagnosis is done via tests.",
            latency_seconds=1.1,
            token_usage=70,
            retrieval_count=2,
            score=3.0,
        ),
        BenchmarkResult(
            pipeline_name="rag",
            question_id="q004",
            predicted_answer="There may be some side effects.",
            latency_seconds=0.8,
            token_usage=50,
            retrieval_count=1,
            score=2.0,
        ),
        BenchmarkResult(
            pipeline_name="rag",
            question_id="q005",
            predicted_answer="Outlook is good.",
            latency_seconds=1.0,
            token_usage=65,
            retrieval_count=1,
            score=3.5,
        ),
    ]
    
    return llm_wiki_results, rag_results


def test_judge_response_parsing():
    """Test the judge response parsing logic."""
    print("=" * 70)
    print("TEST 1: Judge Response Parsing")
    print("=" * 70)
    
    judge = LLMJudge.__new__(LLMJudge)  # Create without init for unit testing
    
    # Test standard format
    response1 = """REASONING: The answer captures the main points but misses some details about the specific medications mentioned in the ground truth.
SCORE: 4"""
    score1, reasoning1 = judge._parse_judge_response(response1)
    assert score1 == 4.0, f"Expected 4.0, got {score1}"
    assert "main points" in reasoning1, f"Reasoning not extracted properly: {reasoning1}"
    print(f"✓ Standard format parsed correctly: score={score1}")
    
    # Test lowercase format
    response2 = """reasoning: Answer is mostly correct.
score: 3"""
    score2, reasoning2 = judge._parse_judge_response(response2)
    assert score2 == 3.0, f"Expected 3.0, got {score2}"
    print(f"✓ Lowercase format parsed correctly: score={score2}")
    
    # Test fallback (just number at end)
    response3 = """The answer is partially correct but misses key information.
4"""
    score3, reasoning3 = judge._parse_judge_response(response3)
    assert score3 == 4.0, f"Expected 4.0, got {score3}"
    print(f"✓ Fallback format parsed correctly: score={score3}")
    
    # Test error case
    try:
        response4 = "Invalid response without score"
        judge._parse_judge_response(response4)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Error handling works: {e}")
    
    print()


def test_metrics_calculator():
    """Test the metrics calculator."""
    print("=" * 70)
    print("TEST 2: Metrics Calculator")
    print("=" * 70)
    
    llm_wiki_results, rag_results = create_mock_benchmark_results()
    calculator = MetricsCalculator()
    
    # Calculate metrics for LLM-Wiki-Agent
    llm_metrics = calculator.calculate_architecture_metrics(llm_wiki_results)
    
    print(f"LLM-Wiki-Agent Metrics:")
    print(f"  num_samples: {llm_metrics.num_samples}")
    print(f"  mean_score: {llm_metrics.mean_score:.3f}")
    print(f"  median_score: {llm_metrics.median_score}")
    print(f"  std_score: {llm_metrics.std_score:.3f}")
    print(f"  min_score: {llm_metrics.min_score}")
    print(f"  max_score: {llm_metrics.max_score}")
    print(f"  mean_latency: {llm_metrics.mean_latency_seconds:.3f}s")
    print(f"  total_tokens: {llm_metrics.total_token_usage}")
    print(f"  mean_retrieval: {llm_metrics.mean_retrieval_count:.1f}")
    print(f"  score_distribution: {llm_metrics.score_distribution}")
    
    # Verify calculations
    assert llm_metrics.num_samples == 5, f"Expected 5 samples, got {llm_metrics.num_samples}"
    assert llm_metrics.mean_score is not None, "Mean score should not be None"
    assert 1 <= llm_metrics.mean_score <= 5, f"Mean score {llm_metrics.mean_score} out of range"
    assert llm_metrics.total_token_usage == 750, f"Expected 750 tokens, got {llm_metrics.total_token_usage}"
    
    print("✓ LLM-Wiki-Agent metrics calculated correctly")
    print()
    
    # Calculate metrics for RAG
    rag_metrics = calculator.calculate_architecture_metrics(rag_results)
    
    print(f"RAG Metrics:")
    print(f"  num_samples: {rag_metrics.num_samples}")
    print(f"  mean_score: {rag_metrics.mean_score:.3f}")
    print(f"  mean_latency: {rag_metrics.mean_latency_seconds:.3f}s")
    print(f"  total_tokens: {rag_metrics.total_token_usage}")
    
    assert rag_metrics.num_samples == 5, f"Expected 5 samples, got {rag_metrics.num_samples}"
    print("✓ RAG metrics calculated correctly")
    print()
    
    # Test calculate_all_metrics
    llm_all, rag_all = calculator.calculate_all_metrics(llm_wiki_results, rag_results)
    assert llm_all.pipeline_name == "llm-wiki-agent"
    assert rag_all.pipeline_name == "rag"
    print("✓ calculate_all_metrics works correctly")
    print()


def test_report_generator():
    """Test the report generator."""
    print("=" * 70)
    print("TEST 3: Report Generator")
    print("=" * 70)
    
    llm_wiki_results, rag_results = create_mock_benchmark_results()
    calculator = MetricsCalculator()
    llm_metrics, rag_metrics = calculator.calculate_all_metrics(llm_wiki_results, rag_results)
    
    # Create report generator with temp directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        from pathlib import Path
        generator = ReportGenerator(results_dir=Path(tmpdir))
        
        # Generate full report
        comparison_data, csv_path, console_output = generator.generate_full_report(
            llm_metrics,
            rag_metrics,
            filename="test_comparison.csv",
            title="Test Benchmark Comparison"
        )
        
        # Verify comparison data structure
        assert "Metric" in comparison_data
        assert "llm-wiki-agent" in comparison_data
        assert "rag" in comparison_data
        assert len(comparison_data["Metric"]) == 16
        print("✓ Comparison data structure is correct")
        
        # Verify CSV was created
        assert csv_path.exists(), f"CSV file not created at {csv_path}"
        print(f"✓ CSV saved to: {csv_path}")
        
        # Read and verify CSV content
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 17, f"Expected 17 lines (header + 16 metrics), got {len(lines)}"
            assert lines[0].strip() == "Metric,llm-wiki-agent,rag"
        print("✓ CSV content verified")
        
        # Verify console output contains expected sections
        assert "SUMMARY:" in console_output
        assert "Mean Score Winner:" in console_output or "N/A" in console_output
        print("✓ Console output generated correctly")
        
        print()
        print("Console Output Preview:")
        print("-" * 70)
        # Print first 20 lines of output
        preview_lines = console_output.split('\n')[:20]
        for line in preview_lines:
            print(line)
        print("...")
        print()


def test_empty_results():
    """Test handling of empty results."""
    print("=" * 70)
    print("TEST 4: Empty Results Handling")
    print("=" * 70)
    
    calculator = MetricsCalculator()
    
    # Test with empty list
    empty_metrics = calculator.calculate_architecture_metrics([])
    assert empty_metrics.num_samples == 0
    assert empty_metrics.mean_score is None
    print("✓ Empty results handled correctly")
    
    # Test with results that have no scores
    from src.data.models import BenchmarkResult
    no_score_results = [
        BenchmarkResult(
            pipeline_name="test",
            question_id="q001",
            predicted_answer="test",
            latency_seconds=1.0,
            token_usage=100,
            retrieval_count=1,
            score=None  # No score
        )
    ]
    no_score_metrics = calculator.calculate_architecture_metrics(no_score_results)
    assert no_score_metrics.num_samples == 1
    assert no_score_metrics.mean_score is None
    print("✓ Results with no scores handled correctly")
    print()


def main():
    """Run all tests."""
    print("\n")
    print("*" * 70)
    print("* EVALUATION MODULE TEST SUITE")
    print("*" * 70)
    print()
    
    try:
        test_judge_response_parsing()
        test_metrics_calculator()
        test_report_generator()
        test_empty_results()
        
        print("*" * 70)
        print("* ALL TESTS PASSED!")
        print("*" * 70)
        print()
        
        # Final demonstration with mock data
        print("=" * 70)
        print("FINAL DEMONSTRATION: Full Pipeline")
        print("=" * 70)
        print()
        
        llm_wiki_results, rag_results = create_mock_benchmark_results()
        
        print(f"Created {len(llm_wiki_results)} LLM-Wiki-Agent results")
        print(f"Created {len(rag_results)} RAG results")
        print()
        
        calculator = MetricsCalculator()
        llm_metrics, rag_metrics = calculator.calculate_all_metrics(llm_wiki_results, rag_results)
        
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path
            generator = ReportGenerator(results_dir=Path(tmpdir))
            comparison_data, csv_path, _ = generator.generate_full_report(
                llm_metrics,
                rag_metrics,
                title="Final Demonstration: LLM-vs-RAG Benchmark"
            )
            
            print(f"\nReport saved to: {csv_path}")
        
        print()
        print("Evaluation module is working correctly!")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
