"""LLM-as-Judge evaluator for scoring answers against ground truth.

This module implements a rigorous LLM-based judge that scores predicted answers
against ground truth on a 1-5 scale.
"""

import re
from dataclasses import dataclass
from typing import Optional

from ..llm_client import LLMClient, get_llm_client


@dataclass
class JudgeResult:
    """Result of an LLM-as-Judge evaluation.
    
    Attributes:
        score: Score from 1-5 (float to allow intermediate values)
        reasoning: Explanation of why this score was given
        question_id: ID of the question being evaluated
        predicted_answer: The answer that was scored
        ground_truth: The ground truth answer
    """
    score: float
    reasoning: str
    question_id: str
    predicted_answer: str
    ground_truth: str


class LLMJudge:
    """LLM-as-Judge evaluator that scores answers on a 1-5 scale.
    
    The judge uses a rigorous prompt that evaluates:
    - Factual accuracy compared to ground truth
    - Completeness of the answer
    - Relevance to the question
    
    Scores:
        5: Excellent - Answer is factually correct, complete, and matches ground truth
        4: Good - Answer is mostly correct with minor omissions or imprecisions
        3: Fair - Answer captures key points but has notable gaps or errors
        2: Poor - Answer has significant errors or misses key information
        1: Very Poor - Answer is incorrect, irrelevant, or empty
    """
    
    # Rigorous judge prompt template
    JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator tasked with scoring answers to questions based on their accuracy and completeness compared to a ground truth answer.

QUESTION:
{question}

GROUND TRUTH ANSWER:
{ground_truth}

PREDICTED ANSWER:
{predicted_answer}

Evaluate the predicted answer against the ground truth using these criteria:

1. FACTUAL ACCURACY: Does the predicted answer contain factually correct information that aligns with the ground truth?
2. COMPLETENESS: Does the predicted answer cover all key points mentioned in the ground truth?
3. RELEVANCE: Does the predicted answer directly address the question asked?

SCORING RUBRIC:
- Score 5 (Excellent): Answer is factually correct, complete, and essentially equivalent to ground truth. All key information is present and accurate.
- Score 4 (Good): Answer is mostly correct with minor omissions or slight imprecisions. Captures the main points but may miss some details.
- Score 3 (Fair): Answer captures some key points but has notable gaps, errors, or imprecisions. Partially aligns with ground truth.
- Score 2 (Poor): Answer has significant factual errors or misses most key information from the ground truth.
- Score 1 (Very Poor): Answer is completely incorrect, irrelevant, contradictory to ground truth, or empty.

First, provide your detailed reasoning comparing the predicted answer to the ground truth. Then, on a new line, output ONLY the score as a single number from 1 to 5.

Your response must follow this exact format:
REASONING: <your detailed analysis here>
SCORE: <single digit 1-5>

Begin your evaluation:"""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        """Initialize the LLM Judge.
        
        Args:
            llm_client: LLMClient instance. If None, uses the global client.
        """
        self.llm_client = llm_client or get_llm_client()
    
    def _parse_judge_response(self, response: str) -> tuple[float, str]:
        """Parse the LLM judge response to extract score and reasoning.
        
        Args:
            response: Raw response from the LLM judge
            
        Returns:
            Tuple of (score, reasoning)
            
        Raises:
            ValueError: If the response cannot be parsed
        """
        # Try to find SCORE: pattern first
        score_match = re.search(r'SCORE:\s*(\d)', response, re.IGNORECASE)
        
        if score_match:
            score = float(score_match.group(1))
            # Extract reasoning (everything before SCORE:)
            reasoning_match = re.search(r'REASONING:\s*(.+?)(?=SCORE:|$)', response, re.IGNORECASE | re.DOTALL)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
            else:
                # If no REASONING: tag, take everything before SCORE:
                reasoning = response[:score_match.start()].strip()
        else:
            # Fallback: try to find just a number 1-5 at the end
            number_match = re.search(r'(\b[1-5]\b)\s*$', response)
            if number_match:
                score = float(number_match.group(1))
                reasoning = response[:number_match.start()].strip()
            else:
                raise ValueError(f"Could not parse score from judge response: {response}")
        
        # Validate score range
        if not (1 <= score <= 5):
            raise ValueError(f"Score {score} is outside valid range 1-5")
        
        return score, reasoning
    
    def evaluate(
        self,
        question: str,
        predicted_answer: str,
        ground_truth: str,
        question_id: str = "unknown"
    ) -> JudgeResult:
        """Evaluate a predicted answer against ground truth.
        
        Args:
            question: The original question
            predicted_answer: The answer generated by a pipeline
            ground_truth: The ground truth answer
            question_id: Optional ID for tracking
            
        Returns:
            JudgeResult with score and reasoning
            
        Raises:
            ValueError: If the judge response cannot be parsed
        """
        # Handle edge cases
        if not predicted_answer or not predicted_answer.strip():
            return JudgeResult(
                score=1.0,
                reasoning="Predicted answer is empty.",
                question_id=question_id,
                predicted_answer=predicted_answer,
                ground_truth=ground_truth
            )
        
        if not ground_truth or not ground_truth.strip():
            return JudgeResult(
                score=3.0,  # Neutral score if no ground truth
                reasoning="Ground truth answer is empty, cannot properly evaluate.",
                question_id=question_id,
                predicted_answer=predicted_answer,
                ground_truth=ground_truth
            )
        
        # Build the judge prompt
        prompt = self.JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            ground_truth=ground_truth,
            predicted_answer=predicted_answer
        )
        
        # Call the LLM judge
        result = self.llm_client.call(
            prompt=prompt,
            max_tokens=1024,  # Enough for reasoning + score
            temperature=0.0   # Deterministic for consistent judging
        )
        
        # Parse the response
        score, reasoning = self._parse_judge_response(result.content)
        
        return JudgeResult(
            score=score,
            reasoning=reasoning,
            question_id=question_id,
            predicted_answer=predicted_answer,
            ground_truth=ground_truth
        )
    
    def evaluate_batch(
        self,
        evaluations: list[tuple[str, str, str, str]]
    ) -> list[JudgeResult]:
        """Evaluate multiple answers in batch.
        
        Args:
            evaluations: List of tuples (question, predicted_answer, ground_truth, question_id)
            
        Returns:
            List of JudgeResult objects
        """
        results = []
        for question, predicted, ground_truth, q_id in evaluations:
            try:
                result = self.evaluate(question, predicted, ground_truth, q_id)
                results.append(result)
            except Exception as e:
                # On error, record a failed evaluation with score 0
                results.append(JudgeResult(
                    score=0.0,
                    reasoning=f"Evaluation failed: {str(e)}",
                    question_id=q_id,
                    predicted_answer=predicted,
                    ground_truth=ground_truth
                ))
        return results
