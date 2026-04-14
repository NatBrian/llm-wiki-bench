"""Tracking module for LLM Wiki Agent.

Provides trajectory logging and metrics tracking for the wiki agent.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..data.models import Trajectory as TrajectoryDataclass


@dataclass
class ThoughtActionObservation:
    """Represents a single thought/action/observation cycle."""
    thought: str
    action: Optional[str] = None
    observation: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class QueryMetrics:
    """Metrics for a single query operation."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    retrieval_count: int = 0
    llm_calls: int = 0


class TrajectoryLogger:
    """Logs full trajectories for agent operations.
    
    Tracks every Thought/Action/Observation cycle during agent execution.
    Compatible with the original wiki-agent's single-shot LLM call pattern.
    """
    
    def __init__(self, log_dir: Optional[Path] = None):
        """Initialize the trajectory logger.
        
        Args:
            log_dir: Directory to store trajectory logs. If None, uses default.
        """
        self.log_dir = log_dir or Path(__file__).parent.parent.parent / "trajectories"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Current session tracking
        self.current_cycles: list[ThoughtActionObservation] = []
        self.current_metrics = QueryMetrics()
        self.current_question_id: Optional[str] = None
    
    def start_query(self, question_id: str) -> None:
        """Start tracking a new query.
        
        Args:
            question_id: ID of the question being processed
        """
        self.current_question_id = question_id
        self.current_cycles = []
        self.current_metrics = QueryMetrics()
    
    def log_cycle(
        self,
        thought: str,
        action: Optional[str] = None,
        observation: Optional[str] = None
    ) -> None:
        """Log a thought/action/observation cycle.
        
        For the original wiki-agent's single-shot pattern, this is called once
        with the full prompt as 'thought' and the LLM response as 'observation'.
        
        Args:
            thought: The agent's thought/prompt
            action: Optional action taken (if any)
            observation: The observation/result from the action
        """
        cycle = ThoughtActionObservation(
            thought=thought,
            action=action,
            observation=observation
        )
        self.current_cycles.append(cycle)
    
    def update_metrics(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: float = 0.0,
        retrieval_count: int = 0
    ) -> None:
        """Update cumulative metrics for the current query.
        
        Args:
            prompt_tokens: Number of prompt tokens used
            completion_tokens: Number of completion tokens generated
            latency_ms: Latency in milliseconds
            retrieval_count: Number of documents/pages retrieved
        """
        self.current_metrics.prompt_tokens += prompt_tokens
        self.current_metrics.completion_tokens += completion_tokens
        self.current_metrics.total_tokens += prompt_tokens + completion_tokens
        self.current_metrics.latency_ms += latency_ms
        self.current_metrics.retrieval_count = max(
            self.current_metrics.retrieval_count, retrieval_count
        )
        self.current_metrics.llm_calls += 1
    
    def end_query(self) -> tuple[list[dict], QueryMetrics]:
        """End the current query and return trajectory + metrics.
        
        Returns:
            Tuple of (trajectory messages, metrics)
        """
        # Convert cycles to OpenAI message format for trajectory
        messages = []
        for cycle in self.current_cycles:
            messages.append({
                "role": "user",
                "content": cycle.thought,
                "timestamp": cycle.timestamp
            })
            if cycle.action:
                messages.append({
                    "role": "assistant",
                    "content": f"[ACTION] {cycle.action}",
                    "timestamp": cycle.timestamp
                })
            if cycle.observation:
                messages.append({
                    "role": "user",
                    "content": f"[OBSERVATION] {cycle.observation}",
                    "timestamp": cycle.timestamp
                })
        
        return messages, self.current_metrics
    
    def save_trajectory(
        self,
        question_id: str,
        messages: list[dict],
        metrics: QueryMetrics,
        answer: str,
        metadata: Optional[dict] = None
    ) -> Path:
        """Save trajectory to disk.
        
        Args:
            question_id: ID of the question
            messages: List of conversation messages
            metrics: Query metrics
            answer: Final answer generated
            metadata: Additional metadata
            
        Returns:
            Path to saved trajectory file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{question_id}_{timestamp}.json"
        filepath = self.log_dir / filename
        
        trajectory_data = {
            "question_id": question_id,
            "timestamp": timestamp,
            "messages": messages,
            "metrics": {
                "prompt_tokens": metrics.prompt_tokens,
                "completion_tokens": metrics.completion_tokens,
                "total_tokens": metrics.total_tokens,
                "latency_ms": metrics.latency_ms,
                "retrieval_count": metrics.retrieval_count,
                "llm_calls": metrics.llm_calls
            },
            "answer": answer,
            "metadata": metadata or {}
        }
        
        filepath.write_text(json.dumps(trajectory_data, indent=2))
        return filepath
    
    def to_dataclass(
        self,
        question_id: str,
        messages: list[dict],
        metadata: Optional[dict] = None
    ) -> TrajectoryDataclass:
        """Convert trajectory to our standard Trajectory dataclass.
        
        Args:
            question_id: ID of the question
            messages: List of conversation messages
            metadata: Additional metadata
            
        Returns:
            Trajectory dataclass instance
        """
        return TrajectoryDataclass(
            question_id=question_id,
            messages=messages,
            metadata=metadata or {}
        )
