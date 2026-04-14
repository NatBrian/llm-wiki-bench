"""Wiki Query module - adapted from llm-wiki-agent tools/query.py.

This module faithfully preserves the original query logic while adapting:
- LLM calls use src/llm_client.py instead of direct litellm
- Query interface accepts Question dataclass and returns BenchmarkResult/Trajectory
- Added tracking for token usage, latency, and retrieval count
"""

import json
import re
from datetime import date
from pathlib import Path
from typing import Optional

from ..llm_client import LLMClient, CallResult
from ..data.models import Question, BenchmarkResult, Trajectory
from .tracking import TrajectoryLogger


def read_file(path: Path) -> str:
    """Read file content, returns empty string if not exists."""
    return path.read_text(encoding="utf-8") if path.exists() else ""


def write_file(path: Path, content: str):
    """Create parent directories if needed and write content to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def find_relevant_pages(
    question: str,
    index_content: str,
    wiki_dir: Path,
    graph_dir: Optional[Path] = None
) -> list[Path]:
    """Extract linked pages from index that seem relevant to the question.

    Faithfully preserves the original keyword-based relevance matching:
    1. Pull all [[links]] and markdown links from index
    2. Match question keywords against page titles:
       - English: Check words > 3 chars
       - Exact substring match for short titles (CJK support)
       - CJK chunks: Contiguous non-ASCII characters
    3. Graph-based expansion: Find neighbors of matched pages via graph edges
    4. Always include overview.md
    5. Cap at 15 pages

    Args:
        question: The question text
        index_content: Content of wiki/index.md
        wiki_dir: Path to wiki directory
        graph_dir: Optional path to graph directory for neighbor expansion

    Returns:
        List of relevant page paths
    """
    # Pull all [[links]] and markdown links from index
    md_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', index_content)
    question_lower = question.lower()
    relevant = []

    for title, href in md_links:
        title_lower = title.lower()
        match = False

        # 1. English/Space-separated: check words > 3 chars
        if any(word in question_lower for word in title_lower.split() if len(word) > 3):
            match = True
        # 2. Exact substring match for the whole title (useful for short CJK titles)
        elif len(title_lower) >= 2 and title_lower in question_lower:
            match = True
        # 3. CJK chunks: find contiguous non-ASCII characters (len >= 2) and check if in question
        elif any(chunk in question_lower for chunk in re.findall(r'[^\x00-\x7F]{2,}', title_lower)):
            match = True

        if match:
            p = wiki_dir / href
            if p.exists() and p not in relevant:
                relevant.append(p)

    # Graph-based expansion: find neighbors of matched pages
    if graph_dir is None:
        graph_dir = wiki_dir.parent / "graph"

    graph_json = graph_dir / "graph.json"
    if graph_json.exists() and relevant:
        try:
            graph_data = json.loads(graph_json.read_text())
            page_ids = {
                p.relative_to(wiki_dir).as_posix().replace('.md', '')
                for p in relevant
            }
            neighbors = set()
            for edge in graph_data.get('edges', []):
                if edge.get('confidence', 0) >= 0.7:
                    if edge['from'] in page_ids:
                        neighbors.add(edge['to'])
                    elif edge['to'] in page_ids:
                        neighbors.add(edge['from'])
            for nid in neighbors:
                np = wiki_dir / f"{nid}.md"
                if np.exists() and np not in relevant:
                    relevant.append(np)
        except (json.JSONDecodeError, KeyError):
            pass

    # Always include overview
    overview = wiki_dir / "overview.md"
    if overview.exists() and overview not in relevant:
        relevant.insert(0, overview)

    return relevant[:15]  # cap to avoid context overflow


class WikiQuerier:
    """Queries the LLM Wiki to synthesize answers.
    
    Faithfully adapted from the original llm-wiki-agent tools/query.py.
    
    The original agent uses this workflow:
    1. Read wiki/index.md
    2. Find relevant pages via keyword matching
    3. Fallback to LLM-based selection if keyword match fails
    4. Read all relevant pages
    5. Call LLM with schema + pages + question
    6. Request well-structured markdown answer with wikilink citations
    7. Optionally save to wiki/syntheses/
    8. Append to log
    
    This adaptation preserves that exact logic while:
    - Using our LLMClient for API calls
    - Tracking token usage and latency per query
    - Accepting Question dataclass as input
    - Returning BenchmarkResult and Trajectory dataclasses
    """
    
    def __init__(
        self,
        wiki_dir: Optional[Path] = None,
        schema_file: Optional[Path] = None,
        graph_dir: Optional[Path] = None,
        client: Optional[LLMClient] = None,
        trajectory_logger: Optional[TrajectoryLogger] = None
    ):
        """Initialize the WikiQuerier.

        Args:
            wiki_dir: Directory for wiki files. Defaults to project root / wiki
            schema_file: Path to CLAUDE.md schema file. If None, uses default
            graph_dir: Directory for graph files. Defaults to project root / graph
            client: LLMClient instance. If None, creates new one
            trajectory_logger: Logger for tracking. If None, creates new one
        """
        # Set up paths
        self.repo_root = Path(__file__).parent.parent.parent
        self.wiki_dir = wiki_dir or self.repo_root / "wiki"
        self.graph_dir = graph_dir or self.repo_root / "graph"
        self.index_file = self.wiki_dir / "index.md"
        self.log_file = self.wiki_dir / "log.md"
        self.syntheses_dir = self.wiki_dir / "syntheses"
        
        # Schema file - check both locations
        if schema_file:
            self.schema_file = schema_file
        elif (self.repo_root / "CLAUDE.md").exists():
            self.schema_file = self.repo_root / "CLAUDE.md"
        elif (self.repo_root / "src" / "llm_wiki" / "CLAUDE.md").exists():
            self.schema_file = self.repo_root / "src" / "llm_wiki" / "CLAUDE.md"
        else:
            self.schema_file = None
        
        # Initialize clients
        self.client = client or LLMClient()
        self.trajectory_logger = trajectory_logger or TrajectoryLogger()
        
        # Create syntheses directory
        self.syntheses_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_schema(self) -> str:
        """Get schema content.
        
        Returns:
            Schema string from CLAUDE.md or default
        """
        if self.schema_file and self.schema_file.exists():
            return read_file(self.schema_file)
        return self._get_default_schema()
    
    def _select_pages_via_llm(
        self,
        question: str,
        index_content: str
    ) -> list[Path]:
        """Fallback: Ask LLM to select relevant pages from index.
        
        Used when keyword matching finds ≤1 page.
        
        Args:
            question: The question text
            index_content: Content of wiki/index.md
            
        Returns:
            List of selected page paths
        """
        print("  selecting relevant pages via API...")
        prompt = (
            f"Given this wiki index:\n\n{index_content}\n\n"
            f"Which pages are most relevant to answering: \"{question}\"\n\n"
            f"Return ONLY a JSON array of relative file paths (as listed in the index), "
            f"e.g. [\"sources/foo.md\", \"concepts/Bar.md\"]. Maximum 10 pages."
        )
        
        # Log the cycle
        self.trajectory_logger.log_cycle(thought=prompt, action="select_pages")
        
        result: CallResult = self.client.call(
            prompt=prompt,
            max_tokens=512,
            model=self.client.default_model_fast
        )
        
        # Update metrics
        self.trajectory_logger.update_metrics(
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            latency_ms=result.latency_ms
        )
        
        # Parse response
        raw = result.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        
        try:
            paths = json.loads(raw)
            relevant_pages = [
                self.wiki_dir / p for p in paths 
                if (self.wiki_dir / p).exists()
            ]
            
            # Log observation
            self.trajectory_logger.log_cycle(
                thought="",
                action="pages_selected",
                observation=f"Selected {len(relevant_pages)} pages"
            )
            
            return relevant_pages
        except (json.JSONDecodeError, TypeError):
            return []
    
    def query(
        self,
        question_text: str,
        question_id: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> tuple[str, dict]:
        """Query the wiki to synthesize an answer.
        
        Faithfully preserves the original query workflow:
        1. Read wiki/index.md
        2. Find relevant pages via keyword matching
        3. Fallback to LLM-based selection if needed
        4. Read all relevant pages
        5. Call LLM with schema + pages + question
        6. Request well-structured markdown answer with wikilink citations
        7. Optionally save to wiki/syntheses/
        8. Append to log
        
        Args:
            question_text: The question to ask
            question_id: Optional ID for tracking
            save_path: Optional path to save answer (relative to wiki/syntheses/)
            
        Returns:
            Tuple of (answer_string, metadata_dict)
        """
        today = date.today().isoformat()
        
        # Start trajectory tracking
        if question_id:
            self.trajectory_logger.start_query(question_id)
        
        # Step 1: Read index
        index_content = read_file(self.index_file)
        if not index_content:
            return "Wiki is empty. Ingest some sources first.", {"error": "wiki_empty"}
        
        # Step 2: Find relevant pages via keyword matching + graph expansion
        relevant_pages = find_relevant_pages(
            question_text, index_content, self.wiki_dir, self.graph_dir
        )
        
        # Fallback to LLM-based selection if no keyword match
        if not relevant_pages or len(relevant_pages) <= 1:
            relevant_pages = self._select_pages_via_llm(question_text, index_content)
        
        # Step 3: Read relevant pages
        pages_context = ""
        for p in relevant_pages:
            rel = p.relative_to(self.repo_root)
            pages_context += f"\n\n### {rel}\n{p.read_text(encoding='utf-8')}"
        
        if not pages_context:
            pages_context = f"\n\n### wiki/index.md\n{index_content}"
        
        # Get schema
        schema = self._get_schema()
        
        # Step 4: Synthesize answer
        print(f"  synthesizing answer from {len(relevant_pages)} pages...")
        
        prompt = f"""You are querying an LLM Wiki to answer a question. Use the wiki pages below to synthesize a thorough answer. Cite sources using [[PageName]] wikilink syntax.

Schema:
{schema}

Wiki pages:
{pages_context}

Question: {question_text}

Write a well-structured markdown answer with headers, bullets, and [[wikilink]] citations. At the end, add a ## Sources section listing the pages you drew from.
"""
        
        # Log the cycle
        self.trajectory_logger.log_cycle(thought=prompt, action="synthesize_answer")
        
        result: CallResult = self.client.call(prompt=prompt, max_tokens=4096)
        
        # Update metrics
        self.trajectory_logger.update_metrics(
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            latency_ms=result.latency_ms,
            retrieval_count=len(relevant_pages)
        )
        
        answer = result.content
        
        # Log observation
        self.trajectory_logger.log_cycle(
            thought="",
            action="answer_generated",
            observation=answer[:500] + "..." if len(answer) > 500 else answer
        )
        
        print("\n" + "=" * 60)
        print(answer)
        print("=" * 60)
        
        # Step 5: Optionally save answer
        if save_path is not None:
            full_save_path = self.syntheses_dir / save_path
            frontmatter = f"""---
title: "{question_text[:80]}"
type: synthesis
tags: []
sources: []
last_updated: {today}
---

"""
            write_file(full_save_path, frontmatter + answer)
            
            # Update index
            index_content = read_file(self.index_file)
            entry = f"- [{question_text[:60]}](syntheses/{save_path}) — synthesis"
            if "## Syntheses" in index_content:
                index_content = index_content.replace("## Syntheses\n", f"## Syntheses\n{entry}\n")
                write_file(self.index_file, index_content)
            print(f"  indexed: syntheses/{save_path}")
        
        # Append to log
        self._append_log(
            f"## [{today}] query | {question_text[:80]}\n\n"
            f"Synthesized answer from {len(relevant_pages)} pages." +
            (f" Saved to syntheses/{save_path}." if save_path else "")
        )
        
        # End trajectory tracking
        messages, metrics = self.trajectory_logger.end_query()
        
        metadata = {
            "relevant_pages_count": len(relevant_pages),
            "relevant_pages": [str(p.relative_to(self.repo_root)) for p in relevant_pages],
            "tokens_used": metrics.total_tokens,
            "latency_ms": metrics.latency_ms,
            "llm_calls": metrics.llm_calls,
            "saved_path": save_path
        }
        
        return answer, metadata
    
    def query_from_question_dataclass(
        self,
        question: Question,
        save_path: Optional[str] = None
    ) -> BenchmarkResult:
        """Query using a Question dataclass and return BenchmarkResult.
        
        Args:
            question: Question dataclass instance
            save_path: Optional path to save answer
            
        Returns:
            BenchmarkResult with answer and metrics
        """
        # Perform query
        answer, metadata = self.query(
            question_text=question.text,
            question_id=question.question_id,
            save_path=save_path
        )
        
        # Get trajectory
        messages, metrics = self.trajectory_logger.end_query()
        
        # Create BenchmarkResult
        result = BenchmarkResult(
            pipeline_name="llm-wiki-agent",
            question_id=question.question_id,
            predicted_answer=answer,
            latency_seconds=metrics.latency_ms / 1000.0,
            token_usage=metrics.total_tokens,
            retrieval_count=metrics.retrieval_count,
            trajectory={
                "messages": messages,
                "metadata": metadata
            }
        )
        
        return result
    
    def _append_log(self, entry: str) -> None:
        """Prepend new entry to wiki/log.md.
        
        Args:
            entry: Log entry text
        """
        existing = read_file(self.log_file)
        write_file(self.log_file, entry.strip() + "\n\n" + existing)
    
    def _get_default_schema(self) -> str:
        """Get default schema if CLAUDE.md not found.
        
        Returns:
            Default schema string
        """
        return """# LLM Wiki Agent — Schema & Workflow Instructions

## Directory Layout

```
wiki/
  index.md    # Catalog of all pages
  log.md      # Append-only chronological record
  overview.md # Living synthesis across all sources
  sources/    # One summary page per source document
  entities/   # People, companies, projects, products
  concepts/   # Ideas, frameworks, methods, theories
  syntheses/  # Saved query answers
```

## Page Format

Every wiki page uses this frontmatter:

```yaml
---
title: "Page Title"
type: source | entity | concept | synthesis
tags: []
sources: []
last_updated: YYYY-MM-DD
---
```

Use `[[PageName]]` wikilinks to link to other wiki pages.

## Query Workflow

1. Read `wiki/index.md` to identify relevant pages
2. Read those pages
3. Synthesize an answer with inline citations as `[[PageName]]` wikilinks
4. Optionally save answer to `wiki/syntheses/<slug>.md`
"""
