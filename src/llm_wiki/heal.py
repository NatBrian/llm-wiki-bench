"""Wiki Healer - adapted from llm-wiki-agent tools/heal.py.

This module faithfully preserves the original healing logic while adapting:
- LLM calls use src/llm_client.py instead of direct litellm
- Added tracking for token usage and latency

The heal tool automatically retrieves "Missing Entity Pages" from the wiki
and generates comprehensive definition pages for them using the LLM.
It resolves broken entity links by scanning existing contexts where the entity is referenced.
"""

import json
from datetime import date
from pathlib import Path
from typing import Optional
from collections import defaultdict

from ..llm_client import LLMClient, CallResult
from .tracking import TrajectoryLogger


def read_file(path: Path) -> str:
    """Read file content, returns empty string if not exists."""
    return path.read_text(encoding="utf-8") if path.exists() else ""


def write_file(path: Path, content: str):
    """Create parent directories if needed and write content to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def extract_wikilinks(content: str) -> list[str]:
    """Extract all [[wikilinks]] from content."""
    import re
    return re.findall(r'\[\[([^\]]+)\]\]', content)


class WikiHealer:
    """Heals the LLM Wiki by generating missing entity pages.

    Faithfully adapted from the original llm-wiki-agent tools/heal.py.

    The original agent:
    1. Runs lint to find missing entities (mentioned 3+ times but no page)
    2. For each missing entity, finds source pages where it's mentioned
    3. Calls LLM with context to generate entity definition page
    4. Writes the entity page to wiki/entities/

    This adaptation preserves that exact logic while:
    - Using our LLMClient for API calls
    - Tracking token usage and latency
    """

    def __init__(
        self,
        wiki_dir: Optional[Path] = None,
        client: Optional[LLMClient] = None,
        trajectory_logger: Optional[TrajectoryLogger] = None
    ):
        """Initialize the WikiHealer.

        Args:
            wiki_dir: Directory for wiki files. Defaults to project root / wiki
            client: LLMClient instance. If None, creates new one
            trajectory_logger: Logger for tracking. If None, creates new one
        """
        # Set up paths
        self.repo_root = Path(__file__).parent.parent.parent
        self.wiki_dir = wiki_dir or self.repo_root / "wiki"
        self.entities_dir = self.wiki_dir / "entities"
        self.log_file = self.wiki_dir / "log.md"

        # Initialize clients
        self.client = client or LLMClient()
        self.trajectory_logger = trajectory_logger or TrajectoryLogger()

        # Create entities directory
        self.entities_dir.mkdir(parents=True, exist_ok=True)

    def all_wiki_pages(self) -> list[Path]:
        """Get all wiki pages excluding index, log, and lint-report."""
        return [
            p for p in self.wiki_dir.rglob("*.md")
            if p.name not in ("index.md", "log.md", "lint-report.md")
        ]

    def find_missing_entities(self, pages: list[Path]) -> list[str]:
        """Find entity-like names mentioned in 3+ pages but lacking their own page.

        Args:
            pages: List of wiki page paths

        Returns:
            List of missing entity names
        """
        mention_counts = defaultdict(int)
        existing_pages = {p.stem.lower() for p in pages}

        for p in pages:
            content = read_file(p)
            links = extract_wikilinks(content)
            for link in links:
                if link.lower() not in existing_pages:
                    mention_counts[link] += 1

        return [name for name, count in mention_counts.items() if count >= 3]

    def search_sources(self, entity: str, pages: list[Path]) -> list[Path]:
        """Find up to 15 pages where this entity is mentioned natively.

        Args:
            entity: Entity name to search for
            pages: List of wiki page paths

        Returns:
            List of pages mentioning the entity
        """
        sources = []
        for p in pages:
            # Skip entity and concept pages - we want source pages
            if "entities" not in str(p.parent) and "concepts" not in str(p.parent):
                content = read_file(p)
                if entity.lower() in content.lower():
                    sources.append(p)
        return sources[:15]

    def heal_missing_entities(self) -> dict:
        """Auto-generate pages for missing entity nodes.

        Returns:
            Dictionary with healing statistics
        """
        pages = self.all_wiki_pages()
        missing_entities = self.find_missing_entities(pages)

        if not missing_entities:
            print("Graph is fully connected. No missing entities found!")
            return {"healed": 0, "missing": 0}

        print(f"Found {len(missing_entities)} missing entity nodes. Commencing auto-heal...")

        today = date.today().isoformat()
        healed_count = 0
        errors = []

        for entity in missing_entities:
            print(f"Healing entity page for: {entity}")
            sources = self.search_sources(entity, pages)

            context = ""
            for s in sources:
                context += f"\n\n### {s.name}\n{read_file(s)[:800]}"

            prompt = f"""You are filling a data gap in the Personal LLM Wiki.
Create an Entity definition page for "{entity}".

Here is how the entity appears in the current sources:
{context}

Format:
---
title: "{entity}"
type: entity
tags: []
sources: {[s.name for s in sources]}
---

# {entity}

Write a comprehensive paragraph defining what `{entity}` means in the context of this wiki, its main significance, and any actions or associations related to it.
"""

            try:
                # Log the cycle
                self.trajectory_logger.log_cycle(thought=prompt, action="heal_entity")

                result: CallResult = self.client.call(
                    prompt=prompt,
                    max_tokens=1500,
                    model=self.client.default_model_fast
                )

                # Update metrics
                self.trajectory_logger.update_metrics(
                    prompt_tokens=result.usage.prompt_tokens,
                    completion_tokens=result.usage.completion_tokens,
                    latency_ms=result.latency_ms
                )

                out_path = self.entities_dir / f"{entity}.md"
                write_file(out_path, result.content)
                print(f" -> Saved to {out_path.relative_to(self.repo_root)}")
                healed_count += 1

                # Log observation
                self.trajectory_logger.log_cycle(
                    thought="",
                    action="entity_created",
                    observation=f"Created entity page for {entity}"
                )
            except Exception as e:
                print(f" [!] Failed to generate {entity}: {e}")
                errors.append(str(e))

        # Append to log
        self._append_log(
            f"## [{today}] heal | Auto-generated entity pages\n\n"
            f"Healed {healed_count} of {len(missing_entities)} missing entities."
        )

        return {
            "healed": healed_count,
            "missing": len(missing_entities),
            "errors": errors
        }

    def _append_log(self, entry: str) -> None:
        """Prepend new entry to wiki/log.md.

        Args:
            entry: Log entry text
        """
        existing = read_file(self.log_file)
        write_file(self.log_file, entry.strip() + "\n\n" + existing)