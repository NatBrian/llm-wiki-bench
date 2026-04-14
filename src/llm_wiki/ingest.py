"""Wiki Ingestor - adapted from llm-wiki-agent tools/ingest.py.

This module faithfully preserves the original ingestion logic while adapting:
- LLM calls use src/llm_client.py instead of direct litellm
- Document ingestion works with src/data/models.py Document dataclass
- Added tracking for token usage and latency
"""

import json
import hashlib
import re
from datetime import date
from pathlib import Path
from typing import Optional

from ..llm_client import LLMClient, CallResult
from ..data.models import Document, BenchmarkResult, Trajectory
from .tracking import TrajectoryLogger


def sha256(text: str) -> str:
    """Compute SHA256 hash of text (first 16 chars)."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def read_file(path: Path) -> str:
    """Read file content, returns empty string if not exists."""
    return path.read_text(encoding="utf-8") if path.exists() else ""


def write_file(path: Path, content: str):
    """Create parent directories if needed and write content to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def parse_json_from_response(text: str) -> dict:
    """Parse JSON from LLM response, stripping markdown code fences."""
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    # Find the outermost JSON object
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON object found in response")
    return json.loads(match.group())


def extract_wikilinks(content: str) -> list[str]:
    """Extract all [[WikiLink]] targets from page content."""
    return re.findall(r'\[\[([^\]]+)\]\]', content)


def all_wiki_pages(wiki_dir: Path) -> set[str]:
    """Return set of all wiki page stems (case-insensitive)."""
    pages = set()
    for p in wiki_dir.rglob("*.md"):
        if p.name not in ("index.md", "log.md", "lint-report.md"):
            pages.add(p.stem.lower())
    return pages


def validate_ingest(
    wiki_dir: Path,
    index_file: Path,
    changed_pages: list[str] | None = None
) -> dict:
    """Validate wiki integrity after an ingest.

    Checks:
      1. Broken wikilinks in changed pages (or all pages if none specified)
      2. Pages not registered in index.md

    Returns dict with 'broken_links' and 'unindexed' lists.
    """
    existing_pages = all_wiki_pages(wiki_dir)
    index_content = read_file(index_file).lower()

    # Determine which pages to scan for broken links
    if changed_pages:
        scan_paths = [wiki_dir / p for p in changed_pages if (wiki_dir / p).exists()]
    else:
        scan_paths = [p for p in wiki_dir.rglob("*.md")
                      if p.name not in ("index.md", "log.md", "lint-report.md")]

    # Check 1: Broken wikilinks
    broken_links = []
    for page_path in scan_paths:
        content = read_file(page_path)
        rel = str(page_path.relative_to(wiki_dir))
        for link in extract_wikilinks(content):
            # Normalize: strip paths, check stem only
            link_stem = Path(link).stem.lower() if '/' in link else link.lower()
            if link_stem not in existing_pages:
                broken_links.append((rel, link))

    # Check 2: Unindexed pages (only check changed pages)
    unindexed = []
    for p in (changed_pages or []):
        page_path = wiki_dir / p
        if page_path.exists():
            # Check if the page filename appears in index.md
            stem = page_path.stem.lower()
            if stem not in index_content and p not in ("log.md", "overview.md"):
                unindexed.append(p)

    return {"broken_links": broken_links, "unindexed": unindexed}


class WikiIngestor:
    """Ingests documents into the LLM Wiki.
    
    Faithfully adapted from the original llm-wiki-agent tools/ingest.py.
    
    The original agent uses a single-shot LLM call pattern:
    1. Read source document
    2. Build wiki context from existing pages
    3. Call LLM with comprehensive prompt requesting JSON output
    4. Parse JSON and execute all writes
    5. Log contradictions if any
    
    This adaptation preserves that exact logic while:
    - Using our LLMClient for API calls
    - Tracking token usage and latency
    - Supporting our Document dataclass as input
    """
    
    def __init__(
        self,
        wiki_dir: Optional[Path] = None,
        schema_file: Optional[Path] = None,
        client: Optional[LLMClient] = None,
        trajectory_logger: Optional[TrajectoryLogger] = None
    ):
        """Initialize the WikiIngestor.
        
        Args:
            wiki_dir: Directory for wiki files. Defaults to project root / wiki
            schema_file: Path to CLAUDE.md schema file. If None, uses default
            client: LLMClient instance. If None, creates new one
            trajectory_logger: Logger for tracking. If None, creates new one
        """
        # Set up paths
        self.repo_root = Path(__file__).parent.parent.parent
        self.wiki_dir = wiki_dir or self.repo_root / "wiki"
        self.log_file = self.wiki_dir / "log.md"
        self.index_file = self.wiki_dir / "index.md"
        self.overview_file = self.wiki_dir / "overview.md"
        self.sources_dir = self.wiki_dir / "sources"
        self.entities_dir = self.wiki_dir / "entities"
        self.concepts_dir = self.wiki_dir / "concepts"
        
        # Schema file - check both locations
        if schema_file:
            self.schema_file = schema_file
        elif (self.repo_root / "CLAUDE.md").exists():
            self.schema_file = self.repo_root / "CLAUDE.md"
        elif (self.repo_root / "src" / "llm_wiki" / "CLAUDE.md").exists():
            self.schema_file = self.repo_root / "src" / "llm_wiki" / "CLAUDE.md"
        else:
            # Use embedded schema
            self.schema_file = None
        
        # Initialize clients
        self.client = client or LLMClient()
        self.trajectory_logger = trajectory_logger or TrajectoryLogger()
        
        # Create wiki directories
        self.wiki_dir.mkdir(parents=True, exist_ok=True)
        self.sources_dir.mkdir(parents=True, exist_ok=True)
        self.entities_dir.mkdir(parents=True, exist_ok=True)
        self.concepts_dir.mkdir(parents=True, exist_ok=True)
    
    def build_wiki_context(self) -> str:
        """Build context from current wiki state.
        
        Aggregates:
        - wiki/index.md
        - wiki/overview.md
        - Up to 5 most recent source pages from wiki/sources/
        
        Returns:
            Concatenated wiki context string
        """
        parts = []
        if self.index_file.exists():
            parts.append(f"## wiki/index.md\n{read_file(self.index_file)}")
        if self.overview_file.exists():
            parts.append(f"## wiki/overview.md\n{read_file(self.overview_file)}")
        
        # Include a few recent source pages for contradiction checking
        if self.sources_dir.exists():
            recent = sorted(
                self.sources_dir.glob("*.md"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )[:5]
            for p in recent:
                parts.append(f"## {p.relative_to(self.repo_root)}\n{p.read_text()}")
        
        return "\n\n---\n\n".join(parts)
    
    def update_index(self, new_entry: str, section: str = "Sources") -> None:
        """Insert new entry under specified section in index.md.
        
        Args:
            new_entry: Entry line to add
            section: Section name (default: "Sources")
        """
        content = read_file(self.index_file)
        if not content:
            content = (
                "# Wiki Index\n\n"
                "## Overview\n- [Overview](overview.md) — living synthesis\n\n"
                "## Sources\n\n"
                "## Entities\n\n"
                "## Concepts\n\n"
                "## Syntheses\n"
            )
        
        section_header = f"## {section}"
        if section_header in content:
            content = content.replace(section_header + "\n", section_header + "\n" + new_entry + "\n")
        else:
            content += f"\n{section_header}\n{new_entry}\n"
        
        write_file(self.index_file, content)
    
    def append_log(self, entry: str) -> None:
        """Prepend new entry to wiki/log.md.
        
        Args:
            entry: Log entry text
        """
        existing = read_file(self.log_file)
        write_file(self.log_file, entry.strip() + "\n\n" + existing)
    
    def ingest_document(
        self,
        document: Document,
        source_content: str,
        question_id: Optional[str] = None
    ) -> tuple[str, dict]:
        """Ingest a document into the wiki.
        
        Faithfully preserves the original ingest logic:
        1. Read source file and compute hash
        2. Build wiki context from existing pages
        3. Read schema from CLAUDE.md
        4. Construct prompt with schema, wiki state, source content, date
        5. Call LLM requesting JSON response
        6. Parse JSON response
        7. Write all output files
        8. Report contradictions if any
        
        Args:
            document: Document dataclass instance
            source_content: Full text content of the source document
            question_id: Optional ID for tracking
            
        Returns:
            Tuple of (ingestion_result_string, metadata_dict)
        """
        today = date.today().isoformat()
        source_hash = sha256(source_content)
        
        # Start trajectory tracking
        if question_id:
            self.trajectory_logger.start_query(question_id)
        
        print(f"\nIngesting: {document.doc_id} (hash: {source_hash})")
        
        # Build wiki context
        wiki_context = self.build_wiki_context()
        
        # Read schema
        if self.schema_file and self.schema_file.exists():
            schema = read_file(self.schema_file)
        else:
            schema = self._get_default_schema()
        
        # Determine relative path for source
        try:
            source_rel = f"documents/{document.doc_id}"
        except (ValueError, TypeError):
            source_rel = f"documents/{document.doc_id}"
        
        # Build prompt - EXACTLY as original
        prompt = f"""You are maintaining an LLM Wiki. Process this source document and integrate its knowledge into the wiki.

Schema and conventions:
{schema}

Current wiki state (index + recent pages):
{wiki_context if wiki_context else "(wiki is empty — this is the first source)"}

New source to ingest (file: {source_rel}):
=== SOURCE START ===
{source_content}
=== SOURCE END ===

Today's date: {today}

Return ONLY a valid JSON object with these fields (no markdown fences, no prose outside the JSON):
{{
  "title": "Human-readable title for this source",
  "slug": "kebab-case-slug-for-filename",
  "source_page": "full markdown content for wiki/sources/<slug>.md — use the source page format from the schema. CRITICAL: Aggressively convert key people, products, concepts and projects into [[Wikilinks]] inline in the text. Omitting [[ ]] for known terms is a failure.",
  "index_entry": "- [Title](sources/slug.md) — one-line summary",
  "overview_update": "full updated content for wiki/overview.md, or null if no update needed",
  "entity_pages": [
    {{"path": "entities/EntityName.md", "content": "full markdown content"}}
  ],
  "concept_pages": [
    {{"path": "concepts/ConceptName.md", "content": "full markdown content"}}
  ],
  "contradictions": ["describe any contradiction with existing wiki content, or empty list"],
  "log_entry": "## [{today}] ingest | <title>\\n\\nAdded source. Key claims: ..."
}}
"""
        
        # Log the cycle (single-shot pattern)
        self.trajectory_logger.log_cycle(thought=prompt, action="ingest")
        
        # Call LLM
        print(f"  calling API (model: {self.client.default_model})")
        result: CallResult = self.client.call(prompt=prompt, max_tokens=8192)
        
        # Update metrics
        self.trajectory_logger.update_metrics(
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            latency_ms=result.latency_ms
        )
        
        # Log observation
        self.trajectory_logger.log_cycle(
            thought="",
            action="write_files",
            observation=result.content[:500] + "..." if len(result.content) > 500 else result.content
        )
        
        # Parse JSON response
        try:
            data = parse_json_from_response(result.content)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"Error parsing API response: {e}")
            debug_path = Path("/tmp/ingest_debug.txt")
            debug_path.write_text(result.content)
            raise
        
        # Write source page
        slug = data["slug"]
        write_file(self.sources_dir / f"{slug}.md", data["source_page"])
        
        # Write entity pages
        for page in data.get("entity_pages", []):
            write_file(self.wiki_dir / page["path"], page["content"])
        
        # Write concept pages
        for page in data.get("concept_pages", []):
            write_file(self.wiki_dir / page["path"], page["content"])
        
        # Update overview
        if data.get("overview_update"):
            write_file(self.overview_file, data["overview_update"])
        
        # Update index
        self.update_index(data["index_entry"], section="Sources")
        
        # Append log
        self.append_log(data["log_entry"])
        
        # Report contradictions
        contradictions = data.get("contradictions", [])
        if contradictions:
            print("\n  ⚠️  Contradictions detected:")
            for c in contradictions:
                print(f"     - {c}")

        # --- Post-ingest validation ---
        created_pages = [f"sources/{slug}.md"]
        for page in data.get("entity_pages", []):
            created_pages.append(page["path"])
        for page in data.get("concept_pages", []):
            created_pages.append(page["path"])
        updated_pages = ["index.md", "log.md"]
        if data.get("overview_update"):
            updated_pages.append("overview.md")

        validation = validate_ingest(self.wiki_dir, self.index_file, created_pages)

        print(f"\n{'='*50}")
        print(f"  ✅ Ingested: {data['title']}")
        print(f"{'='*50}")
        print(f"  Created : {len(created_pages)} pages")
        for p in created_pages:
            print(f"           + wiki/{p}")
        print(f"  Updated : {len(updated_pages)} pages")
        for p in updated_pages:
            print(f"           ~ wiki/{p}")
        if contradictions:
            print(f"  Warnings: {len(contradictions)} contradiction(s)")
        if validation["broken_links"]:
            print(f"  ⚠️  Broken links: {len(validation['broken_links'])}")
            for page, link in validation["broken_links"][:10]:
                print(f"           wiki/{page} → [[{link}]]")
            if len(validation["broken_links"]) > 10:
                print(f"           ... and {len(validation['broken_links']) - 10} more")
        if validation["unindexed"]:
            print(f"  ⚠️  Not in index.md: {len(validation['unindexed'])}")
            for p in validation["unindexed"][:10]:
                print(f"           wiki/{p}")
            if len(validation["unindexed"]) > 10:
                print(f"           ... and {len(validation['unindexed']) - 10} more")
        if not validation["broken_links"] and not validation["unindexed"]:
            print("  ✓ Validation passed — no broken links, all pages indexed")
        print()
        
        # End trajectory tracking
        messages, metrics = self.trajectory_logger.end_query()
        
        metadata = {
            "slug": slug,
            "title": data["title"],
            "source_hash": source_hash,
            "contradictions": contradictions,
            "tokens_used": metrics.total_tokens,
            "latency_ms": metrics.latency_ms,
            "entity_pages_created": len(data.get("entity_pages", [])),
            "concept_pages_created": len(data.get("concept_pages", []))
        }
        
        return f"Ingested: {data['title']}", metadata
    
    def ingest_from_document_dataclass(
        self,
        document: Document,
        question_id: Optional[str] = None
    ) -> tuple[str, dict]:
        """Ingest from a Document dataclass.
        
        For the wiki agent, we need to convert document images to text.
        Since the original wiki-agent works with markdown text sources,
        we create a markdown representation of the document.
        
        Args:
            document: Document dataclass instance
            question_id: Optional ID for tracking
            
        Returns:
            Tuple of (result_string, metadata_dict)
        """
        # Create a markdown representation of the document
        # In a real scenario, you'd use OCR or have pre-extracted text
        # For now, we create a placeholder that describes the document
        source_content = self._document_to_markdown(document)
        
        return self.ingest_document(document, source_content, question_id)
    
    def _document_to_markdown(self, document: Document) -> str:
        """Convert Document dataclass to markdown text.
        
        Since the original wiki-agent expects text documents, we create
        a markdown representation. In production, this would use OCR.
        
        Args:
            document: Document dataclass instance
            
        Returns:
            Markdown representation of the document
        """
        lines = [
            f"# Document: {document.doc_id}",
            f"\nDomain: {document.domain}",
            f"\nPages: {document.page_count}",
            "\n## Content\n",
            "[Note: This document contains image pages. ",
            "Full text extraction would require OCR processing.]",
            "\n\n## Page References\n"
        ]
        
        for page in document.pages:
            lines.append(f"- Page {page.page_number}: {page.image_path}")
        
        return "\n".join(lines)
    
    def _get_default_schema(self) -> str:
        """Get default schema if CLAUDE.md not found.
        
        Returns:
            Default schema string
        """
        return """# LLM Wiki Agent — Schema & Workflow Instructions

## Directory Layout

```
raw/          # Immutable source documents
wiki/         # Agent-maintained layer
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

## Source Page Format

```markdown
---
title: "Source Title"
type: source
tags: []
date: YYYY-MM-DD
source_file: raw/...
---

## Summary
2–4 sentence summary.

## Key Claims
- Claim 1
- Claim 2

## Key Quotes
> "Quote here" — context

## Connections
- [[EntityName]] — how they relate
- [[ConceptName]] — how it connects

## Contradictions
- Contradicts [[OtherPage]] on: ...
```

## Naming Conventions

- Source slugs: `kebab-case` matching source filename
- Entity pages: `TitleCase.md`
- Concept pages: `TitleCase.md`
"""
