# LLM Wiki Agent Analysis

## Repository Overview

**Repository:** https://github.com/SamurAIGPT/llm-wiki-agent

**Purpose:** A coding agent skill that reads source documents from `raw/`, extracts knowledge, and builds a persistent interlinked wiki. The wiki compounds over time with cross-references, contradiction flags, and synthesis.

---

## File-by-File Analysis

### 1. `/README.md`

**Purpose:** Main documentation explaining the project's value proposition, usage, and architecture.

**Key Information:**
- **Core Concept:** Drop source documents into `raw/` and trigger ingestion — the agent builds a structured wiki automatically
- **Directory Structure:**
  ```
  wiki/
  ├── index.md          # catalog of all pages
  ├── log.md            # append-only record of operations
  ├── overview.md       # living synthesis across all sources
  ├── sources/          # one summary page per source document
  ├── entities/         # people, companies, projects — auto-created
  ├── concepts/         # ideas, frameworks, methods — auto-created
  └── syntheses/        # query answers filed back as wiki pages
  graph/
  ├── graph.json        # persistent node/edge data (SHA256-cached)
  └── graph.html        # interactive vis.js visualization
  ```
- **Commands:**
  - `/wiki-ingest <file>` — ingest a source into the wiki
  - `/wiki-query "<question>"` — synthesize answer from wiki pages
  - `/wiki-lint` — find orphans, contradictions, gaps
  - `/wiki-graph` — build knowledge graph visualization

**Dependencies Mentioned:** NetworkX, Louvain, Claude/LLM API, vis.js

---

### 2. `/requirements.txt`

**Purpose:** Python dependencies for standalone tools.

**Contents:**
```
litellm>=1.0.0
networkx>=3.2
```

**Analysis:**
- `litellm`: Unified LLM API client supporting multiple providers (OpenAI, Anthropic, Google, etc.)
- `networkx`: Graph library for community detection (Louvain algorithm)

---

### 3. `/CLAUDE.md` (Schema File for Claude Code)

**Purpose:** Schema and workflow instructions for Claude Code agent.

**Key Sections:**

#### Directory Layout
```
raw/          # Immutable source documents
wiki/         # Agent-maintained layer
  index.md    # Catalog of all pages — update on every ingest
  log.md      # Append-only chronological record
  overview.md # Living synthesis across all sources
  sources/    # One summary page per source document
  entities/   # People, companies, projects, products
  concepts/   # Ideas, frameworks, methods, theories
  syntheses/  # Saved query answers
graph/        # Auto-generated graph data
tools/        # Optional standalone Python scripts
```

#### Page Format (Frontmatter)
Every wiki page uses YAML frontmatter:
```yaml
---
title: "Page Title"
type: source | entity | concept | synthesis
tags: []
sources: []       # list of source slugs that inform this page
last_updated: YYYY-MM-DD
---
```

#### Wikilink Syntax
- Uses `[[PageName]]` syntax for internal links

#### Ingest Workflow Steps
1. Read the source document fully
2. Read `wiki/index.md` and `wiki/overview.md` for context
3. Write `wiki/sources/<slug>.md` using source page format
4. Update `wiki/index.md` — add entry under Sources section
5. Update `wiki/overview.md` — revise synthesis if warranted
6. Update/create entity pages for key people, companies, projects
7. Update/create concept pages for key ideas and frameworks
8. Flag any contradictions with existing wiki content
9. Append to `wiki/log.md`: `## [YYYY-MM-DD] ingest | <Title>`

#### Source Page Format
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

#### Domain-Specific Templates
- **Diary/Journal Template:** Includes Event Summary, Key Decisions, Energy & Mood, Connections, Shifts & Contradictions
- **Meeting Notes Template:** Includes Goal, Key Discussions, Decisions Made, Action Items

#### Query Workflow
1. Read `wiki/index.md` to identify relevant pages
2. Read those pages
3. Synthesize answer with inline `[[PageName]]` citations
4. Ask user if they want answer filed as `wiki/syntheses/<slug>.md`

#### Lint Workflow Checks
- Orphan pages (no inbound `[[links]]`)
- Broken links (`[[WikiLinks]]` pointing to non-existent pages)
- Contradictions between pages
- Stale summaries
- Missing entity pages (mentioned in 3+ pages but no page)
- Data gaps and suggested new sources

#### Graph Workflow
- Pass 1: Parse all `[[wikilinks]]` → deterministic `EXTRACTED` edges
- Pass 2: Infer implicit relationships → `INFERRED` edges with confidence scores
- Run Louvain community detection
- Output `graph/graph.json` + `graph/graph.html`

#### Naming Conventions
- Source slugs: `kebab-case` matching source filename
- Entity pages: `TitleCase.md` (e.g., `OpenAI.md`, `SamAltman.md`)
- Concept pages: `TitleCase.md` (e.g., `ReinforcementLearning.md`, `RAG.md`)

---

### 4. `/AGENTS.md`

**Purpose:** Schema and workflow instructions for Codex/OpenCode agents.

**Analysis:** Nearly identical to `CLAUDE.md` but tailored for OpenAI's Codex and OpenCode agents. Same structure, workflows, and conventions.

---

### 5. `/GEMINI.md`

**Purpose:** Schema and workflow instructions for Gemini CLI.

**Analysis:** Condensed version of `CLAUDE.md` tailored for Google's Gemini CLI. Same core workflows but more concise.

---

### 6. `/tools/ingest.py`

**Purpose:** Standalone Python script to ingest a source document into the wiki.

**Classes/Functions:**

#### `sha256(text: str) -> str`
- Computes SHA256 hash of text (first 16 chars)
- Used for change detection/caching

#### `read_file(path: Path) -> str`
- Reads file content, returns empty string if not exists

#### `call_llm(prompt: str, max_tokens: int = 8192) -> str`
- **LLM Client:** `litellm.completion()`
- **Model:** From env var `LLM_MODEL`, default `"claude-3-5-sonnet-latest"`
- **Parameters:** `model`, `messages`, `max_tokens`
- **Returns:** `response.choices[0].message.content`

#### `write_file(path: Path, content: str)`
- Creates parent directories if needed
- Writes content to file

#### `build_wiki_context() -> str`
- Aggregates current wiki state:
  - `wiki/index.md`
  - `wiki/overview.md`
  - Up to 5 most recent source pages from `wiki/sources/`

#### `parse_json_from_response(text: str) -> dict`
- Strips markdown code fences
- Extracts outermost JSON object using regex
- Parses and returns dict

#### `update_index(new_entry: str, section: str = "Sources")`
- Inserts new entry under specified section in index.md

#### `append_log(entry: str)`
- Prepends new entry to `wiki/log.md`

#### `ingest(source_path: str)`
**Main ingestion logic:**
1. Read source file and compute hash
2. Build wiki context from existing pages
3. Read schema from `CLAUDE.md`
4. Construct prompt with:
   - Schema/conventions
   - Current wiki state
   - New source content
   - Date
5. Call LLM with prompt requesting JSON response containing:
   - `title`: Human-readable title
   - `slug`: kebab-case slug for filename
   - `source_page`: Full markdown for `wiki/sources/<slug>.md`
   - `index_entry`: Index line to add
   - `overview_update`: Updated overview or null
   - `entity_pages`: List of `{path, content}` for entity pages
   - `concept_pages`: List of `{path, content}` for concept pages
   - `contradictions`: List of contradiction descriptions
   - `log_entry`: Log entry text
6. Parse JSON response
7. Write all output files
8. Report contradictions if any

**Agent Loop Logic:**
- Single-shot LLM call (not iterative)
- LLM decides what entities/concepts to create based on source content
- LLM identifies contradictions by comparing against existing wiki context
- Termination: After writing all files and logging

**Document/Knowledge Handling:**
- Documents stored as markdown in `wiki/sources/`
- Knowledge extracted into structured entity/concept pages
- Cross-references via `[[wikilinks]]`
- Contradictions flagged at ingest time

---

### 7. `/tools/query.py`

**Purpose:** Query the wiki to synthesize answers from existing pages.

**Classes/Functions:**

#### `call_llm(prompt: str, model_env: str, default_model: str, max_tokens: int = 4096) -> str`
- **LLM Client:** `litellm.completion()`
- **Model:** From env var (passed as `model_env`), default provided
- **Usage:** Two models used:
  - `LLM_MODEL_FAST` (default: `claude-3-5-haiku-latest`) for page selection
  - `LLM_MODEL` (default: `claude-3-5-sonnet-latest`) for answer synthesis

#### `find_relevant_pages(question: str, index_content: str) -> list[Path]`
**Keyword-based relevance matching:**
1. Extract all links from index (both `[[wikilinks]]` and markdown links)
2. Match question keywords against page titles:
   - English: Check words > 3 chars
   - Exact substring match for short titles (CJK support)
   - CJK chunks: Contiguous non-ASCII characters
3. Always include `overview.md`
4. Cap at 12 pages

**Fallback:** If ≤1 page found, call LLM to select relevant pages from index via JSON array

#### `query(question: str, save_path: str | None = None)`
**Query workflow:**
1. Read `wiki/index.md`
2. Find relevant pages via keyword matching
3. Fallback to LLM-based selection if needed
4. Read all relevant pages
5. Call LLM with:
   - Schema from `CLAUDE.md`
   - Relevant page contents
   - Question
6. Request well-structured markdown answer with `[[wikilink]]` citations
7. Print answer
8. Optionally save to `wiki/syntheses/` with frontmatter
9. Update index if saved
10. Append to log

**LLM Call Format:**
```python
prompt = f"""You are querying an LLM Wiki to answer a question...

Schema:
{schema}

Wiki pages:
{pages_context}

Question: {question}

Write a well-structured markdown answer with headers, bullets, and [[wikilink]] citations. At the end, add a ## Sources section listing the pages you drew from.
"""
```

---

### 8. `/tools/lint.py`

**Purpose:** Lint the wiki for health issues.

**Classes/Functions:**

#### `all_wiki_pages() -> list[Path]`
- Returns all `.md` files in `wiki/` excluding `index.md`, `log.md`, `lint-report.md`

#### `extract_wikilinks(content: str) -> list[str]`
- Regex: `\[\[([^\]]+)\]\]` to find all `[[wikilinks]]`

#### `page_name_to_path(name: str) -> list[Path]`
- Resolves `[[WikiLink]]` to file path(s) by case-insensitive stem matching

#### `find_orphans(pages: list[Path]) -> list[Path]`
- Builds inbound link count for each page
- Returns pages with 0 inbound links (except `overview.md`)

#### `find_broken_links(pages: list[Path]) -> list[tuple[Path, str]]`
- For each wikilink, checks if target page exists
- Returns list of `(page, link)` tuples for broken links

#### `find_missing_entities(pages: list[Path]) -> list[str]`
- Finds entity names mentioned in 3+ pages but lacking their own page
- Uses wikilink extraction and mention counting

#### `call_llm(prompt: str, model_env: str, default_model: str, max_tokens: int = 4096) -> str`
- **LLM Client:** `litellm.completion()`
- **Model:** From env var `LLM_MODEL`, default `claude-3-5-sonnet-latest`

#### `run_lint()`
**Lint workflow:**
1. Get all wiki pages
2. Run deterministic checks:
   - `find_orphans()`
   - `find_broken_links()`
   - `find_missing_entities()`
3. Sample up to 20 pages (truncate to 1500 chars each)
4. Call LLM for semantic checks:
   - Contradictions between pages
   - Stale content
   - Data gaps and suggested sources
   - Concepts needing more depth
5. Compose full report with structural issues + semantic report

**LLM Prompt for Semantic Lint:**
```
You are linting an LLM Wiki. Review the pages below and identify:
1. Contradictions between pages (claims that conflict)
2. Stale content (summaries that newer sources have superseded)
3. Data gaps (important questions the wiki can't answer — suggest specific sources to find)
4. Concepts mentioned but lacking depth

Return a markdown lint report with these sections:
## Contradictions
## Stale Content
## Data Gaps & Suggested Sources
## Concepts Needing More Depth
```

---

### 9. `/tools/build_graph.py`

**Purpose:** Build knowledge graph from wiki pages.

**Classes/Functions:**

#### Constants
```python
TYPE_COLORS = {
    "source": "#4CAF50",
    "entity": "#2196F3",
    "concept": "#FF9800",
    "synthesis": "#9C27B0",
    "unknown": "#9E9E9E",
}

EDGE_COLORS = {
    "EXTRACTED": "#555555",
    "INFERRED": "#FF5722",
    "AMBIGUOUS": "#BDBDBD",
}
```

#### `sha256(text: str) -> str`
- Full SHA256 hash for caching

#### `page_id(path: Path) -> str`
- Converts file path to node ID (relative path without `.md`)

#### `load_cache() -> dict` / `save_cache(cache: dict)`
- Loads/saves cache from `graph/.cache.json`
- Cache stores page hashes and inferred edges

#### `build_nodes(pages: list[Path]) -> list[dict]`
**Node structure:**
```json
{
  "id": "sources/my-source",
  "label": "My Source Title",
  "type": "source",
  "color": "#4CAF50",
  "path": "wiki/sources/my-source.md"
}
```

#### `extract_frontmatter_type(content: str) -> str`
- Regex: `^type:\s*(\S+)` to extract type from YAML frontmatter

#### `build_extracted_edges(pages: list[Path]) -> list[dict]`
**Pass 1: Deterministic wikilink extraction**
- Parse all `[[wikilinks]]` from each page
- Resolve links to node IDs
- Create edges with type `EXTRACTED`, confidence 1.0

**Edge structure:**
```json
{
  "from": "sources/page-a",
  "to": "entities/PersonB",
  "type": "EXTRACTED",
  "color": "#555555",
  "confidence": 1.0
}
```

#### `build_inferred_edges(pages: list[Path], existing_edges: list[dict], cache: dict) -> list[dict]`
**Pass 2: LLM-inferred semantic relationships**
1. Check cache for changed pages (by SHA256 hash)
2. For changed pages only, call LLM to infer implicit relationships
3. Cache results to avoid re-processing unchanged pages

**LLM Prompt:**
```
Analyze this wiki page and identify implicit semantic relationships to other pages in the wiki.

Source page: {src}
Content: {content}

All available pages: {node_list}

Already-extracted edges from this page: {existing_edge_summary}

Return ONLY a JSON array of NEW relationships not already captured by explicit wikilinks:
[
  {"to": "page-id", "relationship": "one-line description", "confidence": 0.0-1.0, "type": "INFERRED or AMBIGUOUS"}
]

Rules:
- Only include pages from the available list above
- Confidence >= 0.7 → INFERRED, < 0.7 → AMBIGUOUS
- Do not repeat edges already in the extracted list
- Return empty array [] if no new relationships found
```

**Model:** `LLM_MODEL_FAST` (default: `claude-3-5-haiku-latest`)

#### `detect_communities(nodes: list[dict], edges: list[dict]) -> dict[str, int]`
- Uses NetworkX + Louvain algorithm
- Returns mapping of node ID → community ID

#### `render_html(nodes: list[dict], edges: list[dict]) -> str`
- Generates self-contained HTML with vis.js visualization
- Includes search, node info panel, stats, legend

#### `build_graph(infer: bool = True, open_browser: bool = False)`
**Full workflow:**
1. Get all wiki pages
2. Load cache
3. Pass 1: Build nodes and extract wikilinks
4. Pass 2: Infer semantic relationships (if enabled)
5. Run Louvain community detection
6. Apply community colors to nodes
7. Save `graph/graph.json`
8. Generate and save `graph/graph.html`
9. Append to log

---

### 10. `/tools/heal.py`

**Purpose:** Auto-generate missing entity pages to heal graph connectivity.

**Classes/Functions:**

#### `call_llm(prompt: str, max_tokens: int = 1500) -> str`
- **LLM Client:** `litellm.completion()`
- **Model:** From env var `LLM_MODEL`, default `claude-3-5-haiku-latest`

#### `search_sources(entity: str, pages: list[Path]) -> list[Path]`
- Finds up to 15 pages where entity is mentioned (excluding existing entity/concept pages)
- Case-insensitive substring match

#### `heal_missing_entities()`
**Workflow:**
1. Call `find_missing_entities()` from lint module
2. For each missing entity:
   - Search for pages mentioning the entity
   - Extract context snippets (first 800 chars from each source)
   - Call LLM to generate entity definition page

**LLM Prompt:**
```
You are filling a data gap in the Personal LLM Wiki. 
Create an Entity definition page for "{entity}".

Here is how the entity appears in the current sources:
{context}

Format:
---
title: "{entity}"
type: entity
tags: []
sources: [source filenames]
---

# {entity}

Write a comprehensive paragraph defining what `{entity}` means in the context of this wiki...
```

---

### 11. `/docs/automated-sync.md`

**Purpose:** Guide for automating wiki synchronization via cron/launchd.

**Key Information:**
- Two-step architecture:
  1. Sync files to `raw/` directory
  2. Batch ingest using `tools/ingest.py`
- Example shell script for orchestration
- macOS launchd configuration for scheduled execution
- Self-healing via `tools/heal.py`

---

## LLM Calling Patterns Summary

### Common Pattern Across All Tools

```python
from litellm import completion

def call_llm(prompt: str, max_tokens: int = ...) -> str:
    model = os.getenv("LLM_MODEL", "<default-model>")
    response = completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens
    )
    return response.choices[0].message.content
```

### Models Used

| Tool | Env Var | Default Model | Purpose |
|------|---------|---------------|---------|
| ingest.py | `LLM_MODEL` | `claude-3-5-sonnet-latest` | Full ingestion with JSON output |
| query.py | `LLM_MODEL_FAST` | `claude-3-5-haiku-latest` | Page selection |
| query.py | `LLM_MODEL` | `claude-3-5-sonnet-latest` | Answer synthesis |
| lint.py | `LLM_MODEL` | `claude-3-5-sonnet-latest` | Semantic lint checks |
| build_graph.py | `LLM_MODEL_FAST` | `claude-3-5-haiku-latest` | Relationship inference |
| heal.py | `LLM_MODEL` | `claude-3-5-haiku-latest` | Entity page generation |

### Response Formats

**ingest.py:** JSON object with structured fields
```json
{
  "title": "...",
  "slug": "...",
  "source_page": "...",
  "index_entry": "...",
  "overview_update": "...",
  "entity_pages": [...],
  "concept_pages": [...],
  "contradictions": [...],
  "log_entry": "..."
}
```

**query.py:** Markdown with `[[wikilink]]` citations

**lint.py:** Markdown report with sections

**build_graph.py:** JSON array of relationship objects
```json
[
  {"to": "page-id", "relationship": "...", "confidence": 0.8, "type": "INFERRED"}
]
```

**heal.py:** Markdown entity page with frontmatter

---

## Agent Loop Logic

### Ingest Agent Loop
1. **Input:** Source file path
2. **Context Gathering:** Read existing wiki (index, overview, recent sources)
3. **LLM Call:** Single comprehensive prompt requesting full JSON response
4. **Parsing:** Extract JSON from response
5. **Execution:** Write all files specified in JSON
6. **Termination:** After all writes complete

**Decision Making:** LLM decides:
- What entities/concepts to create
- How to summarize the source
- Whether overview needs updating
- What contradictions exist

**Tools Available to LLM:** None directly — LLM receives context in prompt and outputs declarative JSON specifying actions

### Query Agent Loop
1. **Input:** Question string
2. **Retrieval:** Keyword-based page matching from index
3. **Fallback:** LLM-based page selection if keyword match fails
4. **Synthesis:** LLM generates answer from retrieved pages
5. **Optional Save:** User can save answer to syntheses/
6. **Termination:** After printing (and optionally saving) answer

### Lint Agent Loop
1. **Deterministic Phase:** Compute orphans, broken links, missing entities
2. **Semantic Phase:** LLM reviews sampled pages for contradictions, gaps, staleness
3. **Reporting:** Combine both phases into markdown report
4. **Termination:** After printing/saving report

### Graph Build Loop
1. **Pass 1 (Deterministic):** Extract all wikilinks → EXTRACTED edges
2. **Cache Check:** Identify changed pages by SHA256 hash
3. **Pass 2 (LLM):** Infer relationships for changed pages only
4. **Community Detection:** Louvain algorithm via NetworkX
5. **Rendering:** Generate JSON + HTML visualization
6. **Termination:** After saving outputs

---

## Document/Knowledge Handling

### Storage Format
- All knowledge stored as markdown files
- YAML frontmatter for metadata (title, type, tags, sources, last_updated)
- Plain markdown body with `[[wikilinks]]` for cross-references

### Knowledge Types
| Type | Location | Purpose |
|------|----------|---------|
| source | `wiki/sources/` | Summaries of ingested documents |
| entity | `wiki/entities/` | People, companies, projects, products |
| concept | `wiki/concepts/` | Ideas, frameworks, methods, theories |
| synthesis | `wiki/syntheses/` | Saved query answers |

### Knowledge Extraction
- **At Ingest:** LLM extracts entities, concepts, claims, quotes from source
- **Structure:** Organized into predefined sections (Summary, Key Claims, Key Quotes, Connections, Contradictions)
- **Cross-references:** `[[wikilinks]]` created automatically by LLM

### Knowledge Retrieval
- **Index-based:** `wiki/index.md` serves as catalog with links to all pages
- **Keyword matching:** Query matches question keywords against page titles
- **LLM fallback:** If keyword match fails, LLM selects relevant pages from index

### Knowledge Validation
- **Contradiction Detection:** At ingest time, LLM compares new source against existing wiki
- **Lint Checks:** Periodic scans for orphans, broken links, missing entities
- **Semantic Lint:** LLM identifies contradictions, stale content, data gaps

---

## Dependencies Summary

### Core Dependencies
- `litellm>=1.0.0`: Unified LLM API client
- `networkx>=3.2`: Graph algorithms (Louvain community detection)

### Implicit Dependencies
- Python 3.x (type hints suggest 3.9+ for `list[type]` syntax)
- `pathlib`: Standard library for path handling
- `json`, `re`, `hashlib`, `argparse`: Standard library modules

### External Services
- LLM API endpoint (Anthropic Claude, OpenAI, Google Gemini, etc. via LiteLLM)
- No vector database
- No traditional RAG pipeline

---

## Key Design Patterns

### 1. Declarative LLM Output
LLM outputs declarative JSON specifying *what* to do, not *how* to do it. Python code executes the actions.

### 2. Two-Pass Graph Construction
- Pass 1: Deterministic extraction (wikilinks)
- Pass 2: Semantic inference (LLM) with caching

### 3. SHA256 Caching
Pages cached by content hash; only changed pages reprocessed.

### 4. Fast/Slow Model Split
- `LLM_MODEL_FAST` (Haiku): Quick tasks (page selection, relationship inference)
- `LLM_MODEL` (Sonnet): Complex reasoning (ingestion, synthesis, semantic lint)

### 5. Wikilink-Based Knowledge Graph
Explicit `[[wikilinks]]` form the backbone; LLM infers implicit relationships as supplement.

### 6. Contradiction-at-Ingest
Unlike RAG where contradictions surface at query time (if ever), this system flags them immediately during ingestion.

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_MODEL` | `claude-3-5-sonnet-latest` | Primary model for complex tasks |
| `LLM_MODEL_FAST` | `claude-3-5-haiku-latest` | Fast model for quick tasks |
| `ANTHROPIC_API_KEY` | — | Required for Claude models |
| `OPENAI_API_KEY` | — | Required for OpenAI models |
| `GEMINI_API_KEY` | — | Required for Gemini models |

Note: LiteLLM supports many providers; actual key depends on model prefix (e.g., `claude/`, `gpt/`, `gemini/`).
