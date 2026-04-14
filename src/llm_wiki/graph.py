"""Knowledge Graph Builder - adapted from llm-wiki-agent tools/build_graph.py.

This module faithfully preserves the original graph building logic while adapting:
- LLM calls use src/llm_client.py instead of direct litellm
- Added tracking for token usage and latency
"""

import json
import hashlib
import re
from datetime import date
from pathlib import Path
from typing import Optional

from ..llm_client import LLMClient, CallResult
from .tracking import TrajectoryLogger

try:
    import networkx as nx
    from networkx.algorithms import community as nx_community
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


# Node type → color mapping
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

COMMUNITY_COLORS = [
    "#E91E63", "#00BCD4", "#8BC34A", "#FF5722", "#673AB7",
    "#FFC107", "#009688", "#F44336", "#3F51B5", "#CDDC39",
]


def read_file(path: Path) -> str:
    """Read file content, returns empty string if not exists."""
    return path.read_text(encoding="utf-8") if path.exists() else ""


def sha256(text: str) -> str:
    """Compute SHA256 hash of text."""
    return hashlib.sha256(text.encode()).hexdigest()


def extract_wikilinks(content: str) -> list[str]:
    """Extract all [[wikilinks]] from content."""
    return list(set(re.findall(r'\[\[([^\]]+)\]\]', content)))


def extract_frontmatter_type(content: str) -> str:
    """Extract page type from frontmatter."""
    match = re.search(r'^type:\s*(\S+)', content, re.MULTILINE)
    return match.group(1).strip('"\'') if match else "unknown"


class WikiGraphBuilder:
    """Builds knowledge graph from wiki pages.
    
    Faithfully adapted from the original llm-wiki-agent tools/build_graph.py.
    
    The original agent uses this workflow:
    1. Pass 1: Extract deterministic edges from explicit [[wikilinks]]
    2. Pass 2: Infer semantic relationships via LLM (optional)
    3. Run Louvain community detection
    4. Output graph/graph.json + graph/graph.html
    
    This adaptation preserves that exact logic while:
    - Using our LLMClient for API calls
    - Tracking token usage and latency
    - Caching inferred edges by page hash
    """
    
    def __init__(
        self,
        wiki_dir: Optional[Path] = None,
        graph_dir: Optional[Path] = None,
        client: Optional[LLMClient] = None,
        trajectory_logger: Optional[TrajectoryLogger] = None
    ):
        """Initialize the WikiGraphBuilder.
        
        Args:
            wiki_dir: Directory for wiki files. Defaults to project root / wiki
            graph_dir: Directory for graph output. Defaults to project root / graph
            client: LLMClient instance. If None, creates new one
            trajectory_logger: Logger for tracking. If None, creates new one
        """
        # Set up paths
        self.repo_root = Path(__file__).parent.parent.parent
        self.wiki_dir = wiki_dir or self.repo_root / "wiki"
        self.graph_dir = graph_dir or self.repo_root / "graph"
        self.graph_json = self.graph_dir / "graph.json"
        self.graph_html = self.graph_dir / "graph.html"
        self.cache_file = self.graph_dir / ".cache.json"
        self.inferred_edges_file = self.graph_dir / ".inferred_edges.jsonl"
        self.log_file = self.wiki_dir / "log.md"
        
        # Initialize clients
        self.client = client or LLMClient()
        self.trajectory_logger = trajectory_logger or TrajectoryLogger()
        
        # Create graph directory
        self.graph_dir.mkdir(parents=True, exist_ok=True)
    
    def all_wiki_pages(self) -> list[Path]:
        """Get all wiki pages excluding index, log, and lint-report."""
        return [
            p for p in self.wiki_dir.rglob("*.md")
            if p.name not in ("index.md", "log.md", "lint-report.md")
        ]
    
    def page_id(self, path: Path) -> str:
        """Convert page path to node ID."""
        return path.relative_to(self.wiki_dir).as_posix().replace(".md", "")
    
    def load_cache(self) -> dict:
        """Load inference cache from disk."""
        if self.cache_file.exists():
            try:
                return json.loads(self.cache_file.read_text())
            except (json.JSONDecodeError, IOError):
                return {}
        return {}
    
    def save_cache(self, cache: dict):
        """Save inference cache to disk."""
        self.cache_file.write_text(json.dumps(cache, indent=2))

    def load_checkpoint(self) -> tuple[list[dict], set[str]]:
        """Load previously inferred edges from JSONL checkpoint file.

        Returns:
            Tuple of (edges list, set of completed page IDs)
        """
        edges = []
        completed = set()
        if self.inferred_edges_file.exists():
            for line in self.inferred_edges_file.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    completed.add(record["page_id"])
                    edges.extend(record.get("edges", []))
                except (json.JSONDecodeError, KeyError):
                    continue
        return edges, completed

    def append_checkpoint(self, page_id_str: str, edges: list[dict]) -> None:
        """Append one page's inferred edges to the JSONL checkpoint.

        Args:
            page_id_str: Page ID that was processed
            edges: List of inferred edges for this page
        """
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "page_id": page_id_str,
            "edges": edges,
            "ts": date.today().isoformat()
        }
        with open(self.inferred_edges_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    def build_nodes(self, pages: list[Path]) -> list[dict]:
        """Build node list from wiki pages."""
        nodes = []
        for p in pages:
            content = read_file(p)
            node_type = extract_frontmatter_type(content)
            title_match = re.search(r'^title:\s*"?([^"\n]+)"?', content, re.MULTILINE)
            label = title_match.group(1).strip() if title_match else p.stem
            nodes.append({
                "id": self.page_id(p),
                "label": label,
                "type": node_type,
                "color": TYPE_COLORS.get(node_type, TYPE_COLORS["unknown"]),
                "path": str(p.relative_to(self.repo_root)),
            })
        return nodes
    
    def build_extracted_edges(self, pages: list[Path]) -> list[dict]:
        """Pass 1: Build deterministic edges from explicit wikilinks."""
        # Build a map from stem (lower) -> page_id for resolution
        stem_map = {p.stem.lower(): self.page_id(p) for p in pages}
        edges = []
        seen = set()
        
        for p in pages:
            content = read_file(p)
            src = self.page_id(p)
            for link in extract_wikilinks(content):
                target = stem_map.get(link.lower())
                if target and target != src:
                    key = (src, target)
                    if key not in seen:
                        seen.add(key)
                        edges.append({
                            "from": src,
                            "to": target,
                            "type": "EXTRACTED",
                            "color": EDGE_COLORS["EXTRACTED"],
                            "confidence": 1.0,
                        })
        return edges
    
    def build_inferred_edges(
        self,
        pages: list[Path],
        existing_edges: list[dict],
        cache: dict,
        resume: bool = True
    ) -> list[dict]:
        """Pass 2: Infer semantic relationships via LLM with checkpoint/resume.

        Args:
            pages: List of wiki page paths
            existing_edges: List of already-extracted edges
            cache: Cache dictionary for SHA256 hashes
            resume: Whether to resume from checkpoint (default True)

        Returns:
            List of inferred edges
        """
        # Load checkpoint if resuming
        checkpoint_edges, completed_ids = ([], set())
        if resume:
            checkpoint_edges, completed_ids = self.load_checkpoint()
            if completed_ids:
                print(f"  checkpoint: {len(completed_ids)} pages already done, {len(checkpoint_edges)} edges loaded")

        new_edges = list(checkpoint_edges)

        # Only process pages that changed since last run
        changed_pages = []
        for p in pages:
            content = read_file(p)
            h = sha256(content)
            pid = self.page_id(p)
            entry = cache.get(str(p))

            if (not isinstance(entry, dict) or entry.get("hash") != h) and pid not in completed_ids:
                changed_pages.append(p)
            elif pid in completed_ids:
                # Already in checkpoint, skip
                continue
            else:
                # Page unchanged: load its inferred edges from cache
                src = self.page_id(p)
                for rel in entry.get("edges", []):
                    new_edges.append({
                        "from": src,
                        "to": rel["to"],
                        "type": rel.get("type", "INFERRED"),
                        "title": rel.get("relationship", ""),
                        "label": "",
                        "color": EDGE_COLORS.get(rel.get("type", "INFERRED"), EDGE_COLORS["INFERRED"]),
                        "confidence": float(rel.get("confidence", 0.7)),
                    })

        if not changed_pages:
            print("  no changed pages — skipping semantic inference")
            return new_edges

        total_pages = len(changed_pages)
        already_done = len(completed_ids)
        grand_total = total_pages + already_done
        print(f"  inferring relationships for {total_pages} remaining pages (of {grand_total} total)...")

        # Build a summary of existing nodes for context
        node_list = "\n".join(
            f"- {self.page_id(p)} ({extract_frontmatter_type(read_file(p))})"
            for p in pages
        )
        existing_edge_summary = "\n".join(
            f"- {e['from']} → {e['to']} (EXTRACTED)"
            for e in existing_edges[:30]
        )

        for i, p in enumerate(changed_pages, 1):
            content = read_file(p)[:2000]  # truncate for context efficiency
            src = self.page_id(p)
            global_idx = already_done + i
            print(f"    [{global_idx}/{grand_total}] Inferring for '{src}'... ", end="", flush=True)

            prompt = f"""Analyze this wiki page and identify implicit semantic relationships to other pages in the wiki.

Source page: {src}
Content:
{content}

All available pages:
{node_list}

Already-extracted edges from this page:
{existing_edge_summary}

Return ONLY a JSON object containing an "edges" array of NEW relationships not already captured by explicit wikilinks. The response must be STRICTLY valid JSON formatted exactly like this:
{{
  "edges": [
    {{"to": "page-id", "relationship": "one-line description", "confidence": 0.0-1.0, "type": "INFERRED or AMBIGUOUS"}}
  ]
}}

CRITICAL INSTRUCTION:
YOU MUST RETURN ONLY A RAW JSON STRING BEGINNING WITH {{ AND ENDING WITH }}.
DO NOT OUTPUT BULLET POINTS. DO NOT OUTPUT MARKDOWN LISTS.
ANY CONVERSATIONAL PREAMBLE WILL CAUSE A SYSTEM CRASH.

Rules:
- Only include pages from the available list above
- Confidence >= 0.7 → INFERRED, < 0.7 → AMBIGUOUS
- Do not repeat edges already in the extracted list
- Return {{"edges": []}} if no new relationships found
"""
            page_edges = []
            try:
                # Log the cycle
                self.trajectory_logger.log_cycle(thought=prompt, action="infer_edges")

                result: CallResult = self.client.call(
                    prompt=prompt,
                    max_tokens=1024,
                    model=self.client.default_model_fast
                )

                # Update metrics
                self.trajectory_logger.update_metrics(
                    prompt_tokens=result.usage.prompt_tokens,
                    completion_tokens=result.usage.completion_tokens,
                    latency_ms=result.latency_ms
                )

                raw = result.content.strip()
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)

                # Robust JSON extraction
                match = re.search(r"\{.*\}", raw, re.DOTALL)
                if match:
                    raw = match.group(0)

                inferred = json.loads(raw)
                edges_list = inferred.get("edges", [])
                for rel in edges_list:
                    if isinstance(rel, dict) and "to" in rel:
                        edge = {
                            "from": src,
                            "to": rel["to"],
                            "type": rel.get("type", "INFERRED"),
                            "title": rel.get("relationship", ""),
                            "label": "",
                            "color": EDGE_COLORS.get(rel.get("type", "INFERRED"), EDGE_COLORS["INFERRED"]),
                            "confidence": rel.get("confidence", 0.7),
                        }
                        page_edges.append(edge)
                        new_edges.append(edge)
                print(f"-> Found {len(page_edges)} edges.")

                # Save properly to cache
                cache[str(p)] = {
                    "hash": sha256(read_file(p)),
                    "edges": edges_list
                }

                # Log observation
                self.trajectory_logger.log_cycle(
                    thought="",
                    action="edges_inferred",
                    observation=f"Inferred {len(page_edges)} edges for {src}"
                )
            except json.JSONDecodeError as jde:
                print(f"-> [WARN] Invalid JSON: {str(jde)[:60]}")
            except Exception as e:
                err_msg = str(e).replace('\n', ' ')[:80]
                print(f"-> [ERROR] {err_msg}")

            # Persist checkpoint immediately after each page
            self.append_checkpoint(src, page_edges)

        return new_edges
    
    def detect_communities(self, nodes: list[dict], edges: list[dict]) -> dict[str, int]:
        """Assign community IDs to nodes using Louvain algorithm."""
        if not HAS_NETWORKX:
            print("  Warning: networkx not installed. Community detection skipped.")
            return {}
        
        G = nx.Graph()
        for n in nodes:
            G.add_node(n["id"])
        for e in edges:
            G.add_edge(e["from"], e["to"])
        
        if G.number_of_edges() == 0:
            return {}
        
        try:
            communities = nx_community.louvain_communities(G, seed=42)
            node_to_community = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    node_to_community[node] = i
            return node_to_community
        except Exception:
            return {}

    def deduplicate_edges(self, edges: list[dict]) -> list[dict]:
        """Merge duplicate and bidirectional edges, keeping highest confidence.

        Args:
            edges: List of edge dictionaries

        Returns:
            Deduplicated list of edges
        """
        best = {}  # (min(a,b), max(a,b)) -> edge
        for e in edges:
            a, b = e["from"], e["to"]
            key = (min(a, b), max(a, b))
            existing = best.get(key)
            if not existing or e.get("confidence", 0) > existing.get("confidence", 0):
                best[key] = e
        return list(best.values())

    def render_html(self, nodes: list[dict], edges: list[dict]) -> str:
        """Generate self-contained vis.js HTML visualization."""
        nodes_json = json.dumps(nodes, indent=2)
        edges_json = json.dumps(edges, indent=2)
        
        legend_items = "".join(
            f'<span style="background:{color};padding:3px 8px;margin:2px;border-radius:3px;font-size:12px">{t}</span>'
            for t, color in TYPE_COLORS.items() if t != "unknown"
        )
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>LLM Wiki — Knowledge Graph</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  body {{ margin: 0; background: #1a1a2e; font-family: sans-serif; color: #eee; }}
  #graph {{ width: 100vw; height: 100vh; }}
  #controls {{
    position: fixed; top: 10px; left: 10px; background: rgba(0,0,0,0.7);
    padding: 12px; border-radius: 8px; z-index: 10; max-width: 260px;
  }}
  #controls h3 {{ margin: 0 0 8px; font-size: 14px; }}
  #search {{ width: 100%; padding: 4px; margin-bottom: 8px; background: #333; color: #eee; border: 1px solid #555; border-radius: 4px; }}
  #info {{
    position: fixed; bottom: 10px; left: 10px; background: rgba(0,0,0,0.8);
    padding: 12px; border-radius: 8px; z-index: 10; max-width: 320px;
    display: none;
  }}
  #stats {{ position: fixed; top: 10px; right: 10px; background: rgba(0,0,0,0.7); padding: 10px; border-radius: 8px; font-size: 12px; }}
</style>
</head>
<body>
<div id="controls">
  <h3>LLM Wiki Graph</h3>
  <input id="search" type="text" placeholder="Search nodes..." oninput="searchNodes(this.value)">
  <div>{legend_items}</div>
  <div style="margin-top:8px;font-size:11px;color:#aaa">
    <span style="background:#555;padding:2px 6px;border-radius:3px;margin-right:4px">──</span> Explicit link<br>
    <span style="background:#FF5722;padding:2px 6px;border-radius:3px;margin-right:4px">──</span> Inferred
  </div>
</div>
<div id="graph"></div>
<div id="info">
  <b id="info-title"></b><br>
  <span id="info-type" style="font-size:12px;color:#aaa"></span><br>
  <span id="info-path" style="font-size:11px;color:#666"></span>
</div>
<div id="stats"></div>
<script>
const nodes = new vis.DataSet({nodes_json});
const edges = new vis.DataSet({edges_json});

const container = document.getElementById("graph");
const network = new vis.Network(container, {{ nodes, edges }}, {{
  nodes: {{
    shape: "dot",
    size: 12,
    font: {{ color: "#eee", size: 13 }},
    borderWidth: 2,
  }},
  edges: {{
    width: 1.2,
    smooth: {{ type: "continuous" }},
    arrows: {{ to: {{ enabled: true, scaleFactor: 0.5 }} }},
  }},
  physics: {{
    stabilization: {{ iterations: 150 }},
    barnesHut: {{ gravitationalConstant: -8000, springLength: 120 }},
  }},
  interaction: {{ hover: true, tooltipDelay: 200 }},
}});

network.on("click", params => {{
  if (params.nodes.length > 0) {{
    const node = nodes.get(params.nodes[0]);
    document.getElementById("info").style.display = "block";
    document.getElementById("info-title").textContent = node.label;
    document.getElementById("info-type").textContent = node.type;
    document.getElementById("info-path").textContent = node.path;
  }} else {{
    document.getElementById("info").style.display = "none";
  }}
}});

document.getElementById("stats").textContent =
  `${{nodes.length}} nodes · ${{edges.length}} edges`;

function searchNodes(q) {{
  const lower = q.toLowerCase();
  nodes.forEach(n => {{
    nodes.update({{ id: n.id, opacity: (!q || n.label.toLowerCase().includes(lower)) ? 1 : 0.15 }});
  }});
}}
</script>
</body>
</html>"""
    
    def append_log(self, entry: str):
        """Prepend new entry to wiki/log.md."""
        existing = read_file(self.log_file)
        write_file(self.log_file, entry.strip() + "\n\n" + existing)
    
    def build_graph(self, infer: bool = True) -> dict:
        """Build the knowledge graph.
        
        Args:
            infer: Whether to run semantic inference (Pass 2)
            
        Returns:
            Graph data dictionary with nodes, edges, and metadata
        """
        pages = self.all_wiki_pages()
        today = date.today().isoformat()
        
        if not pages:
            print("Wiki is empty. Ingest some sources first.")
            return {"nodes": [], "edges": [], "built": today}
        
        print(f"Building graph from {len(pages)} wiki pages...")
        
        cache = self.load_cache()
        
        # Pass 1: extracted edges
        print("  Pass 1: extracting wikilinks...")
        nodes = self.build_nodes(pages)
        edges = self.build_extracted_edges(pages)
        print(f"  → {len(edges)} extracted edges")
        
        # Pass 2: inferred edges
        if infer:
            print("  Pass 2: inferring semantic relationships...")
            inferred = self.build_inferred_edges(pages, edges, cache)
            edges.extend(inferred)
            print(f"  → {len(inferred)} inferred edges")
            self.save_cache(cache)

        # Deduplicate edges
        before_dedup = len(edges)
        edges = self.deduplicate_edges(edges)
        if before_dedup != len(edges):
            print(f"  dedup: {before_dedup} → {len(edges)} edges")

        # Community detection
        print("  Running Louvain community detection...")
        communities = self.detect_communities(nodes, edges)
        for node in nodes:
            comm_id = communities.get(node["id"], -1)
            if comm_id >= 0:
                node["color"] = COMMUNITY_COLORS[comm_id % len(COMMUNITY_COLORS)]
            node["group"] = comm_id
        
        # Save graph.json
        graph_data = {"nodes": nodes, "edges": edges, "built": today}
        self.graph_json.write_text(json.dumps(graph_data, indent=2))
        print(f"  saved: graph/graph.json  ({len(nodes)} nodes, {len(edges)} edges)")
        
        # Save graph.html
        html = self.render_html(nodes, edges)
        self.graph_html.write_text(html)
        print(f"  saved: graph/graph.html")
        
        # Append to log
        extracted_count = len([e for e in edges if e['type'] == 'EXTRACTED'])
        inferred_count = len([e for e in edges if e['type'] == 'INFERRED'])
        self.append_log(
            f"## [{today}] graph | Knowledge graph rebuilt\n\n"
            f"{len(nodes)} nodes, {len(edges)} edges "
            f"({extracted_count} extracted, {inferred_count} inferred)."
        )
        
        return graph_data


def write_file(path: Path, content: str):
    """Create parent directories if needed and write content to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
