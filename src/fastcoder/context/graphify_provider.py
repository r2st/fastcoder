"""Graphify provider — optional knowledge-graph-aware context retrieval.

Wraps the optional `graphify` package behind a stable interface so the
ContextManager can call it without caring whether graphify is installed,
enabled, or has produced a graph yet.

Behavioural contract:
- All public methods are safe to call regardless of install/enable state.
- When disabled, missing, or below-threshold, every method returns None
  or an empty result — the caller should fall back to the legacy file-dump
  path. We never raise on the hot path.
- The first call against a target repo builds the graph (AST-only by
  default to avoid surprise LLM token spend; semantic extraction is a
  separate opt-in driven by GraphifyConfig.semantic_extraction).
- Subsequent calls reuse the cached graph from disk.
- Build failures are logged and recorded so we don't retry on every call.

See: https://github.com/safishamsi/graphify
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Optional

from fastcoder.types.config import GraphifyConfig

logger = logging.getLogger(__name__)


class GraphifyProvider:
    """Lazily build and query a graphify graph for a target repo.

    Thread-safe: build is guarded by a lock so concurrent stories don't
    trigger duplicate graph construction.
    """

    def __init__(self, config: GraphifyConfig, project_dir: str | Path):
        self.config = config
        self.project_dir = Path(project_dir).resolve()
        self.cache_dir = self.project_dir / config.cache_dir
        self.graph_path = self.cache_dir / "graph.json"

        self._build_lock = threading.Lock()
        self._build_attempted = False
        self._build_failed = False
        self._graph: Any = None  # NetworkX graph, loaded lazily

    # ── Public API ───────────────────────────────────────────────

    def is_available(self) -> bool:
        """True iff graphify is enabled, installed, and not in a failed state."""
        if not self.config.enabled:
            return False
        if self._build_failed:
            return False
        return _graphify_installed()

    def query_context(
        self,
        question: str,
        *,
        budget: Optional[int] = None,
    ) -> Optional[str]:
        """Return a graph-derived context block for the question, or None.

        Returns None whenever graphify is disabled, the package is missing,
        the corpus is below the size threshold, the graph could not be
        built, or no relevant nodes were found. Callers should fall back
        to whatever legacy context they had on hand.

        Args:
            question: Free-text query (story description, task summary, …).
            budget: Override token budget for this query. Defaults to
                config.query_token_budget.
        """
        if not self.is_available():
            return None

        if not self._ensure_graph():
            return None

        try:
            return self._query(question, budget=budget or self.config.query_token_budget)
        except Exception as e:  # pragma: no cover — defensive
            logger.warning("graphify query failed: %s", e)
            return None

    # ── Build ────────────────────────────────────────────────────

    def _ensure_graph(self) -> bool:
        """Load or build the graph. Returns True if a graph is available."""
        if self._graph is not None:
            return True

        with self._build_lock:
            # Re-check inside the lock — another thread may have built it.
            if self._graph is not None:
                return True
            if self._build_attempted:
                # Previous attempt finished (success or fail).
                return self._graph is not None

            self._build_attempted = True

            if self.graph_path.exists():
                self._graph = _load_graph(self.graph_path)
                if self._graph is not None:
                    logger.info("graphify: loaded existing graph from %s", self.graph_path)
                    return True
                logger.warning("graphify: cache exists but failed to load; rebuilding")

            built = self._build_graph()
            if not built:
                self._build_failed = True
                return False
            self._graph = _load_graph(self.graph_path)
            return self._graph is not None

    def _build_graph(self) -> bool:
        """Run AST extraction on the project. Skip if corpus is too small.

        Returns True if a graph was successfully written to disk.
        """
        try:
            from graphify.build import build_from_json
            from graphify.cluster import cluster
            from graphify.detect import detect
            from graphify.export import to_json
            from graphify.extract import collect_files, extract
        except ImportError:
            logger.warning("graphify not installed — context queries disabled")
            return False

        try:
            detection = detect(self.project_dir)
        except Exception as e:
            logger.warning("graphify detect failed: %s", e)
            return False

        words = detection.get("total_words", 0)
        if words < self.config.min_corpus_words:
            logger.info(
                "graphify: corpus too small (%d words < %d threshold) — skipping graph build",
                words,
                self.config.min_corpus_words,
            )
            return False

        code_files: list[Path] = []
        for f in detection.get("files", {}).get("code", []):
            p = Path(f)
            code_files.extend(collect_files(p) if p.is_dir() else [p])
        if not code_files:
            logger.info("graphify: no code files detected — skipping")
            return False

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            extraction = extract(code_files, cache_root=self.cache_dir)
            # Phase 2 ships AST-only. Semantic extraction (which costs LLM
            # tokens) can be wired in a follow-up that drives Kimi or
            # in-process Claude with proper guardrails.
            graph = build_from_json(extraction)
            communities = cluster(graph)
            to_json(graph, communities, str(self.graph_path))
            logger.info(
                "graphify: built graph (%d nodes, %d edges) at %s",
                graph.number_of_nodes(),
                graph.number_of_edges(),
                self.graph_path,
            )
            return True
        except Exception as e:
            logger.warning("graphify build failed: %s", e)
            return False

    # ── Query ────────────────────────────────────────────────────

    def _query(self, question: str, *, budget: int) -> Optional[str]:
        """Run a BFS traversal scored by term overlap with the question.

        Returns a formatted context string suitable for injection into the
        LLM prompt, or None if no relevant nodes were found.
        """
        graph = self._graph
        if graph is None or graph.number_of_nodes() == 0:
            return None

        terms = [t.lower() for t in _tokenize(question) if len(t) > 3]
        if not terms:
            return None

        scored: list[tuple[int, str]] = []
        for nid, data in graph.nodes(data=True):
            label = (data.get("label") or "").lower()
            score = sum(1 for t in terms if t in label)
            if score > 0:
                scored.append((score, nid))
        scored.sort(reverse=True)
        start_nodes = [nid for _, nid in scored[:3]]
        if not start_nodes:
            return None

        # BFS to depth 3
        subgraph_nodes: set[str] = set(start_nodes)
        subgraph_edges: list[tuple[str, str]] = []
        frontier = set(start_nodes)
        for _ in range(3):
            next_frontier: set[str] = set()
            for n in frontier:
                for neighbor in graph.neighbors(n):
                    if neighbor not in subgraph_nodes:
                        next_frontier.add(neighbor)
                        subgraph_edges.append((n, neighbor))
            subgraph_nodes.update(next_frontier)
            frontier = next_frontier

        # Rank by relevance, format, and truncate to budget.
        def rel(nid: str) -> int:
            label = (graph.nodes[nid].get("label") or "").lower()
            return sum(1 for t in terms if t in label)

        ranked = sorted(subgraph_nodes, key=rel, reverse=True)

        char_budget = budget * 4  # ~4 chars/token
        lines: list[str] = [
            "Knowledge-graph context (graphify):",
            f"Started from {len(start_nodes)} relevant nodes; expanded to {len(subgraph_nodes)} nodes.",
            "",
            "Nodes:",
        ]
        for nid in ranked:
            d = graph.nodes[nid]
            label = d.get("label", nid)
            src = d.get("source_file") or ""
            loc = d.get("source_location") or ""
            lines.append(f"  - {label}  [{src}{':' + str(loc) if loc else ''}]")
        lines.append("")
        lines.append("Edges:")
        for u, v in subgraph_edges:
            if u not in subgraph_nodes or v not in subgraph_nodes:
                continue
            d = graph.edges[u, v]
            rel_kind = d.get("relation", "")
            conf = d.get("confidence", "")
            ulabel = graph.nodes[u].get("label", u)
            vlabel = graph.nodes[v].get("label", v)
            lines.append(f"  {ulabel} --{rel_kind} [{conf}]--> {vlabel}")

        text = "\n".join(lines)
        if len(text) > char_budget:
            text = text[:char_budget].rstrip() + "\n... (truncated to query budget)"
        return text


# ── Helpers ──────────────────────────────────────────────────────


def _graphify_installed() -> bool:
    try:
        import graphify  # noqa: F401

        return True
    except ImportError:
        return False


def _load_graph(path: Path) -> Any:
    """Load a graphify graph.json from disk. Returns None on any failure."""
    try:
        import json

        import networkx as nx
        from networkx.readwrite import json_graph
    except ImportError:
        return None
    try:
        data = json.loads(path.read_text())
        return json_graph.node_link_graph(data, edges="links")
    except Exception as e:  # pragma: no cover — defensive
        logger.warning("graphify: failed to load %s: %s", path, e)
        return None


def _tokenize(text: str) -> list[str]:
    """Cheap tokenizer for query terms."""
    import re

    return re.findall(r"[A-Za-z][A-Za-z0-9_]+", text)
