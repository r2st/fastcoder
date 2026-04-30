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

import json
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
        self.semantic_marker = self.cache_dir / "semantic_done.json"

        self._build_lock = threading.Lock()
        self._build_attempted = False
        self._build_failed = False
        self._graph: Any = None  # NetworkX graph, loaded lazily
        self._communities: Optional[dict[int, list[str]]] = None
        self._god_nodes_cache: Optional[list[dict[str, Any]]] = None

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

    # ── Risk / hint helpers (used by Planner & Reviewer) ─────────

    def god_nodes(self, top_n: int = 10) -> list[dict[str, Any]]:
        """Return the most-connected nodes — the codebase's core abstractions.

        Returns an empty list when graphify is unavailable or the graph
        has not been built. Cached after first call.
        """
        if not self.is_available() or not self._ensure_graph():
            return []
        if self._god_nodes_cache is not None:
            return self._god_nodes_cache
        graph = self._graph
        scored = sorted(
            ((graph.degree(n), n) for n in graph.nodes()),
            reverse=True,
        )
        result: list[dict[str, Any]] = []
        for degree, nid in scored[:top_n]:
            d = graph.nodes[nid]
            result.append(
                {
                    "id": nid,
                    "label": d.get("label", nid),
                    "source_file": d.get("source_file"),
                    "degree": int(degree),
                }
            )
        self._god_nodes_cache = result
        return result

    def community_hint(self, text: str, *, top_n: int = 3) -> Optional[str]:
        """Return a short hint about which communities this text touches.

        Used by the Planner to nudge task decomposition along community
        boundaries. Returns None when graphify is unavailable or no
        match is found.
        """
        if not self.is_available() or not self._ensure_graph():
            return None
        graph = self._graph
        if graph.number_of_nodes() == 0:
            return None

        terms = [t.lower() for t in _tokenize(text) if len(t) > 3]
        if not terms:
            return None

        # Build node -> community map from graph node attributes (graphify
        # writes "community" into to_json output).
        community_hits: dict[Any, int] = {}
        community_labels: dict[Any, str] = {}
        for nid, data in graph.nodes(data=True):
            cid = data.get("community")
            if cid is None:
                continue
            label = (data.get("label") or "").lower()
            score = sum(1 for t in terms if t in label)
            if score:
                community_hits[cid] = community_hits.get(cid, 0) + score
                community_labels.setdefault(cid, data.get("community_label") or f"community-{cid}")

        if not community_hits:
            return None

        ranked = sorted(community_hits.items(), key=lambda kv: -kv[1])[:top_n]
        lines = ["Module landscape hint (from knowledge graph):"]
        for cid, score in ranked:
            label = community_labels.get(cid, f"community-{cid}")
            lines.append(f"  - touches {label} (score {score})")
        lines.append(
            "Consider decomposing tasks along these module boundaries."
        )
        return "\n".join(lines)

    def blast_radius(self, file_paths: list[str]) -> Optional[str]:
        """Return a blast-radius warning if changes touch high-degree nodes.

        Used by the Reviewer to flag risky PRs. Returns None when graphify
        is unavailable, no nodes match, or no hits are above the high-
        degree threshold.
        """
        if not self.is_available() or not self._ensure_graph():
            return None
        graph = self._graph
        if graph.number_of_nodes() == 0:
            return None

        # Normalise paths to suffix matches; node source_file may be
        # absolute or repo-relative.
        targets = {Path(p).as_posix() for p in file_paths if p}
        if not targets:
            return None

        # Compute the degree threshold once (we count anything in the
        # 90th percentile as "high-degree").
        degrees = sorted((graph.degree(n) for n in graph.nodes()), reverse=True)
        if not degrees:
            return None
        threshold = degrees[max(0, int(len(degrees) * 0.1) - 1)]
        threshold = max(threshold, 5)  # never flag trivially small graphs

        hits: list[tuple[str, int, str]] = []  # (label, degree, source_file)
        for nid, data in graph.nodes(data=True):
            sf = data.get("source_file") or ""
            sf_pos = Path(sf).as_posix() if sf else ""
            if not sf_pos:
                continue
            if not any(sf_pos.endswith(t) or t.endswith(sf_pos) for t in targets):
                continue
            d = graph.degree(nid)
            if d >= threshold:
                hits.append((data.get("label", nid), int(d), sf_pos))

        if not hits:
            return None

        hits.sort(key=lambda h: -h[1])
        lines = [
            "Blast radius (from knowledge graph):",
            f"  This PR touches {len(hits)} high-degree node(s) — review carefully.",
        ]
        for label, degree, sf in hits[:5]:
            lines.append(f"  - {label} ({degree} edges) in {sf}")
        return "\n".join(lines)

    # ── Semantic enrichment ──────────────────────────────────────

    def semantic_enrichment_done(self) -> bool:
        return self.semantic_marker.exists()

    def enrich_semantic(
        self,
        llm_complete: Any,
        *,
        max_input_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        """Run LLM-driven semantic enrichment over the existing graph.

        Walks code/doc files in chunks, asks the configured LLM to extract
        semantic edges that AST cannot find, and merges results back into
        graph.json. Stops early once `max_input_tokens` is exhausted.

        This is intended to be triggered explicitly (e.g. from the admin
        panel), never as a side-effect of context queries — that way the
        token spend is observable and capped.

        Args:
            llm_complete: Async callable matching ContextManager's
                ``llm_complete(messages, metadata) -> CompletionResponse``.
            max_input_tokens: Override config.semantic_max_input_tokens.

        Returns:
            Summary dict: {nodes_added, edges_added, files_processed,
            input_tokens_used, output_tokens_used, stopped_early}.
        """
        result = {
            "nodes_added": 0,
            "edges_added": 0,
            "files_processed": 0,
            "input_tokens_used": 0,
            "output_tokens_used": 0,
            "stopped_early": False,
            "error": None,
        }
        if not self.is_available():
            result["error"] = "graphify unavailable"
            return result
        if not self.config.semantic_extraction:
            result["error"] = "semantic_extraction disabled in config"
            return result
        if not self._ensure_graph():
            result["error"] = "graph not built"
            return result

        budget = max_input_tokens or self.config.semantic_max_input_tokens

        try:
            from graphify.detect import detect
        except ImportError:
            result["error"] = "graphify not installed"
            return result

        try:
            detection = detect(self.project_dir)
        except Exception as e:
            result["error"] = f"detect failed: {e}"
            return result

        # Files to enrich: docs, papers, plus code (semantic edges code
        # AST misses). Skip files we've already enriched (tracked in
        # marker file).
        already: set[str] = set()
        if self.semantic_marker.exists():
            try:
                marker = json.loads(self.semantic_marker.read_text())
                already = set(marker.get("processed_files", []))
            except Exception:
                already = set()

        files = []
        for cat in ("document", "paper", "code"):
            files.extend(detection.get("files", {}).get(cat, []))
        files = [f for f in files if f not in already]
        if not files:
            self.semantic_marker.write_text(
                json.dumps({"processed_files": list(already)})
            )
            return result

        # Chunk files for sequential processing — small chunks so a
        # single oversized response can't blow our budget.
        chunk_size = 6
        chunks = [files[i : i + chunk_size] for i in range(0, len(files), chunk_size)]

        new_nodes: list[dict[str, Any]] = []
        new_edges: list[dict[str, Any]] = []
        processed: list[str] = []
        tokens_used = 0

        # Best-effort import; if not available the loop just exits empty.
        try:
            from fastcoder.types.llm import CompletionRequest, Message
        except ImportError:  # pragma: no cover
            result["error"] = "fastcoder LLM types unavailable"
            return result

        import asyncio

        for chunk in chunks:
            if tokens_used >= budget:
                result["stopped_early"] = True
                break

            prompt = _build_semantic_prompt(chunk, self.project_dir)
            estimated = max(1, len(prompt) // 4)
            if tokens_used + estimated > budget:
                result["stopped_early"] = True
                break

            try:
                request = CompletionRequest(
                    model="",
                    messages=[
                        Message(role="system", content=_SEMANTIC_SYSTEM_PROMPT),
                        Message(role="user", content=prompt),
                    ],
                    max_tokens=2000,
                    temperature=0.2,
                )
                # llm_complete is typically async. Run it sync if needed.
                response = asyncio.run(
                    _await_completion(llm_complete, request, {"purpose": "graphify_enrich"})
                )
            except Exception as e:
                logger.warning("semantic enrichment chunk failed: %s", e)
                continue

            usage = getattr(response, "usage", None) or {}
            in_tok = int(usage.get("input_tokens", estimated)) if isinstance(usage, dict) else estimated
            out_tok = int(usage.get("output_tokens", 0)) if isinstance(usage, dict) else 0
            tokens_used += in_tok
            result["input_tokens_used"] += in_tok
            result["output_tokens_used"] += out_tok

            chunk_nodes, chunk_edges = _parse_semantic_response(
                getattr(response, "content", "") or ""
            )
            new_nodes.extend(chunk_nodes)
            new_edges.extend(chunk_edges)
            processed.extend(chunk)
            result["files_processed"] += len(chunk)

        # Merge into graph & persist
        if new_nodes or new_edges:
            self._merge_into_graph(new_nodes, new_edges)
            result["nodes_added"] = len(new_nodes)
            result["edges_added"] = len(new_edges)

        # Update marker
        all_processed = sorted(already.union(processed))
        try:
            self.semantic_marker.write_text(
                json.dumps({"processed_files": all_processed})
            )
        except Exception as e:  # pragma: no cover — defensive
            logger.warning("failed to write semantic marker: %s", e)

        return result

    def _merge_into_graph(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> None:
        """Merge nodes/edges into the in-memory graph and persist to disk."""
        graph = self._graph
        if graph is None:
            return
        for n in nodes:
            nid = n.get("id")
            if not nid or nid in graph.nodes:
                continue
            attrs = {k: v for k, v in n.items() if k != "id"}
            graph.add_node(nid, **attrs)
        for e in edges:
            u = e.get("source")
            v = e.get("target")
            if not u or not v or u not in graph.nodes or v not in graph.nodes:
                continue
            attrs = {k: val for k, val in e.items() if k not in ("source", "target")}
            graph.add_edge(u, v, **attrs)
        # Persist (compat: newer networkx accepts edges= ; older does not)
        try:
            from networkx.readwrite import json_graph

            try:
                data = json_graph.node_link_data(graph, edges="links")
            except TypeError:
                data = json_graph.node_link_data(graph)
            self.graph_path.write_text(json.dumps(data, indent=2))
        except Exception as e:  # pragma: no cover — defensive
            logger.warning("failed to persist enriched graph: %s", e)

    # ── Git post-commit hook ─────────────────────────────────────

    @property
    def hook_path(self) -> Path:
        return self.project_dir / ".git" / "hooks" / "post-commit"

    @property
    def hook_marker(self) -> str:
        return "# fastcoder-graphify-hook"

    def sync_hook_state(self) -> str:
        """Install or remove the hook based on current config. Returns status string."""
        if self.config.enabled and self.config.auto_rebuild_on_commit:
            return self.install_git_hook()
        return self.uninstall_git_hook()

    def install_git_hook(self) -> str:
        """Install a post-commit hook that incrementally rebuilds the graph.

        Idempotent. Appends to an existing post-commit hook rather than
        replacing it. Returns status string.
        """
        hooks_dir = self.project_dir / ".git" / "hooks"
        if not hooks_dir.exists():
            return "skipped: not a git repo"
        hook = self.hook_path
        existing = hook.read_text() if hook.exists() else ""
        if self.hook_marker in existing:
            return "already installed"

        block = (
            f"\n{self.hook_marker} BEGIN\n"
            f"# Auto-rebuild graphify graph after commit. "
            f"Remove this block to disable.\n"
            f"if command -v graphify >/dev/null 2>&1; then\n"
            f"  ( cd {self.project_dir!s} && graphify --update >/dev/null 2>&1 ) &\n"
            f"fi\n"
            f"{self.hook_marker} END\n"
        )

        if existing:
            new_content = existing.rstrip() + "\n" + block
        else:
            new_content = "#!/usr/bin/env bash\nset -e\n" + block

        hook.write_text(new_content)
        try:
            hook.chmod(0o755)
        except OSError:  # pragma: no cover
            pass
        return "installed"

    def uninstall_git_hook(self) -> str:
        """Remove the fastcoder block from the post-commit hook (if present)."""
        hook = self.hook_path
        if not hook.exists():
            return "not present"
        content = hook.read_text()
        marker_begin = f"{self.hook_marker} BEGIN"
        marker_end = f"{self.hook_marker} END"
        if marker_begin not in content:
            return "not present"

        lines = content.splitlines(keepends=True)
        out_lines: list[str] = []
        skipping = False
        for line in lines:
            if marker_begin in line:
                skipping = True
                continue
            if marker_end in line:
                skipping = False
                continue
            if not skipping:
                out_lines.append(line)
        hook.write_text("".join(out_lines))
        return "uninstalled"


# ── Helpers ──────────────────────────────────────────────────────

_SEMANTIC_SYSTEM_PROMPT = """You are a knowledge-graph extraction assistant.
Read the supplied source files and emit semantic edges between named
entities (functions, classes, concepts) that cannot be inferred from
imports alone — call relationships across modules, shared data
structures, design patterns, conceptual bridges between docs and code.

Output JSON only, no markdown. Schema:
{"nodes": [{"id":"<lowercase_snake_id>","label":"<readable>","source_file":"<rel>"}],
 "edges": [{"source":"<id>","target":"<id>","relation":"calls|implements|references|conceptually_related_to|shares_data_with","confidence":"EXTRACTED|INFERRED|AMBIGUOUS","confidence_score":0.0-1.0}]}
Keep output under 200 nodes/edges per response. Prefer high-confidence
edges. Skip if the chunk yields nothing useful."""


def _build_semantic_prompt(files: list[str], project_dir: Path) -> str:
    parts = ["Extract semantic edges from the following files.\n"]
    for f in files:
        try:
            content = Path(f).read_text(errors="replace")
        except Exception:
            continue
        # Cap per-file size so a single huge file can't blow the budget.
        if len(content) > 8000:
            content = content[:8000] + "\n... [truncated]\n"
        try:
            rel = Path(f).resolve().relative_to(project_dir)
            display = str(rel)
        except ValueError:
            display = f
        parts.append(f"\n=== FILE: {display} ===\n{content}\n")
    return "".join(parts)


def _parse_semantic_response(text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Best-effort extract of {nodes, edges} JSON from a model response."""
    if not text:
        return [], []
    # Try fenced block first, then raw.
    candidates = [text]
    fence_start = text.find("{")
    fence_end = text.rfind("}")
    if 0 <= fence_start < fence_end:
        candidates.append(text[fence_start : fence_end + 1])
    for cand in candidates:
        try:
            parsed = json.loads(cand)
            nodes = parsed.get("nodes") or []
            edges = parsed.get("edges") or []
            if isinstance(nodes, list) and isinstance(edges, list):
                return nodes, edges
        except Exception:
            continue
    return [], []


async def _await_completion(llm_complete: Any, request: Any, metadata: dict) -> Any:
    """Adapter: tolerate llm_complete signatures that take either
    (messages, metadata) or (CompletionRequest)."""
    try:
        result = llm_complete(request)
    except TypeError:
        result = llm_complete(getattr(request, "messages", []), metadata)
    if asyncio_iscoroutine(result):
        return await result
    return result


def asyncio_iscoroutine(obj: Any) -> bool:
    import asyncio

    return asyncio.iscoroutine(obj)


# ── Helpers (existing) ────────────────────────────────────────────


def _graphify_installed() -> bool:
    try:
        import graphify  # noqa: F401

        return True
    except ImportError:
        return False


def _load_graph(path: Path) -> Any:
    """Load a graphify graph.json from disk. Returns None on any failure.

    networkx renamed the link-array key in 3.4; we try the new signature
    first and fall back so this works on either version.
    """
    try:
        import json

        from networkx.readwrite import json_graph
    except ImportError:
        return None
    try:
        data = json.loads(path.read_text())
    except Exception as e:  # pragma: no cover — defensive
        logger.warning("graphify: failed to read %s: %s", path, e)
        return None
    # networkx >= 3.4 accepts edges= ; older versions need the legacy default.
    try:
        return json_graph.node_link_graph(data, edges="links")
    except TypeError:
        try:
            return json_graph.node_link_graph(data)
        except Exception as e:  # pragma: no cover — defensive
            logger.warning("graphify: failed to parse %s: %s", path, e)
            return None
    except Exception as e:  # pragma: no cover — defensive
        logger.warning("graphify: failed to parse %s: %s", path, e)
        return None


def _tokenize(text: str) -> list[str]:
    """Cheap tokenizer for query terms."""
    import re

    return re.findall(r"[A-Za-z][A-Za-z0-9_]+", text)
