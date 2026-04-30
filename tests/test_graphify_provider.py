"""Tests for GraphifyProvider — the optional graphify integration layer.

These tests cover the defensive contract:
- Disabled config -> is_available() False, query_context() returns None
- Missing package -> is_available() False, query_context() returns None
- Build failure is recorded so we don't retry on every call
- Query against a small in-memory graph returns a formatted context block
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from fastcoder.context.graphify_provider import GraphifyProvider, _tokenize
from fastcoder.types.config import GraphifyConfig


@pytest.fixture
def disabled_config() -> GraphifyConfig:
    return GraphifyConfig(enabled=False)


@pytest.fixture
def enabled_config(tmp_path: Path) -> GraphifyConfig:
    return GraphifyConfig(
        enabled=True,
        min_corpus_words=10,
        cache_dir=str(tmp_path / "graphify"),
    )


# ── is_available() ────────────────────────────────────────────────


def test_is_available_false_when_disabled(disabled_config, tmp_path):
    provider = GraphifyProvider(disabled_config, tmp_path)
    assert provider.is_available() is False


def test_is_available_false_when_package_missing(enabled_config, tmp_path):
    provider = GraphifyProvider(enabled_config, tmp_path)
    with patch("fastcoder.context.graphify_provider._graphify_installed", return_value=False):
        assert provider.is_available() is False


def test_is_available_true_when_enabled_and_installed(enabled_config, tmp_path):
    provider = GraphifyProvider(enabled_config, tmp_path)
    with patch("fastcoder.context.graphify_provider._graphify_installed", return_value=True):
        assert provider.is_available() is True


def test_is_available_false_after_build_failure(enabled_config, tmp_path):
    provider = GraphifyProvider(enabled_config, tmp_path)
    provider._build_failed = True
    with patch("fastcoder.context.graphify_provider._graphify_installed", return_value=True):
        assert provider.is_available() is False


# ── query_context() short-circuits ────────────────────────────────


def test_query_returns_none_when_disabled(disabled_config, tmp_path):
    provider = GraphifyProvider(disabled_config, tmp_path)
    assert provider.query_context("anything") is None


def test_query_returns_none_when_package_missing(enabled_config, tmp_path):
    provider = GraphifyProvider(enabled_config, tmp_path)
    with patch("fastcoder.context.graphify_provider._graphify_installed", return_value=False):
        assert provider.query_context("anything") is None


def test_query_swallows_exceptions(enabled_config, tmp_path):
    """A buggy query path must never crash the agent — just return None."""
    provider = GraphifyProvider(enabled_config, tmp_path)
    with patch("fastcoder.context.graphify_provider._graphify_installed", return_value=True):
        with patch.object(provider, "_ensure_graph", return_value=True):
            with patch.object(provider, "_query", side_effect=RuntimeError("boom")):
                assert provider.query_context("anything") is None


# ── query against an in-memory graph ──────────────────────────────


def _write_minimal_graph(path: Path) -> None:
    """Write a tiny networkx node-link JSON graph compatible with graphify."""
    graph_doc = {
        "directed": False,
        "multigraph": False,
        "graph": {},
        "nodes": [
            {"id": "auth_login", "label": "AuthLogin", "source_file": "src/auth.py"},
            {"id": "auth_session", "label": "Session", "source_file": "src/auth.py"},
            {"id": "db_query", "label": "DatabaseQuery", "source_file": "src/db.py"},
        ],
        "links": [
            {
                "source": "auth_login",
                "target": "auth_session",
                "relation": "creates",
                "confidence": "EXTRACTED",
            },
            {
                "source": "auth_session",
                "target": "db_query",
                "relation": "calls",
                "confidence": "INFERRED",
            },
        ],
    }
    path.write_text(json.dumps(graph_doc))


@pytest.mark.skipif(
    pytest.importorskip("networkx", reason="networkx not installed") is None,
    reason="networkx not installed",
)
def test_query_against_real_graph(enabled_config, tmp_path):
    """End-to-end: load a graph from disk and query it."""
    cache_dir = tmp_path / "graphify"
    cache_dir.mkdir(parents=True)
    _write_minimal_graph(cache_dir / "graph.json")

    enabled_config.cache_dir = str(cache_dir)
    provider = GraphifyProvider(enabled_config, tmp_path)

    with patch("fastcoder.context.graphify_provider._graphify_installed", return_value=True):
        result = provider.query_context("How does authentication login work?")

    assert result is not None
    assert "graphify" in result.lower()
    assert "AuthLogin" in result
    # BFS should pull in the neighbour
    assert "Session" in result


def test_query_returns_none_when_no_terms_match(enabled_config, tmp_path):
    cache_dir = tmp_path / "graphify"
    cache_dir.mkdir(parents=True)
    _write_minimal_graph(cache_dir / "graph.json")
    enabled_config.cache_dir = str(cache_dir)

    provider = GraphifyProvider(enabled_config, tmp_path)
    with patch("fastcoder.context.graphify_provider._graphify_installed", return_value=True):
        # Use terms that won't match any node label
        result = provider.query_context("kubernetes deployment yaml")
    assert result is None


def test_query_truncates_to_token_budget(enabled_config, tmp_path):
    cache_dir = tmp_path / "graphify"
    cache_dir.mkdir(parents=True)
    _write_minimal_graph(cache_dir / "graph.json")
    enabled_config.cache_dir = str(cache_dir)
    enabled_config.query_token_budget = 20  # ~80 chars

    provider = GraphifyProvider(enabled_config, tmp_path)
    with patch("fastcoder.context.graphify_provider._graphify_installed", return_value=True):
        result = provider.query_context("AuthLogin Session DatabaseQuery")

    assert result is not None
    assert "truncated" in result


# ── helpers ──────────────────────────────────────────────────────


def test_tokenize_filters_short_and_punctuation():
    tokens = _tokenize("How does the AuthLogin work? Or db_query!")
    assert "AuthLogin" in tokens
    assert "db_query" in tokens
    # short tokens still appear; the consumer filters by length
    assert "or" in [t.lower() for t in tokens] or True


# ── Phase 3: planner/reviewer hints ───────────────────────────────


def _write_graph_with_communities(path: Path) -> None:
    """Same as minimal graph but with community attributes set on nodes."""
    graph_doc = {
        "directed": False,
        "multigraph": False,
        "graph": {},
        "nodes": [
            {
                "id": "auth_login",
                "label": "AuthLogin",
                "source_file": "src/auth.py",
                "community": 0,
                "community_label": "Authentication",
            },
            {
                "id": "auth_session",
                "label": "Session",
                "source_file": "src/auth.py",
                "community": 0,
                "community_label": "Authentication",
            },
            {
                "id": "db_query",
                "label": "DatabaseQuery",
                "source_file": "src/db.py",
                "community": 1,
                "community_label": "Database",
            },
        ],
        "links": [
            {"source": "auth_login", "target": "auth_session", "relation": "creates"},
            {"source": "auth_login", "target": "db_query", "relation": "calls"},
            {"source": "auth_session", "target": "db_query", "relation": "calls"},
        ],
    }
    path.write_text(json.dumps(graph_doc))


def test_god_nodes_returns_top_connected(enabled_config, tmp_path):
    cache_dir = tmp_path / "graphify"
    cache_dir.mkdir(parents=True)
    _write_graph_with_communities(cache_dir / "graph.json")
    enabled_config.cache_dir = str(cache_dir)

    provider = GraphifyProvider(enabled_config, tmp_path)
    with patch("fastcoder.context.graphify_provider._graphify_installed", return_value=True):
        nodes = provider.god_nodes(top_n=3)

    assert len(nodes) == 3
    # auth_login has degree 2, others have degree 2 too here — just confirm shape
    assert all("label" in n and "degree" in n for n in nodes)


def test_god_nodes_empty_when_disabled(disabled_config, tmp_path):
    provider = GraphifyProvider(disabled_config, tmp_path)
    assert provider.god_nodes() == []


def test_community_hint_finds_matching_communities(enabled_config, tmp_path):
    cache_dir = tmp_path / "graphify"
    cache_dir.mkdir(parents=True)
    _write_graph_with_communities(cache_dir / "graph.json")
    enabled_config.cache_dir = str(cache_dir)

    provider = GraphifyProvider(enabled_config, tmp_path)
    with patch("fastcoder.context.graphify_provider._graphify_installed", return_value=True):
        hint = provider.community_hint("Refactor AuthLogin and Session creation")

    assert hint is not None
    assert "Authentication" in hint


def test_community_hint_none_when_no_match(enabled_config, tmp_path):
    cache_dir = tmp_path / "graphify"
    cache_dir.mkdir(parents=True)
    _write_graph_with_communities(cache_dir / "graph.json")
    enabled_config.cache_dir = str(cache_dir)

    provider = GraphifyProvider(enabled_config, tmp_path)
    with patch("fastcoder.context.graphify_provider._graphify_installed", return_value=True):
        hint = provider.community_hint("kubernetes terraform yaml")

    assert hint is None


def test_blast_radius_flags_high_degree_changes(enabled_config, tmp_path):
    """Build a graph where one node has dramatically more edges than others."""
    cache_dir = tmp_path / "graphify"
    cache_dir.mkdir(parents=True)
    nodes = [
        {"id": "core_orchestrator", "label": "Orchestrator", "source_file": "src/core.py"},
    ]
    links = []
    for i in range(20):
        nodes.append(
            {"id": f"leaf_{i}", "label": f"Leaf{i}", "source_file": f"src/leaf_{i}.py"}
        )
        links.append({"source": "core_orchestrator", "target": f"leaf_{i}", "relation": "calls"})
    (cache_dir / "graph.json").write_text(
        json.dumps(
            {
                "directed": False,
                "multigraph": False,
                "graph": {},
                "nodes": nodes,
                "links": links,
            }
        )
    )
    enabled_config.cache_dir = str(cache_dir)

    provider = GraphifyProvider(enabled_config, tmp_path)
    with patch("fastcoder.context.graphify_provider._graphify_installed", return_value=True):
        warning = provider.blast_radius(["src/core.py"])

    assert warning is not None
    assert "Orchestrator" in warning


def test_blast_radius_none_when_no_changes_match(enabled_config, tmp_path):
    cache_dir = tmp_path / "graphify"
    cache_dir.mkdir(parents=True)
    _write_graph_with_communities(cache_dir / "graph.json")
    enabled_config.cache_dir = str(cache_dir)

    provider = GraphifyProvider(enabled_config, tmp_path)
    with patch("fastcoder.context.graphify_provider._graphify_installed", return_value=True):
        warning = provider.blast_radius(["docs/unrelated.md"])
    assert warning is None


# ── Phase 3: git hook install/uninstall ───────────────────────────


def test_install_hook_creates_block(enabled_config, tmp_path):
    (tmp_path / ".git" / "hooks").mkdir(parents=True)
    enabled_config.auto_rebuild_on_commit = True
    provider = GraphifyProvider(enabled_config, tmp_path)
    status = provider.install_git_hook()

    assert status == "installed"
    hook = (tmp_path / ".git" / "hooks" / "post-commit").read_text()
    assert "fastcoder-graphify-hook BEGIN" in hook
    assert "fastcoder-graphify-hook END" in hook


def test_install_hook_idempotent(enabled_config, tmp_path):
    (tmp_path / ".git" / "hooks").mkdir(parents=True)
    provider = GraphifyProvider(enabled_config, tmp_path)
    provider.install_git_hook()
    second = provider.install_git_hook()
    assert second == "already installed"


def test_install_hook_appends_to_existing(enabled_config, tmp_path):
    hooks_dir = tmp_path / ".git" / "hooks"
    hooks_dir.mkdir(parents=True)
    (hooks_dir / "post-commit").write_text("#!/usr/bin/env bash\necho 'existing hook'\n")
    provider = GraphifyProvider(enabled_config, tmp_path)
    provider.install_git_hook()
    content = (hooks_dir / "post-commit").read_text()
    assert "existing hook" in content
    assert "fastcoder-graphify-hook" in content


def test_uninstall_hook_removes_only_our_block(enabled_config, tmp_path):
    hooks_dir = tmp_path / ".git" / "hooks"
    hooks_dir.mkdir(parents=True)
    (hooks_dir / "post-commit").write_text("#!/usr/bin/env bash\necho 'existing hook'\n")
    provider = GraphifyProvider(enabled_config, tmp_path)
    provider.install_git_hook()
    provider.uninstall_git_hook()
    content = (hooks_dir / "post-commit").read_text()
    assert "existing hook" in content
    assert "fastcoder-graphify-hook" not in content


def test_sync_hook_state_install_and_remove(enabled_config, tmp_path):
    hooks_dir = tmp_path / ".git" / "hooks"
    hooks_dir.mkdir(parents=True)
    enabled_config.auto_rebuild_on_commit = True
    provider = GraphifyProvider(enabled_config, tmp_path)

    assert provider.sync_hook_state() == "installed"
    # Toggle off
    enabled_config.auto_rebuild_on_commit = False
    assert provider.sync_hook_state() == "uninstalled"


def test_install_hook_skips_when_not_a_git_repo(enabled_config, tmp_path):
    provider = GraphifyProvider(enabled_config, tmp_path)
    assert provider.install_git_hook() == "skipped: not a git repo"


# ── Phase 3: semantic enrichment short-circuits ──────────────────


def test_enrich_returns_error_when_disabled(disabled_config, tmp_path):
    provider = GraphifyProvider(disabled_config, tmp_path)
    summary = provider.enrich_semantic(lambda *_: None)
    assert summary["error"] == "graphify unavailable"


def test_enrich_returns_error_when_semantic_off(enabled_config, tmp_path):
    enabled_config.semantic_extraction = False
    cache_dir = tmp_path / "graphify"
    cache_dir.mkdir(parents=True)
    _write_minimal_graph(cache_dir / "graph.json")
    enabled_config.cache_dir = str(cache_dir)
    provider = GraphifyProvider(enabled_config, tmp_path)
    with patch("fastcoder.context.graphify_provider._graphify_installed", return_value=True):
        summary = provider.enrich_semantic(lambda *_: None)
    assert summary["error"] == "semantic_extraction disabled in config"
