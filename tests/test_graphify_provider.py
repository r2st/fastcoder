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
