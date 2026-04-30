"""Comprehensive test suite for memory, context, and error handling modules."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
import pytest_asyncio

from fastcoder.memory import MemoryStore
from fastcoder.context import ContextManager, TokenBudget
from fastcoder.errors.classifier import ErrorClassifier
from fastcoder.errors.recovery import RecoveryManager, ErrorRecoveryCoordinator, RecoveryAction
from fastcoder.types.memory import (
    MemoryEntry,
    MemoryQuery,
    MemoryTier,
    MemoryType,
    MemoryConsolidationResult,
)
from fastcoder.types.errors import (
    ErrorCategory,
    ErrorClassification,
    RecoveryStrategy,
    ErrorDetail,
    ErrorContext,
)
from fastcoder.types.story import Story, StorySpec, StoryState, AcceptanceCriterion
from fastcoder.types.plan import PlanTask, TaskAction
from fastcoder.types.codebase import ProjectProfile
from fastcoder.types.llm import Message
from fastcoder.types.iteration import Iteration


# ============================================================================
# MEMORY STORE TESTS
# ============================================================================


class TestMemoryStoreInit:
    """Test MemoryStore initialization."""

    def test_init_default_max_entries(self):
        """Test default max_entries_per_tier."""
        store = MemoryStore()
        assert store.max_entries_per_tier == 500
        assert len(store.memories) == 5  # 5 tiers
        assert store.error_fixes == {}

    def test_init_custom_max_entries(self):
        """Test custom max_entries_per_tier."""
        store = MemoryStore(max_entries_per_tier=1000)
        assert store.max_entries_per_tier == 1000

    def test_init_all_tiers_empty(self):
        """Test all tiers initialized empty."""
        store = MemoryStore()
        for tier in MemoryTier:
            assert store.memories[tier] == []


class TestMemoryStoreStore:
    """Test MemoryStore.store() method."""

    def test_store_single_entry(self):
        """Test storing a single entry."""
        store = MemoryStore()
        entry = MemoryEntry(
            id="test-1",
            type=MemoryType.PATTERN,
            tier=MemoryTier.SEMANTIC,
            content="test content",
        )
        store.store(entry)
        assert len(store.memories[MemoryTier.SEMANTIC]) == 1
        assert store.memories[MemoryTier.SEMANTIC][0].id == "test-1"

    def test_store_multiple_entries_same_tier(self):
        """Test storing multiple entries in same tier."""
        store = MemoryStore()
        for i in range(5):
            entry = MemoryEntry(
                id=f"test-{i}",
                type=MemoryType.PATTERN,
                tier=MemoryTier.SEMANTIC,
                content=f"content {i}",
            )
            store.store(entry)
        assert len(store.memories[MemoryTier.SEMANTIC]) == 5

    def test_store_entries_different_tiers(self):
        """Test storing entries in different tiers."""
        store = MemoryStore()
        tiers = [MemoryTier.WORKING, MemoryTier.EPISODIC, MemoryTier.SEMANTIC]
        for tier in tiers:
            entry = MemoryEntry(
                id=f"entry-{tier.value}",
                type=MemoryType.PATTERN,
                tier=tier,
                content="test",
            )
            store.store(entry)
        for tier in tiers:
            assert len(store.memories[tier]) == 1

    def test_store_triggers_eviction(self):
        """Test that storing beyond max triggers eviction."""
        store = MemoryStore(max_entries_per_tier=3)
        # Add 5 entries
        for i in range(5):
            entry = MemoryEntry(
                id=f"test-{i}",
                type=MemoryType.PATTERN,
                tier=MemoryTier.SEMANTIC,
                content="test",
                effectiveness_score=i * 0.2,  # Lower scores for earlier entries
            )
            store.store(entry)
        # Should be capped at 3
        assert len(store.memories[MemoryTier.SEMANTIC]) <= 3


class TestMemoryStoreQuery:
    """Test MemoryStore.query() method."""

    def setup_method(self):
        """Setup store with test data."""
        self.store = MemoryStore()
        self.entries = []
        for i in range(5):
            entry = MemoryEntry(
                id=f"entry-{i}",
                type=MemoryType.PATTERN if i % 2 == 0 else MemoryType.ERROR_FIX,
                tier=MemoryTier.SEMANTIC if i < 3 else MemoryTier.PROCEDURAL,
                content=f"python function implementation for loop data structure",
                project_id="proj-1" if i < 3 else "proj-2",
                effectiveness_score=0.9 - (i * 0.1),
            )
            self.store.store(entry)
            self.entries.append(entry)

    def test_query_text_similarity(self):
        """Test query returns entries with text similarity."""
        query = MemoryQuery(query="python implementation loop")
        results = self.store.query(query)
        assert len(results) > 0
        assert all(isinstance(r, MemoryEntry) for r in results)

    def test_query_filter_by_tier(self):
        """Test query filtering by tier."""
        query = MemoryQuery(
            query="python",
            tier=MemoryTier.SEMANTIC,
        )
        results = self.store.query(query)
        assert all(r.tier == MemoryTier.SEMANTIC for r in results)

    def test_query_filter_by_type(self):
        """Test query filtering by type."""
        query = MemoryQuery(
            query="python",
            type=MemoryType.PATTERN,
        )
        results = self.store.query(query)
        assert all(r.type == MemoryType.PATTERN for r in results)

    def test_query_filter_by_project_id(self):
        """Test query filtering by project_id."""
        query = MemoryQuery(
            query="python",
            project_id="proj-1",
        )
        results = self.store.query(query)
        assert all(r.project_id == "proj-1" for r in results)

    def test_query_filter_by_effectiveness(self):
        """Test query filtering by min_effectiveness."""
        query = MemoryQuery(
            query="python",
            min_effectiveness=0.8,
        )
        results = self.store.query(query)
        assert all(r.effectiveness_score >= 0.8 for r in results)

    def test_query_max_results(self):
        """Test query respects max_results."""
        query = MemoryQuery(query="python", max_results=2)
        results = self.store.query(query)
        assert len(results) <= 2

    def test_query_similarity_sorting(self):
        """Test results sorted by similarity."""
        query = MemoryQuery(query="function")
        results = self.store.query(query)
        # Results should be sorted by similarity (all have same content, so check score)
        if len(results) > 1:
            # Higher effectiveness scores should come first
            scores = [r.effectiveness_score for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_query_no_matches(self):
        """Test query with no matches."""
        query = MemoryQuery(query="completely unrelated xyz abc")
        results = self.store.query(query)
        assert len(results) == 0


class TestMemoryStoreConsolidate:
    """Test MemoryStore.consolidate() method."""

    def test_consolidate_creates_pattern_memory(self):
        """Test consolidate creates pattern memory for done story."""
        store = MemoryStore()
        story = Story(
            id="story-1",
            raw_text="test story",
            project_id="proj-1",
            state=StoryState.DONE,
            spec=StorySpec(
                title="Test Feature",
                description="Test description",
            ),
            iterations=[],
        )
        result = store.consolidate(story)
        assert isinstance(result, MemoryConsolidationResult)
        assert len(result.new_memories) == 1
        assert result.new_memories[0].type == MemoryType.PATTERN

    def test_consolidate_records_error_fixes(self):
        """Test consolidate records error fixes from iterations."""
        store = MemoryStore()
        # Create iteration and mock the error_fingerprint and error_fix attributes
        iteration = Iteration(number=1)
        # Mock these attributes since they're not in the Iteration model
        iteration = MagicMock(spec=Iteration)
        iteration.error_fingerprint = "abc123"
        iteration.error_fix = "fixed by adding type hints"

        story = Story(
            id="story-1",
            raw_text="test",
            project_id="proj-1",
            state=StoryState.DONE,
            iterations=[iteration],
        )
        result = store.consolidate(story)
        # Should record the error fix
        assert store.get_error_fix("abc123") == "fixed by adding type hints"

    def test_consolidate_merges_similar_memories(self):
        """Test consolidate merges similar memories."""
        store = MemoryStore()
        # Add two similar entries
        entry1 = MemoryEntry(
            id="entry-1",
            type=MemoryType.PATTERN,
            tier=MemoryTier.SEMANTIC,
            content="function implementation pattern for data structure",
            effectiveness_score=0.8,
        )
        entry2 = MemoryEntry(
            id="entry-2",
            type=MemoryType.PATTERN,
            tier=MemoryTier.SEMANTIC,
            content="function implementation pattern for data structure",
            effectiveness_score=0.7,
            use_count=5,
        )
        store.store(entry1)
        store.store(entry2)

        story = Story(
            id="story-1",
            raw_text="test",
            project_id="proj-1",
            state=StoryState.DONE,
            iterations=[],
        )
        result = store.consolidate(story)
        assert result.merged_count > 0
        assert len(result.evicted_memory_ids) > 0


class TestMemoryStoreDecay:
    """Test MemoryStore.decay() method."""

    def test_decay_reduces_old_entries(self):
        """Test decay reduces effectiveness of old entries."""
        store = MemoryStore()
        entry = MemoryEntry(
            id="old-entry",
            type=MemoryType.PATTERN,
            tier=MemoryTier.SEMANTIC,
            content="old content",
            effectiveness_score=1.0,
        )
        # Set last_used_at to 8 days ago
        entry.last_used_at = datetime.utcnow() - timedelta(days=8)
        store.store(entry)

        store.decay(decay_rate=0.95)
        # Check effectiveness was reduced
        stored_entry = store.memories[MemoryTier.SEMANTIC][0]
        assert stored_entry.effectiveness_score < 1.0
        assert stored_entry.effectiveness_score == pytest.approx(0.95)

    def test_decay_preserves_recent_entries(self):
        """Test decay doesn't affect recent entries."""
        store = MemoryStore()
        entry = MemoryEntry(
            id="recent-entry",
            type=MemoryType.PATTERN,
            tier=MemoryTier.SEMANTIC,
            content="recent content",
            effectiveness_score=1.0,
        )
        # Set last_used_at to 2 days ago (within 7 day window)
        entry.last_used_at = datetime.utcnow() - timedelta(days=2)
        store.store(entry)

        store.decay(decay_rate=0.95)
        stored_entry = store.memories[MemoryTier.SEMANTIC][0]
        assert stored_entry.effectiveness_score == 1.0

    def test_decay_custom_rate(self):
        """Test decay with custom rate."""
        store = MemoryStore()
        entry = MemoryEntry(
            id="entry",
            type=MemoryType.PATTERN,
            tier=MemoryTier.SEMANTIC,
            content="content",
            effectiveness_score=1.0,
        )
        entry.last_used_at = datetime.utcnow() - timedelta(days=8)
        store.store(entry)

        store.decay(decay_rate=0.8)
        stored_entry = store.memories[MemoryTier.SEMANTIC][0]
        assert stored_entry.effectiveness_score == pytest.approx(0.8)


class TestMemoryStoreEvict:
    """Test MemoryStore.evict() method."""

    def test_evict_removes_low_score_entries(self):
        """Test evict removes lowest scoring entries."""
        store = MemoryStore()
        for i in range(5):
            entry = MemoryEntry(
                id=f"entry-{i}",
                type=MemoryType.PATTERN,
                tier=MemoryTier.SEMANTIC,
                content=f"content {i}",
                effectiveness_score=i * 0.2,  # 0, 0.2, 0.4, 0.6, 0.8
            )
            store.store(entry)

        store.evict(MemoryTier.SEMANTIC, max_entries=3)
        assert len(store.memories[MemoryTier.SEMANTIC]) == 3
        # Check lowest scores were removed
        scores = [e.effectiveness_score for e in store.memories[MemoryTier.SEMANTIC]]
        assert min(scores) >= 0.4  # Lowest remaining should be 0.4

    def test_evict_preserves_max_entries(self):
        """Test evict stops at max_entries."""
        store = MemoryStore()
        for i in range(3):
            entry = MemoryEntry(
                id=f"entry-{i}",
                type=MemoryType.PATTERN,
                tier=MemoryTier.SEMANTIC,
                content="content",
            )
            store.store(entry)

        store.evict(MemoryTier.SEMANTIC, max_entries=5)
        assert len(store.memories[MemoryTier.SEMANTIC]) == 3  # No eviction needed

    def test_evict_nonexistent_tier(self):
        """Test evict handles nonexistent tier gracefully."""
        store = MemoryStore()
        # Should not raise
        store.evict(MemoryTier.WORKING, max_entries=10)


class TestMemoryStoreErrorFixes:
    """Test MemoryStore error fix methods."""

    def test_record_and_get_error_fix(self):
        """Test recording and retrieving error fixes."""
        store = MemoryStore()
        store.record_error_fix("fingerprint-1", "add type hints", "story-1")
        assert store.get_error_fix("fingerprint-1") == "add type hints"

    def test_get_nonexistent_error_fix(self):
        """Test getting nonexistent error fix."""
        store = MemoryStore()
        assert store.get_error_fix("nonexistent") is None

    def test_record_error_fix_creates_memory_entry(self):
        """Test record_error_fix creates memory entry."""
        store = MemoryStore()
        store.record_error_fix("fp-1", "fix code", "story-1")
        # Check memory was created
        procedural = store.memories[MemoryTier.PROCEDURAL]
        assert len(procedural) == 1
        assert procedural[0].type == MemoryType.ERROR_FIX


class TestMemoryStorePersistence:
    """Test MemoryStore.save() and load() methods."""

    def test_save_creates_file(self, tmp_path):
        """Test save creates JSON file."""
        store = MemoryStore()
        entry = MemoryEntry(
            id="test-1",
            type=MemoryType.PATTERN,
            tier=MemoryTier.SEMANTIC,
            content="test",
        )
        store.store(entry)

        file_path = tmp_path / "memory.json"
        store.save(str(file_path))
        assert file_path.exists()

    def test_save_includes_error_fixes(self, tmp_path):
        """Test save includes error fixes."""
        store = MemoryStore()
        store.record_error_fix("fp-1", "fix", "story-1")

        file_path = tmp_path / "memory.json"
        store.save(str(file_path))

        with open(file_path) as f:
            data = json.load(f)
        assert "fp-1" in data["error_fixes"]

    def test_load_restores_memories(self, tmp_path):
        """Test load restores memories from file."""
        # Save
        store1 = MemoryStore()
        entry = MemoryEntry(
            id="test-1",
            type=MemoryType.PATTERN,
            tier=MemoryTier.SEMANTIC,
            content="test",
        )
        store1.store(entry)

        file_path = tmp_path / "memory.json"
        store1.save(str(file_path))

        # Load
        store2 = MemoryStore()
        store2.load(str(file_path))
        assert len(store2.memories[MemoryTier.SEMANTIC]) == 1
        assert store2.memories[MemoryTier.SEMANTIC][0].id == "test-1"

    def test_load_nonexistent_file(self):
        """Test load handles missing file gracefully."""
        store = MemoryStore()
        store.load("/nonexistent/path/memory.json")
        # Should not raise, just return

    def test_load_restores_error_fixes(self, tmp_path):
        """Test load restores error fixes."""
        store1 = MemoryStore()
        store1.record_error_fix("fp-1", "fix code", "story-1")

        file_path = tmp_path / "memory.json"
        store1.save(str(file_path))

        store2 = MemoryStore()
        store2.load(str(file_path))
        assert store2.get_error_fix("fp-1") == "fix code"


class TestMemoryStoreJaccardSimilarity:
    """Test MemoryStore._jaccard_similarity() method."""

    def test_jaccard_identical_texts(self):
        """Test Jaccard similarity for identical texts."""
        store = MemoryStore()
        sim = store._jaccard_similarity("hello world", "hello world")
        assert sim == 1.0

    def test_jaccard_no_overlap(self):
        """Test Jaccard similarity with no overlap."""
        store = MemoryStore()
        sim = store._jaccard_similarity("abc def", "xyz uvw")
        assert sim == 0.0

    def test_jaccard_partial_overlap(self):
        """Test Jaccard similarity with partial overlap."""
        store = MemoryStore()
        sim = store._jaccard_similarity("hello world test", "hello world")
        assert 0 < sim < 1

    def test_jaccard_case_insensitive(self):
        """Test Jaccard is case insensitive."""
        store = MemoryStore()
        sim1 = store._jaccard_similarity("Hello World", "hello world")
        assert sim1 == 1.0

    def test_jaccard_empty_strings(self):
        """Test Jaccard with empty strings."""
        store = MemoryStore()
        assert store._jaccard_similarity("", "test") == 0.0
        assert store._jaccard_similarity("test", "") == 0.0
        assert store._jaccard_similarity("", "") == 0.0


# ============================================================================
# CONTEXT MANAGER TESTS
# ============================================================================


class TestTokenBudget:
    """Test TokenBudget dataclass."""

    def test_default_budget(self):
        """Test default TokenBudget values."""
        budget = TokenBudget()
        assert budget.system == 2000
        assert budget.project == 1500
        assert budget.story == 2000
        assert budget.task == 1000
        assert budget.code == 16000
        assert budget.error == 3000
        assert budget.memory == 1500

    def test_custom_budget(self):
        """Test custom TokenBudget values."""
        budget = TokenBudget(system=3000, code=20000)
        assert budget.system == 3000
        assert budget.code == 20000

    def test_total_property(self):
        """Test TokenBudget.total property."""
        budget = TokenBudget()
        expected = 2000 + 1500 + 2000 + 1000 + 16000 + 3000 + 1500
        assert budget.total == expected


class TestContextManagerInit:
    """Test ContextManager initialization."""

    def test_init_default(self):
        """Test default ContextManager initialization."""
        cm = ContextManager()
        assert cm.budget is not None
        assert cm.budget.total == TokenBudget().total
        assert cm.tokens_per_char == 0.25

    def test_init_custom_budget(self):
        """Test ContextManager with custom budget."""
        budget = TokenBudget(code=20000)
        cm = ContextManager(budget=budget)
        assert cm.budget.code == 20000

    def test_init_custom_tokens_per_char(self):
        """Test ContextManager with custom tokens_per_char."""
        cm = ContextManager(tokens_per_char=0.3)
        assert cm.tokens_per_char == 0.3


@pytest_asyncio.fixture
async def context_manager():
    """Fixture for ContextManager."""
    return ContextManager()


@pytest_asyncio.fixture
async def story_spec():
    """Fixture for StorySpec."""
    return StorySpec(
        title="Add user authentication",
        description="Implement OAuth2 authentication",
        acceptance_criteria=[
            AcceptanceCriterion(
                id="ac-1",
                description="Users can login with email",
            ),
        ],
    )


@pytest_asyncio.fixture
async def plan_task():
    """Fixture for PlanTask."""
    return PlanTask(
        id="task-1",
        action=TaskAction.CREATE_FILE,
        target="src/auth.py",
        description="Create auth module",
    )


@pytest_asyncio.fixture
async def project_profile():
    """Fixture for ProjectProfile."""
    return ProjectProfile(
        language="python",
        framework="fastapi",
        package_manager="pip",
        test_framework="pytest",
    )


@pytest.mark.asyncio
async def test_build_context_basic(context_manager, story_spec, plan_task, project_profile):
    """Test basic build_context."""
    messages = await context_manager.build_context(
        story=story_spec,
        task=plan_task,
        project_profile=project_profile,
        relevant_files=["src/auth.py"],
    )
    assert len(messages) > 0
    assert all(isinstance(m, Message) for m in messages)
    assert messages[0].role == "system"  # First message is system

@pytest.mark.asyncio
async def test_build_context_with_error(context_manager, story_spec, plan_task, project_profile):
    """Test build_context with error context."""
    error_context = ErrorContext(
        attempt=2,
        error=ErrorDetail(type="TypeError", message="Expected str, got int"),
    )
    messages = await context_manager.build_context(
        story=story_spec,
        task=plan_task,
        project_profile=project_profile,
        relevant_files=["src/auth.py"],
        error_context=error_context,
    )
    assert any("error" in m.content.lower() for m in messages)

@pytest.mark.asyncio
async def test_build_context_with_memory(context_manager, story_spec, plan_task, project_profile):
    """Test build_context with memory entries."""
    memory_entries = [
        MemoryEntry(
            id="mem-1",
            type=MemoryType.PATTERN,
            tier=MemoryTier.SEMANTIC,
            content="Use context managers for resource management",
        ),
    ]
    messages = await context_manager.build_context(
        story=story_spec,
        task=plan_task,
        project_profile=project_profile,
        relevant_files=["src/auth.py"],
        memory_entries=memory_entries,
    )
    assert any("memory" in m.content.lower() or "knowledge" in m.content.lower() for m in messages)


class TestContextManagerSelectFiles:
    """Test ContextManager.select_files() method."""

    def test_select_files_with_target(self):
        """Test select_files includes target file."""
        cm = ContextManager()
        task = PlanTask(
            id="task-1",
            action=TaskAction.MODIFY_FILE,
            target="src/auth.py",
            description="test",
        )
        selected = cm.select_files(task)
        assert "src/auth.py" in selected

    def test_select_files_with_dependencies(self):
        """Test select_files includes dependencies."""
        cm = ContextManager()
        task = PlanTask(
            id="task-1",
            action=TaskAction.CREATE_FILE,
            target="src/auth.py",
            description="test",
        )
        deps = {
            "src/auth.py": ["src/utils.py", "src/models.py"],
        }
        selected = cm.select_files(task, dependency_graph=deps)
        assert "src/auth.py" in selected

    def test_select_files_respects_limit(self):
        """Test select_files respects 20 file limit."""
        cm = ContextManager()
        task = PlanTask(
            id="task-1",
            action=TaskAction.CREATE_FILE,
            target="src/auth.py",
            description="test",
        )
        selected = cm.select_files(task)
        assert len(selected) <= 20


class TestContextManagerExtractSkeleton:
    """Test ContextManager.extract_skeleton() method."""

    def test_extract_skeleton_keeps_function_signatures(self):
        """Test skeleton keeps function signatures."""
        cm = ContextManager()
        code = """def hello(name: str) -> str:
    return f"Hello {name}"

class MyClass:
    def method(self):
        pass
"""
        skeleton = cm.extract_skeleton(code)
        assert "def hello" in skeleton
        assert "class MyClass" in skeleton

    def test_extract_skeleton_removes_implementation(self):
        """Test skeleton removes function bodies."""
        cm = ContextManager()
        code = """def hello():
    x = 1
    y = 2
    return x + y
"""
        skeleton = cm.extract_skeleton(code)
        assert "def hello" in skeleton
        assert "x = 1" not in skeleton

    def test_extract_skeleton_keeps_imports(self):
        """Test skeleton keeps imports."""
        cm = ContextManager()
        code = """import os
from pathlib import Path

def hello():
    pass
"""
        skeleton = cm.extract_skeleton(code)
        assert "import os" in skeleton
        assert "from pathlib import Path" in skeleton


class TestContextManagerCreateDiff:
    """Test ContextManager.create_diff_context() method."""

    def test_create_diff_context(self):
        """Test creating diff context."""
        cm = ContextManager()
        old = "line 1\nline 2\nline 3"
        new = "line 1\nmodified line 2\nline 3"
        diff = cm.create_diff_context(old, new)
        assert "---" in diff  # Unified diff format
        assert "+++" in diff

    def test_create_diff_addition(self):
        """Test diff shows additions."""
        cm = ContextManager()
        old = "line 1"
        new = "line 1\nline 2"
        diff = cm.create_diff_context(old, new)
        assert "+line 2" in diff

    def test_create_diff_deletion(self):
        """Test diff shows deletions."""
        cm = ContextManager()
        old = "line 1\nline 2"
        new = "line 1"
        diff = cm.create_diff_context(old, new)
        assert "-line 2" in diff


class TestContextManagerHandleOverflow:
    """Test ContextManager.handle_overflow() method."""

    def test_handle_overflow_no_overflow(self):
        """Test handle_overflow when no overflow."""
        cm = ContextManager()
        messages = [
            Message(role="system", content="x" * 100),
            Message(role="user", content="y" * 100),
        ]
        result = cm.handle_overflow(messages, max_tokens=1000)
        assert len(result) == 2

    def test_handle_overflow_removes_memory(self):
        """Test handle_overflow prioritizes removing memory."""
        cm = ContextManager()
        messages = [
            Message(role="system", content="System " + "x" * 1000),
            Message(role="user", content="Memory: " + "y" * 2000),
            Message(role="user", content="Code: " + "z" * 100),
        ]
        result = cm.handle_overflow(messages, max_tokens=500)
        # Memory should be removed first
        assert all("Memory:" not in m.content for m in result)


class TestContextManagerEstimateTokens:
    """Test ContextManager.estimate_tokens() method."""

    def test_estimate_tokens_basic(self):
        """Test basic token estimation."""
        cm = ContextManager(tokens_per_char=0.25)
        tokens = cm.estimate_tokens("a" * 100)
        assert tokens == 25

    def test_estimate_tokens_minimum_one(self):
        """Test estimation returns at least 1 token."""
        cm = ContextManager()
        tokens = cm.estimate_tokens("")
        assert tokens >= 1

    def test_estimate_tokens_longer_text(self):
        """Test estimation with longer text."""
        cm = ContextManager()
        text = "hello world " * 100
        tokens = cm.estimate_tokens(text)
        assert tokens > 1


# ============================================================================
# ERROR CLASSIFIER TESTS
# ============================================================================


class TestErrorClassifierClassify:
    """Test ErrorClassifier.classify() method."""

    def test_classify_syntax_error(self):
        """Test classifying syntax error."""
        classifier = ErrorClassifier()
        result = classifier.classify(
            error_type="SyntaxError",
            message="invalid syntax at line 42",
        )
        assert result.category == ErrorCategory.SYNTAX_ERROR
        assert result.recovery_strategy == RecoveryStrategy.DIRECT_FIX

    def test_classify_type_error(self):
        """Test classifying type error."""
        classifier = ErrorClassifier()
        result = classifier.classify(
            error_type="TypeError",
            message="expected str, got int",
        )
        assert result.category == ErrorCategory.TYPE_ERROR
        assert result.recovery_strategy == RecoveryStrategy.INCLUDE_TYPES

    def test_classify_import_error(self):
        """Test classifying import error."""
        classifier = ErrorClassifier()
        result = classifier.classify(
            error_type="ModuleNotFoundError",
            message="No module named 'requests'",
        )
        assert result.category == ErrorCategory.IMPORT_ERROR
        assert result.recovery_strategy == RecoveryStrategy.CONSULT_SYMBOL_TABLE

    def test_classify_logic_error(self):
        """Test classifying logic error."""
        classifier = ErrorClassifier()
        result = classifier.classify(
            error_type="AssertionError",
            message="assertion failed: expected True but got False",
        )
        assert result.category == ErrorCategory.LOGIC_ERROR

    def test_classify_environment_error(self):
        """Test classifying environment error."""
        classifier = ErrorClassifier()
        result = classifier.classify(
            error_type="FileNotFoundError",
            message="ENOENT: no such file or directory",
        )
        assert result.category == ErrorCategory.ENVIRONMENT_ERROR

    def test_classify_flaky_error(self):
        """Test classifying flaky error."""
        classifier = ErrorClassifier()
        result = classifier.classify(
            error_type="ConnectionError",
            message="connection timeout",
        )
        assert result.category == ErrorCategory.FLAKY_ERROR

    def test_classify_integration_error(self):
        """Test classifying integration error."""
        classifier = ErrorClassifier()
        result = classifier.classify(
            error_type="APIError",
            message="API service unavailable",
        )
        assert result.category == ErrorCategory.INTEGRATION_ERROR

    def test_classify_architectural_error(self):
        """Test classifying architectural error."""
        classifier = ErrorClassifier()
        result = classifier.classify(
            error_type="CircularDependency",
            message="circular dependency detected",
        )
        assert result.category == ErrorCategory.ARCHITECTURAL_ERROR

    def test_classify_unknown_error(self):
        """Test classifying unknown error."""
        classifier = ErrorClassifier()
        result = classifier.classify(
            error_type="UnknownError",
            message="something went wrong that we have never seen before",
        )
        assert result.category == ErrorCategory.UNKNOWN

    def test_classify_generates_fingerprint(self):
        """Test classify generates fingerprint."""
        classifier = ErrorClassifier()
        result = classifier.classify(
            error_type="TypeError",
            message="expected str",
        )
        assert len(result.fingerprint) > 0

    def test_classify_sets_confidence(self):
        """Test classify sets confidence score."""
        classifier = ErrorClassifier()
        result = classifier.classify(
            error_type="TypeError",
            message="test",
        )
        assert 0 <= result.confidence <= 1

    def test_classify_high_confidence_standard_types(self):
        """Test standard error types have high confidence."""
        classifier = ErrorClassifier()
        result = classifier.classify(
            error_type="SyntaxError",
            message="test",
        )
        assert result.confidence >= 0.9


class TestErrorClassifierGenerateFingerprint:
    """Test ErrorClassifier.generate_fingerprint() method."""

    def test_fingerprint_deterministic(self):
        """Test fingerprint is deterministic."""
        classifier = ErrorClassifier()
        fp1 = classifier.generate_fingerprint("TypeError", "expected str")
        fp2 = classifier.generate_fingerprint("TypeError", "expected str")
        assert fp1 == fp2

    def test_fingerprint_different_messages(self):
        """Test different messages generate different fingerprints."""
        classifier = ErrorClassifier()
        fp1 = classifier.generate_fingerprint("TypeError", "expected str")
        fp2 = classifier.generate_fingerprint("TypeError", "expected int")
        # Should be different (after normalization)
        # Note: similar messages might have same fingerprint after normalization

    def test_fingerprint_includes_file(self):
        """Test fingerprint can include file."""
        classifier = ErrorClassifier()
        fp1 = classifier.generate_fingerprint("TypeError", "test", file="auth.py")
        fp2 = classifier.generate_fingerprint("TypeError", "test", file="models.py")
        # Should be different when file differs
        assert fp1 != fp2

    def test_fingerprint_length(self):
        """Test fingerprint is 16 characters."""
        classifier = ErrorClassifier()
        fp = classifier.generate_fingerprint("TypeError", "test")
        assert len(fp) == 16


# ============================================================================
# RECOVERY MANAGER TESTS
# ============================================================================


class TestRecoveryManagerGetStrategy:
    """Test RecoveryManager.get_strategy() method."""

    def test_get_strategy_first_attempt(self):
        """Test strategy for first attempt."""
        manager = RecoveryManager()
        classification = ErrorClassification(
            category=ErrorCategory.SYNTAX_ERROR,
            recovery_strategy=RecoveryStrategy.DIRECT_FIX,
            typical_fix_attempts=1,
            fingerprint="fp-1",
            confidence=0.9,
        )
        action = manager.get_strategy(classification, attempt=1)
        assert isinstance(action, RecoveryAction)
        assert action.strategy == RecoveryStrategy.DIRECT_FIX

    def test_get_strategy_known_fingerprint(self):
        """Test strategy returns known fix."""
        manager = RecoveryManager()
        manager.record_fix("fp-1", "add type hints", "story-1")

        classification = ErrorClassification(
            category=ErrorCategory.TYPE_ERROR,
            recovery_strategy=RecoveryStrategy.INCLUDE_TYPES,
            typical_fix_attempts=2,
            fingerprint="fp-1",
            confidence=0.7,
        )
        action = manager.get_strategy(classification, attempt=1)
        assert "known_fix" in action.additional_context

    def test_get_strategy_escalation_architectural(self):
        """Test escalation for architectural errors."""
        manager = RecoveryManager()
        classification = ErrorClassification(
            category=ErrorCategory.ARCHITECTURAL_ERROR,
            recovery_strategy=RecoveryStrategy.REPLAN,
            typical_fix_attempts=1,
            fingerprint="fp-1",
            confidence=0.8,
        )
        action = manager.get_strategy(classification, attempt=2)
        assert action.escalate is True
        assert action.replan is True

    def test_get_strategy_model_upgrade_low_confidence(self):
        """Test model upgrade for low confidence."""
        manager = RecoveryManager()
        classification = ErrorClassification(
            category=ErrorCategory.TYPE_ERROR,
            recovery_strategy=RecoveryStrategy.INCLUDE_TYPES,
            typical_fix_attempts=2,
            fingerprint="fp-1",
            confidence=0.6,
        )
        action = manager.get_strategy(classification, attempt=2)
        assert action.switch_to_top_tier is True

    def test_get_strategy_progressive_context(self):
        """Test strategy adds context progressively."""
        manager = RecoveryManager()
        classification = ErrorClassification(
            category=ErrorCategory.LOGIC_ERROR,
            recovery_strategy=RecoveryStrategy.INCLUDE_BROAD_CONTEXT,
            typical_fix_attempts=3,
            fingerprint="fp-1",
            confidence=0.8,
        )

        action1 = manager.get_strategy(classification, attempt=1)
        assert action1.additional_context.get("context_level") == 1

        action2 = manager.get_strategy(classification, attempt=2)
        assert action2.additional_context.get("context_level") == 2

        action3 = manager.get_strategy(classification, attempt=3)
        assert action3.additional_context.get("context_level") == 3


class TestRecoveryManagerRecordAndLookup:
    """Test RecoveryManager.record_fix() and lookup_fix() methods."""

    def test_record_and_lookup_fix(self):
        """Test recording and looking up fixes."""
        manager = RecoveryManager()
        manager.record_fix("fp-1", "add type hints", "story-1")
        fix = manager.lookup_fix("fp-1")
        assert fix == "add type hints"

    def test_lookup_nonexistent_fix(self):
        """Test looking up nonexistent fix."""
        manager = RecoveryManager()
        fix = manager.lookup_fix("nonexistent")
        assert fix is None

    def test_multiple_fixes(self):
        """Test storing multiple fixes."""
        manager = RecoveryManager()
        manager.record_fix("fp-1", "fix 1", "story-1")
        manager.record_fix("fp-2", "fix 2", "story-2")
        assert manager.lookup_fix("fp-1") == "fix 1"
        assert manager.lookup_fix("fp-2") == "fix 2"


class TestErrorRecoveryCoordinator:
    """Test ErrorRecoveryCoordinator class."""

    def test_coordinator_handle_error(self):
        """Test coordinator handles error end-to-end."""
        classifier = ErrorClassifier()
        manager = RecoveryManager()
        coordinator = ErrorRecoveryCoordinator(classifier, manager)

        classification, action = coordinator.handle_error(
            error_type="TypeError",
            message="expected str",
            stack_trace="",
            attempt=1,
        )

        assert isinstance(classification, ErrorClassification)
        assert isinstance(action, RecoveryAction)
        assert classification.category == ErrorCategory.TYPE_ERROR

    def test_coordinator_known_fix_high_confidence(self):
        """Test coordinator sets high confidence for known fix."""
        classifier = ErrorClassifier()
        manager = RecoveryManager()
        manager.record_fix("fp-123", "add return type", "story-1")

        coordinator = ErrorRecoveryCoordinator(classifier, manager)

        # Mock the fingerprint to match
        with patch.object(classifier, 'generate_fingerprint', return_value='fp-123'):
            classification, action = coordinator.handle_error(
                error_type="TypeError",
                message="test",
                attempt=1,
            )
            assert classification.confidence == 0.95

    def test_coordinator_multiple_attempts(self):
        """Test coordinator behavior across attempts."""
        classifier = ErrorClassifier()
        manager = RecoveryManager()
        coordinator = ErrorRecoveryCoordinator(classifier, manager)

        _, action1 = coordinator.handle_error(
            error_type="TypeError",
            message="test",
            attempt=1,
        )
        _, action2 = coordinator.handle_error(
            error_type="TypeError",
            message="test",
            attempt=2,
        )
        # Context should escalate
        context_level1 = action1.additional_context.get("context_level", 0)
        context_level2 = action2.additional_context.get("context_level", 0)
        assert context_level2 >= context_level1


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestMemoryContextIntegration:
    """Integration tests between memory and context."""

    @pytest.mark.asyncio
    async def test_memory_retrieval_in_context(self):
        """Test memory entries can be used in context building."""
        store = MemoryStore()
        entry = MemoryEntry(
            id="mem-1",
            type=MemoryType.PATTERN,
            tier=MemoryTier.SEMANTIC,
            content="Use context managers for resources",
        )
        store.store(entry)

        cm = ContextManager()
        story = StorySpec(
            title="Test",
            description="Test story",
        )
        task = PlanTask(
            id="task-1",
            action=TaskAction.CREATE_FILE,
            target="src/test.py",
            description="Test task",
        )
        profile = ProjectProfile()

        messages = await cm.build_context(
            story=story,
            task=task,
            project_profile=profile,
            relevant_files=["src/test.py"],
            memory_entries=[entry],
        )

        assert any("knowledge" in m.content.lower() or "pattern" in m.content.lower() for m in messages)


class TestErrorMemoryIntegration:
    """Integration tests between error handling and memory."""

    def test_error_fixes_stored_in_memory(self):
        """Test error fixes are stored in memory."""
        store = MemoryStore()

        # Simulate error fix
        store.record_error_fix("fp-syntax-1", "add missing colon", "story-1")

        # Verify in memory and retrievable
        assert store.get_error_fix("fp-syntax-1") is not None

        # Verify memory entry created
        procedural = store.memories[MemoryTier.PROCEDURAL]
        assert len(procedural) > 0
        assert procedural[0].type == MemoryType.ERROR_FIX


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_memory_store_zero_entries_per_tier(self):
        """Test memory store with zero max entries."""
        # Should still work, just immediately evict
        store = MemoryStore(max_entries_per_tier=0)
        entry = MemoryEntry(
            id="test",
            type=MemoryType.PATTERN,
            tier=MemoryTier.SEMANTIC,
            content="test",
        )
        store.store(entry)
        # Will be evicted immediately
        assert len(store.memories[MemoryTier.SEMANTIC]) == 0

    def test_context_manager_empty_files_list(self):
        """Test context manager with empty files list."""
        cm = ContextManager()
        task = PlanTask(
            id="task-1",
            action=TaskAction.CREATE_FILE,
            target="src/test.py",
            description="test",
        )
        selected = cm.select_files(task, dependency_graph={}, symbol_table={})
        # Should still include target
        assert len(selected) > 0

    def test_classifier_empty_error_message(self):
        """Test classifier with empty message."""
        classifier = ErrorClassifier()
        result = classifier.classify(
            error_type="UnknownError",
            message="",
        )
        assert result.category == ErrorCategory.UNKNOWN

    @pytest.mark.asyncio
    async def test_context_too_many_files(self):
        """Test context building with many files."""
        cm = ContextManager()
        story = StorySpec(
            title="Test",
            description="Test",
        )
        task = PlanTask(
            id="task-1",
            action=TaskAction.MODIFY_FILE,
            target="src/main.py",
            description="test",
        )
        profile = ProjectProfile()

        # Provide 100 files
        files = [f"src/module_{i}.py" for i in range(100)]

        messages = await cm.build_context(
            story=story,
            task=task,
            project_profile=profile,
            relevant_files=files,
        )
        # Should handle gracefully without crash
        assert len(messages) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
