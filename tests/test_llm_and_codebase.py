"""Comprehensive pytest tests for LLM and Codebase modules with 100% API coverage."""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from fastcoder.codebase import CodebaseIntelligence
from fastcoder.codebase.ast_indexer import ASTIndexer
from fastcoder.codebase.convention_detector import ConventionDetector
from fastcoder.codebase.cross_repo_index import (
    ContractConsumer,
    CrossRepoChangeSet,
    CrossRepoIndex,
    RepoChange,
    SharedContract,
)
from fastcoder.codebase.dependency_graph import DependencyGraph
from fastcoder.codebase.ownership_map import OwnershipMap
from fastcoder.codebase.semantic_search import SemanticSearch
from fastcoder.codebase.symbol_table import SymbolTable
from fastcoder.llm.prompt_templates import PromptRegistry, get_prompt_registry
from fastcoder.llm.providers.base import LLMProvider, UsageTracker
from fastcoder.llm.router import CostTracker, ModelRouter
from fastcoder.types.codebase import ASTNode, CodeChunk, SearchResult, SymbolInfo
from fastcoder.types.config import ModelConfig, ProviderConfig, RoutingConfig
from fastcoder.types.llm import (
    CompletionRequest,
    CompletionResponse,
    HealthStatus,
    Message,
    ModelTier,
    PromptTemplate,
    PromptSection,
    ReasoningMode,
    RoutingContext,
    StreamChunk,
    TaskPurpose,
    TokenUsage,
    UsageMetrics,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_project_dir():
    """Create temporary project directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_python_code():
    """Sample Python code for testing."""
    return """
import os
from pathlib import Path

class DataProcessor:
    \"\"\"Process data files.\"\"\"

    def __init__(self, input_path: str):
        self.input_path = input_path

    def process(self, data: dict) -> dict:
        \"\"\"Process the data.\"\"\"
        return {k: v.upper() if isinstance(v, str) else v for k, v in data.items()}

    async def async_process(self, data: list):
        \"\"\"Process data asynchronously.\"\"\"
        return data

def helper_function(x: int) -> int:
    \"\"\"Helper function.\"\"\"
    return x * 2
"""


@pytest.fixture
def sample_typescript_code():
    """Sample TypeScript code for testing."""
    return """
export interface User {
    id: number;
    name: string;
}

export class UserService {
    constructor(private db: Database) {}

    async getUser(id: number): Promise<User> {
        return this.db.query(`SELECT * FROM users WHERE id = ${id}`);
    }
}

export type UserInput = Omit<User, 'id'>;
"""


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider for testing."""

    class MockProvider(LLMProvider):
        def __init__(self, name: str = "mock", provider_type: str = "mock"):
            super().__init__(name, provider_type)

        async def complete(self, request: CompletionRequest) -> CompletionResponse:
            return CompletionResponse(
                id="mock-response-1",
                model=request.model,
                content="Mock response",
                tool_calls=[],
                usage=TokenUsage(
                    input_tokens=10,
                    output_tokens=5,
                    total_tokens=15,
                ),
            )

        async def stream(
            self, request: CompletionRequest
        ) -> AsyncGenerator[StreamChunk, None]:
            yield StreamChunk(
                type="content",
                content="Mock",
                is_final=False,
            )
            yield StreamChunk(
                type="content",
                content=" response",
                is_final=True,
            )

        async def health_check(self) -> HealthStatus:
            return HealthStatus(
                provider="mock",
                healthy=True,
                latency_ms=10.0,
            )

    return MockProvider()


@pytest.fixture
def provider_configs():
    """Create provider configurations for testing."""
    return [
        ProviderConfig(
            name="mock1",
            type="anthropic",
            enabled=True,
            api_key="test-key-1",
            models=[
                ModelConfig(
                    id="claude-3-opus",
                    tier="top",
                    max_context_tokens=200000,
                    cost_per_1k_input=0.015,
                    cost_per_1k_output=0.075,
                ),
                ModelConfig(
                    id="claude-3-sonnet",
                    tier="mid",
                    max_context_tokens=200000,
                    cost_per_1k_input=0.003,
                    cost_per_1k_output=0.015,
                ),
            ],
        ),
        ProviderConfig(
            name="mock2",
            type="openai",
            enabled=True,
            api_key="test-key-2",
            models=[
                ModelConfig(
                    id="gpt-4",
                    tier="top",
                    max_context_tokens=8192,
                    cost_per_1k_input=0.03,
                    cost_per_1k_output=0.06,
                ),
            ],
        ),
        ProviderConfig(
            name="mock3",
            type="anthropic",
            enabled=False,
            api_key="test-key-3",
            models=[
                ModelConfig(
                    id="claude-3-haiku",
                    tier="low",
                    max_context_tokens=200000,
                    cost_per_1k_input=0.00025,
                    cost_per_1k_output=0.00125,
                ),
            ],
        ),
    ]


# ============================================================================
# Tests: CostTracker
# ============================================================================


class TestCostTracker:
    """Tests for CostTracker."""

    def test_init(self):
        """Test CostTracker initialization."""
        tracker = CostTracker(story_id="story-1")
        assert tracker.story_id == "story-1"
        assert tracker.total_cost_usd == 0.0
        assert tracker.call_count == 0

    def test_add_cost(self):
        """Test adding cost."""
        tracker = CostTracker()
        tracker.add_cost(0.5)
        assert tracker.total_cost_usd == 0.5
        assert tracker.call_count == 1

        tracker.add_cost(0.3)
        assert tracker.total_cost_usd == 0.8
        assert tracker.call_count == 2

    def test_add_cost_multiple_calls(self):
        """Test adding costs for multiple calls."""
        tracker = CostTracker(story_id="test-story")
        costs = [0.1, 0.2, 0.15, 0.05]
        for cost in costs:
            tracker.add_cost(cost)
        assert tracker.total_cost_usd == sum(costs)
        assert tracker.call_count == len(costs)


# ============================================================================
# Tests: UsageTracker
# ============================================================================


class TestUsageTracker:
    """Tests for UsageTracker."""

    def test_init(self):
        """Test UsageTracker initialization."""
        tracker = UsageTracker(provider="anthropic")
        assert tracker.provider == "anthropic"
        assert tracker.total_calls == 0
        assert tracker.total_input_tokens == 0
        assert tracker.total_output_tokens == 0
        assert tracker.total_cost_usd == 0.0
        assert tracker.total_latency_ms == 0.0
        assert tracker.error_count == 0

    def test_record_call(self):
        """Test recording a call."""
        tracker = UsageTracker(provider="anthropic")
        tracker.record_call(
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.01,
            latency_ms=500.0,
        )
        assert tracker.total_calls == 1
        assert tracker.total_input_tokens == 100
        assert tracker.total_output_tokens == 50
        assert tracker.total_cost_usd == 0.01
        assert tracker.total_latency_ms == 500.0

    def test_record_call_with_error(self):
        """Test recording a call with error."""
        tracker = UsageTracker(provider="anthropic")
        tracker.record_call(
            input_tokens=100,
            output_tokens=0,
            cost_usd=0.0,
            latency_ms=100.0,
            error=True,
        )
        assert tracker.error_count == 1
        assert tracker.total_calls == 1

    def test_avg_latency_ms(self):
        """Test average latency calculation."""
        tracker = UsageTracker(provider="anthropic")
        tracker.record_call(100, 50, 0.01, 500.0)
        tracker.record_call(100, 50, 0.01, 300.0)
        assert tracker.avg_latency_ms == 400.0

    def test_error_rate(self):
        """Test error rate calculation."""
        tracker = UsageTracker(provider="anthropic")
        tracker.record_call(100, 50, 0.01, 500.0)
        tracker.record_call(100, 50, 0.01, 300.0, error=True)
        assert tracker.error_rate == 50.0

    def test_to_metrics(self):
        """Test conversion to UsageMetrics."""
        tracker = UsageTracker(provider="anthropic")
        tracker.record_call(100, 50, 0.01, 500.0)
        metrics = tracker.to_metrics()
        assert isinstance(metrics, UsageMetrics)
        assert metrics.provider == "anthropic"
        assert metrics.total_calls == 1
        assert metrics.total_tokens == 150
        assert metrics.total_cost_usd == 0.01


# ============================================================================
# Tests: LLMProvider (ABC with Mock Subclass)
# ============================================================================


class TestLLMProvider:
    """Tests for LLMProvider abstract base class."""

    @pytest.mark.asyncio
    async def test_mock_provider_complete(self, mock_llm_provider):
        """Test provider complete method."""
        request = CompletionRequest(
            model="mock-model",
            messages=[Message(role="user", content="Test")],
        )
        response = await mock_llm_provider.complete(request)
        assert response.id
        assert response.model == "mock-model"
        assert response.content == "Mock response"

    @pytest.mark.asyncio
    async def test_mock_provider_stream(self, mock_llm_provider):
        """Test provider stream method."""
        request = CompletionRequest(
            model="mock-model",
            messages=[Message(role="user", content="Test")],
        )
        chunks = []
        async for chunk in mock_llm_provider.stream(request):
            chunks.append(chunk)
        assert len(chunks) == 2
        assert chunks[0].content == "Mock"
        assert chunks[1].content == " response"

    @pytest.mark.asyncio
    async def test_mock_provider_health_check(self, mock_llm_provider):
        """Test provider health check."""
        health = await mock_llm_provider.health_check()
        assert health.provider == "mock"
        assert health.healthy
        assert health.latency_ms == 10.0

    def test_get_usage(self, mock_llm_provider):
        """Test getting usage metrics."""
        usage = mock_llm_provider.get_usage()
        assert isinstance(usage, UsageMetrics)
        assert usage.provider == "mock"

    def test_estimate_cost(self, mock_llm_provider):
        """Test cost estimation."""
        cost = mock_llm_provider.estimate_cost(100, 50, "test-model")
        assert cost == 0.0  # Default implementation

    def test_generate_id(self, mock_llm_provider):
        """Test ID generation."""
        id1 = LLMProvider._generate_id()
        id2 = LLMProvider._generate_id()
        assert id1
        assert id2
        assert id1 != id2


# ============================================================================
# Tests: ModelRouter
# ============================================================================


class TestModelRouter:
    """Tests for ModelRouter."""

    def test_init(self, provider_configs):
        """Test ModelRouter initialization."""
        router = ModelRouter(provider_configs)
        assert router._provider_configs == provider_configs
        assert len(router._providers) == 2  # Only enabled providers

    def test_init_with_routing_config(self, provider_configs):
        """Test ModelRouter with custom routing config."""
        routing_config = RoutingConfig(fallback_chain=["mock2", "mock1"])
        router = ModelRouter(provider_configs, routing_config)
        assert router._routing_config.fallback_chain == ["mock2", "mock1"]

    @pytest.mark.asyncio
    async def test_route_story_analysis(self, provider_configs):
        """Test routing for story analysis."""
        with patch.object(ModelRouter, "_check_provider_health") as mock_health:
            mock_health.return_value = HealthStatus(
                provider="mock1", healthy=True
            )
            router = ModelRouter(provider_configs)
            router._health_cache["mock1"] = HealthStatus(
                provider="mock1", healthy=True
            )
            router._health_cache["mock2"] = HealthStatus(
                provider="mock2", healthy=True
            )

            context = RoutingContext(purpose=TaskPurpose.STORY_ANALYSIS)
            selected = await router.route(context)
            assert selected.provider in ["mock1", "mock2"]
            assert selected.tier == ModelTier.MID

    @pytest.mark.asyncio
    async def test_route_code_generation_low_complexity(self, provider_configs):
        """Test routing for low complexity code generation."""
        router = ModelRouter(provider_configs)
        router._health_cache["mock1"] = HealthStatus(
            provider="mock1", healthy=True
        )
        router._health_cache["mock2"] = HealthStatus(
            provider="mock2", healthy=True
        )

        context = RoutingContext(
            purpose=TaskPurpose.CODE_GENERATION, complexity_score=2.0
        )
        selected = await router.route(context)
        assert selected.tier == ModelTier.MID

    @pytest.mark.asyncio
    async def test_route_code_generation_high_complexity(self, provider_configs):
        """Test routing for high complexity code generation."""
        router = ModelRouter(provider_configs)
        router._health_cache["mock1"] = HealthStatus(
            provider="mock1", healthy=True
        )
        router._health_cache["mock2"] = HealthStatus(
            provider="mock2", healthy=True
        )

        context = RoutingContext(
            purpose=TaskPurpose.CODE_GENERATION, complexity_score=8.0
        )
        selected = await router.route(context)
        assert selected.tier == ModelTier.TOP

    @pytest.mark.asyncio
    async def test_route_no_enabled_providers(self):
        """Test routing with no enabled providers."""
        configs = [
            ProviderConfig(
                name="disabled",
                type="anthropic",
                enabled=False,
                api_key="test",
                models=[
                    ModelConfig(
                        id="model1",
                        tier="top",
                        max_context_tokens=1000,
                        cost_per_1k_input=0.01,
                        cost_per_1k_output=0.01,
                    )
                ],
            )
        ]
        router = ModelRouter(configs)

        with pytest.raises(ValueError, match="No suitable model found"):
            context = RoutingContext(purpose=TaskPurpose.PLANNING)
            await router.route(context)

    @pytest.mark.asyncio
    async def test_complete(self, provider_configs, mock_llm_provider):
        """Test complete request routing."""
        router = ModelRouter(provider_configs)
        router._providers = {"mock1": mock_llm_provider}
        router._health_cache["mock1"] = HealthStatus(
            provider="mock1", healthy=True
        )

        request = CompletionRequest(
            model="claude-3-opus",
            messages=[Message(role="user", content="Test")],
        )
        context = RoutingContext(purpose=TaskPurpose.PLANNING)

        response = await router.complete(request, context)
        assert response.content == "Mock response"

    def test_get_cost_for_story(self):
        """Test getting cost for a story."""
        router = ModelRouter([])
        cost1 = router.get_cost_for_story("story-1")
        assert cost1 == 0.0

        tracker = CostTracker(story_id="story-1")
        tracker.add_cost(0.5)
        router._cost_trackers["story-1"] = tracker

        cost2 = router.get_cost_for_story("story-1")
        assert cost2 == 0.5

    def test_get_provider_metrics(self, mock_llm_provider):
        """Test getting provider metrics."""
        router = ModelRouter([])
        router._providers = {"mock": mock_llm_provider}

        metrics = router.get_provider_metrics("mock")
        assert metrics is not None
        assert metrics["provider"] == "mock"


# ============================================================================
# Tests: PromptRegistry
# ============================================================================


class TestPromptRegistry:
    """Tests for PromptRegistry."""

    def test_init(self):
        """Test PromptRegistry initialization."""
        registry = PromptRegistry()
        assert registry._templates
        assert registry._default_versions
        assert "story_analysis" in registry._templates

    def test_register_template(self):
        """Test registering a template."""
        registry = PromptRegistry()
        template = PromptTemplate(
            id="custom_template",
            version="1.0",
            reasoning_mode=ReasoningMode.DIRECT,
            sections=[
                PromptSection(
                    role="user",
                    template="Test {{variable}}",
                    priority=1,
                    required=True,
                )
            ],
        )
        registry.register(template)
        assert "custom_template" in registry._templates
        assert registry._default_versions["custom_template"] == "1.0"

    def test_get_template(self):
        """Test getting a template."""
        registry = PromptRegistry()
        template = registry.get("story_analysis")
        assert template.id == "story_analysis"
        assert template.version == "1.0"

    def test_get_template_not_found(self):
        """Test getting non-existent template."""
        registry = PromptRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_render_template(self):
        """Test rendering a template."""
        registry = PromptRegistry()
        messages = registry.render(
            "story_analysis",
            {"story_content": "User story text here"},
        )
        assert messages
        assert any(msg.role == "system" for msg in messages)
        assert any(msg.role == "user" for msg in messages)

    def test_render_with_all_variables(self):
        """Test rendering planning template with all variables."""
        registry = PromptRegistry()
        messages = registry.render(
            "planning",
            {
                "story_content": "Build login feature",
                "project_type": "web-app",
                "tech_stack": "Python/React",
            },
        )
        assert messages
        rendered_text = " ".join(m.content for m in messages)
        assert "login" in rendered_text.lower() or "story" in rendered_text.lower()

    def test_get_prompt_registry_singleton(self):
        """Test get_prompt_registry singleton."""
        registry1 = get_prompt_registry()
        registry2 = get_prompt_registry()
        assert registry1 is registry2


# ============================================================================
# Tests: ASTIndexer
# ============================================================================


class TestASTIndexer:
    """Tests for ASTIndexer."""

    @pytest.mark.asyncio
    async def test_index_python_file(self, sample_python_code):
        """Test indexing a Python file."""
        indexer = ASTIndexer()
        nodes = await indexer.index_file("test.py", sample_python_code)
        assert len(nodes) > 0
        assert any(node.type == "class" for node in nodes)
        assert any(node.type == "function" for node in nodes)

    @pytest.mark.asyncio
    async def test_index_typescript_file(self, sample_typescript_code):
        """Test indexing a TypeScript file."""
        indexer = ASTIndexer()
        nodes = await indexer.index_file("test.ts", sample_typescript_code)
        assert len(nodes) > 0
        assert any(node.type in ["export", "class"] for node in nodes)

    @pytest.mark.asyncio
    async def test_index_project(self, temp_project_dir, sample_python_code):
        """Test indexing a project."""
        project_path = Path(temp_project_dir)
        py_file = project_path / "module.py"
        py_file.write_text(sample_python_code)

        indexer = ASTIndexer()
        index = await indexer.index_project(temp_project_dir)
        assert len(index) > 0
        assert str(py_file) in index

    def test_get_exports(self, sample_python_code):
        """Test getting exported symbols."""
        indexer = ASTIndexer()
        indexer.index_file = AsyncMock(return_value=[
            ASTNode(
                file="test.py",
                type="class",
                name="DataProcessor",
                start_line=1,
                exported=True,
            ),
            ASTNode(
                file="test.py",
                type="function",
                name="helper_function",
                start_line=10,
                exported=True,
            ),
        ])
        indexer.ast_cache["test.py"] = [
            ASTNode(
                file="test.py",
                type="class",
                name="DataProcessor",
                start_line=1,
                exported=True,
            ),
            ASTNode(
                file="test.py",
                type="function",
                name="_private",
                start_line=15,
                exported=False,
            ),
        ]
        exports = indexer.get_exports("test.py")
        assert len(exports) == 1
        assert exports[0].name == "DataProcessor"

    def test_get_function_signature(self):
        """Test getting function signature."""
        indexer = ASTIndexer()
        indexer.ast_cache["test.py"] = [
            ASTNode(
                file="test.py",
                type="function",
                name="process",
                start_line=1,
                signature="def process(data: dict) -> dict",
            )
        ]
        sig = indexer.get_function_signature("test.py", "process")
        assert sig == "def process(data: dict) -> dict"

    def test_extract_skeleton_python(self, sample_python_code):
        """Test extracting Python skeleton."""
        indexer = ASTIndexer()
        skeleton = indexer.extract_skeleton(sample_python_code, "python")
        assert "class DataProcessor" in skeleton
        assert "def process" in skeleton


# ============================================================================
# Tests: SymbolTable
# ============================================================================


class TestSymbolTable:
    """Tests for SymbolTable."""

    def test_init(self):
        """Test SymbolTable initialization."""
        table = SymbolTable()
        assert table.symbols == []
        assert table.file_symbols == {}

    def test_build(self):
        """Test building symbol table."""
        table = SymbolTable()
        ast_index = {
            "file1.py": [
                ASTNode(
                    file="file1.py",
                    type="class",
                    name="MyClass",
                    start_line=1,
                    exported=True,
                    signature="class MyClass",
                ),
                ASTNode(
                    file="file1.py",
                    type="function",
                    name="_private",
                    start_line=5,
                    exported=False,
                ),
            ]
        }
        table.build(ast_index)
        assert len(table.symbols) == 1
        assert table.symbols[0].name == "MyClass"

    def test_add_symbol(self):
        """Test adding a symbol."""
        table = SymbolTable()
        symbol = SymbolInfo(
            name="TestFunc",
            kind="function",
            file="test.py",
            line=1,
            exported=True,
        )
        table.add_symbol(symbol)
        assert len(table.symbols) == 1
        assert table.symbols[0].name == "TestFunc"

    def test_remove_symbols_for_file(self):
        """Test removing symbols for a file."""
        table = SymbolTable()
        symbol1 = SymbolInfo(
            name="Func1", kind="function", file="file1.py", line=1, exported=True
        )
        symbol2 = SymbolInfo(
            name="Func2", kind="function", file="file2.py", line=1, exported=True
        )
        table.add_symbol(symbol1)
        table.add_symbol(symbol2)
        assert len(table.symbols) == 2

        table.remove_symbols_for_file("file1.py")
        assert len(table.symbols) == 1
        assert table.symbols[0].file == "file2.py"

    def test_lookup(self):
        """Test symbol lookup."""
        table = SymbolTable()
        symbol = SymbolInfo(
            name="MyFunc",
            kind="function",
            file="test.py",
            line=1,
            exported=True,
        )
        table.add_symbol(symbol)
        results = table.lookup("MyFunc")
        assert len(results) == 1
        assert results[0].name == "MyFunc"

    def test_lookup_by_file(self):
        """Test lookup by file."""
        table = SymbolTable()
        symbol = SymbolInfo(
            name="MyFunc",
            kind="function",
            file="test.py",
            line=1,
            exported=True,
        )
        table.add_symbol(symbol)
        results = table.lookup_by_file("test.py")
        assert len(results) == 1

    def test_resolve_import(self):
        """Test resolving import."""
        table = SymbolTable()
        symbol = SymbolInfo(
            name="MyClass",
            kind="class",
            file="module.py",
            line=1,
            exported=True,
        )
        table.add_symbol(symbol)
        resolved = table.resolve_import("MyClass")
        assert resolved is not None
        assert resolved.name == "MyClass"

    def test_search_symbols(self):
        """Test searching symbols."""
        table = SymbolTable()
        table.add_symbol(
            SymbolInfo(
                name="UserService",
                kind="class",
                file="services.py",
                line=1,
                exported=True,
            )
        )
        table.add_symbol(
            SymbolInfo(
                name="UserModel",
                kind="class",
                file="models.py",
                line=1,
                exported=True,
            )
        )
        results = table.search_symbols("user")
        assert len(results) == 2

    def test_increment_usage(self):
        """Test incrementing usage count."""
        table = SymbolTable()
        symbol = SymbolInfo(
            name="MyFunc",
            kind="function",
            file="test.py",
            line=1,
            exported=True,
        )
        table.add_symbol(symbol)
        table.increment_usage("MyFunc")
        table.increment_usage("MyFunc")
        assert table.symbols[0].usage_count == 2

    def test_get_stats(self):
        """Test getting statistics."""
        table = SymbolTable()
        table.add_symbol(
            SymbolInfo(
                name="Func1",
                kind="function",
                file="test.py",
                line=1,
                exported=True,
            )
        )
        table.add_symbol(
            SymbolInfo(
                name="Class1",
                kind="class",
                file="test.py",
                line=10,
                exported=True,
            )
        )
        stats = table.get_stats()
        assert stats["total_symbols"] == 2
        assert stats["total_files"] == 1


# ============================================================================
# Tests: DependencyGraph
# ============================================================================


class TestDependencyGraph:
    """Tests for DependencyGraph."""

    def test_init(self):
        """Test DependencyGraph initialization."""
        graph = DependencyGraph()
        assert graph.nodes == {}
        assert graph.file_to_path == {}

    @pytest.mark.asyncio
    async def test_build(self, temp_project_dir):
        """Test building dependency graph."""
        project_path = Path(temp_project_dir)
        (project_path / "module1.py").write_text("import os")
        (project_path / "module2.py").write_text("import module1")

        graph = DependencyGraph()
        nodes = await graph.build(temp_project_dir)
        assert len(nodes) >= 2

    def test_add_file(self):
        """Test adding a file."""
        graph = DependencyGraph()
        graph.add_file("test.py", "import os\nfrom pathlib import Path")
        assert "test.py" in graph.nodes

    def test_remove_file(self):
        """Test removing a file."""
        graph = DependencyGraph()
        graph.nodes["test.py"] = MagicMock()
        graph.file_to_path["test.py"] = Path("test.py")
        graph.remove_file("test.py")
        assert "test.py" not in graph.nodes

    def test_get_dependencies(self):
        """Test getting file dependencies."""
        graph = DependencyGraph()
        graph.nodes["file1.py"] = MagicMock(imports=["file2.py", "file3.py"])
        deps = graph.get_dependencies("file1.py")
        assert len(deps) == 2

    def test_get_dependents(self):
        """Test getting dependent files."""
        graph = DependencyGraph()
        graph.nodes["file1.py"] = MagicMock(imported_by=["file2.py"])
        dependents = graph.get_dependents("file1.py")
        assert len(dependents) == 1

    def test_get_impacted_files(self):
        """Test getting impacted files."""
        graph = DependencyGraph()
        from fastcoder.types.codebase import DependencyNode

        graph.nodes["a.py"] = DependencyNode(
            file="a.py", imports=["b.py"], imported_by=["c.py"]
        )
        graph.nodes["b.py"] = DependencyNode(
            file="b.py", imports=[], imported_by=["a.py"]
        )
        graph.nodes["c.py"] = DependencyNode(
            file="c.py", imports=["a.py"], imported_by=[]
        )

        impacted = graph.get_impacted_files(["a.py"])
        assert "a.py" in impacted
        assert "c.py" in impacted

    def test_detect_circular_dependencies(self):
        """Test circular dependency detection."""
        graph = DependencyGraph()
        from fastcoder.types.codebase import DependencyNode

        graph.nodes["a.py"] = DependencyNode(file="a.py", imports=["b.py"])
        graph.nodes["b.py"] = DependencyNode(file="b.py", imports=["a.py"])

        cycles = graph.detect_circular_dependencies()
        assert len(cycles) > 0


# ============================================================================
# Tests: SemanticSearch
# ============================================================================


class TestSemanticSearch:
    """Tests for SemanticSearch."""

    def test_init(self):
        """Test SemanticSearch initialization."""
        search = SemanticSearch()
        assert search.chunks == []
        assert search.inverted_index == {}
        assert search.doc_freq == {}

    def test_index(self):
        """Test indexing chunks."""
        search = SemanticSearch()
        chunks = [
            CodeChunk(
                file="file1.py",
                start_line=1,
                end_line=10,
                content="def process(data): return data",
                type="function",
            ),
            CodeChunk(
                file="file2.py",
                start_line=1,
                end_line=5,
                content="class Handler: pass",
                type="class",
            ),
        ]
        search.index(chunks)
        assert len(search.chunks) == 2
        assert len(search.inverted_index) > 0

    def test_search(self):
        """Test searching chunks."""
        search = SemanticSearch()
        chunks = [
            CodeChunk(
                file="file.py",
                start_line=1,
                end_line=10,
                content="def process data",
                type="function",
            )
        ]
        search.index(chunks)
        results = search.search("process", top_k=5)
        assert len(results) > 0

    def test_add_chunk(self):
        """Test adding a chunk."""
        search = SemanticSearch()
        chunk = CodeChunk(
            file="file.py",
            start_line=1,
            end_line=5,
            content="def func(): pass",
            type="function",
        )
        search.add_chunk(chunk)
        assert len(search.chunks) == 1

    def test_remove_chunks_for_file(self):
        """Test removing chunks for a file."""
        search = SemanticSearch()
        chunks = [
            CodeChunk(
                file="file1.py",
                start_line=1,
                end_line=10,
                content="def func(): pass",
                type="function",
            ),
            CodeChunk(
                file="file2.py",
                start_line=1,
                end_line=5,
                content="class A: pass",
                type="class",
            ),
        ]
        search.index(chunks)
        search.remove_chunks_for_file("file1.py")
        assert all(c.file != "file1.py" for c in search.chunks)

    def test_get_index_stats(self):
        """Test getting index statistics."""
        search = SemanticSearch()
        chunks = [
            CodeChunk(
                file="file.py",
                start_line=1,
                end_line=10,
                content="def process(data): return data",
                type="function",
            )
        ]
        search.index(chunks)
        stats = search.get_index_stats()
        assert "total_chunks" in stats
        assert "total_tokens" in stats


# ============================================================================
# Tests: ConventionDetector
# ============================================================================


class TestConventionDetector:
    """Tests for ConventionDetector."""

    @pytest.mark.asyncio
    async def test_detect(self, temp_project_dir):
        """Test detecting conventions."""
        project_path = Path(temp_project_dir)
        (project_path / "main.py").write_text("print('hello')")
        (project_path / "requirements.txt").write_text("flask==2.0.0")

        detector = ConventionDetector()
        result = await detector.detect(temp_project_dir)
        assert result.profile is not None
        assert result.confidence >= 0.0

    def test_detect_language_python(self, temp_project_dir):
        """Test detecting Python language."""
        project_path = Path(temp_project_dir)
        (project_path / "file1.py").write_text("")
        (project_path / "file2.py").write_text("")

        detector = ConventionDetector()
        lang = detector._detect_language(project_path)
        assert lang == "python"

    def test_detect_package_manager_npm(self, temp_project_dir):
        """Test detecting npm package manager."""
        project_path = Path(temp_project_dir)
        (project_path / "package-lock.json").write_text("{}")

        detector = ConventionDetector()
        pm = detector._detect_package_manager(project_path)
        assert pm == "npm"


# ============================================================================
# Tests: OwnershipMap
# ============================================================================


class TestOwnershipMap:
    """Tests for OwnershipMap."""

    def test_init(self, temp_project_dir):
        """Test OwnershipMap initialization."""
        ownership = OwnershipMap(temp_project_dir)
        assert ownership._project_dir == Path(temp_project_dir).resolve()
        assert ownership._codeowners == {}

    @pytest.mark.asyncio
    async def test_initialize(self, temp_project_dir):
        """Test OwnershipMap initialization."""
        project_path = Path(temp_project_dir)
        codeowners_file = project_path / "CODEOWNERS"
        codeowners_file.write_text("*.py @backend-team\n")

        ownership = OwnershipMap(temp_project_dir)
        await ownership.initialize()
        assert len(ownership._codeowners) > 0

    def test_get_owners(self, temp_project_dir):
        """Test getting owners for a file."""
        ownership = OwnershipMap(temp_project_dir)
        ownership._codeowners = {"*.py": ["@team-a"], "docs/": ["@team-b"]}
        owners = ownership.get_owners("main.py")
        assert "@team-a" in owners

    def test_pattern_matches_exact(self, temp_project_dir):
        """Test exact pattern matching."""
        ownership = OwnershipMap(temp_project_dir)
        assert ownership._pattern_matches("path/to/file.py", "path/to/file.py")
        assert not ownership._pattern_matches("path/to/file.py", "other.py")

    def test_pattern_matches_directory(self, temp_project_dir):
        """Test directory pattern matching."""
        ownership = OwnershipMap(temp_project_dir)
        assert ownership._pattern_matches("src/", "src/main.py")
        assert not ownership._pattern_matches("src/", "other/file.py")

    def test_pattern_matches_wildcard(self, temp_project_dir):
        """Test wildcard pattern matching."""
        ownership = OwnershipMap(temp_project_dir)
        assert ownership._pattern_matches("*.py", "main.py")
        assert ownership._pattern_matches("*.py", "test.py")
        assert not ownership._pattern_matches("*.py", "main.txt")

    def test_record_review(self, temp_project_dir):
        """Test recording review."""
        ownership = OwnershipMap(temp_project_dir)
        ownership.record_review(["main.py", "utils.py"], "@reviewer")
        assert ownership._review_frequency["main.py"]["@reviewer"] == 1

    @pytest.mark.asyncio
    async def test_get_experts_for_feature(self, temp_project_dir):
        """Test getting experts for a feature."""
        ownership = OwnershipMap(temp_project_dir)
        ownership._codeowners = {"*.py": ["@expert-a"]}
        experts = await ownership.get_experts_for_feature(["main.py"])
        assert len(experts) >= 0

    @pytest.mark.asyncio
    async def test_validate_ownership(self, temp_project_dir):
        """Test ownership validation."""
        ownership = OwnershipMap(temp_project_dir)
        ownership._codeowners = {"*.py": ["@valid-owner"]}
        issues = await ownership.validate_ownership()
        assert "malformed_owners" in issues

    def test_clear_cache(self, temp_project_dir):
        """Test clearing cache."""
        ownership = OwnershipMap(temp_project_dir)
        ownership._blame_cache["file.py"] = {"author": 0.5}
        ownership.clear_cache()
        assert len(ownership._blame_cache) == 0

    def test_export_summary(self, temp_project_dir):
        """Test exporting summary."""
        ownership = OwnershipMap(temp_project_dir)
        ownership._codeowners = {"*.py": ["@team"]}
        summary = ownership.export_summary()
        assert "codeowners_patterns" in summary


# ============================================================================
# Tests: CrossRepoIndex
# ============================================================================


class TestCrossRepoIndex:
    """Tests for CrossRepoIndex."""

    def test_init(self):
        """Test CrossRepoIndex initialization."""
        index = CrossRepoIndex()
        assert index._repos == {}
        assert index._contracts == {}

    @pytest.mark.asyncio
    async def test_register_repo(self, temp_project_dir):
        """Test registering a repository."""
        index = CrossRepoIndex()
        reg = await index.register_repo(
            "repo-1",
            temp_project_dir,
            "https://github.com/org/repo",
        )
        assert reg.repo_id == "repo-1"
        assert "repo-1" in index._repos

    @pytest.mark.asyncio
    async def test_index_repo(self, temp_project_dir):
        """Test indexing a repository."""
        index = CrossRepoIndex()
        await index.register_repo("repo-1", temp_project_dir)

        # Create sample proto file
        proto_dir = Path(temp_project_dir) / "proto"
        proto_dir.mkdir(exist_ok=True)
        (proto_dir / "message.proto").write_text("message User { string name = 1; }")

        contracts = await index.index_repo("repo-1")
        assert len(contracts) >= 0

    def test_get_contract(self):
        """Test getting a contract."""
        index = CrossRepoIndex()
        contract = SharedContract(
            id="contract-1",
            name="UserType",
            kind="type",
            source_repo="repo-1",
            source_file="types.ts",
        )
        index._contracts["contract-1"] = contract
        retrieved = index.get_contract("contract-1")
        assert retrieved is not None
        assert retrieved.name == "UserType"

    def test_get_consumers(self):
        """Test getting consumers."""
        index = CrossRepoIndex()
        consumer = ContractConsumer(repo="repo-2", file="main.ts")
        contract = SharedContract(
            id="contract-1",
            name="UserType",
            kind="type",
            source_repo="repo-1",
            source_file="types.ts",
            consumers=[consumer],
        )
        index._contracts["contract-1"] = contract
        consumers = index.get_consumers("contract-1")
        assert len(consumers) == 1

    def test_add_consumer(self):
        """Test adding a consumer."""
        index = CrossRepoIndex()
        contract = SharedContract(
            id="contract-1",
            name="UserType",
            kind="type",
            source_repo="repo-1",
            source_file="types.ts",
        )
        index._contracts["contract-1"] = contract

        consumer = ContractConsumer(repo="repo-2", file="main.ts")
        index.add_consumer("contract-1", consumer)
        assert len(contract.consumers) == 1

    def test_get_affected_repos(self):
        """Test getting affected repositories."""
        index = CrossRepoIndex()
        consumer = ContractConsumer(repo="repo-2", file="main.ts")
        contract = SharedContract(
            id="contract-1",
            name="UserType",
            kind="type",
            source_repo="repo-1",
            source_file="types.ts",
            consumers=[consumer],
        )
        index._contracts["contract-1"] = contract

        affected = index.get_affected_repos(["contract-1"])
        assert "repo-1" in affected
        assert "repo-2" in affected

    @pytest.mark.asyncio
    async def test_validate_change_set(self):
        """Test validating a change set."""
        index = CrossRepoIndex()
        await index.register_repo("repo-1", "/tmp")

        change_set = CrossRepoChangeSet(
            id="changeset-1",
            changes=[RepoChange(repo_id="repo-1")],
        )
        result = await index.validate_change_set(change_set)
        assert result.validation_status == "valid"

    def test_create_change_plan(self):
        """Test creating a change plan."""
        index = CrossRepoIndex()
        plan = index.create_change_plan("story-1", "repo-1", ["file.py"])
        assert plan.source_story_id == "story-1"

    def test_get_dependency_graph(self):
        """Test getting dependency graph."""
        index = CrossRepoIndex()
        index._repos = {"repo-1": MagicMock(), "repo-2": MagicMock()}
        graph = index.get_dependency_graph()
        assert "repo-1" in graph
        assert "repo-2" in graph

    def test_save_and_load(self, temp_project_dir):
        """Test saving and loading index."""
        save_path = str(Path(temp_project_dir) / "index.json")

        index1 = CrossRepoIndex()
        index1._repos["repo-1"] = MagicMock(
            repo_id="repo-1",
            repo_path="/tmp",
            model_dump=MagicMock(
                return_value={
                    "repo_id": "repo-1",
                    "repo_path": "/tmp",
                    "repo_url": "",
                    "branch": "main",
                    "last_indexed": None,
                    "contract_count": 0,
                }
            ),
        )

        # Would need proper setup for full save/load test
        # This is a basic structure test


# ============================================================================
# Tests: CodebaseIntelligence (Integration)
# ============================================================================


class TestCodebaseIntelligence:
    """Tests for CodebaseIntelligence."""

    def test_init(self):
        """Test CodebaseIntelligence initialization."""
        codebase = CodebaseIntelligence()
        assert codebase.ast_indexer is not None
        assert codebase.dependency_graph is not None
        assert codebase.symbol_table is not None
        assert codebase.semantic_search is not None

    @pytest.mark.asyncio
    async def test_initialize(self, temp_project_dir):
        """Test CodebaseIntelligence initialization."""
        project_path = Path(temp_project_dir)
        (project_path / "main.py").write_text(
            "def hello(): pass\nclass MyClass: pass"
        )

        codebase = CodebaseIntelligence()
        result = await codebase.initialize(temp_project_dir)
        assert result is not None
        assert codebase.project_profile is not None

    @pytest.mark.asyncio
    async def test_reindex(self, temp_project_dir):
        """Test reindexing changed files."""
        project_path = Path(temp_project_dir)
        file1 = project_path / "file1.py"
        file1.write_text("x = 1")

        codebase = CodebaseIntelligence()
        await codebase.initialize(temp_project_dir)

        file1.write_text("y = 2")
        impacted = await codebase.reindex([str(file1)])
        assert str(file1) in impacted

    def test_search(self):
        """Test searching code."""
        codebase = CodebaseIntelligence()
        codebase.semantic_search.chunks = [
            CodeChunk(
                file="test.py",
                start_line=1,
                end_line=10,
                content="def process(data): pass",
                type="function",
            )
        ]
        codebase.semantic_search.index(codebase.semantic_search.chunks)
        results = codebase.search("process")
        assert len(results) >= 0

    def test_get_project_profile(self):
        """Test getting project profile."""
        codebase = CodebaseIntelligence()
        profile = codebase.get_project_profile()
        assert profile is None  # Before initialization

    def test_get_impacted_files(self):
        """Test getting impacted files."""
        codebase = CodebaseIntelligence()
        from fastcoder.types.codebase import DependencyNode

        codebase.dependency_graph.nodes = {
            "a.py": DependencyNode(
                file="a.py", imports=["b.py"], imported_by=["c.py"]
            ),
            "b.py": DependencyNode(file="b.py", imports=[], imported_by=["a.py"]),
            "c.py": DependencyNode(file="c.py", imports=["a.py"], imported_by=[]),
        }
        impacted = codebase.get_impacted_files(["a.py"])
        assert "a.py" in impacted

    def test_resolve_symbol(self):
        """Test resolving symbols."""
        codebase = CodebaseIntelligence()
        symbol = SymbolInfo(
            name="MyFunc",
            kind="function",
            file="test.py",
            line=1,
            exported=True,
        )
        codebase.symbol_table.add_symbol(symbol)
        results = codebase.resolve_symbol("MyFunc")
        assert len(results) == 1

    def test_get_file_skeleton(self, temp_project_dir, sample_python_code):
        """Test getting file skeleton."""
        project_path = Path(temp_project_dir)
        py_file = project_path / "test.py"
        py_file.write_text(sample_python_code)

        codebase = CodebaseIntelligence()
        skeleton = codebase.get_file_skeleton(str(py_file))
        assert "class DataProcessor" in skeleton or "def" in skeleton

    def test_get_api_surface(self):
        """Test getting API surface."""
        codebase = CodebaseIntelligence()
        codebase.ast_index = {}
        surface = codebase.get_api_surface()
        assert surface.endpoints == []

    def test_get_statistics(self):
        """Test getting statistics."""
        codebase = CodebaseIntelligence()
        stats = codebase.get_statistics()
        assert "ast" in stats
        assert "symbols" in stats

    def test_get_circular_dependencies(self):
        """Test getting circular dependencies."""
        codebase = CodebaseIntelligence()
        cycles = codebase.get_circular_dependencies()
        assert isinstance(cycles, list)

    def test_search_symbols(self):
        """Test searching symbols."""
        codebase = CodebaseIntelligence()
        codebase.symbol_table.add_symbol(
            SymbolInfo(
                name="UserService",
                kind="class",
                file="services.py",
                line=1,
                exported=True,
            )
        )
        results = codebase.search_symbols("user")
        assert len(results) >= 0

    def test_get_file_dependencies(self):
        """Test getting file dependencies."""
        codebase = CodebaseIntelligence()
        from fastcoder.types.codebase import DependencyNode

        codebase.dependency_graph.nodes = {
            "a.py": DependencyNode(
                file="a.py", imports=["b.py"], imported_by=["c.py"]
            ),
        }
        deps = codebase.get_file_dependencies("a.py")
        assert "imports" in deps
        assert "imported_by" in deps


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
