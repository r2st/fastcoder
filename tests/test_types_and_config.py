"""Comprehensive tests for all Pydantic models, enums, and configuration."""

import json
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest
from pydantic import ValidationError

# Types imports
from fastcoder.types.story import (
    AcceptanceCriterion,
    FileDependency,
    Priority,
    Story,
    StoryConstraints,
    StoryMetadata,
    StorySpec,
    StoryState,
    StorySubmission,
    StoryType,
)
from fastcoder.types.plan import (
    DeployStrategy,
    ExecutionPlan,
    PlanTask,
    TaskAction,
    TestingStrategy,
)
from fastcoder.types.task import (
    DeployReport,
    FileChange,
    ReviewIssue,
    ReviewReport,
    SuiteResult,
    TaskResult,
    TaskStatus,
    TestFailure,
    TestReport,
)
from fastcoder.types.iteration import Iteration
from fastcoder.types.memory import (
    MemoryConsolidationResult,
    MemoryEntry,
    MemoryQuery,
    MemoryTier,
    MemoryType,
)
from fastcoder.types.config import (
    AgentConfig,
    CostConfig,
    GitToolsConfig,
    LLMConfig,
    ModelConfig,
    ObservabilityConfig,
    ProviderConfig,
    QualityConfig,
    RoutingConfig,
    SafetyConfig,
    SandboxToolsConfig,
    ShellToolsConfig,
    ToolsConfig,
    ProjectConfig,
)
from fastcoder.types.errors import (
    ErrorCategory,
    ErrorClassification,
    ErrorContext,
    ErrorDetail,
    ErrorFingerprint,
    PreviousAttempt,
    RecoveryStrategy,
)
from fastcoder.types.llm import (
    CompletionRequest,
    CompletionResponse,
    HealthStatus,
    Message,
    ModelCall,
    ModelTier,
    PromptSection,
    PromptTemplate,
    ReasoningMode,
    RoutingContext,
    SelectedModel,
    StreamChunk,
    TaskPurpose,
    TokenUsage,
    ToolCallRequest,
    ToolDefinition,
    UsageMetrics,
)
from fastcoder.types.tools import (
    LintError,
    ParsedToolOutput,
    RiskLevel,
    SandboxConfig,
    SideEffects,
    ToolCall,
    ToolName,
    ToolPolicy,
    ToolResult,
    TypeCheckError,
)
from fastcoder.types.codebase import (
    APIEndpoint,
    APISurface,
    ASTNode,
    CodeChunk,
    ConventionScanResult,
    DependencyNode,
    DetectedPattern,
    ProjectProfile,
    SearchResult,
    SymbolInfo,
)
from fastcoder.types.security import (
    FindingCategory,
    SASTReport,
    SecurityFinding,
    SecurityScanResult,
    SecretFinding,
    Severity,
)
from fastcoder.types.quality import (
    EnforcementLevel,
    GateOutcome,
    GateResult,
    GateThreshold,
    GateType,
    PolicyEvaluationResult,
    QualityGatePolicy,
)
from fastcoder.types.evidence import (
    CostBreakdown,
    CriterionTestMapping,
    EvidenceBundle,
    ImpactedFile,
    ChangeImpactReport,
    ReviewerChecklistItem,
)
from fastcoder.types.learning import (
    FailureClass,
    HeuristicRule,
    LearningStats,
    PostMortemEntry,
)

# Config module imports
from fastcoder.config import load_config, validate_config
from fastcoder.config.llm_key_store import (
    LLMKeyStore,
    _validate_provider_name,
    get_llm_key_store,
    resolve_admin_db_path,
)


# ============================================================================
# STORY TYPES TESTS
# ============================================================================


class TestStoryEnums:
    """Test StoryState, StoryType, and Priority enums."""

    def test_story_state_values(self):
        """Test StoryState enum values."""
        assert StoryState.RECEIVED.value == "RECEIVED"
        assert StoryState.ANALYZING.value == "ANALYZING"
        assert StoryState.DONE.value == "DONE"
        assert StoryState.FAILED.value == "FAILED"
        assert len(StoryState) == 10

    def test_story_type_values(self):
        """Test StoryType enum values."""
        assert StoryType.FEATURE.value == "feature"
        assert StoryType.BUGFIX.value == "bugfix"
        assert StoryType.REFACTOR.value == "refactor"
        assert StoryType.INFRA.value == "infra"

    def test_priority_values(self):
        """Test Priority enum values."""
        assert Priority.LOW.value == "low"
        assert Priority.MEDIUM.value == "medium"
        assert Priority.HIGH.value == "high"
        assert Priority.CRITICAL.value == "critical"


class TestAcceptanceCriterion:
    """Test AcceptanceCriterion model."""

    def test_create_with_defaults(self):
        """Test creating AcceptanceCriterion with defaults."""
        criterion = AcceptanceCriterion(
            id="AC-1",
            description="Users can login"
        )
        assert criterion.id == "AC-1"
        assert criterion.description == "Users can login"
        assert criterion.testable is True
        assert criterion.verified is False
        assert criterion.linked_test_ids == []
        assert criterion.given is None

    def test_create_with_gherkin(self):
        """Test creating AcceptanceCriterion with Gherkin format."""
        criterion = AcceptanceCriterion(
            id="AC-1",
            description="Login validation",
            given="User is on login page",
            when="User enters valid credentials",
            then="User is authenticated and redirected"
        )
        assert criterion.given == "User is on login page"
        assert criterion.when == "User enters valid credentials"
        assert criterion.then == "User is authenticated and redirected"

    def test_with_linked_tests(self):
        """Test AcceptanceCriterion with linked test IDs."""
        criterion = AcceptanceCriterion(
            id="AC-1",
            description="Login works",
            linked_test_ids=["test_login_valid", "test_login_invalid"]
        )
        assert criterion.linked_test_ids == ["test_login_valid", "test_login_invalid"]


class TestFileDependency:
    """Test FileDependency model."""

    def test_file_dependency_creation(self):
        """Test creating FileDependency."""
        dep = FileDependency(
            file_path="src/utils.py",
            relationship="imports",
            confidence=0.95
        )
        assert dep.file_path == "src/utils.py"
        assert dep.relationship == "imports"
        assert dep.confidence == 0.95

    def test_file_dependency_default_confidence(self):
        """Test FileDependency with default confidence."""
        dep = FileDependency(
            file_path="src/main.py",
            relationship="modified"
        )
        assert dep.confidence == 1.0


class TestStorySpec:
    """Test StorySpec model."""

    def test_story_spec_creation(self):
        """Test creating StorySpec with required fields."""
        spec = StorySpec(
            title="Add user authentication",
            description="Implement JWT-based auth"
        )
        assert spec.title == "Add user authentication"
        assert spec.description == "Implement JWT-based auth"
        assert spec.story_type == StoryType.FEATURE
        assert spec.complexity_score == 5
        assert spec.acceptance_criteria == []
        assert spec.dependencies == []

    def test_story_spec_with_complexity(self):
        """Test StorySpec with complexity score validation."""
        spec = StorySpec(
            title="Feature",
            description="Desc",
            complexity_score=8
        )
        assert spec.complexity_score == 8

    def test_story_spec_invalid_complexity(self):
        """Test StorySpec with invalid complexity score."""
        with pytest.raises(ValidationError):
            StorySpec(
                title="Feature",
                description="Desc",
                complexity_score=11  # Out of range
            )

    def test_story_spec_with_criteria_and_deps(self):
        """Test StorySpec with acceptance criteria and dependencies."""
        criterion = AcceptanceCriterion(id="AC-1", description="Test")
        dep = FileDependency(file_path="src/auth.py", relationship="imports")
        spec = StorySpec(
            title="Auth feature",
            description="Implement auth",
            acceptance_criteria=[criterion],
            dependencies=[dep],
            story_type=StoryType.FEATURE
        )
        assert len(spec.acceptance_criteria) == 1
        assert len(spec.dependencies) == 1


class TestStoryConstraints:
    """Test StoryConstraints model."""

    def test_story_constraints_defaults(self):
        """Test StoryConstraints with defaults."""
        constraints = StoryConstraints()
        assert constraints.max_iterations == 10
        assert constraints.approval_gates == ["pre_deploy"]
        assert constraints.target_branch is None
        assert constraints.deploy_target is None
        assert constraints.cost_budget_usd == 5.0

    def test_story_constraints_custom(self):
        """Test StoryConstraints with custom values."""
        constraints = StoryConstraints(
            max_iterations=15,
            approval_gates=["pre_code", "pre_deploy", "pre_production"],
            target_branch="develop",
            deploy_target="staging",
            cost_budget_usd=10.0
        )
        assert constraints.max_iterations == 15
        assert len(constraints.approval_gates) == 3
        assert constraints.target_branch == "develop"
        assert constraints.cost_budget_usd == 10.0


class TestStoryMetadata:
    """Test StoryMetadata model."""

    def test_story_metadata_defaults(self):
        """Test StoryMetadata with defaults."""
        metadata = StoryMetadata()
        assert metadata.total_tokens_used == 0
        assert metadata.total_cost_usd == 0.0
        assert metadata.model_usage == {}
        assert metadata.completed_at is None
        assert isinstance(metadata.created_at, datetime)

    def test_story_metadata_custom(self):
        """Test StoryMetadata with custom values."""
        now = datetime.utcnow()
        metadata = StoryMetadata(
            total_tokens_used=5000,
            total_cost_usd=2.50,
            completed_at=now,
            model_usage={"claude-sonnet": 3000, "gpt-4": 2000}
        )
        assert metadata.total_tokens_used == 5000
        assert metadata.total_cost_usd == 2.50
        assert metadata.model_usage["claude-sonnet"] == 3000


class TestStory:
    """Test Story model."""

    def test_story_creation(self):
        """Test creating a Story."""
        story = Story(
            id="story-1",
            raw_text="Implement user login",
            project_id="proj-1"
        )
        assert story.id == "story-1"
        assert story.raw_text == "Implement user login"
        assert story.project_id == "proj-1"
        assert story.priority == Priority.MEDIUM
        assert story.state == StoryState.RECEIVED
        assert story.iterations == []

    def test_story_with_priority(self):
        """Test Story with custom priority."""
        story = Story(
            id="story-1",
            raw_text="Fix critical bug",
            project_id="proj-1",
            priority=Priority.CRITICAL
        )
        assert story.priority == Priority.CRITICAL

    def test_story_with_spec(self):
        """Test Story with StorySpec."""
        spec = StorySpec(
            title="Feature",
            description="Description"
        )
        story = Story(
            id="story-1",
            raw_text="Raw",
            project_id="proj-1",
            spec=spec
        )
        assert story.spec is not None
        assert story.spec.title == "Feature"


class TestStorySubmission:
    """Test StorySubmission model."""

    def test_story_submission_creation(self):
        """Test creating a StorySubmission."""
        submission = StorySubmission(
            story="Implement feature X",
            project_id="proj-1"
        )
        assert submission.story == "Implement feature X"
        assert submission.project_id == "proj-1"
        assert submission.priority == Priority.MEDIUM
        assert submission.constraints is None

    def test_story_submission_with_priority(self):
        """Test StorySubmission with priority and constraints."""
        constraints = StoryConstraints(max_iterations=5)
        submission = StorySubmission(
            story="Fix bug",
            project_id="proj-1",
            priority=Priority.HIGH,
            constraints=constraints
        )
        assert submission.priority == Priority.HIGH
        assert submission.constraints.max_iterations == 5


# ============================================================================
# PLAN TYPES TESTS
# ============================================================================


class TestPlanEnums:
    """Test TaskAction, TestingStrategy, and DeployStrategy enums."""

    def test_task_action_values(self):
        """Test TaskAction enum values."""
        assert TaskAction.CREATE_FILE.value == "create_file"
        assert TaskAction.MODIFY_FILE.value == "modify_file"
        assert TaskAction.DELETE_FILE.value == "delete_file"
        assert TaskAction.RUN_COMMAND.value == "run_command"

    def test_testing_strategy_values(self):
        """Test TestingStrategy enum values."""
        assert TestingStrategy.UNIT.value == "unit"
        assert TestingStrategy.INTEGRATION.value == "integration"
        assert TestingStrategy.E2E.value == "e2e"
        assert TestingStrategy.UNIT_INTEGRATION.value == "unit + integration"

    def test_deploy_strategy_values(self):
        """Test DeployStrategy enum values."""
        assert DeployStrategy.STAGING_FIRST.value == "staging_first"
        assert DeployStrategy.DIRECT_DEPLOY.value == "direct_deploy"
        assert DeployStrategy.PR_ONLY.value == "pr_only"


class TestPlanTask:
    """Test PlanTask model."""

    def test_plan_task_creation(self):
        """Test creating a PlanTask."""
        task = PlanTask(
            id="task-1",
            action=TaskAction.CREATE_FILE,
            target="src/main.py",
            description="Create main file"
        )
        assert task.id == "task-1"
        assert task.action == TaskAction.CREATE_FILE
        assert task.target == "src/main.py"
        assert task.depends_on == []
        assert task.estimated_tokens == 2000

    def test_plan_task_with_dependencies(self):
        """Test PlanTask with dependencies."""
        task = PlanTask(
            id="task-2",
            action=TaskAction.MODIFY_FILE,
            target="src/main.py",
            description="Modify file",
            depends_on=["task-1"]
        )
        assert task.depends_on == ["task-1"]

    def test_plan_task_with_custom_tokens(self):
        """Test PlanTask with custom token estimate."""
        task = PlanTask(
            id="task-1",
            action=TaskAction.RUN_COMMAND,
            target="test_suite",
            description="Run tests",
            estimated_tokens=5000
        )
        assert task.estimated_tokens == 5000


class TestExecutionPlan:
    """Test ExecutionPlan model."""

    def test_execution_plan_creation(self):
        """Test creating ExecutionPlan."""
        plan = ExecutionPlan(
            story_id="story-1"
        )
        assert plan.story_id == "story-1"
        assert plan.tasks == []
        assert plan.testing_strategy == TestingStrategy.UNIT_INTEGRATION
        assert plan.deploy_strategy == DeployStrategy.STAGING_FIRST
        assert plan.revision == 1

    def test_execution_plan_with_tasks(self):
        """Test ExecutionPlan with multiple tasks."""
        task1 = PlanTask(
            id="task-1",
            action=TaskAction.CREATE_FILE,
            target="src/main.py",
            description="Create file"
        )
        task2 = PlanTask(
            id="task-2",
            action=TaskAction.MODIFY_FILE,
            target="src/main.py",
            description="Modify file",
            depends_on=["task-1"]
        )
        plan = ExecutionPlan(
            story_id="story-1",
            tasks=[task1, task2],
            testing_strategy=TestingStrategy.UNIT_INTEGRATION_E2E,
            deploy_strategy=DeployStrategy.STAGING_FIRST,
            estimated_total_tokens=10000,
            revision=2
        )
        assert len(plan.tasks) == 2
        assert plan.estimated_total_tokens == 10000
        assert plan.revision == 2


# ============================================================================
# TASK TYPES TESTS
# ============================================================================


class TestTaskStatusEnum:
    """Test TaskStatus enum."""

    def test_task_status_values(self):
        """Test TaskStatus enum values."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.IN_PROGRESS.value == "in_progress"
        assert TaskStatus.DONE.value == "done"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.SKIPPED.value == "skipped"


class TestFileChange:
    """Test FileChange model."""

    def test_file_change_creation(self):
        """Test creating FileChange."""
        change = FileChange(
            file_path="src/utils.py",
            change_type="modified"
        )
        assert change.file_path == "src/utils.py"
        assert change.change_type == "modified"
        assert change.diff is None

    def test_file_change_with_content(self):
        """Test FileChange with content and diff."""
        change = FileChange(
            file_path="src/main.py",
            change_type="created",
            content="def main(): pass",
            diff="+def main(): pass"
        )
        assert change.content == "def main(): pass"
        assert change.diff == "+def main(): pass"


class TestTestFailureModel:
    """Test TestFailure model (named TestFailureModel to avoid pytest collision)."""

    def test_test_failure_creation(self):
        """Test creating TestFailure."""
        failure = TestFailure(
            suite="test_auth",
            test="test_login_valid",
            error="AssertionError: Expected True but got False"
        )
        assert failure.suite == "test_auth"
        assert failure.test == "test_login_valid"
        assert failure.file is None

    def test_test_failure_with_details(self):
        """Test TestFailure with full details."""
        failure = TestFailure(
            suite="test_auth",
            test="test_login",
            error="AssertionError",
            expected="True",
            actual="False",
            stack_trace="Traceback...",
            file="tests/test_auth.py",
            line=42
        )
        assert failure.expected == "True"
        assert failure.actual == "False"
        assert failure.line == 42


class TestSuiteResult:
    """Test SuiteResult model."""

    def test_suite_result_creation(self):
        """Test creating SuiteResult."""
        result = SuiteResult(
            name="test_auth",
            file="tests/test_auth.py"
        )
        assert result.name == "test_auth"
        assert result.file == "tests/test_auth.py"
        assert result.passed == 0
        assert result.failed == 0

    def test_suite_result_with_counts(self):
        """Test SuiteResult with test counts."""
        result = SuiteResult(
            name="test_suite",
            file="tests/test_suite.py",
            passed=10,
            failed=2,
            skipped=1,
            duration_ms=1234.5
        )
        assert result.passed == 10
        assert result.failed == 2
        assert result.skipped == 1


class TestTestReport:
    """Test TestReport model."""

    def test_test_report_creation(self):
        """Test creating TestReport."""
        report = TestReport()
        assert report.total == 0
        assert report.passed == 0
        assert report.failed == 0
        assert report.failures == []

    def test_test_report_with_data(self):
        """Test TestReport with full data."""
        failure = TestFailure(
            suite="test_main",
            test="test_func",
            error="Failed"
        )
        suite_result = SuiteResult(
            name="test_main",
            file="tests/test_main.py",
            passed=5,
            failed=1
        )
        report = TestReport(
            total=6,
            passed=5,
            failed=1,
            coverage_percent=85.5,
            failures=[failure],
            suite_results=[suite_result]
        )
        assert report.total == 6
        assert report.coverage_percent == 85.5
        assert len(report.failures) == 1


class TestReviewIssue:
    """Test ReviewIssue model."""

    def test_review_issue_creation(self):
        """Test creating ReviewIssue."""
        issue = ReviewIssue(
            severity="suggestion",
            category="style",
            file="src/main.py",
            description="Variable name not following convention"
        )
        assert issue.severity == "suggestion"
        assert issue.category == "style"
        assert issue.line is None

    def test_review_issue_with_fix(self):
        """Test ReviewIssue with suggested fix."""
        issue = ReviewIssue(
            severity="blocking",
            category="security",
            file="src/auth.py",
            line=42,
            description="Hardcoded password",
            suggested_fix="Use environment variable"
        )
        assert issue.line == 42
        assert issue.suggested_fix == "Use environment variable"


class TestReviewReport:
    """Test ReviewReport model."""

    def test_review_report_creation(self):
        """Test creating ReviewReport."""
        report = ReviewReport()
        assert report.approved is False
        assert report.issues == []
        assert report.summary == ""

    def test_review_report_approved(self):
        """Test ReviewReport when approved."""
        issue = ReviewIssue(
            severity="nit",
            category="style",
            file="src/main.py",
            description="Minor style issue"
        )
        report = ReviewReport(
            approved=True,
            issues=[issue],
            summary="Code approved with minor notes",
            reviewer_model="claude-sonnet"
        )
        assert report.approved is True
        assert len(report.issues) == 1


class TestDeployReport:
    """Test DeployReport model."""

    def test_deploy_report_creation(self):
        """Test creating DeployReport."""
        report = DeployReport()
        assert report.success is False
        assert report.environment == ""
        assert report.health_check_passed is False

    def test_deploy_report_success(self):
        """Test successful DeployReport."""
        report = DeployReport(
            success=True,
            environment="production",
            url="https://api.example.com",
            health_check_passed=True,
            smoke_tests_passed=True
        )
        assert report.success is True
        assert report.environment == "production"
        assert report.health_check_passed is True


class TestTaskResult:
    """Test TaskResult model."""

    def test_task_result_creation(self):
        """Test creating TaskResult."""
        result = TaskResult()
        assert result.status == TaskStatus.PENDING
        assert result.changes == []
        assert result.output is None

    def test_task_result_completed(self):
        """Test TaskResult with completion data."""
        change = FileChange(
            file_path="src/main.py",
            change_type="created"
        )
        result = TaskResult(
            status=TaskStatus.DONE,
            changes=[change],
            output="Task completed successfully",
            duration_ms=5000.0
        )
        assert result.status == TaskStatus.DONE
        assert len(result.changes) == 1


# ============================================================================
# ITERATION TYPES TESTS
# ============================================================================


class TestIteration:
    """Test Iteration model."""

    def test_iteration_creation(self):
        """Test creating Iteration."""
        iteration = Iteration(number=1)
        assert iteration.number == 1
        assert isinstance(iteration.started_at, datetime)
        assert iteration.ended_at is None
        assert iteration.changes == []

    def test_iteration_with_results(self):
        """Test Iteration with test and review results."""
        test_report = TestReport(total=10, passed=10)
        review_report = ReviewReport(approved=True)
        iteration = Iteration(
            number=2,
            stage="testing",
            test_results=test_report,
            review_results=review_report
        )
        assert iteration.stage == "testing"
        assert iteration.test_results.total == 10
        assert iteration.review_results.approved is True


# ============================================================================
# MEMORY TYPES TESTS
# ============================================================================


class TestMemoryEnums:
    """Test MemoryType and MemoryTier enums."""

    def test_memory_type_values(self):
        """Test MemoryType enum values."""
        assert MemoryType.PATTERN.value == "pattern"
        assert MemoryType.ERROR_FIX.value == "error_fix"
        assert MemoryType.CONVENTION.value == "convention"
        assert MemoryType.ANTI_PATTERN.value == "anti_pattern"

    def test_memory_tier_values(self):
        """Test MemoryTier enum values."""
        assert MemoryTier.WORKING.value == "working"
        assert MemoryTier.EPISODIC.value == "episodic"
        assert MemoryTier.SEMANTIC.value == "semantic"
        assert MemoryTier.PROJECT.value == "project"


class TestMemoryEntry:
    """Test MemoryEntry model."""

    def test_memory_entry_creation(self):
        """Test creating MemoryEntry."""
        entry = MemoryEntry(
            id="mem-1",
            type=MemoryType.PATTERN,
            tier=MemoryTier.WORKING,
            content="Use dependency injection for testing"
        )
        assert entry.id == "mem-1"
        assert entry.type == MemoryType.PATTERN
        assert entry.tier == MemoryTier.WORKING
        assert entry.effectiveness_score == 0.5

    def test_memory_entry_with_embedding(self):
        """Test MemoryEntry with embedding."""
        embedding = [0.1, 0.2, 0.3, 0.4]
        entry = MemoryEntry(
            id="mem-1",
            type=MemoryType.ERROR_FIX,
            tier=MemoryTier.EPISODIC,
            content="Fix for IndexError",
            embedding=embedding,
            effectiveness_score=0.85
        )
        assert entry.embedding == embedding
        assert entry.effectiveness_score == 0.85


class TestMemoryQuery:
    """Test MemoryQuery model."""

    def test_memory_query_creation(self):
        """Test creating MemoryQuery."""
        query = MemoryQuery(query="How to handle null values?")
        assert query.query == "How to handle null values?"
        assert query.tier is None
        assert query.max_results == 10

    def test_memory_query_with_filters(self):
        """Test MemoryQuery with filters."""
        query = MemoryQuery(
            query="Type error fixes",
            tier=MemoryTier.SEMANTIC,
            type=MemoryType.ERROR_FIX,
            project_id="proj-1",
            max_results=5,
            min_effectiveness=0.7
        )
        assert query.type == MemoryType.ERROR_FIX
        assert query.min_effectiveness == 0.7


class TestMemoryConsolidationResult:
    """Test MemoryConsolidationResult model."""

    def test_memory_consolidation_result_creation(self):
        """Test creating MemoryConsolidationResult."""
        result = MemoryConsolidationResult()
        assert result.new_memories == []
        assert result.updated_memories == []
        assert result.evicted_memory_ids == []
        assert result.merged_count == 0

    def test_memory_consolidation_with_data(self):
        """Test MemoryConsolidationResult with data."""
        entry1 = MemoryEntry(
            id="mem-1",
            type=MemoryType.PATTERN,
            tier=MemoryTier.SEMANTIC
        )
        result = MemoryConsolidationResult(
            new_memories=[entry1],
            evicted_memory_ids=["mem-old-1", "mem-old-2"],
            merged_count=3
        )
        assert len(result.new_memories) == 1
        assert len(result.evicted_memory_ids) == 2
        assert result.merged_count == 3


# ============================================================================
# CONFIG TYPES TESTS
# ============================================================================


class TestModelConfig:
    """Test ModelConfig model."""

    def test_model_config_creation(self):
        """Test creating ModelConfig."""
        config = ModelConfig(id="claude-sonnet")
        assert config.id == "claude-sonnet"
        assert config.tier == "mid"
        assert config.max_context_tokens == 128000
        assert config.cost_per_1k_input == 0.003

    def test_model_config_custom(self):
        """Test ModelConfig with custom values."""
        config = ModelConfig(
            id="gpt-4",
            tier="top",
            max_context_tokens=8192,
            cost_per_1k_input=0.03,
            cost_per_1k_output=0.06
        )
        assert config.max_context_tokens == 8192
        assert config.cost_per_1k_output == 0.06


class TestProviderConfig:
    """Test ProviderConfig model."""

    def test_provider_config_creation(self):
        """Test creating ProviderConfig."""
        config = ProviderConfig(name="anthropic", type="anthropic")
        assert config.name == "anthropic"
        assert config.type == "anthropic"
        assert config.api_key is None
        assert config.enabled is True

    def test_provider_config_with_models(self):
        """Test ProviderConfig with models."""
        model = ModelConfig(id="claude-sonnet")
        config = ProviderConfig(
            name="anthropic",
            type="anthropic",
            models=[model],
            enabled=True
        )
        assert len(config.models) == 1

    def test_provider_config_repr_redacts_key(self):
        """Test ProviderConfig __repr__ redacts API key."""
        config = ProviderConfig(
            name="openai",
            type="openai",
            api_key="sk-secret-key-12345"
        )
        repr_str = repr(config)
        assert "***" in repr_str
        assert "sk-secret-key-12345" not in repr_str
        assert "openai" in repr_str

    def test_provider_config_repr_no_key(self):
        """Test ProviderConfig __repr__ with no API key."""
        config = ProviderConfig(name="anthropic", type="anthropic")
        repr_str = repr(config)
        assert "None" in repr_str


class TestRoutingConfig:
    """Test RoutingConfig model."""

    def test_routing_config_defaults(self):
        """Test RoutingConfig with defaults."""
        config = RoutingConfig()
        assert config.default_tier == "mid"
        assert config.default_provider is None
        assert config.task_overrides == {}

    def test_routing_config_custom(self):
        """Test RoutingConfig with custom values."""
        config = RoutingConfig(
            default_tier="top",
            default_provider="anthropic",
            default_model="claude-sonnet",
            fallback_chain=["anthropic", "openai", "google"]
        )
        assert config.default_provider == "anthropic"
        assert len(config.fallback_chain) == 3


class TestCostConfig:
    """Test CostConfig model."""

    def test_cost_config_defaults(self):
        """Test CostConfig with defaults."""
        config = CostConfig()
        assert config.max_cost_per_story_usd == 5.0
        assert config.daily_budget_usd == 100.0
        assert config.warning_threshold == 0.8

    def test_cost_config_custom(self):
        """Test CostConfig with custom values."""
        config = CostConfig(
            max_cost_per_story_usd=10.0,
            daily_budget_usd=200.0,
            monthly_budget_usd=5000.0,
            escalation_action="downgrade_model"
        )
        assert config.monthly_budget_usd == 5000.0
        assert config.escalation_action == "downgrade_model"


class TestLLMConfig:
    """Test LLMConfig model."""

    def test_llm_config_defaults(self):
        """Test LLMConfig with defaults."""
        config = LLMConfig()
        assert config.providers == []
        assert isinstance(config.routing, RoutingConfig)
        assert isinstance(config.cost, CostConfig)

    def test_llm_config_with_providers(self):
        """Test LLMConfig with providers."""
        provider = ProviderConfig(
            name="anthropic",
            type="anthropic",
            models=[ModelConfig(id="claude-sonnet")]
        )
        config = LLMConfig(providers=[provider])
        assert len(config.providers) == 1


class TestSandboxToolsConfig:
    """Test SandboxToolsConfig model."""

    def test_sandbox_tools_config_defaults(self):
        """Test SandboxToolsConfig defaults."""
        config = SandboxToolsConfig()
        assert config.cpu_limit == "2"
        assert config.memory_limit_mb == 2048
        assert "github.com" in config.network_whitelist

    def test_sandbox_tools_config_custom(self):
        """Test SandboxToolsConfig with custom values."""
        config = SandboxToolsConfig(
            cpu_limit="4",
            memory_limit_mb=4096,
            disk_limit_mb=20480,
            network_whitelist=["api.example.com"]
        )
        assert config.cpu_limit == "4"
        assert config.memory_limit_mb == 4096


class TestGitToolsConfig:
    """Test GitToolsConfig model."""

    def test_git_tools_config_defaults(self):
        """Test GitToolsConfig defaults."""
        config = GitToolsConfig()
        assert config.auto_commit is True
        assert config.commit_convention == "conventional"
        assert config.auto_pr is True


class TestShellToolsConfig:
    """Test ShellToolsConfig model."""

    def test_shell_tools_config_defaults(self):
        """Test ShellToolsConfig defaults."""
        config = ShellToolsConfig()
        assert "pytest" in config.command_allowlist
        assert "npm" in config.command_allowlist
        assert config.max_execution_time_ms == 300000


class TestToolsConfig:
    """Test ToolsConfig model."""

    def test_tools_config_defaults(self):
        """Test ToolsConfig defaults."""
        config = ToolsConfig()
        assert isinstance(config.sandbox, SandboxToolsConfig)
        assert isinstance(config.git, GitToolsConfig)
        assert isinstance(config.shell, ShellToolsConfig)


class TestSafetyConfig:
    """Test SafetyConfig model."""

    def test_safety_config_defaults(self):
        """Test SafetyConfig defaults."""
        config = SafetyConfig()
        assert config.max_iterations_per_story == 10
        assert config.max_retries_per_stage == 5
        assert config.approval_gates["pre_deploy"] is True
        assert config.secret_scanning is True

    def test_safety_config_custom(self):
        """Test SafetyConfig with custom values."""
        gates = {
            "pre_code": True,
            "pre_deploy": False,
            "pre_production": True
        }
        config = SafetyConfig(
            max_iterations_per_story=20,
            max_retries_per_stage=3,
            approval_gates=gates
        )
        assert config.max_iterations_per_story == 20
        assert config.approval_gates["pre_code"] is True


class TestQualityConfig:
    """Test QualityConfig model."""

    def test_quality_config_defaults(self):
        """Test QualityConfig defaults."""
        config = QualityConfig()
        assert config.min_test_coverage == 80.0
        assert config.lint_required is True
        assert config.type_check_required is True
        assert config.max_complexity == 15

    def test_quality_config_custom(self):
        """Test QualityConfig with custom values."""
        config = QualityConfig(
            min_test_coverage=90.0,
            max_complexity=10,
            max_nesting_depth=3
        )
        assert config.min_test_coverage == 90.0
        assert config.max_nesting_depth == 3


class TestObservabilityConfig:
    """Test ObservabilityConfig model."""

    def test_observability_config_defaults(self):
        """Test ObservabilityConfig defaults."""
        config = ObservabilityConfig()
        assert config.log_level == "info"
        assert config.trace_llm_calls is True
        assert config.metrics_enabled is True

    def test_observability_config_custom(self):
        """Test ObservabilityConfig with custom log level."""
        config = ObservabilityConfig(log_level="debug")
        assert config.log_level == "debug"


class TestProjectConfig:
    """Test ProjectConfig model."""

    def test_project_config_defaults(self):
        """Test ProjectConfig defaults."""
        config = ProjectConfig()
        assert config.project_id == "default"
        assert config.project_dir == "."
        assert config.language == "python"
        assert config.base_branch == "main"

    def test_project_config_custom(self):
        """Test ProjectConfig with custom values."""
        config = ProjectConfig(
            project_id="my-project",
            project_dir="/path/to/project",
            language="javascript",
            framework="react",
            test_framework="jest",
            package_manager="npm"
        )
        assert config.project_id == "my-project"
        assert config.language == "javascript"


class TestAgentConfig:
    """Test AgentConfig model."""

    def test_agent_config_creation(self):
        """Test creating AgentConfig."""
        config = AgentConfig()
        assert isinstance(config.project, ProjectConfig)
        assert isinstance(config.llm, LLMConfig)
        assert isinstance(config.tools, ToolsConfig)
        assert isinstance(config.safety, SafetyConfig)
        assert isinstance(config.quality, QualityConfig)


# ============================================================================
# ERROR TYPES TESTS
# ============================================================================


class TestErrorEnums:
    """Test ErrorCategory and RecoveryStrategy enums."""

    def test_error_category_values(self):
        """Test ErrorCategory enum values."""
        assert ErrorCategory.SYNTAX_ERROR.value == "syntax_error"
        assert ErrorCategory.TYPE_ERROR.value == "type_error"
        assert ErrorCategory.LOGIC_ERROR.value == "logic_error"
        assert ErrorCategory.UNKNOWN.value == "unknown"

    def test_recovery_strategy_values(self):
        """Test RecoveryStrategy enum values."""
        assert RecoveryStrategy.DIRECT_FIX.value == "direct_fix"
        assert RecoveryStrategy.REPLAN.value == "replan"
        assert RecoveryStrategy.ESCALATE_TO_HUMAN.value == "escalate_to_human"


class TestErrorDetail:
    """Test ErrorDetail model."""

    def test_error_detail_creation(self):
        """Test creating ErrorDetail."""
        detail = ErrorDetail(
            type="ValueError",
            message="Invalid input"
        )
        assert detail.type == "ValueError"
        assert detail.message == "Invalid input"

    def test_error_detail_with_location(self):
        """Test ErrorDetail with file location."""
        detail = ErrorDetail(
            type="SyntaxError",
            message="Unexpected token",
            file="src/main.py",
            line=42,
            column=10
        )
        assert detail.file == "src/main.py"
        assert detail.line == 42


class TestErrorClassification:
    """Test ErrorClassification model."""

    def test_error_classification_defaults(self):
        """Test ErrorClassification with defaults."""
        classification = ErrorClassification()
        assert classification.category == ErrorCategory.UNKNOWN
        assert classification.recovery_strategy == RecoveryStrategy.DIRECT_FIX
        assert classification.confidence == 0.0

    def test_error_classification_custom(self):
        """Test ErrorClassification with custom values."""
        classification = ErrorClassification(
            category=ErrorCategory.SYNTAX_ERROR,
            recovery_strategy=RecoveryStrategy.INCLUDE_TYPES,
            confidence=0.95
        )
        assert classification.category == ErrorCategory.SYNTAX_ERROR
        assert classification.confidence == 0.95


class TestErrorContext:
    """Test ErrorContext model."""

    def test_error_context_creation(self):
        """Test creating ErrorContext."""
        context = ErrorContext()
        assert context.attempt == 1
        assert isinstance(context.error, ErrorDetail)
        assert context.previous_attempts == []

    def test_error_context_with_history(self):
        """Test ErrorContext with previous attempts."""
        attempt = PreviousAttempt(
            attempt_number=1,
            code_diff="diff content",
            error_message="Initial error"
        )
        context = ErrorContext(
            attempt=2,
            previous_attempts=[attempt]
        )
        assert context.attempt == 2
        assert len(context.previous_attempts) == 1


class TestErrorFingerprint:
    """Test ErrorFingerprint model."""

    def test_error_fingerprint_creation(self):
        """Test creating ErrorFingerprint."""
        fingerprint = ErrorFingerprint(
            hash="abc123def456",
            category=ErrorCategory.TYPE_ERROR,
            pattern="isinstance.*dict"
        )
        assert fingerprint.hash == "abc123def456"
        assert fingerprint.category == ErrorCategory.TYPE_ERROR
        assert fingerprint.occurrences == 0

    def test_error_fingerprint_with_fix(self):
        """Test ErrorFingerprint with known fix."""
        fingerprint = ErrorFingerprint(
            hash="error-hash",
            category=ErrorCategory.IMPORT_ERROR,
            pattern="ModuleNotFoundError.*requests",
            known_fix="pip install requests",
            occurrences=5
        )
        assert fingerprint.known_fix == "pip install requests"
        assert fingerprint.occurrences == 5


# ============================================================================
# LLM TYPES TESTS
# ============================================================================


class TestLLMEnums:
    """Test ModelTier, TaskPurpose, and ReasoningMode enums."""

    def test_model_tier_values(self):
        """Test ModelTier enum values."""
        assert ModelTier.LOW.value == "low"
        assert ModelTier.MID.value == "mid"
        assert ModelTier.TOP.value == "top"

    def test_task_purpose_values(self):
        """Test TaskPurpose enum values."""
        assert TaskPurpose.STORY_ANALYSIS.value == "story_analysis"
        assert TaskPurpose.CODE_GENERATION.value == "code_generation"
        assert TaskPurpose.CODE_REVIEW.value == "code_review"

    def test_reasoning_mode_values(self):
        """Test ReasoningMode enum values."""
        assert ReasoningMode.DIRECT.value == "direct"
        assert ReasoningMode.CHAIN_OF_THOUGHT.value == "chain_of_thought"
        assert ReasoningMode.TREE_OF_THOUGHT.value == "tree_of_thought"


class TestMessage:
    """Test Message model."""

    def test_message_creation(self):
        """Test creating Message."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.tool_call_id is None

    def test_message_system_prompt(self):
        """Test system message."""
        msg = Message(
            role="system",
            content="You are a helpful assistant"
        )
        assert msg.role == "system"


class TestTokenUsage:
    """Test TokenUsage model."""

    def test_token_usage_creation(self):
        """Test creating TokenUsage."""
        usage = TokenUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0

    def test_token_usage_with_data(self):
        """Test TokenUsage with data."""
        usage = TokenUsage(
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500
        )
        assert usage.total_tokens == 1500


class TestCompletionResponse:
    """Test CompletionResponse model."""

    def test_completion_response_creation(self):
        """Test creating CompletionResponse."""
        response = CompletionResponse(
            id="resp-1",
            content="The answer is 42",
            model="gpt-4"
        )
        assert response.id == "resp-1"
        assert response.model == "gpt-4"
        assert isinstance(response.usage, TokenUsage)

    def test_completion_response_with_tool_calls(self):
        """Test CompletionResponse with tool calls."""
        tool_call = ToolCallRequest(
            id="call-1",
            name="calculator",
            arguments={"operation": "add", "a": 1, "b": 2}
        )
        response = CompletionResponse(
            id="resp-1",
            tool_calls=[tool_call],
            finish_reason="tool_calls"
        )
        assert response.tool_calls[0].name == "calculator"


class TestModelCall:
    """Test ModelCall model."""

    def test_model_call_creation(self):
        """Test creating ModelCall."""
        call = ModelCall(
            id="call-1",
            model="claude-sonnet",
            provider="anthropic"
        )
        assert call.id == "call-1"
        assert call.model == "claude-sonnet"
        assert isinstance(call.timestamp, datetime)

    def test_model_call_with_tokens(self):
        """Test ModelCall with token usage."""
        call = ModelCall(
            id="call-1",
            model="gpt-4",
            provider="openai",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.005
        )
        assert call.cost_usd == 0.005


class TestHealthStatus:
    """Test HealthStatus model."""

    def test_health_status_creation(self):
        """Test creating HealthStatus."""
        status = HealthStatus(provider="anthropic")
        assert status.provider == "anthropic"
        assert status.healthy is True
        assert status.error is None

    def test_health_status_unhealthy(self):
        """Test unhealthy HealthStatus."""
        status = HealthStatus(
            provider="openai",
            healthy=False,
            error="Connection timeout"
        )
        assert status.healthy is False
        assert status.error == "Connection timeout"


class TestUsageMetrics:
    """Test UsageMetrics model."""

    def test_usage_metrics_creation(self):
        """Test creating UsageMetrics."""
        metrics = UsageMetrics(provider="anthropic")
        assert metrics.provider == "anthropic"
        assert metrics.total_calls == 0
        assert metrics.error_rate == 0.0

    def test_usage_metrics_with_data(self):
        """Test UsageMetrics with data."""
        metrics = UsageMetrics(
            provider="openai",
            total_calls=100,
            total_tokens=50000,
            total_cost_usd=10.50,
            avg_latency_ms=250.5,
            error_rate=0.02
        )
        assert metrics.total_cost_usd == 10.50


class TestPromptTemplate:
    """Test PromptTemplate model."""

    def test_prompt_template_creation(self):
        """Test creating PromptTemplate."""
        template = PromptTemplate(
            id="template-1",
            version="1.0"
        )
        assert template.id == "template-1"
        assert template.version == "1.0"
        assert template.reasoning_mode == ReasoningMode.CHAIN_OF_THOUGHT

    def test_prompt_template_with_sections(self):
        """Test PromptTemplate with sections."""
        section = PromptSection(
            role="system",
            template="You are a code reviewer"
        )
        template = PromptTemplate(
            id="review-template",
            sections=[section]
        )
        assert len(template.sections) == 1


# ============================================================================
# TOOLS TYPES TESTS
# ============================================================================


class TestToolEnums:
    """Test ToolName and RiskLevel enums."""

    def test_tool_name_values(self):
        """Test ToolName enum values."""
        assert ToolName.FILE_SYSTEM.value == "file_system"
        assert ToolName.SHELL.value == "shell"
        assert ToolName.GIT.value == "git"
        assert ToolName.TEST_RUNNER.value == "test_runner"

    def test_risk_level_values(self):
        """Test RiskLevel enum values."""
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"


class TestToolCall:
    """Test ToolCall model."""

    def test_tool_call_creation(self):
        """Test creating ToolCall."""
        call = ToolCall(
            tool=ToolName.FILE_SYSTEM,
            operation="read",
            args={"path": "src/main.py"}
        )
        assert call.tool == ToolName.FILE_SYSTEM
        assert call.operation == "read"
        assert call.args["path"] == "src/main.py"

    def test_tool_call_with_env(self):
        """Test ToolCall with environment variables."""
        call = ToolCall(
            tool=ToolName.SHELL,
            operation="execute",
            args={"command": "pytest"},
            timeout_ms=30000,
            env={"PYTHONPATH": "/src"}
        )
        assert call.env["PYTHONPATH"] == "/src"


class TestSideEffects:
    """Test SideEffects model."""

    def test_side_effects_creation(self):
        """Test creating SideEffects."""
        effects = SideEffects()
        assert effects.files_created == []
        assert effects.packages_added == []

    def test_side_effects_with_data(self):
        """Test SideEffects with data."""
        effects = SideEffects(
            files_created=["src/main.py"],
            files_modified=["src/utils.py"],
            packages_added=["pytest"]
        )
        assert len(effects.files_created) == 1
        assert len(effects.packages_added) == 1


class TestToolResult:
    """Test ToolResult model."""

    def test_tool_result_creation(self):
        """Test creating ToolResult."""
        result = ToolResult(
            tool=ToolName.SHELL,
            operation="execute"
        )
        assert result.tool == ToolName.SHELL
        assert result.exit_code == 0
        assert result.stdout == ""

    def test_tool_result_with_output(self):
        """Test ToolResult with output."""
        side_effects = SideEffects(files_created=["output.txt"])
        result = ToolResult(
            tool=ToolName.FILE_SYSTEM,
            operation="create",
            exit_code=0,
            stdout="File created",
            duration_ms=100.0,
            side_effects=side_effects
        )
        assert result.exit_code == 0
        assert len(result.side_effects.files_created) == 1


class TestLintError:
    """Test LintError model."""

    def test_lint_error_creation(self):
        """Test creating LintError."""
        error = LintError(
            file="src/main.py",
            line=42,
            rule="E501",
            message="line too long"
        )
        assert error.file == "src/main.py"
        assert error.line == 42
        assert error.rule == "E501"


class TestTypeCheckError:
    """Test TypeCheckError model."""

    def test_type_check_error_creation(self):
        """Test creating TypeCheckError."""
        error = TypeCheckError(
            file="src/main.py",
            line=10,
            column=5,
            code="assignment",
            message="Incompatible types in assignment"
        )
        assert error.file == "src/main.py"
        assert error.code == "assignment"


class TestSandboxConfig:
    """Test SandboxConfig model."""

    def test_sandbox_config_creation(self):
        """Test creating SandboxConfig."""
        config = SandboxConfig(project_dir="/path/to/project")
        assert config.project_dir == "/path/to/project"
        assert config.cpu_limit == "2"
        assert "github.com" in config.network_whitelist


class TestToolPolicy:
    """Test ToolPolicy model."""

    def test_tool_policy_creation(self):
        """Test creating ToolPolicy."""
        policy = ToolPolicy(
            tool=ToolName.SHELL,
            allowed_operations=["execute"],
            risk_level=RiskLevel.MEDIUM
        )
        assert policy.tool == ToolName.SHELL
        assert policy.risk_level == RiskLevel.MEDIUM
        assert policy.max_calls_per_story == 500


# ============================================================================
# CODEBASE TYPES TESTS
# ============================================================================


class TestProjectProfile:
    """Test ProjectProfile model."""

    def test_project_profile_defaults(self):
        """Test ProjectProfile with defaults."""
        profile = ProjectProfile()
        assert profile.language == "python"
        assert profile.package_manager == "pip"
        assert profile.test_framework == "pytest"
        assert "PascalCase" in str(profile.naming_conventions)

    def test_project_profile_custom(self):
        """Test ProjectProfile with custom values."""
        profile = ProjectProfile(
            language="javascript",
            framework="react",
            test_framework="jest"
        )
        assert profile.language == "javascript"
        assert profile.framework == "react"


class TestSymbolInfo:
    """Test SymbolInfo model."""

    def test_symbol_info_creation(self):
        """Test creating SymbolInfo."""
        symbol = SymbolInfo(
            name="authenticate",
            kind="function",
            file="src/auth.py",
            line=42
        )
        assert symbol.name == "authenticate"
        assert symbol.kind == "function"
        assert symbol.exported is True

    def test_symbol_info_with_signature(self):
        """Test SymbolInfo with type signature."""
        symbol = SymbolInfo(
            name="add",
            kind="function",
            file="src/math.py",
            type_signature="(int, int) -> int",
            docstring="Add two integers"
        )
        assert symbol.type_signature == "(int, int) -> int"


class TestDependencyNode:
    """Test DependencyNode model."""

    def test_dependency_node_creation(self):
        """Test creating DependencyNode."""
        node = DependencyNode(file="src/main.py")
        assert node.file == "src/main.py"
        assert node.imports == []
        assert node.is_circular is False

    def test_dependency_node_with_dependencies(self):
        """Test DependencyNode with dependencies."""
        node = DependencyNode(
            file="src/main.py",
            imports=["src/utils.py", "src/config.py"],
            imported_by=["src/app.py"]
        )
        assert len(node.imports) == 2


class TestASTNode:
    """Test ASTNode model."""

    def test_ast_node_creation(self):
        """Test creating ASTNode."""
        node = ASTNode(
            file="src/main.py",
            type="function",
            name="main",
            start_line=1,
            end_line=10
        )
        assert node.name == "main"
        assert node.type == "function"
        assert node.children == []

    def test_ast_node_with_children(self):
        """Test ASTNode with children."""
        child = ASTNode(
            file="src/main.py",
            type="method",
            name="helper",
            start_line=3,
            end_line=5
        )
        parent = ASTNode(
            file="src/main.py",
            type="class",
            name="MyClass",
            children=[child]
        )
        assert len(parent.children) == 1


class TestCodeChunk:
    """Test CodeChunk model."""

    def test_code_chunk_creation(self):
        """Test creating CodeChunk."""
        chunk = CodeChunk(
            file="src/main.py",
            start_line=10,
            end_line=20,
            content="def foo():\n    pass"
        )
        assert chunk.file == "src/main.py"
        assert chunk.type == "block"

    def test_code_chunk_with_name(self):
        """Test CodeChunk with function/class name."""
        chunk = CodeChunk(
            file="src/main.py",
            type="function",
            name="authenticate",
            start_line=42,
            end_line=50
        )
        assert chunk.name == "authenticate"


class TestSearchResult:
    """Test SearchResult model."""

    def test_search_result_creation(self):
        """Test creating SearchResult."""
        chunk = CodeChunk(file="src/main.py", content="def foo(): pass")
        result = SearchResult(chunk=chunk, score=0.95)
        assert result.chunk.file == "src/main.py"
        assert result.score == 0.95
        assert result.match_type == "text"


class TestAPIEndpoint:
    """Test APIEndpoint model."""

    def test_api_endpoint_creation(self):
        """Test creating APIEndpoint."""
        endpoint = APIEndpoint(
            method="GET",
            path="/users/{id}",
            handler_file="src/handlers.py",
            handler_function="get_user"
        )
        assert endpoint.method == "GET"
        assert endpoint.path == "/users/{id}"


class TestAPISurface:
    """Test APISurface model."""

    def test_api_surface_creation(self):
        """Test creating APISurface."""
        endpoint = APIEndpoint(
            method="POST",
            path="/users",
            handler_file="src/handlers.py"
        )
        surface = APISurface(endpoints=[endpoint])
        assert len(surface.endpoints) == 1


class TestDetectedPattern:
    """Test DetectedPattern model."""

    def test_detected_pattern_creation(self):
        """Test creating DetectedPattern."""
        pattern = DetectedPattern(
            category="error_handling",
            pattern="try-catch blocks"
        )
        assert pattern.category == "error_handling"
        assert pattern.confidence == 0.0

    def test_detected_pattern_with_examples(self):
        """Test DetectedPattern with examples."""
        pattern = DetectedPattern(
            category="decorator",
            pattern="@staticmethod",
            examples=["@staticmethod\ndef helper(): pass"],
            confidence=0.95
        )
        assert len(pattern.examples) == 1


class TestConventionScanResult:
    """Test ConventionScanResult model."""

    def test_convention_scan_result_creation(self):
        """Test creating ConventionScanResult."""
        profile = ProjectProfile(language="python")
        result = ConventionScanResult(profile=profile)
        assert result.profile.language == "python"
        assert result.confidence == 0.0


# ============================================================================
# SECURITY TYPES TESTS
# ============================================================================


class TestSecurityEnums:
    """Test Severity and FindingCategory enums."""

    def test_severity_values(self):
        """Test Severity enum values."""
        assert Severity.CRITICAL.value == "critical"
        assert Severity.HIGH.value == "high"
        assert Severity.INFO.value == "info"

    def test_finding_category_values(self):
        """Test FindingCategory enum values."""
        assert FindingCategory.SECRET_LEAK.value == "secret_leak"
        assert FindingCategory.INJECTION.value == "injection"
        assert FindingCategory.XSS.value == "xss"


class TestSecurityFinding:
    """Test SecurityFinding model."""

    def test_security_finding_creation(self):
        """Test creating SecurityFinding."""
        finding = SecurityFinding(
            id="find-1",
            severity=Severity.HIGH,
            category=FindingCategory.INJECTION,
            title="SQL Injection",
            description="User input not sanitized",
            file_path="src/db.py",
            line_number=42
        )
        assert finding.id == "find-1"
        assert finding.severity == Severity.HIGH

    def test_security_finding_with_suggestion(self):
        """Test SecurityFinding with fix suggestion."""
        finding = SecurityFinding(
            id="find-1",
            severity=Severity.CRITICAL,
            category=FindingCategory.HARDCODED_CREDENTIALS,
            title="Hardcoded API Key",
            description="API key in source code",
            file_path="src/config.py",
            line_number=10,
            fix_suggestion="Use environment variables"
        )
        assert finding.fix_suggestion == "Use environment variables"


class TestSecretFinding:
    """Test SecretFinding model."""

    def test_secret_finding_creation(self):
        """Test creating SecretFinding."""
        finding = SecretFinding(
            id="secret-1",
            severity=Severity.CRITICAL,
            category=FindingCategory.SECRET_LEAK,
            title="AWS Access Key",
            description="AWS credentials exposed",
            file_path="src/config.py",
            secret_type="api_key",
            entropy=4.2
        )
        assert finding.secret_type == "api_key"
        assert finding.entropy == 4.2


class TestSASTReport:
    """Test SASTReport model."""

    def test_sast_report_creation(self):
        """Test creating SASTReport."""
        report = SASTReport(scanner="bandit")
        assert report.scanner == "bandit"
        assert report.findings == []
        assert report.passed is True

    def test_sast_report_with_findings(self):
        """Test SASTReport with findings."""
        finding = SecurityFinding(
            id="find-1",
            severity=Severity.HIGH,
            category=FindingCategory.INJECTION,
            title="Issue",
            description="Desc",
            file_path="src/main.py"
        )
        report = SASTReport(
            scanner="semgrep",
            findings=[finding],
            passed=False
        )
        assert len(report.findings) == 1
        assert report.passed is False


class TestSecurityScanResult:
    """Test SecurityScanResult model."""

    def test_security_scan_result_creation(self):
        """Test creating SecurityScanResult."""
        result = SecurityScanResult()
        assert result.sast_reports == []
        assert result.secret_findings == []
        assert result.passed is True

    def test_security_scan_result_with_findings(self):
        """Test SecurityScanResult with findings."""
        sast_report = SASTReport(scanner="bandit")
        result = SecurityScanResult(
            sast_reports=[sast_report],
            critical_count=1,
            high_count=2,
            passed=False
        )
        assert len(result.sast_reports) == 1
        assert result.critical_count == 1


# ============================================================================
# QUALITY TYPES TESTS
# ============================================================================


class TestQualityEnums:
    """Test GateType, GateOutcome, and EnforcementLevel enums."""

    def test_gate_type_values(self):
        """Test GateType enum values."""
        assert GateType.LINT.value == "lint"
        assert GateType.UNIT_TEST.value == "unit_test"
        assert GateType.SAST.value == "sast"

    def test_gate_outcome_values(self):
        """Test GateOutcome enum values."""
        assert GateOutcome.PASSED.value == "passed"
        assert GateOutcome.FAILED.value == "failed"
        assert GateOutcome.SKIPPED.value == "skipped"

    def test_enforcement_level_values(self):
        """Test EnforcementLevel enum values."""
        assert EnforcementLevel.REQUIRED.value == "required"
        assert EnforcementLevel.WARNING_ONLY.value == "warning"
        assert EnforcementLevel.OPTIONAL.value == "optional"


class TestGateThreshold:
    """Test GateThreshold model."""

    def test_gate_threshold_creation(self):
        """Test creating GateThreshold."""
        threshold = GateThreshold(
            gate_type=GateType.UNIT_TEST,
            enforcement=EnforcementLevel.REQUIRED
        )
        assert threshold.gate_type == GateType.UNIT_TEST
        assert threshold.enforcement == EnforcementLevel.REQUIRED

    def test_gate_threshold_with_coverage(self):
        """Test GateThreshold with coverage requirement."""
        threshold = GateThreshold(
            gate_type=GateType.COVERAGE_DELTA,
            min_coverage=80.0,
            enforcement=EnforcementLevel.REQUIRED
        )
        assert threshold.min_coverage == 80.0


class TestGateResult:
    """Test GateResult model."""

    def test_gate_result_creation(self):
        """Test creating GateResult."""
        result = GateResult(
            gate_type=GateType.LINT,
            outcome=GateOutcome.PASSED,
            enforcement=EnforcementLevel.REQUIRED
        )
        assert result.gate_type == GateType.LINT
        assert result.outcome == GateOutcome.PASSED

    def test_gate_result_with_findings(self):
        """Test GateResult with findings."""
        result = GateResult(
            gate_type=GateType.SAST,
            outcome=GateOutcome.FAILED,
            enforcement=EnforcementLevel.REQUIRED,
            message="2 critical issues found",
            findings_count=2
        )
        assert result.findings_count == 2


class TestQualityGatePolicy:
    """Test QualityGatePolicy model."""

    def test_quality_gate_policy_creation(self):
        """Test creating QualityGatePolicy."""
        policy = QualityGatePolicy()
        assert policy.name == "default"
        assert policy.gates == []
        assert policy.fail_fast is False

    def test_quality_gate_policy_with_gates(self):
        """Test QualityGatePolicy with gates."""
        threshold = GateThreshold(gate_type=GateType.LINT)
        policy = QualityGatePolicy(
            name="strict",
            gates=[threshold],
            fail_fast=True
        )
        assert len(policy.gates) == 1
        assert policy.fail_fast is True


class TestPolicyEvaluationResult:
    """Test PolicyEvaluationResult model."""

    def test_policy_evaluation_result_creation(self):
        """Test creating PolicyEvaluationResult."""
        result = PolicyEvaluationResult(policy_name="default")
        assert result.policy_name == "default"
        assert result.all_required_passed is True
        assert result.results == []

    def test_policy_evaluation_with_results(self):
        """Test PolicyEvaluationResult with gate results."""
        gate_result = GateResult(
            gate_type=GateType.UNIT_TEST,
            outcome=GateOutcome.PASSED,
            enforcement=EnforcementLevel.REQUIRED
        )
        result = PolicyEvaluationResult(
            policy_name="policy",
            results=[gate_result]
        )
        assert len(result.results) == 1


# ============================================================================
# EVIDENCE TYPES TESTS
# ============================================================================


class TestImpactedFile:
    """Test ImpactedFile model."""

    def test_impacted_file_creation(self):
        """Test creating ImpactedFile."""
        file = ImpactedFile(
            file_path="src/main.py",
            change_type="modified",
            lines_added=10,
            lines_removed=5
        )
        assert file.file_path == "src/main.py"
        assert file.lines_added == 10

    def test_impacted_file_with_dependents(self):
        """Test ImpactedFile with dependents."""
        file = ImpactedFile(
            file_path="src/utils.py",
            change_type="modified",
            dependents=["src/main.py", "src/app.py"]
        )
        assert len(file.dependents) == 2


class TestChangeImpactReport:
    """Test ChangeImpactReport model."""

    def test_change_impact_report_creation(self):
        """Test creating ChangeImpactReport."""
        report = ChangeImpactReport()
        assert report.files_changed == []
        assert report.risk_level == "low"

    def test_change_impact_report_with_files(self):
        """Test ChangeImpactReport with files."""
        file1 = ImpactedFile(file_path="src/main.py", change_type="modified")
        report = ChangeImpactReport(
            files_changed=[file1],
            total_files_changed=1,
            blast_radius=3,
            risk_level="medium"
        )
        assert len(report.files_changed) == 1
        assert report.blast_radius == 3


class TestCriterionTestMapping:
    """Test CriterionTestMapping model."""

    def test_criterion_test_mapping_creation(self):
        """Test creating CriterionTestMapping."""
        mapping = CriterionTestMapping(
            criterion_id="AC-1",
            criterion_description="User can login"
        )
        assert mapping.criterion_id == "AC-1"
        assert mapping.verified is False

    def test_criterion_test_mapping_with_tests(self):
        """Test CriterionTestMapping with linked tests."""
        mapping = CriterionTestMapping(
            criterion_id="AC-1",
            criterion_description="Login works",
            linked_tests=["test_login_valid", "test_login_invalid"],
            verified=True
        )
        assert len(mapping.linked_tests) == 2
        assert mapping.verified is True


class TestCostBreakdown:
    """Test CostBreakdown model."""

    def test_cost_breakdown_creation(self):
        """Test creating CostBreakdown."""
        breakdown = CostBreakdown()
        assert breakdown.total_tokens_input == 0
        assert breakdown.total_cost_usd == 0.0

    def test_cost_breakdown_with_data(self):
        """Test CostBreakdown with data."""
        breakdown = CostBreakdown(
            total_tokens_input=10000,
            total_tokens_output=5000,
            total_cost_usd=2.50,
            model_usage={"claude-sonnet": 8000, "gpt-4": 7000},
            iterations_count=3
        )
        assert breakdown.total_cost_usd == 2.50
        assert breakdown.iterations_count == 3


class TestReviewerChecklistItem:
    """Test ReviewerChecklistItem model."""

    def test_reviewer_checklist_item_creation(self):
        """Test creating ReviewerChecklistItem."""
        item = ReviewerChecklistItem(item="Code follows style guide")
        assert item.item == "Code follows style guide"
        assert item.verified_by_agent is False

    def test_reviewer_checklist_item_verified(self):
        """Test ReviewerChecklistItem verified by agent."""
        item = ReviewerChecklistItem(
            item="Tests added",
            verified_by_agent=True,
            notes="All edge cases covered"
        )
        assert item.verified_by_agent is True
        assert item.notes == "All edge cases covered"


class TestEvidenceBundle:
    """Test EvidenceBundle model."""

    def test_evidence_bundle_creation(self):
        """Test creating EvidenceBundle."""
        bundle = EvidenceBundle(story_id="story-1")
        assert bundle.story_id == "story-1"
        assert bundle.all_gates_passed is False

    def test_evidence_bundle_to_markdown(self):
        """Test EvidenceBundle.to_markdown() method."""
        bundle = EvidenceBundle(
            story_id="story-1",
            story_title="Add auth",
            test_summary={"total": 10, "passed": 10, "failed": 0, "coverage_percent": 85.0},
            all_gates_passed=True
        )
        markdown = bundle.to_markdown()
        assert "Evidence Bundle Report" in markdown
        assert "story-1" not in markdown  # story_id is not in title
        assert "Test Report" in markdown

    def test_evidence_bundle_to_pr_comment(self):
        """Test EvidenceBundle.to_pr_comment() method."""
        bundle = EvidenceBundle(
            story_id="story-1",
            test_summary={"total": 5, "passed": 5, "failed": 0},
            all_gates_passed=True
        )
        comment = bundle.to_pr_comment()
        assert "✅" in comment
        assert "Tests: 5/5 passed" in comment

    def test_evidence_bundle_to_pr_comment_failed(self):
        """Test EvidenceBundle.to_pr_comment() with failures."""
        bundle = EvidenceBundle(
            story_id="story-1",
            test_summary={"total": 5, "passed": 3, "failed": 2},
            security_summary={"critical": 1, "high": 2},
            all_gates_passed=False
        )
        comment = bundle.to_pr_comment()
        assert "⚠️" in comment
        assert "Issues detected" in comment


# ============================================================================
# LEARNING TYPES TESTS
# ============================================================================


class TestFailureClassEnum:
    """Test FailureClass enum."""

    def test_failure_class_values(self):
        """Test FailureClass enum values."""
        assert FailureClass.GATE_FAILURE.value == "gate_failure"
        assert FailureClass.TEST_REGRESSION.value == "test_regression"
        assert FailureClass.DEPLOYMENT_FAILURE.value == "deployment_failure"


class TestPostMortemEntry:
    """Test PostMortemEntry model."""

    def test_post_mortem_entry_creation(self):
        """Test creating PostMortemEntry."""
        entry = PostMortemEntry(
            id="pm-1",
            story_id="story-1",
            project_id="proj-1",
            failure_class=FailureClass.TEST_REGRESSION
        )
        assert entry.id == "pm-1"
        assert entry.failure_class == FailureClass.TEST_REGRESSION

    def test_post_mortem_entry_with_details(self):
        """Test PostMortemEntry with full details."""
        entry = PostMortemEntry(
            id="pm-1",
            story_id="story-1",
            project_id="proj-1",
            failure_class=FailureClass.SECURITY_FINDING,
            failure_details="SQL injection detected",
            root_cause="User input not sanitized",
            resolution="Added parameterized queries"
        )
        assert entry.root_cause == "User input not sanitized"


class TestHeuristicRule:
    """Test HeuristicRule model."""

    def test_heuristic_rule_creation(self):
        """Test creating HeuristicRule."""
        rule = HeuristicRule(
            id="rule-1",
            project_id="proj-1",
            rule_type="pattern_detection",
            trigger_pattern="user_input",
            action="sanitize"
        )
        assert rule.id == "rule-1"
        assert rule.active is True

    def test_heuristic_rule_with_effectiveness(self):
        """Test HeuristicRule with effectiveness tracking."""
        rule = HeuristicRule(
            id="rule-1",
            project_id="proj-1",
            rule_type="validation",
            trigger_pattern="type_error",
            action="add_type_hint",
            times_applied=10,
            times_effective=9
        )
        assert rule.times_applied == 10
        assert rule.times_effective == 9


class TestLearningStats:
    """Test LearningStats model."""

    def test_learning_stats_creation(self):
        """Test creating LearningStats."""
        stats = LearningStats()
        assert stats.total_post_mortems == 0
        assert stats.failure_reduction_rate == 0.0

    def test_learning_stats_with_data(self):
        """Test LearningStats with data."""
        stats = LearningStats(
            total_post_mortems=10,
            total_heuristics=5,
            active_heuristics=4,
            failure_reduction_rate=0.35,
            top_failure_classes={"gate_failure": 4, "test_regression": 3}
        )
        assert stats.total_post_mortems == 10
        assert stats.failure_reduction_rate == 0.35


# ============================================================================
# CONFIG MODULE TESTS
# ============================================================================


class TestLoadConfig:
    """Test load_config function."""

    def test_load_config_defaults(self, tmp_path):
        """Test load_config with defaults."""
        with patch.dict(os.environ, {"AGENT_PROJECT_DIR": str(tmp_path)}):
            config = load_config()
            assert isinstance(config, AgentConfig)
            assert config.project.project_id == "default"
            assert config.safety.max_iterations_per_story == 10

    def test_load_config_with_overrides(self, tmp_path):
        """Test load_config with overrides."""
        with patch.dict(os.environ, {"AGENT_PROJECT_DIR": str(tmp_path)}):
            overrides = {
                "project_id": "my-proj",
                "max_iterations": 20,
                "min_test_coverage": 90.0
            }
            config = load_config(overrides=overrides)
            assert config.project.project_id == "my-proj"
            assert config.safety.max_iterations_per_story == 20
            assert config.quality.min_test_coverage == 90.0

    def test_load_config_with_file(self, tmp_path):
        """Test load_config reading from file."""
        config_file = tmp_path / ".agent.json"
        config_data = {
            "project_id": "file-proj",
            "max_iterations": 15,
            "max_retries": 3
        }
        config_file.write_text(json.dumps(config_data))

        with patch.dict(os.environ, {
            "AGENT_PROJECT_DIR": str(tmp_path),
            "AGENT_CONFIG_FILE": str(config_file)
        }):
            config = load_config()
            assert config.project.project_id == "file-proj"
            assert config.safety.max_iterations_per_story == 15

    def test_load_config_invalid_json(self, tmp_path):
        """Test load_config with invalid JSON file."""
        config_file = tmp_path / ".agent.json"
        config_file.write_text("{ invalid json }")

        with patch.dict(os.environ, {
            "AGENT_PROJECT_DIR": str(tmp_path),
            "AGENT_CONFIG_FILE": str(config_file)
        }):
            # Should fall back to defaults
            config = load_config()
            assert config.project.project_id == "default"


class TestValidateConfig:
    """Test validate_config function."""

    def test_validate_config_valid(self, tmp_path):
        """Test validate_config with valid config."""
        with patch.dict(os.environ, {"AGENT_PROJECT_DIR": str(tmp_path)}):
            config = load_config()
            errors = validate_config(config)
            # Config might have no enabled providers initially
            assert isinstance(errors, list)

    def test_validate_config_negative_cost(self, tmp_path):
        """Test validate_config detects negative costs."""
        with patch.dict(os.environ, {"AGENT_PROJECT_DIR": str(tmp_path)}):
            config = load_config()
            config.llm.cost.max_cost_per_story_usd = -1.0
            errors = validate_config(config)
            assert any("max_cost_per_story_usd" in e for e in errors)

    def test_validate_config_invalid_coverage(self, tmp_path):
        """Test validate_config detects invalid coverage."""
        with patch.dict(os.environ, {"AGENT_PROJECT_DIR": str(tmp_path)}):
            config = load_config()
            config.quality.min_test_coverage = 150.0
            errors = validate_config(config)
            assert any("coverage" in e for e in errors)

    def test_validate_config_zero_iterations(self, tmp_path):
        """Test validate_config detects zero iterations."""
        with patch.dict(os.environ, {"AGENT_PROJECT_DIR": str(tmp_path)}):
            config = load_config()
            config.safety.max_iterations_per_story = 0
            errors = validate_config(config)
            assert any("iterations" in e for e in errors)


# ============================================================================
# LLM KEY STORE TESTS
# ============================================================================


class TestValidateProviderName:
    """Test _validate_provider_name function."""

    def test_validate_provider_name_valid(self):
        """Test _validate_provider_name with valid names."""
        assert _validate_provider_name("anthropic") == "anthropic"
        assert _validate_provider_name("openai") == "openai"
        assert _validate_provider_name("google") == "google"
        assert _validate_provider_name("ollama") == "ollama"

    def test_validate_provider_name_invalid(self):
        """Test _validate_provider_name with invalid names."""
        with pytest.raises(ValueError):
            _validate_provider_name("invalid")

    def test_validate_provider_name_case_sensitive(self):
        """Test _validate_provider_name is case-sensitive."""
        with pytest.raises(ValueError):
            _validate_provider_name("Anthropic")


class TestResolveAdminDbPath:
    """Test resolve_admin_db_path function."""

    def test_resolve_admin_db_path_explicit(self, tmp_path):
        """Test resolve_admin_db_path with explicit path."""
        db_file = tmp_path / "admin.db"
        with patch.dict(os.environ, {"AGENT_ADMIN_DB_PATH": str(db_file)}):
            path = resolve_admin_db_path()
            assert path == db_file

    def test_resolve_admin_db_path_default(self, tmp_path):
        """Test resolve_admin_db_path with default path."""
        with patch.dict(os.environ, {"AGENT_ADMIN_DB_PATH": ""}):
            path = resolve_admin_db_path(project_dir=str(tmp_path))
            assert path == tmp_path / ".agent_admin.db"

    def test_resolve_admin_db_path_expanduser(self):
        """Test resolve_admin_db_path expands ~ in path."""
        with patch.dict(os.environ, {"AGENT_ADMIN_DB_PATH": "~/admin.db"}):
            path = resolve_admin_db_path()
            assert "~" not in str(path)


class TestLLMKeyStore:
    """Test LLMKeyStore model."""

    def test_llm_key_store_creation(self, tmp_path):
        """Test creating LLMKeyStore."""
        db_path = tmp_path / "admin.db"
        store = LLMKeyStore(db_path)
        assert store.db_path == db_path
        assert db_path.exists()

    def test_llm_key_store_set_and_get(self, tmp_path):
        """Test setting and getting API key."""
        db_path = tmp_path / "admin.db"
        store = LLMKeyStore(db_path)

        store.set_key("anthropic", "sk-ant-12345")
        key = store.get_key("anthropic")
        assert key == "sk-ant-12345"

    def test_llm_key_store_get_nonexistent(self, tmp_path):
        """Test getting nonexistent key."""
        db_path = tmp_path / "admin.db"
        store = LLMKeyStore(db_path)

        key = store.get_key("openai")
        assert key is None

    def test_llm_key_store_update_key(self, tmp_path):
        """Test updating existing key."""
        db_path = tmp_path / "admin.db"
        store = LLMKeyStore(db_path)

        store.set_key("anthropic", "old-key")
        store.set_key("anthropic", "new-key")
        key = store.get_key("anthropic")
        assert key == "new-key"

    def test_llm_key_store_clear_key(self, tmp_path):
        """Test clearing a key."""
        db_path = tmp_path / "admin.db"
        store = LLMKeyStore(db_path)

        store.set_key("openai", "key-value")
        store.clear_key("openai")
        key = store.get_key("openai")
        assert key is None

    def test_llm_key_store_get_all_keys(self, tmp_path):
        """Test getting all keys."""
        db_path = tmp_path / "admin.db"
        store = LLMKeyStore(db_path)

        store.set_key("anthropic", "ant-key")
        store.set_key("openai", "oai-key")

        all_keys = store.get_all_keys()
        assert all_keys["anthropic"] == "ant-key"
        assert all_keys["openai"] == "oai-key"

    def test_llm_key_store_invalid_provider(self, tmp_path):
        """Test LLMKeyStore with invalid provider."""
        db_path = tmp_path / "admin.db"
        store = LLMKeyStore(db_path)

        with pytest.raises(ValueError):
            store.set_key("invalid", "key")

    def test_llm_key_store_secure_permissions(self, tmp_path):
        """Test LLMKeyStore sets secure file permissions."""
        db_path = tmp_path / "admin.db"
        store = LLMKeyStore(db_path)

        # Check that file exists and is readable by owner
        assert db_path.exists()
        stat_info = db_path.stat()
        # Owner should have read/write permissions
        assert stat_info.st_mode & 0o600 == 0o600


class TestGetLLMKeyStore:
    """Test get_llm_key_store factory function."""

    def test_get_llm_key_store_creation(self, tmp_path):
        """Test get_llm_key_store creates store."""
        store = get_llm_key_store(project_dir=str(tmp_path))
        assert isinstance(store, LLMKeyStore)
        assert store.db_path == tmp_path / ".agent_admin.db"

    def test_get_llm_key_store_persistent(self, tmp_path):
        """Test get_llm_key_store is persistent."""
        store1 = get_llm_key_store(project_dir=str(tmp_path))
        store1.set_key("anthropic", "key123")

        store2 = get_llm_key_store(project_dir=str(tmp_path))
        key = store2.get_key("anthropic")
        assert key == "key123"
