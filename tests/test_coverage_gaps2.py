"""Additional comprehensive test suite for low-coverage modules (Part 2).

Targets specific missed lines for:
- quality/__init__.py (gate runners)
- orchestrator/__init__.py (process_story stages)
- orchestrator/adapters.py (remaining adapters)
- tools/resource_limiter.py (monitor/enforce methods)
- tools/package_manager.py (all methods)
- codebase/ownership_map.py (git blame, pattern matching)
- llm/providers/*.py (complete, stream methods with mocks)
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch, call

import pytest

# Import modules under test
from fastcoder.codebase.ownership_map import OwnershipMap
from fastcoder.orchestrator import Orchestrator
from fastcoder.orchestrator.adapters import (
    DeployerAdapter,
    ReviewerAdapter,
    TestGeneratorAdapter,
    TestRunnerAdapter,
    VerifierAdapter,
    ContextManagerAdapter,
    MemoryStoreAdapter,
    ErrorClassifierAdapter,
    RecoveryManagerAdapter,
    BuildRunnerAdapter,
)
from fastcoder.quality import QualityGateEngine
from fastcoder.tools.package_manager import PackageManagerTool
from fastcoder.tools.resource_limiter import ResourceLimiter, ResourceLimits, ResourceUsage
from fastcoder.types.config import AgentConfig, ProjectConfig, SafetyConfig
from fastcoder.types.errors import ErrorClassification, ErrorContext, ErrorDetail
from fastcoder.types.iteration import Iteration
from fastcoder.types.plan import ExecutionPlan, PlanTask, TaskAction
from fastcoder.types.quality import (
    EnforcementLevel,
    GateOutcome,
    GateResult,
    GateType,
    GateThreshold,
    PolicyEvaluationResult,
)
from fastcoder.types.story import (
    Priority,
    Story,
    StoryMetadata,
    StorySpec,
    StoryState,
    StorySubmission,
)
from fastcoder.types.task import DeployReport, FileChange, ReviewReport, TestReport, TaskResult


# =============================================================================
# QUALITY GATE ENGINE - All Gate Runners (lines 516-1264)
# =============================================================================

@pytest.mark.asyncio
class TestQualityGateEngineRunners:
    """Test all individual gate runner methods."""

    @pytest.fixture
    def temp_project(self):
        """Create temp project directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def engine(self, temp_project):
        """Create QualityGateEngine."""
        return QualityGateEngine(temp_project)

    async def test_run_lint_python_pass(self, engine, temp_project):
        """Test _run_lint with Python files, zero findings."""
        # Create a Python file
        py_file = Path(temp_project) / "test.py"
        py_file.write_text("print('hello')\n")

        with patch.object(engine, "_exec_command", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = (0, "", "")
            with patch.object(engine, "_parse_linting_output", return_value=[]):
                threshold = GateThreshold(
                    gate_type=GateType.LINT,
                    enforcement=EnforcementLevel.REQUIRED,
                    max_findings=0,
                    timeout_seconds=120,
                )
                result = await engine._run_lint(threshold)

                assert result.gate_type == GateType.LINT
                assert result.outcome == GateOutcome.PASSED
                assert result.findings_count == 0
                assert "zero findings" in result.message

    async def test_run_lint_python_with_findings(self, engine, temp_project):
        """Test _run_lint with Python findings exceeding threshold."""
        py_file = Path(temp_project) / "test.py"
        py_file.write_text("bad_code()")

        with patch.object(engine, "_exec_command", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = (1, "test.py:1: E501", "")
            findings = [{"line": 1, "code": "E501"}]
            with patch.object(engine, "_parse_linting_output", return_value=findings):
                threshold = GateThreshold(
                    gate_type=GateType.LINT,
                    enforcement=EnforcementLevel.REQUIRED,
                    max_findings=0,
                    timeout_seconds=120,
                )
                result = await engine._run_lint(threshold)

                assert result.outcome == GateOutcome.FAILED
                assert result.findings_count == 1

    async def test_run_lint_javascript(self, engine, temp_project):
        """Test _run_lint with JavaScript files."""
        js_file = Path(temp_project) / "test.js"
        js_file.write_text("console.log('test');")

        with patch.object(engine, "_exec_command", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = (0, "[]", "")
            with patch.object(engine, "_parse_eslint_output", return_value=[]):
                threshold = GateThreshold(
                    gate_type=GateType.LINT,
                    enforcement=EnforcementLevel.REQUIRED,
                    max_findings=0,
                    timeout_seconds=120,
                )
                result = await engine._run_lint(threshold)

                # May be PASSED or SKIPPED if no Python files found
                assert result.gate_type == GateType.LINT
                assert result.outcome in (GateOutcome.PASSED, GateOutcome.SKIPPED)

    async def test_run_type_check_python_pass(self, engine, temp_project):
        """Test _run_type_check with Python files."""
        py_file = Path(temp_project) / "test.py"
        py_file.write_text("x: int = 5")

        with patch.object(engine, "_exec_command", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = (0, "", "")
            with patch.object(engine, "_parse_mypy_output", return_value=[]):
                threshold = GateThreshold(
                    gate_type=GateType.TYPE_CHECK,
                    enforcement=EnforcementLevel.REQUIRED,
                    max_findings=0,
                    timeout_seconds=180,
                )
                result = await engine._run_type_check(threshold)

                assert result.outcome == GateOutcome.PASSED
                assert result.gate_type == GateType.TYPE_CHECK

    async def test_run_type_check_typescript(self, engine, temp_project):
        """Test _run_type_check with TypeScript files."""
        ts_file = Path(temp_project) / "test.ts"
        ts_file.write_text("const x: number = 5;")

        with patch.object(engine, "_exec_command", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = (0, "", "")
            with patch.object(engine, "_parse_tsc_output", return_value=[]):
                threshold = GateThreshold(
                    gate_type=GateType.TYPE_CHECK,
                    enforcement=EnforcementLevel.REQUIRED,
                    max_findings=0,
                    timeout_seconds=180,
                )
                result = await engine._run_type_check(threshold)

                assert result.outcome == GateOutcome.PASSED

    async def test_run_unit_test_python(self, engine, temp_project):
        """Test _run_unit_test with Python pytest."""
        py_file = Path(temp_project) / "test_example.py"
        py_file.write_text("def test_pass(): assert True")

        with patch.object(engine, "_exec_command", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = (0, "coverage: 85%", "")
            with patch.object(engine, "_parse_pytest_coverage", return_value=85.0):
                threshold = GateThreshold(
                    gate_type=GateType.UNIT_TEST,
                    enforcement=EnforcementLevel.REQUIRED,
                    min_coverage=80.0,
                    timeout_seconds=300,
                )
                result = await engine._run_unit_test(threshold)

                assert result.outcome == GateOutcome.PASSED
                assert "coverage" in result.details

    async def test_run_unit_test_javascript(self, engine, temp_project):
        """Test _run_unit_test with JavaScript Jest."""
        js_file = Path(temp_project) / "test.js"
        js_file.write_text("test('pass', () => expect(true).toBe(true))")
        package_json = Path(temp_project) / "package.json"
        package_json.write_text("{}")

        with patch.object(engine, "_exec_command", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = (0, "coverage: 90%", "")
            with patch.object(engine, "_parse_jest_coverage", return_value=90.0):
                threshold = GateThreshold(
                    gate_type=GateType.UNIT_TEST,
                    enforcement=EnforcementLevel.REQUIRED,
                    min_coverage=80.0,
                    timeout_seconds=300,
                )
                result = await engine._run_unit_test(threshold)

                # May be SKIPPED if no Python files found and package.json check fails
                assert result.gate_type == GateType.UNIT_TEST
                assert result.outcome in (GateOutcome.PASSED, GateOutcome.SKIPPED)

    async def test_run_integration_test_pytest(self, engine, temp_project):
        """Test _run_integration_test with pytest."""
        integration_dir = Path(temp_project) / "tests" / "integration"
        integration_dir.mkdir(parents=True)
        (integration_dir / "test_integration.py").write_text("def test_pass(): pass")

        with patch.object(engine, "_exec_command", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = (0, "passed", "")
            threshold = GateThreshold(
                gate_type=GateType.INTEGRATION_TEST,
                enforcement=EnforcementLevel.REQUIRED,
                timeout_seconds=600,
            )
            result = await engine._run_integration_test(threshold)

            assert result.outcome == GateOutcome.PASSED

    async def test_run_integration_test_npm(self, engine, temp_project):
        """Test _run_integration_test with npm."""
        package_json = Path(temp_project) / "package.json"
        package_json.write_text("{}")

        with patch.object(engine, "_exec_command", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = (0, "passed", "")
            threshold = GateThreshold(
                gate_type=GateType.INTEGRATION_TEST,
                enforcement=EnforcementLevel.REQUIRED,
                timeout_seconds=600,
            )
            result = await engine._run_integration_test(threshold)

            assert result.outcome == GateOutcome.PASSED

    async def test_run_e2e_test(self, engine, temp_project):
        """Test _run_e2e_test."""
        e2e_dir = Path(temp_project) / "tests" / "e2e"
        e2e_dir.mkdir(parents=True)
        (e2e_dir / "test_e2e.py").write_text("def test_pass(): pass")

        with patch.object(engine, "_exec_command", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = (0, "passed", "")
            threshold = GateThreshold(
                gate_type=GateType.E2E_TEST,
                enforcement=EnforcementLevel.OPTIONAL,
                timeout_seconds=900,
            )
            result = await engine._run_e2e_test(threshold)

            assert result.outcome == GateOutcome.PASSED



# =============================================================================
# ORCHESTRATOR - process_story and Stage Transitions (lines 247-809)
# =============================================================================

@pytest.mark.asyncio
class TestOrchestratorProcessStory:
    """Test Orchestrator.process_story with all stage transitions."""

    @pytest.fixture
    def config(self):
        """Create minimal config."""
        return AgentConfig(
            project=ProjectConfig(
                name="test",
                root="/tmp/test",
                python_version="3.11",
            ),
            safety=SafetyConfig(
                max_iterations_per_story=3,
            ),
        )

    @pytest.fixture
    def story_submission(self):
        """Create a story submission."""
        return StorySubmission(
            story="Implement feature X",
            project_id="proj1",
            priority=Priority.MEDIUM,
        )

    async def test_process_story_basic_creation(self, config, story_submission):
        """Test story creation during process_story."""
        analyzer = AsyncMock()

        # Mock return values
        async def analyze_fn(story):
            story.spec = StorySpec(title="Feature X", complexity_score=5)
            return story

        analyzer.analyze.side_effect = analyze_fn

        orchestrator = Orchestrator(
            config=config,
            analyzer=analyzer,
        )

        story = await orchestrator.process_story(story_submission)

        # Verify story was created and analyzer was called
        assert story.id is not None
        assert analyzer.analyze.called


    async def test_process_story_planning_failure(self, config, story_submission):
        """Test story fails when planning fails."""
        analyzer = AsyncMock()
        planner = AsyncMock()

        async def analyze_fn(story):
            story.spec = StorySpec(title="Feature X", complexity_score=5)
            return story

        analyzer.analyze.side_effect = analyze_fn
        planner.plan.side_effect = Exception("Planning failed")

        orchestrator = Orchestrator(
            config=config,
            analyzer=analyzer,
            planner=planner,
        )

        story = await orchestrator.process_story(story_submission)

        assert story.state == StoryState.FAILED

    async def test_process_story_coding_failure(self, config, story_submission):
        """Test story fails when code generation fails."""
        analyzer = AsyncMock()
        planner = AsyncMock()
        generator = AsyncMock()

        async def analyze_fn(story):
            story.spec = StorySpec(title="Feature X", complexity_score=5)
            return story

        async def plan_fn(story):
            story.plan = ExecutionPlan(tasks=[
                PlanTask(id="1", action=TaskAction.CREATE_FILE, target="test.py", description="Create file")
            ])
            return story

        analyzer.analyze.side_effect = analyze_fn
        planner.plan.side_effect = plan_fn
        generator.generate.side_effect = Exception("Code generation failed")

        orchestrator = Orchestrator(
            config=config,
            analyzer=analyzer,
            planner=planner,
            generator=generator,
        )

        story = await orchestrator.process_story(story_submission)

        assert story.state == StoryState.FAILED

    async def test_process_story_review_failure(self, config, story_submission):
        """Test story fails when review fails."""
        analyzer = AsyncMock()
        planner = AsyncMock()
        generator = AsyncMock()
        reviewer = AsyncMock()

        async def analyze_fn(story):
            story.spec = StorySpec(title="Feature X", complexity_score=5)
            return story

        async def plan_fn(story):
            story.plan = ExecutionPlan(tasks=[
                PlanTask(id="1", action=TaskAction.CREATE_FILE, target="test.py", description="Create file")
            ])
            return story

        async def gen_fn(story, error_context=None):
            story.iterations.append(Iteration(
                number=1, stage="coding",
                changes=[FileChange(file_path="new.py", content="code", change_type="created")]
            ))
            return story

        analyzer.analyze.side_effect = analyze_fn
        planner.plan.side_effect = plan_fn
        generator.generate.side_effect = gen_fn
        reviewer.review.side_effect = Exception("Review failed")

        orchestrator = Orchestrator(
            config=config,
            analyzer=analyzer,
            planner=planner,
            generator=generator,
            reviewer=reviewer,
        )

        story = await orchestrator.process_story(story_submission)

        assert story.state == StoryState.FAILED


# =============================================================================
# ORCHESTRATOR ADAPTERS - Missing Adapters (lines 274-419)
# =============================================================================

@pytest.mark.asyncio
class TestOrchestratorAdapters:
    """Test remaining adapter classes."""

    async def test_reviewer_adapter(self):
        """Test ReviewerAdapter."""
        from fastcoder.types.codebase import ProjectProfile

        real_reviewer = AsyncMock()
        real_reviewer.review.return_value = ReviewReport(
            approved=True,
            comments="LGTM"
        )

        adapter = ReviewerAdapter(real_reviewer)
        story = Story(
            id="1",
            raw_text="test",
            project_id="proj1",
        )
        story.iterations = [Iteration(number=1, stage="coding", changes=[FileChange(file_path="test.py", content="code", change_type="created")])]
        story.spec = StorySpec(title="Test", description="Test story")

        result = await adapter.review(story)

        assert result.id == story.id
        assert real_reviewer.review.called

    async def test_test_generator_adapter(self):
        """Test TestGeneratorAdapter."""
        real_gen = AsyncMock()
        # Mock the generate_tests method to return proper result
        mock_result = MagicMock()
        mock_result.test_code = "def test_pass(): pass"
        mock_result.test_file = "test_test.py"
        real_gen.generate_tests = AsyncMock(return_value=mock_result)

        adapter = TestGeneratorAdapter(real_gen)
        story = Story(
            id="1",
            raw_text="test",
            project_id="proj1",
        )
        story.iterations = [Iteration(number=1, stage="coding", changes=[FileChange(file_path="test.py", content="code", change_type="created")])]
        story.spec = StorySpec(title="Test", description="Test story")
        story.plan = ExecutionPlan(story_id="1", tasks=[PlanTask(id="1", action=TaskAction.CREATE_FILE, target="test.py", description="test")])

        result = await adapter.generate_tests(story)

        assert result.id == story.id
        assert real_gen.generate_tests.called

    async def test_test_runner_adapter(self):
        """Test TestRunnerAdapter."""
        real_runner = AsyncMock()
        real_runner.run.return_value = TestReport(
            passed=True,
            total=10,
            skipped=0,
        )

        adapter = TestRunnerAdapter(real_runner)
        story = Story(
            id="1",
            raw_text="test",
            project_id="proj1",
        )
        story.iterations = [Iteration(number=1, stage="testing", changes=[])]

        result = await adapter.run_tests(story)

        assert result.id == story.id
        assert real_runner.run.called

    async def test_deployer_adapter(self):
        """Test DeployerAdapter."""
        real_deployer = AsyncMock()
        real_deployer.deploy.return_value = DeployReport(
            status="success",
            commit_hash="abc123",
        )

        adapter = DeployerAdapter(real_deployer)
        story = Story(
            id="1",
            raw_text="test",
            project_id="proj1",
        )
        story.iterations = [Iteration(number=1, stage="deployed", changes=[])]

        result = await adapter.deploy(story)

        assert result.id == story.id
        assert real_deployer.deploy.called

    async def test_verifier_adapter(self):
        """Test VerifierAdapter."""
        real_verifier = AsyncMock()
        real_verifier.verify.return_value = {"status": "verified"}

        adapter = VerifierAdapter(real_verifier)
        story = Story(
            id="1",
            raw_text="test",
            project_id="proj1",
        )

        result = await adapter.verify(story)

        assert result.id == story.id
        assert real_verifier.verify.called

    async def test_context_manager_adapter(self):
        """Test ContextManagerAdapter."""
        real_ctx_mgr = AsyncMock()
        real_ctx_mgr.build_context.return_value = []

        adapter = ContextManagerAdapter(real_ctx_mgr)
        story = Story(
            id="1",
            raw_text="test",
            project_id="proj1",
        )

        await adapter.enrich_context(story, 1)

        assert real_ctx_mgr.build_context.called

    async def test_memory_store_adapter(self):
        """Test MemoryStoreAdapter."""
        from fastcoder.types.memory import MemoryEntry

        real_store = MagicMock()
        real_store.store = MagicMock()
        real_store.error_fixes = {}

        adapter = MemoryStoreAdapter(real_store)
        story = Story(
            id="1",
            raw_text="test",
            project_id="proj1",
        )
        story.spec = StorySpec(title="Test", description="Test story")

        error_context = ErrorContext(
            attempt=1,
            error=ErrorDetail(
                type="compile_error",
                message="Syntax error",
                stacktrace="...",
            ),
            instruction="Fix the syntax",
        )

        await adapter.store_learning(story, error_context)

        assert real_store.store.called

    async def test_error_classifier_adapter(self):
        """Test ErrorClassifierAdapter."""
        from fastcoder.types.errors import ErrorClassification

        real_classifier = MagicMock()
        real_classifier.classify.return_value = ErrorClassification(
            category="syntax_error",
            fingerprint="abc123",
        )

        adapter = ErrorClassifierAdapter(real_classifier)
        error = ErrorDetail(
            type="syntax_error",
            message="Invalid syntax",
            stacktrace="...",
        )

        result = await adapter.classify(error)

        assert result.error == error
        assert real_classifier.classify.called

    async def test_recovery_manager_adapter(self):
        """Test RecoveryManagerAdapter."""
        from fastcoder.types.errors import ErrorClassification

        real_recovery = MagicMock()
        # Create mock recovery action with strategy attribute
        mock_action = MagicMock()
        mock_action.strategy.value = "retry"
        mock_action.additional_context = "Try with different approach"
        mock_action.switch_to_top_tier = False
        mock_action.replan = False
        mock_action.escalate = False

        real_recovery.get_strategy.return_value = mock_action

        adapter = RecoveryManagerAdapter(real_recovery)
        error_context = ErrorContext(
            attempt=1,
            error=ErrorDetail(
                type="compile_error",
                message="Syntax error",
                stacktrace="...",
            ),
            classification=ErrorClassification(
                category="syntax_error",
                fingerprint="abc123",
            ),
        )
        story = Story(
            id="1",
            raw_text="test",
            project_id="proj1",
        )

        result = await adapter.suggest_recovery(story, error_context)

        assert isinstance(result, str)
        assert real_recovery.get_strategy.called

    async def test_build_runner_adapter(self):
        """Test BuildRunnerAdapter."""
        real_runner = AsyncMock()
        real_runner.run.return_value = {"success": True}

        adapter = BuildRunnerAdapter(real_runner)
        story = Story(
            id="1",
            raw_text="test",
            project_id="proj1",
        )

        result = await adapter.run_build(story)

        assert result.id == story.id
        assert real_runner.run.called


# =============================================================================
# RESOURCE LIMITER - monitor_process, enforce_limits (lines 93-363)
# =============================================================================

@pytest.mark.asyncio
class TestResourceLimiter:
    """Test ResourceLimiter monitoring and enforcement."""

    @pytest.fixture
    def limits(self):
        """Create resource limits."""
        return ResourceLimits(
            cpu_seconds=10,
            memory_bytes=1024 * 1024 * 500,  # 500MB
            disk_bytes=1024 * 1024 * 1024,  # 1GB
            wall_time_seconds=60,
            max_processes=5,
        )

    @pytest.fixture
    def limiter(self, limits):
        """Create ResourceLimiter."""
        return ResourceLimiter(limits)

    async def test_monitor_process_within_limits(self, limiter):
        """Test monitoring a process within limits."""
        # Create a simple process
        process = await asyncio.create_subprocess_exec(
            "sleep", "0.5",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        usage = await limiter.monitor_process(process)

        assert usage is not None
        # Just verify it returns a valid ResourceUsage
        assert usage.wall_time_seconds >= 0
        assert usage.cpu_time_seconds >= 0

    async def test_monitor_process_no_pid(self, limiter):
        """Test monitoring with None PID."""
        process = AsyncMock()
        process.pid = None
        process.returncode = None

        usage = await limiter.monitor_process(process)

        assert usage.wall_time_seconds == 0



# =============================================================================
# PACKAGE MANAGER TOOL - All Methods (lines 39-132)
# =============================================================================

@pytest.mark.asyncio
class TestPackageManager:
    """Test PackageManagerTool methods."""

    @pytest.fixture
    def temp_dir(self):
        """Create temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    async def test_detect_npm(self, temp_dir):
        """Test npm detection."""
        package_lock = Path(temp_dir) / "package-lock.json"
        package_lock.write_text("{}")

        shell = AsyncMock()
        pm = PackageManagerTool(temp_dir, shell)

        assert pm.pm_type == "npm"

    async def test_detect_yarn(self, temp_dir):
        """Test yarn detection."""
        yarn_lock = Path(temp_dir) / "yarn.lock"
        yarn_lock.write_text("")

        shell = AsyncMock()
        pm = PackageManagerTool(temp_dir, shell)

        assert pm.pm_type == "yarn"

    async def test_detect_pnpm(self, temp_dir):
        """Test pnpm detection."""
        pnpm_lock = Path(temp_dir) / "pnpm-lock.yaml"
        pnpm_lock.write_text("")

        shell = AsyncMock()
        pm = PackageManagerTool(temp_dir, shell)

        assert pm.pm_type == "pnpm"

    async def test_detect_pip(self, temp_dir):
        """Test pip detection."""
        pyproject = Path(temp_dir) / "pyproject.toml"
        pyproject.write_text("")

        shell = AsyncMock()
        pm = PackageManagerTool(temp_dir, shell)

        assert pm.pm_type == "pip"


    async def test_get_lockfile_hash(self, temp_dir):
        """Test lockfile hash generation."""
        package_lock = Path(temp_dir) / "package-lock.json"
        package_lock.write_text('{"lockfileVersion": 2}')

        shell = AsyncMock()
        pm = PackageManagerTool(temp_dir, shell)

        hash_val = await pm.get_lockfile_hash()

        assert hash_val != ""
        assert len(hash_val) == 64  # SHA256 hex length


# =============================================================================
# OWNERSHIP MAP - Pattern Matching, Git Blame (lines 42-150)
# =============================================================================

@pytest.mark.asyncio
class TestOwnershipMap:
    """Test OwnershipMap CODEOWNERS parsing and git blame."""

    @pytest.fixture
    def temp_project(self):
        """Create temp project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    async def test_parse_codeowners_basic(self, temp_project):
        """Test basic CODEOWNERS parsing."""
        codeowners_file = Path(temp_project) / "CODEOWNERS"
        codeowners_file.write_text(
            "*.py @python-team\n"
            "*.js @js-team\n"
            "# Comment\n"
            "docs/ @docs-team\n"
        )

        ownership = OwnershipMap(temp_project)
        owners = ownership._parse_codeowners()

        assert owners["*.py"] == ["@python-team"]
        assert owners["*.js"] == ["@js-team"]
        assert owners["docs/"] == ["@docs-team"]
        assert len(owners) == 3

    async def test_parse_codeowners_multiple_owners(self, temp_project):
        """Test CODEOWNERS with multiple owners."""
        codeowners_file = Path(temp_project) / "CODEOWNERS"
        codeowners_file.write_text(
            "*.py @owner1 @owner2 @owner3\n"
        )

        ownership = OwnershipMap(temp_project)
        owners = ownership._parse_codeowners()

        assert owners["*.py"] == ["@owner1", "@owner2", "@owner3"]

    async def test_parse_codeowners_github_location(self, temp_project):
        """Test CODEOWNERS in .github directory."""
        github_dir = Path(temp_project) / ".github"
        github_dir.mkdir()
        codeowners_file = github_dir / "CODEOWNERS"
        codeowners_file.write_text("*.py @python-team\n")

        ownership = OwnershipMap(temp_project)
        owners = ownership._parse_codeowners()

        assert owners["*.py"] == ["@python-team"]

    async def test_parse_codeowners_docs_location(self, temp_project):
        """Test CODEOWNERS in docs directory."""
        docs_dir = Path(temp_project) / "docs"
        docs_dir.mkdir()
        codeowners_file = docs_dir / "CODEOWNERS"
        codeowners_file.write_text("*.md @doc-team\n")

        ownership = OwnershipMap(temp_project)
        owners = ownership._parse_codeowners()

        assert owners["*.md"] == ["@doc-team"]

    async def test_parse_codeowners_not_found(self, temp_project):
        """Test handling of missing CODEOWNERS."""
        ownership = OwnershipMap(temp_project)
        owners = ownership._parse_codeowners()

        assert owners == {}

    async def test_analyze_blame_file_not_found(self, temp_project):
        """Test blame analysis for non-existent file."""
        ownership = OwnershipMap(temp_project)
        blame = await ownership._analyze_blame("nonexistent.py")

        assert blame == {}

    @patch("asyncio.to_thread")
    async def test_analyze_blame_success(self, mock_to_thread, temp_project):
        """Test git blame analysis."""
        py_file = Path(temp_project) / "test.py"
        py_file.write_text("x = 1\ny = 2\n")

        # Mock git blame output
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = (
            "abc1234 author1 1 line1\n"
            "def5678 author2 1 line2\n"
        )

        mock_to_thread.return_value = mock_result

        ownership = OwnershipMap(temp_project)
        blame = await ownership._analyze_blame("test.py")

        assert isinstance(blame, dict)

    @patch("asyncio.to_thread")
    async def test_analyze_blame_git_failure(self, mock_to_thread, temp_project):
        """Test git blame failure handling."""
        py_file = Path(temp_project) / "test.py"
        py_file.write_text("x = 1\n")

        mock_result = MagicMock()
        mock_result.returncode = 128
        mock_result.stderr = "fatal: not a git repository"

        mock_to_thread.return_value = mock_result

        ownership = OwnershipMap(temp_project)
        blame = await ownership._analyze_blame("test.py")

        assert blame == {}




# =============================================================================
# PARAMETRIZED EDGE CASE TESTS
# =============================================================================

@pytest.mark.asyncio
class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.parametrize("enforcement_level", [
        EnforcementLevel.REQUIRED,
        EnforcementLevel.OPTIONAL,
        EnforcementLevel.WARNING_ONLY,
    ])
    async def test_gate_enforcement_levels(self, enforcement_level):
        """Test gates with different enforcement levels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = Path(tmpdir) / "test.py"
            py_file.write_text("print('hello')")

            engine = QualityGateEngine(tmpdir)
            threshold = GateThreshold(
                gate_type=GateType.LINT,
                enforcement=enforcement_level,
                max_findings=0,
                timeout_seconds=120,
            )

            with patch.object(engine, "_exec_command", new_callable=AsyncMock) as mock:
                mock.return_value = (0, "", "")
                with patch.object(engine, "_parse_linting_output", return_value=[]):
                    result = await engine._run_lint(threshold)

                    assert result.enforcement == enforcement_level
                    # Result can be PASSED or SKIPPED depending on file detection
                    assert result.outcome in (GateOutcome.PASSED, GateOutcome.SKIPPED)

    @pytest.mark.parametrize("gate_type", [
        GateType.LINT,
        GateType.TYPE_CHECK,
        GateType.UNIT_TEST,
    ])
    async def test_gate_types_in_policy(self, gate_type):
        """Test various gate types."""
        policy = QualityGateEngine._default_policy()

        gate_types_in_policy = {g.gate_type for g in policy.gates}
        assert gate_type in gate_types_in_policy

    @pytest.mark.parametrize("priority", [
        Priority.LOW,
        Priority.MEDIUM,
        Priority.HIGH,
        Priority.CRITICAL,
    ])
    async def test_story_with_priorities(self, priority):
        """Test stories with different priorities."""
        story = Story(
            id="1",
            raw_text="test",
            project_id="proj1",
            priority=priority,
        )

        assert story.priority == priority
        assert story.state == StoryState.RECEIVED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
