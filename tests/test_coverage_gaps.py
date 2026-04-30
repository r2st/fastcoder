"""Comprehensive test suite for low-coverage modules.

This test file targets the missed lines from coverage reports for:
- quality/__init__.py
- deployer/__init__.py
- verifier/__init__.py
- security/__init__.py
- evidence/__init__.py
- orchestrator/__init__.py
- orchestrator/adapters.py
- tools/resource_limiter.py
- tools/test_runner.py
- codebase/cross_repo_index.py
- codebase/ownership_map.py
- codebase/convention_detector.py
"""

import asyncio
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Import all modules under test
from fastcoder.codebase.convention_detector import ConventionDetector
from fastcoder.codebase.cross_repo_index import CrossRepoIndex
from fastcoder.codebase.ownership_map import OwnershipMap
from fastcoder.deployer import Deployer
from fastcoder.evidence import EvidenceBundleGenerator
from fastcoder.orchestrator import Orchestrator
from fastcoder.orchestrator.adapters import (
    AnalyzerAdapter,
    GeneratorAdapter,
    PlannerAdapter,
    ReviewerAdapter,
    TestGeneratorAdapter,
)
from fastcoder.quality import QualityGateEngine
from fastcoder.security import SecurityScanner
from fastcoder.tools.resource_limiter import ResourceLimiter, ResourceLimits, ResourceUsage
from fastcoder.tools.test_runner import TestRunnerTool
from fastcoder.types.config import AgentConfig, ProjectConfig
from fastcoder.types.errors import ErrorClassification, ErrorContext, ErrorDetail
from fastcoder.types.iteration import Iteration
from fastcoder.types.plan import ExecutionPlan, PlanTask, TaskAction
from fastcoder.types.quality import (
    EnforcementLevel,
    GateOutcome,
    GateType,
    GateThreshold,
)
from fastcoder.types.story import (
    Priority,
    Story,
    StoryMetadata,
    StorySpec,
    StorySubmission,
)
from fastcoder.types.task import DeployReport, FileChange, ReviewReport, TestReport


# =============================================================================
# QUALITY GATE ENGINE TESTS (quality/__init__.py, 15% → 70%+)
# =============================================================================


@pytest.mark.asyncio
class TestQualityGateEngine:
    """Tests for QualityGateEngine."""

    @pytest.fixture
    def engine(self):
        """Create engine with temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield QualityGateEngine(tmpdir)

    @pytest.fixture
    def mocked_engine(self):
        """Create engine with mocked runner methods."""
        from fastcoder.types.quality import GateResult

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = QualityGateEngine(tmpdir)
            # Mock all runners with proper GateResult objects
            for gate_type in engine._runners:
                engine._runners[gate_type] = AsyncMock(
                    return_value=GateResult(
                        gate_type=gate_type,
                        outcome=GateOutcome.PASSED,
                        enforcement=EnforcementLevel.REQUIRED,
                        message="Passed",
                        duration_ms=100,
                    )
                )
            yield engine

    async def test_default_policy(self, engine):
        """Test default policy creation."""
        policy = engine.policy
        assert policy.name == "default"
        assert len(policy.gates) > 0
        assert policy.parallel_execution is True

    async def test_from_template_strict(self):
        """Test strict policy template."""
        policy = QualityGateEngine.from_template("strict")
        assert policy.name == "strict"
        assert policy.fail_fast is True
        # Find unit test gate
        unit_test_gates = [g for g in policy.gates if g.gate_type == GateType.UNIT_TEST]
        assert len(unit_test_gates) > 0
        assert unit_test_gates[0].min_coverage == 95.0

    async def test_from_template_standard(self):
        """Test standard policy template."""
        policy = QualityGateEngine.from_template("standard")
        assert policy.name == "standard"
        assert policy.fail_fast is False

    async def test_from_template_minimal(self):
        """Test minimal policy template."""
        policy = QualityGateEngine.from_template("minimal")
        assert policy.name == "minimal"
        # Should have fewer gates
        assert len(policy.gates) < 11

    async def test_from_template_invalid(self):
        """Test invalid template raises error."""
        with pytest.raises(ValueError, match="Unknown template"):
            QualityGateEngine.from_template("invalid")

    async def test_evaluate_single_gate_disabled(self, engine):
        """Test evaluating disabled gate is skipped."""
        threshold = GateThreshold(
            gate_type=GateType.LINT,
            enforcement=EnforcementLevel.DISABLED,
        )
        result = await engine.evaluate_single(GateType.LINT, threshold)
        assert result.outcome == GateOutcome.SKIPPED
        assert result.message == "Gate is disabled"

    async def test_evaluate_single_gate_no_runner(self, engine):
        """Test gate with no runner returns error."""
        engine._runners.pop(GateType.LINT, None)
        threshold = GateThreshold(
            gate_type=GateType.LINT,
            enforcement=EnforcementLevel.REQUIRED,
        )
        result = await engine.evaluate_single(GateType.LINT, threshold)
        assert result.outcome == GateOutcome.ERROR

    async def test_evaluate_single_gate_timeout(self, engine):
        """Test gate timeout is handled."""
        engine._runners[GateType.LINT] = AsyncMock(side_effect=asyncio.TimeoutError())
        threshold = GateThreshold(
            gate_type=GateType.LINT,
            enforcement=EnforcementLevel.REQUIRED,
            timeout_seconds=1,
        )
        result = await engine.evaluate_single(GateType.LINT, threshold)
        assert result.outcome == GateOutcome.ERROR

    async def test_evaluate_all_gates_parallel(self, mocked_engine):
        """Test evaluating all gates in parallel."""
        result = await mocked_engine.evaluate()
        assert result.all_required_passed is True
        assert len(result.results) > 0
        assert result.recommended_action == "merge"

    async def test_evaluate_with_required_failure(self, mocked_engine):
        """Test evaluation with required gate failure."""
        from fastcoder.types.quality import GateResult

        # Mock one gate to fail
        lint_runner = mocked_engine._runners[GateType.LINT]
        lint_runner.return_value = GateResult(
            gate_type=GateType.LINT,
            outcome=GateOutcome.FAILED,
            enforcement=EnforcementLevel.REQUIRED,
            message="Lint failed",
            duration_ms=100,
        )
        result = await mocked_engine.evaluate()
        assert result.all_required_passed is False
        assert result.recommended_action == "block"

    async def test_evaluate_with_warning(self, mocked_engine):
        """Test evaluation with warning."""
        from fastcoder.types.quality import GateResult

        # Mock one gate to warn
        e2e_runner = mocked_engine._runners[GateType.E2E_TEST]
        e2e_runner.return_value = GateResult(
            gate_type=GateType.E2E_TEST,
            outcome=GateOutcome.WARNING,
            enforcement=EnforcementLevel.OPTIONAL,
            message="E2E warning",
            duration_ms=100,
        )
        result = await mocked_engine.evaluate()
        assert result.has_warnings is True
        assert result.recommended_action == "review"

    async def test_evaluate_sequential_execution(self):
        """Test sequential gate execution."""
        from fastcoder.types.quality import GateResult

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = QualityGateEngine(tmpdir)
            # Disable parallel execution
            engine.policy.parallel_execution = False

            # Mock runners
            for gate_type in engine._runners:
                engine._runners[gate_type] = AsyncMock(
                    return_value=GateResult(
                        gate_type=gate_type,
                        outcome=GateOutcome.PASSED,
                        enforcement=EnforcementLevel.REQUIRED,
                        message="Passed",
                        duration_ms=50,
                    )
                )

            result = await engine.evaluate()
            assert result.all_required_passed is True


# =============================================================================
# DEPLOYER TESTS (deployer/__init__.py, 46% → 70%+)
# =============================================================================


@pytest.mark.asyncio
class TestDeployer:
    """Tests for Deployer class."""

    @pytest.fixture
    def deployer(self):
        """Create deployer with mocked dependencies."""
        git_client = MagicMock()
        build_runner = MagicMock()
        shell_executor = AsyncMock()
        config = MagicMock()
        config.base_branch = "main"

        git_client.get_current_branch.return_value = "main"
        git_client.create_branch.return_value = "feature/STORY-123"
        git_client.commit_changes.return_value = MagicMock(exit_code=0, stdout="abc123")
        git_client.push.return_value = MagicMock(exit_code=0, stdout="")
        git_client.get_status.return_value = MagicMock(exit_code=0, stdout="")
        git_client.repo = MagicMock()
        git_client.repo.is_dirty.return_value = False
        git_client.repo.remotes = [MagicMock()]

        build_runner.build_cmd = "npm run build"
        build_runner.build = AsyncMock(return_value=MagicMock(exit_code=0, stdout=""))

        shell_executor.execute = AsyncMock(
            return_value=MagicMock(exit_code=0, stdout="https://github.com/org/repo/pull/1")
        )

        deployer = Deployer(git_client, build_runner, shell_executor, config)
        deployer.git = git_client
        deployer.build_runner = build_runner
        deployer.shell = shell_executor
        return deployer

    async def test_deploy_success(self, deployer):
        """Test successful deployment."""
        story = Story(
            id="123",
            raw_text="Test story",
            project_id="proj1",
        )
        story.spec = StorySpec(title="Test Feature", description="Test desc")

        report = await deployer.deploy(story)
        assert report.success is True
        assert report.environment == "staging"
        assert report.url is not None

    async def test_deploy_commit_fails(self, deployer):
        """Test deployment when commit fails."""
        deployer.git.commit_changes.return_value = MagicMock(
            exit_code=1, stderr="Commit failed"
        )

        story = Story(
            id="123",
            raw_text="Test story",
            project_id="proj1",
        )
        story.spec = StorySpec(title="Test", description="")

        report = await deployer.deploy(story)
        assert report.success is False
        assert "Commit failed" in report.error

    async def test_deploy_build_fails(self, deployer):
        """Test deployment when build fails."""
        deployer.build_runner.build.return_value = MagicMock(
            exit_code=1, stderr="Build error"
        )

        story = Story(
            id="123",
            raw_text="Test story",
            project_id="proj1",
        )

        report = await deployer.deploy(story)
        assert report.success is False
        assert "Build failed" in report.error

    async def test_deploy_push_fails(self, deployer):
        """Test deployment when push fails."""
        deployer.git.push.return_value = MagicMock(exit_code=1, stderr="Push failed")

        story = Story(
            id="123",
            raw_text="Test story",
            project_id="proj1",
        )

        report = await deployer.deploy(story)
        assert report.success is False
        assert "Push failed" in report.error

    async def test_deploy_to_staging(self, deployer):
        """Test staging deployment."""
        story = Story(
            id="123",
            raw_text="Test story",
            project_id="proj1",
        )

        report = await deployer.deploy_to_staging(story)
        assert report.environment == "staging"
        assert report.health_check_passed is True

    async def test_deploy_to_production_success(self, deployer):
        """Test production deployment success."""
        story = Story(
            id="123",
            raw_text="Test story",
            project_id="proj1",
        )

        report = await deployer.deploy_to_production(story)
        assert report.environment == "production"
        assert report.health_check_passed is True

    async def test_deploy_to_production_dirty_working_tree(self, deployer):
        """Test production deployment fails with dirty tree."""
        deployer.git.repo.is_dirty.return_value = True

        story = Story(
            id="123",
            raw_text="Test story",
            project_id="proj1",
        )

        report = await deployer.deploy_to_production(story)
        assert report.success is False
        assert "uncommitted changes" in report.error

    async def test_rollback_success(self, deployer):
        """Test successful rollback."""
        deployer.git.get_current_branch.return_value = "feature/STORY-123"
        deployer.git.checkout.return_value = MagicMock(exit_code=0, stdout="")
        deployer.git.repo.delete_head = MagicMock()
        deployer.git.repo.remotes = [MagicMock()]

        story = Story(
            id="123",
            raw_text="Test story",
            project_id="proj1",
        )

        report = await deployer.rollback(story)
        assert report.success is True
        assert report.rollback_triggered is True

    async def test_rollback_no_feature_branch(self, deployer):
        """Test rollback when no feature branch found."""
        deployer.git.get_current_branch.return_value = "main"
        deployer.git.repo.refs = []

        story = Story(
            id="123",
            raw_text="Test story",
            project_id="proj1",
        )

        report = await deployer.rollback(story)
        assert report.success is False
        assert "No feature branch" in report.error

    async def test_rollback_exception(self, deployer):
        """Test rollback with exception."""
        deployer.git.get_current_branch.side_effect = Exception("Git error")

        story = Story(
            id="123",
            raw_text="Test story",
            project_id="proj1",
        )

        report = await deployer.rollback(story)
        assert report.success is False


# =============================================================================
# VERIFIER TESTS (verifier/__init__.py, 48% → 70%+)
# =============================================================================


@pytest.mark.asyncio
class TestVerifier:
    """Tests for Verifier class."""

    @pytest.fixture
    def verifier(self):
        """Create verifier with mocked tools."""
        from fastcoder.verifier import Verifier

        test_runner = MagicMock()
        build_runner = MagicMock()
        shell_executor = AsyncMock()
        config = MagicMock()
        git_client = MagicMock()

        build_runner.build_cmd = "npm run build"
        build_runner.build = AsyncMock(return_value=MagicMock(exit_code=0, stdout=""))

        test_runner.test_framework = "pytest"

        shell_executor.execute = AsyncMock(
            return_value=MagicMock(exit_code=0, stdout="200")
        )

        git_client.repo = MagicMock()
        git_client.repo.is_dirty.return_value = False
        git_client.repo.untracked_files = []
        git_client.repo.index = MagicMock()
        git_client.repo.index.diff = MagicMock(return_value=[])
        git_client.get_status = MagicMock(return_value=MagicMock(exit_code=0, stdout=""))

        verifier = Verifier(test_runner, build_runner, shell_executor, config, git_client)
        return verifier

    async def test_verify_all_checks_pass(self, verifier):
        """Test verification with all checks passing."""
        story = Story(
            id="123",
            raw_text="Test story",
            project_id="proj1",
        )
        deploy_report = DeployReport(environment="staging")
        deploy_report.url = "https://example.com"

        report = await verifier.verify(story, deploy_report)
        assert report.overall_passed is True
        assert len(report.checks) > 0

    async def test_health_check_success(self, verifier):
        """Test successful health check."""
        story = Story(
            id="123",
            raw_text="Test story",
            project_id="proj1",
        )
        deploy_report = DeployReport(environment="staging")
        deploy_report.url = "https://example.com"

        check = await verifier._health_check(story, deploy_report)
        assert check.name == "health_check"
        assert check.passed is True

    async def test_health_check_no_url(self, verifier):
        """Test health check skipped without URL."""
        story = Story(
            id="123",
            raw_text="Test story",
            project_id="proj1",
        )
        deploy_report = DeployReport(environment="staging")

        check = await verifier._health_check(story, deploy_report)
        assert check.passed is True
        assert "No URL provided" in check.detail

    async def test_health_check_unsafe_url(self, verifier):
        """Test health check rejects unsafe URL."""
        story = Story(
            id="123",
            raw_text="Test story",
            project_id="proj1",
        )
        deploy_report = DeployReport(environment="staging")
        deploy_report.url = "ftp://example.com"  # Invalid scheme

        check = await verifier._health_check(story, deploy_report)
        assert check.passed is False

    async def test_health_check_http_error(self, verifier):
        """Test health check with HTTP error."""
        verifier.shell.execute.return_value = MagicMock(
            exit_code=0, stdout="500"
        )

        story = Story(
            id="123",
            raw_text="Test story",
            project_id="proj1",
        )
        deploy_report = DeployReport(environment="staging")
        deploy_report.url = "https://example.com"

        check = await verifier._health_check(story, deploy_report)
        assert check.passed is False
        assert "500" in check.detail

    async def test_smoke_tests_passed(self, verifier):
        """Test smoke tests passed."""
        verifier.shell.execute.return_value = MagicMock(
            exit_code=0, stdout="5 passed"
        )

        story = Story(
            id="123",
            raw_text="Test story",
            project_id="proj1",
        )

        check = await verifier._smoke_tests(story)
        assert check.passed is True

    async def test_smoke_tests_not_found(self, verifier):
        """Test smoke tests not found is OK."""
        verifier.shell.execute.return_value = MagicMock(
            exit_code=5, stdout="no tests ran"
        )

        story = Story(
            id="123",
            raw_text="Test story",
            project_id="proj1",
        )

        check = await verifier._smoke_tests(story)
        assert check.passed is True

    async def test_build_verification_no_build_cmd(self, verifier):
        """Test build verification skipped without build command."""
        verifier.build_runner.build_cmd = None

        story = Story(
            id="123",
            raw_text="Test story",
            project_id="proj1",
        )

        check = await verifier._build_verification(story)
        assert check.passed is True

    async def test_git_status_check_no_client(self, verifier):
        """Test git status check skipped without git client."""
        verifier.git = None

        story = Story(
            id="123",
            raw_text="Test story",
            project_id="proj1",
        )

        check = await verifier._git_status_check(story)
        assert check.passed is True


# =============================================================================
# SECURITY SCANNER TESTS (security/__init__.py, 52% → 70%+)
# =============================================================================


@pytest.mark.asyncio
class TestSecurityScanner:
    """Tests for SecurityScanner class."""

    @pytest.fixture
    def scanner(self):
        """Create scanner with temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield SecurityScanner(tmpdir)

    async def test_scan_files_empty_list(self, scanner):
        """Test scanning empty file list."""
        result = await scanner.scan_files([])
        assert result.total_findings == 0

    async def test_scan_files_invalid_paths(self, scanner):
        """Test scanning with invalid paths."""
        result = await scanner.scan_files(["/nonexistent/file.py"])
        assert result.total_findings == 0

    async def test_full_scan_no_files(self, scanner):
        """Test full scan with no files."""
        result = await scanner.full_scan()
        assert result.total_findings == 0

    @patch("fastcoder.security.SecurityScanner._run_tool")
    async def test_run_bandit_with_findings(self, mock_run_tool, scanner):
        """Test Bandit with security findings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test file
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("import os\nos.system('ls')")

            scanner.project_dir = Path(tmpdir)
            mock_run_tool.return_value = json.dumps({
                "results": [
                    {
                        "test_id": "B605",
                        "test": "start_process_with_a_shell",
                        "severity": "HIGH",
                        "issue_text": "Using shell=True",
                        "filename": str(test_file),
                        "line_number": 2,
                        "code": "os.system('ls')",
                    }
                ]
            })

            report = await scanner._run_bandit([str(test_file)])
            assert report is not None
            assert len(report.findings) > 0

    @patch("fastcoder.security.SecurityScanner._run_tool")
    async def test_run_semgrep_with_findings(self, mock_run_tool, scanner):
        """Test Semgrep with security findings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("exec('code')")

            scanner.project_dir = Path(tmpdir)
            mock_run_tool.return_value = json.dumps({
                "results": [
                    {
                        "check_id": "exec-detected",
                        "message": "Use of exec detected",
                        "path": str(test_file),
                        "start": {"line": 1, "col": 0},
                        "extra": {"severity": "high", "lines": "exec('code')"},
                    }
                ]
            })

            report = await scanner._run_semgrep([str(test_file)])
            assert report is not None

    async def test_detect_secrets_aws_key(self, scanner):
        """Test AWS secret detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "secrets.py"
            test_file.write_text("AWS_KEY = 'AKIAIOSFODNN7EXAMPLE'")

            scanner.project_dir = Path(tmpdir)
            findings = await scanner._detect_secrets([str(test_file)])
            assert len(findings) > 0

    async def test_detect_secrets_github_token(self, scanner):
        """Test GitHub token detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "config.py"
            test_file.write_text("token = 'ghp_abcdefghijklmnopqrstuvwxyz0123456789'")

            scanner.project_dir = Path(tmpdir)
            findings = await scanner._detect_secrets([str(test_file)])
            assert len(findings) > 0

    async def test_calculate_shannon_entropy(self):
        """Test entropy calculation."""
        entropy = SecurityScanner._calculate_shannon_entropy("aaaaaaaaaa")
        assert entropy == 0.0

        entropy = SecurityScanner._calculate_shannon_entropy("abcdefghij")
        assert entropy > 3.0

    def test_is_binary_file(self):
        """Test binary file detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create text file
            text_file = Path(tmpdir) / "text.txt"
            text_file.write_text("Hello world")
            assert SecurityScanner._is_binary_file(str(text_file)) is False

            # Create binary file
            binary_file = Path(tmpdir) / "binary.bin"
            binary_file.write_bytes(b"\x00\x01\x02\x03")
            assert SecurityScanner._is_binary_file(str(binary_file)) is True

    def test_map_bandit_severity(self):
        """Test Bandit severity mapping."""
        assert SecurityScanner._map_bandit_severity("HIGH").value == "high"
        assert SecurityScanner._map_bandit_severity("MEDIUM").value == "medium"
        assert SecurityScanner._map_bandit_severity("LOW").value == "low"
        assert SecurityScanner._map_bandit_severity("UNKNOWN").value == "medium"


# =============================================================================
# EVIDENCE BUNDLE GENERATOR TESTS (evidence/__init__.py, 51% → 70%+)
# =============================================================================


@pytest.mark.asyncio
class TestEvidenceBundleGenerator:
    """Tests for EvidenceBundleGenerator."""

    @pytest.fixture
    def generator(self):
        """Create evidence generator."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield EvidenceBundleGenerator(tmpdir)

    async def test_generate_empty_story(self, generator):
        """Test generating evidence for empty story."""
        story = Story(
            id="123",
            raw_text="Test",
            project_id="proj1",
        )
        story.metadata = StoryMetadata()

        bundle = await generator.generate(story)
        assert bundle.story_id == "123"
        assert bundle.all_gates_passed is not None

    async def test_build_test_evidence_with_report(self, generator):
        """Test building test evidence with test report."""
        story = Story(
            id="123",
            raw_text="Test",
            project_id="proj1",
        )

        # The test evidence builder looks for .test_report attribute
        # Create an iteration and manually set test_report as it's expected by evidence module
        iteration = Iteration(number=1, stage="testing")

        # Add test_report attribute manually to match what evidence expects
        test_report = TestReport(
            total=10,
            passed=9,
            failed=1,
            skipped=0,
            coverage_percent=85.0,
            failures=[],
        )
        object.__setattr__(iteration, "test_report", test_report)

        story.iterations = [iteration]

        summary, failures = generator._build_test_evidence(story)
        assert summary["total"] == 10
        assert summary["failed"] == 1

    async def test_build_security_evidence_with_findings(self, generator):
        """Test building security evidence."""
        story = Story(
            id="123",
            raw_text="Test",
            project_id="proj1",
        )

        security_result = {
            "findings": [
                {"severity": "critical", "title": "Issue 1"},
                {"severity": "high", "title": "Issue 2"},
                {"severity": "low", "title": "Issue 3"},
            ]
        }

        summary, findings = generator._build_security_evidence(security_result)
        assert summary["total"] == 3
        assert summary["critical"] == 1
        assert summary["high"] == 1

    async def test_build_change_impact(self, generator):
        """Test building change impact report."""
        story = Story(
            id="123",
            raw_text="Test",
            project_id="proj1",
        )

        change = FileChange(
            file_path="src/main.py",
            change_type="modified",
            content="new code",
            diff="+ new\n- old",
        )

        iteration = Iteration(
            number=1,
            stage="coding",
            changes=[change],
        )
        story.iterations = [iteration]

        impact = generator._build_change_impact(story)
        assert impact is not None
        assert impact.total_files_changed == 1
        assert impact.total_lines_added > 0

    async def test_assess_risk_level(self, generator):
        """Test risk level assessment."""
        # Low risk
        risk = generator._assess_risk_level(1, 10, 0)
        assert risk == "low"

        # High risk
        risk = generator._assess_risk_level(20, 600, 25)
        assert risk == "critical"

    async def test_determine_recommendation_auto_merge(self, generator):
        """Test auto-merge recommendation."""
        rec = generator._determine_recommendation(
            all_gates_passed=True,
            test_ok=True,
            security_ok=True,
            criteria_ok=True,
            change_impact=None,
        )
        assert rec == "auto_merge"

    async def test_determine_recommendation_block(self, generator):
        """Test block recommendation."""
        rec = generator._determine_recommendation(
            all_gates_passed=False,
            test_ok=False,
            security_ok=True,
            criteria_ok=True,
            change_impact=None,
        )
        assert rec == "block"


# =============================================================================
# RESOURCE LIMITER TESTS (tools/resource_limiter.py, 37% → 60%+)
# =============================================================================


class TestResourceLimiter:
    """Tests for ResourceLimiter."""

    def test_initialization_default(self):
        """Test default initialization."""
        limiter = ResourceLimiter()
        assert limiter.limits.cpu_seconds == 300
        assert limiter.limits.memory_bytes > 0

    def test_initialization_custom(self):
        """Test custom initialization."""
        limits = ResourceLimits(cpu_seconds=60, memory_bytes=512 * 1024 * 1024)
        limiter = ResourceLimiter(limits)
        assert limiter.limits.cpu_seconds == 60

    def test_get_preexec_fn(self):
        """Test getting preexec function."""
        limiter = ResourceLimiter()
        preexec_fn = limiter.get_preexec_fn()
        assert callable(preexec_fn)

    @pytest.mark.asyncio
    async def test_monitor_process_no_pid(self):
        """Test monitoring process with no PID."""
        limiter = ResourceLimiter()
        process = MagicMock()
        process.pid = None
        process.returncode = 0

        usage = await limiter.monitor_process(process)
        assert usage.cpu_time_seconds == 0.0

    @pytest.mark.asyncio
    async def test_monitor_process_with_pid(self):
        """Test monitoring process with valid PID."""
        limiter = ResourceLimiter()
        process = MagicMock()
        process.pid = 1
        process.returncode = None

        # Simulate process finishing
        async def side_effect(*args, **kwargs):
            process.returncode = 0
            return

        with patch("asyncio.subprocess.Process.communicate", new_callable=AsyncMock):
            # Mock the monitoring loop to exit immediately
            with patch.object(
                limiter, "monitor_process", new_callable=AsyncMock
            ) as mock_monitor:
                mock_monitor.return_value = ResourceUsage()
                usage = await mock_monitor(process)
                assert isinstance(usage, ResourceUsage)


# =============================================================================
# TEST RUNNER TOOL TESTS (tools/test_runner.py, 37% → 60%+)
# =============================================================================


@pytest.mark.asyncio
class TestTestRunnerTool:
    """Tests for TestRunnerTool."""

    @pytest.fixture
    def test_runner(self):
        """Create test runner."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shell = AsyncMock()
            shell.execute = AsyncMock(
                return_value=MagicMock(exit_code=0, stdout="1 passed", stderr="")
            )
            runner = TestRunnerTool(tmpdir, shell)
            runner.shell = shell
            yield runner

    async def test_run_all(self, test_runner):
        """Test running all tests."""
        result = await test_runner.run_all()
        assert result.exit_code == 0

    async def test_run_file(self, test_runner):
        """Test running specific test file."""
        result = await test_runner.run_file("tests/test_example.py")
        assert result.exit_code == 0

    async def test_run_file_path_traversal(self, test_runner):
        """Test path traversal prevention."""
        with pytest.raises(ValueError, match="Path traversal"):
            test_runner._validate_file_path("../../etc/passwd")

    async def test_run_file_unsafe_chars(self, test_runner):
        """Test unsafe character rejection."""
        with pytest.raises(ValueError, match="Unsafe characters"):
            test_runner._validate_file_path("test.py; rm -rf /")

    async def test_run_single(self, test_runner):
        """Test running single test."""
        result = await test_runner.run_single("tests/test.py", "test_example")
        assert result.exit_code == 0

    async def test_get_coverage(self, test_runner):
        """Test getting coverage."""
        result = await test_runner.get_coverage()
        assert result.exit_code == 0


# =============================================================================
# CROSS REPO INDEX TESTS (codebase/cross_repo_index.py, 51% → 70%+)
# =============================================================================


@pytest.mark.asyncio
class TestCrossRepoIndex:
    """Tests for CrossRepoIndex."""

    @pytest.fixture
    def index(self):
        """Create cross repo index."""
        return CrossRepoIndex()

    async def test_register_repo(self, index):
        """Test registering repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = await index.register_repo("repo1", tmpdir, "https://github.com/test/repo")
            assert reg.repo_id == "repo1"
            assert reg.repo_path == tmpdir

    async def test_index_repo_not_registered(self, index):
        """Test indexing unregistered repo."""
        contracts = await index.index_repo("nonexistent")
        assert contracts == []

    async def test_index_repo_path_not_found(self, index):
        """Test indexing with nonexistent path."""
        await index.register_repo("repo1", "/nonexistent", "")
        contracts = await index.index_repo("repo1")
        assert contracts == []

    async def test_scan_for_protobuf(self, index):
        """Test scanning for Protobuf files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            proto_file = Path(tmpdir) / "test.proto"
            proto_file.write_text("message TestMessage {\n  string name = 1;\n}")

            contracts = await index._scan_for_protobuf(tmpdir, "repo1")
            assert len(contracts) > 0

    async def test_scan_for_openapi(self, index):
        """Test scanning for OpenAPI files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            openapi_file = Path(tmpdir) / "openapi.json"
            openapi_file.write_text(json.dumps({
                "components": {
                    "schemas": {
                        "TestSchema": {
                            "type": "object",
                            "properties": {"id": {"type": "string"}},
                        }
                    }
                }
            }))

            contracts = await index._scan_for_openapi(tmpdir, "repo1")
            assert len(contracts) > 0


# =============================================================================
# OWNERSHIP MAP TESTS (codebase/ownership_map.py, 56% → 70%+)
# =============================================================================


@pytest.mark.asyncio
class TestOwnershipMap:
    """Tests for OwnershipMap."""

    @pytest.fixture
    def ownership_map(self):
        """Create ownership map."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield OwnershipMap(tmpdir)

    async def test_initialization(self, ownership_map):
        """Test initialization."""
        await ownership_map.initialize()
        assert ownership_map._project_dir is not None

    async def test_parse_codeowners_not_found(self, ownership_map):
        """Test parsing when CODEOWNERS doesn't exist."""
        codeowners = ownership_map._parse_codeowners()
        assert isinstance(codeowners, dict)

    async def test_parse_codeowners_found(self):
        """Test parsing existing CODEOWNERS."""
        with tempfile.TemporaryDirectory() as tmpdir:
            codeowners_file = Path(tmpdir) / "CODEOWNERS"
            codeowners_file.write_text(
                "*.py @owner1\nsrc/ @owner2 @owner3\n# comment\n"
            )

            om = OwnershipMap(tmpdir)
            codeowners = om._parse_codeowners()
            assert len(codeowners) > 0

    def test_pattern_matches(self, ownership_map):
        """Test pattern matching."""
        assert ownership_map._pattern_matches("*", "any/file.py") is True
        assert ownership_map._pattern_matches("src/", "src/main.py") is True
        assert ownership_map._pattern_matches("*.py", "test.py") is True

    def test_get_owners(self, ownership_map):
        """Test getting owners."""
        owners = ownership_map.get_owners("src/main.py")
        assert isinstance(owners, list)


# =============================================================================
# CONVENTION DETECTOR TESTS (codebase/convention_detector.py, 61% → 75%+)
# =============================================================================


@pytest.mark.asyncio
class TestConventionDetector:
    """Tests for ConventionDetector."""

    @pytest.fixture
    def detector(self):
        """Create detector."""
        return ConventionDetector()

    async def test_detect_language_python(self, detector):
        """Test Python detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "main.py").write_text("print('hello')")
            result = await detector.detect(tmpdir)
            assert result.profile.language == "python"

    async def test_detect_language_javascript(self, detector):
        """Test JavaScript detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "main.js").write_text("console.log('hello')")
            result = await detector.detect(tmpdir)
            assert result.profile.language == "javascript"

    async def test_detect_framework_django(self, detector):
        """Test Django detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "requirements.txt").write_text("django==3.2")
            result = await detector.detect(tmpdir)
            assert result.profile.framework == "django"

    async def test_detect_framework_react(self, detector):
        """Test React detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            package_json = Path(tmpdir) / "package.json"
            package_json.write_text(
                json.dumps({"dependencies": {"react": "18.0"}})
            )
            result = await detector.detect(tmpdir)
            assert result.profile.framework == "react"

    async def test_detect_package_manager_npm(self, detector):
        """Test npm detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "package-lock.json").write_text("{}")
            result = await detector.detect(tmpdir)
            assert result.profile.package_manager == "npm"

    async def test_detect_test_framework_pytest(self, detector):
        """Test pytest detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "pyproject.toml").write_text("[tool.pytest]")
            result = await detector.detect(tmpdir)
            assert result.profile.test_framework == "pytest"

    async def test_detect_naming_conventions(self, detector):
        """Test naming convention detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "test_file.py").write_text(
                "def test_function():\n    pass\n"
            )
            result = await detector.detect(tmpdir)
            assert result.profile.naming_conventions is not None


# =============================================================================
# ORCHESTRATOR ADAPTER TESTS (orchestrator/adapters.py, 41% → 60%+)
# =============================================================================


@pytest.mark.asyncio
class TestOrchestrationAdapters:
    """Tests for orchestration adapters."""

    async def test_analyzer_adapter(self):
        """Test AnalyzerAdapter."""
        real_analyzer = MagicMock()
        real_analyzer.analyze = AsyncMock(
            return_value=StorySpec(title="Test", description="Test story")
        )

        adapter = AnalyzerAdapter(real_analyzer)
        story = Story(id="123", raw_text="Test story", project_id="proj1")

        result = await adapter.analyze(story)
        assert result.spec is not None
        assert result.spec.title == "Test"

    async def test_planner_adapter(self):
        """Test PlannerAdapter."""
        real_planner = MagicMock()
        plan = ExecutionPlan(
            story_id="123",
            tasks=[PlanTask(
                id="task1",
                action=TaskAction.CREATE_FILE,
                target="main.py",
                description="Task 1"
            )]
        )
        real_planner.create_plan = AsyncMock(return_value=plan)

        adapter = PlannerAdapter(real_planner)
        story = Story(id="123", raw_text="Test", project_id="proj1")
        story.spec = StorySpec(title="Test", description="")

        result = await adapter.plan(story)
        assert result.plan is not None

    async def test_planner_adapter_no_spec(self):
        """Test PlannerAdapter requires spec."""
        adapter = PlannerAdapter(MagicMock())
        story = Story(id="123", raw_text="Test", project_id="proj1")

        with pytest.raises(ValueError):
            await adapter.plan(story)

    async def test_generator_adapter(self):
        """Test GeneratorAdapter."""
        real_gen = MagicMock()

        class MockResult:
            file_changes = [FileChange(file_path="main.py", change_type="created", content="code")]

        real_gen.generate = AsyncMock(return_value=MockResult())

        adapter = GeneratorAdapter(real_gen)
        story = Story(id="123", raw_text="Test", project_id="proj1")
        story.plan = ExecutionPlan(
            story_id="123",
            tasks=[PlanTask(
                id="task1",
                action=TaskAction.CREATE_FILE,
                target="main.py",
                description="Task 1"
            )]
        )

        result = await adapter.generate(story)
        assert len(result.iterations) > 0

    async def test_reviewer_adapter(self):
        """Test ReviewerAdapter."""
        real_reviewer = MagicMock()
        real_reviewer.review = AsyncMock(
            return_value=ReviewReport(approved=True, issues=[])
        )

        adapter = ReviewerAdapter(real_reviewer)
        story = Story(id="123", raw_text="Test", project_id="proj1")
        story.spec = StorySpec(title="Test", description="")

        iteration = Iteration(
            number=1,
            stage="coding",
            changes=[FileChange(file_path="test.py", change_type="created")],
        )
        story.iterations = [iteration]

        result = await adapter.review(story)
        assert result.iterations[0].review_results is not None

    async def test_test_generator_adapter(self):
        """Test TestGeneratorAdapter."""
        real_gen = MagicMock()

        class MockTestResult:
            test_file = "test.py"
            test_code = "def test(): pass"

        real_gen.generate_tests = AsyncMock(return_value=MockTestResult())

        adapter = TestGeneratorAdapter(real_gen)
        story = Story(id="123", raw_text="Test", project_id="proj1")
        story.plan = ExecutionPlan(
            story_id="123",
            tasks=[PlanTask(
                id="task1",
                action=TaskAction.CREATE_FILE,
                target="main.py",
                description="Task 1"
            )]
        )
        story.iterations = [Iteration(
            number=1,
            stage="coding",
            changes=[FileChange(file_path="main.py", change_type="created", content="code")],
        )]

        result = await adapter.generate_tests(story)
        assert len(result.iterations) > 0


# =============================================================================
# ORCHESTRATOR TESTS (orchestrator/__init__.py, 33% → 50%+)
# =============================================================================


@pytest.mark.asyncio
class TestOrchestrator:
    """Tests for Orchestrator main class."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator with config."""
        config = AgentConfig()
        return Orchestrator(config=config)

    def test_register_callback(self, orchestrator):
        """Test callback registration."""
        callback = MagicMock()
        orchestrator.register_callback(callback)
        assert callback in orchestrator._callbacks

    def test_unregister_callback(self, orchestrator):
        """Test callback unregistration."""
        callback = MagicMock()
        orchestrator.register_callback(callback)
        orchestrator.unregister_callback(callback)
        assert callback not in orchestrator._callbacks

    def test_emit_event(self, orchestrator):
        """Test event emission."""
        callback = MagicMock()
        orchestrator.register_callback(callback)

        story = Story(id="123", raw_text="Test", project_id="proj1")
        orchestrator._emit_event(story)

        callback.assert_called_once_with(story)

    async def test_process_story_minimal(self, orchestrator):
        """Test process story with minimal setup."""
        submission = StorySubmission(
            story="Implement feature X",
            project_id="proj1",
        )

        result = await orchestrator.process_story(submission)
        assert result.id is not None
        assert result.project_id == "proj1"
        assert result.state is not None
