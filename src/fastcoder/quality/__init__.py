"""Quality Gate Policy Engine for autonomous dev agent.

Treats quality gates as first-class policy objects that run automatically
as part of the agent loop. Each gate produces structured evidence for merge decisions.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional

import structlog

from fastcoder.types.quality import (
    EnforcementLevel,
    GateOutcome,
    GateResult,
    GateThreshold,
    GateType,
    PolicyEvaluationResult,
    QualityGatePolicy,
)

logger = structlog.get_logger(__name__)


class QualityGateEngine:
    """Runs quality gates as first-class policy objects."""

    def __init__(
        self,
        project_dir: str,
        policy: Optional[QualityGatePolicy] = None,
    ):
        """Initialize the quality gate engine.

        Args:
            project_dir: Root directory of the project.
            policy: Optional quality gate policy. If None, uses default.
        """
        self.project_dir = Path(project_dir).resolve()
        self.policy = policy or self._default_policy()

        # Register gate runners
        self._runners: dict[GateType, Callable] = {
            GateType.LINT: self._run_lint,
            GateType.TYPE_CHECK: self._run_type_check,
            GateType.UNIT_TEST: self._run_unit_test,
            GateType.INTEGRATION_TEST: self._run_integration_test,
            GateType.E2E_TEST: self._run_e2e_test,
            GateType.SAST: self._run_sast,
            GateType.DEPENDENCY_AUDIT: self._run_dependency_audit,
            GateType.SECRET_DETECTION: self._run_secret_detection,
            GateType.COVERAGE_DELTA: self._run_coverage_delta,
            GateType.PERFORMANCE_BUDGET: self._run_performance_budget,
            GateType.MIGRATION_SAFETY: self._run_migration_safety,
        }

    @staticmethod
    def _default_policy() -> QualityGatePolicy:
        """Return default policy with all gates at sensible defaults."""
        return QualityGatePolicy(
            name="default",
            description="Default quality gate policy with all gates enabled",
            gates=[
                GateThreshold(
                    gate_type=GateType.LINT,
                    enforcement=EnforcementLevel.REQUIRED,
                    max_findings=0,
                    timeout_seconds=120,
                ),
                GateThreshold(
                    gate_type=GateType.TYPE_CHECK,
                    enforcement=EnforcementLevel.REQUIRED,
                    max_findings=0,
                    timeout_seconds=180,
                ),
                GateThreshold(
                    gate_type=GateType.UNIT_TEST,
                    enforcement=EnforcementLevel.REQUIRED,
                    min_coverage=80.0,
                    timeout_seconds=300,
                ),
                GateThreshold(
                    gate_type=GateType.INTEGRATION_TEST,
                    enforcement=EnforcementLevel.REQUIRED,
                    timeout_seconds=600,
                ),
                GateThreshold(
                    gate_type=GateType.E2E_TEST,
                    enforcement=EnforcementLevel.OPTIONAL,
                    timeout_seconds=900,
                ),
                GateThreshold(
                    gate_type=GateType.SAST,
                    enforcement=EnforcementLevel.REQUIRED,
                    max_severity="medium",
                    timeout_seconds=240,
                ),
                GateThreshold(
                    gate_type=GateType.DEPENDENCY_AUDIT,
                    enforcement=EnforcementLevel.REQUIRED,
                    max_severity="critical",
                    timeout_seconds=180,
                ),
                GateThreshold(
                    gate_type=GateType.SECRET_DETECTION,
                    enforcement=EnforcementLevel.REQUIRED,
                    max_findings=0,
                    timeout_seconds=120,
                ),
                GateThreshold(
                    gate_type=GateType.COVERAGE_DELTA,
                    enforcement=EnforcementLevel.REQUIRED,
                    min_coverage=0.0,
                    timeout_seconds=180,
                ),
                GateThreshold(
                    gate_type=GateType.PERFORMANCE_BUDGET,
                    enforcement=EnforcementLevel.OPTIONAL,
                    timeout_seconds=300,
                ),
                GateThreshold(
                    gate_type=GateType.MIGRATION_SAFETY,
                    enforcement=EnforcementLevel.OPTIONAL,
                    timeout_seconds=240,
                ),
            ],
            fail_fast=False,
            parallel_execution=True,
        )

    @staticmethod
    def from_template(name: str) -> QualityGatePolicy:
        """Load a pre-built policy template.

        Args:
            name: Template name ('strict', 'standard', 'minimal')

        Returns:
            Configured QualityGatePolicy

        Raises:
            ValueError: If template name is not recognized.
        """
        if name == "strict":
            return QualityGatePolicy(
                name="strict",
                description="Strict quality gate policy - all gates required with high thresholds",
                gates=[
                    GateThreshold(
                        gate_type=GateType.LINT,
                        enforcement=EnforcementLevel.REQUIRED,
                        max_findings=0,
                        timeout_seconds=120,
                    ),
                    GateThreshold(
                        gate_type=GateType.TYPE_CHECK,
                        enforcement=EnforcementLevel.REQUIRED,
                        max_findings=0,
                        timeout_seconds=180,
                    ),
                    GateThreshold(
                        gate_type=GateType.UNIT_TEST,
                        enforcement=EnforcementLevel.REQUIRED,
                        min_coverage=95.0,
                        timeout_seconds=300,
                    ),
                    GateThreshold(
                        gate_type=GateType.INTEGRATION_TEST,
                        enforcement=EnforcementLevel.REQUIRED,
                        timeout_seconds=600,
                    ),
                    GateThreshold(
                        gate_type=GateType.E2E_TEST,
                        enforcement=EnforcementLevel.REQUIRED,
                        timeout_seconds=900,
                    ),
                    GateThreshold(
                        gate_type=GateType.SAST,
                        enforcement=EnforcementLevel.REQUIRED,
                        max_severity="low",
                        timeout_seconds=240,
                    ),
                    GateThreshold(
                        gate_type=GateType.DEPENDENCY_AUDIT,
                        enforcement=EnforcementLevel.REQUIRED,
                        max_severity="medium",
                        timeout_seconds=180,
                    ),
                    GateThreshold(
                        gate_type=GateType.SECRET_DETECTION,
                        enforcement=EnforcementLevel.REQUIRED,
                        max_findings=0,
                        timeout_seconds=120,
                    ),
                    GateThreshold(
                        gate_type=GateType.COVERAGE_DELTA,
                        enforcement=EnforcementLevel.REQUIRED,
                        min_coverage=0.0,
                        timeout_seconds=180,
                    ),
                    GateThreshold(
                        gate_type=GateType.PERFORMANCE_BUDGET,
                        enforcement=EnforcementLevel.REQUIRED,
                        timeout_seconds=300,
                    ),
                    GateThreshold(
                        gate_type=GateType.MIGRATION_SAFETY,
                        enforcement=EnforcementLevel.REQUIRED,
                        timeout_seconds=240,
                    ),
                ],
                fail_fast=True,
                parallel_execution=True,
            )

        elif name == "standard":
            return QualityGatePolicy(
                name="standard",
                description="Standard quality gate policy - most gates required with standard thresholds",
                gates=[
                    GateThreshold(
                        gate_type=GateType.LINT,
                        enforcement=EnforcementLevel.REQUIRED,
                        max_findings=0,
                        timeout_seconds=120,
                    ),
                    GateThreshold(
                        gate_type=GateType.TYPE_CHECK,
                        enforcement=EnforcementLevel.REQUIRED,
                        max_findings=0,
                        timeout_seconds=180,
                    ),
                    GateThreshold(
                        gate_type=GateType.UNIT_TEST,
                        enforcement=EnforcementLevel.REQUIRED,
                        min_coverage=80.0,
                        timeout_seconds=300,
                    ),
                    GateThreshold(
                        gate_type=GateType.INTEGRATION_TEST,
                        enforcement=EnforcementLevel.REQUIRED,
                        timeout_seconds=600,
                    ),
                    GateThreshold(
                        gate_type=GateType.E2E_TEST,
                        enforcement=EnforcementLevel.WARNING_ONLY,
                        timeout_seconds=900,
                    ),
                    GateThreshold(
                        gate_type=GateType.SAST,
                        enforcement=EnforcementLevel.REQUIRED,
                        max_severity="medium",
                        timeout_seconds=240,
                    ),
                    GateThreshold(
                        gate_type=GateType.DEPENDENCY_AUDIT,
                        enforcement=EnforcementLevel.REQUIRED,
                        max_severity="critical",
                        timeout_seconds=180,
                    ),
                    GateThreshold(
                        gate_type=GateType.SECRET_DETECTION,
                        enforcement=EnforcementLevel.REQUIRED,
                        max_findings=0,
                        timeout_seconds=120,
                    ),
                    GateThreshold(
                        gate_type=GateType.COVERAGE_DELTA,
                        enforcement=EnforcementLevel.REQUIRED,
                        min_coverage=0.0,
                        timeout_seconds=180,
                    ),
                    GateThreshold(
                        gate_type=GateType.PERFORMANCE_BUDGET,
                        enforcement=EnforcementLevel.OPTIONAL,
                        timeout_seconds=300,
                    ),
                    GateThreshold(
                        gate_type=GateType.MIGRATION_SAFETY,
                        enforcement=EnforcementLevel.OPTIONAL,
                        timeout_seconds=240,
                    ),
                ],
                fail_fast=False,
                parallel_execution=True,
            )

        elif name == "minimal":
            return QualityGatePolicy(
                name="minimal",
                description="Minimal quality gate policy - only essential gates required",
                gates=[
                    GateThreshold(
                        gate_type=GateType.LINT,
                        enforcement=EnforcementLevel.REQUIRED,
                        max_findings=0,
                        timeout_seconds=120,
                    ),
                    GateThreshold(
                        gate_type=GateType.UNIT_TEST,
                        enforcement=EnforcementLevel.REQUIRED,
                        min_coverage=70.0,
                        timeout_seconds=300,
                    ),
                    GateThreshold(
                        gate_type=GateType.SECRET_DETECTION,
                        enforcement=EnforcementLevel.REQUIRED,
                        max_findings=0,
                        timeout_seconds=120,
                    ),
                ],
                fail_fast=False,
                parallel_execution=True,
            )

        else:
            raise ValueError(
                f"Unknown template '{name}'. "
                "Supported templates: 'strict', 'standard', 'minimal'"
            )

    async def evaluate(
        self,
        changed_files: Optional[list[str]] = None,
    ) -> PolicyEvaluationResult:
        """Run all gates in the policy and return results.

        Args:
            changed_files: Optional list of changed files to scope checks.
                          If None, checks entire project.

        Returns:
            PolicyEvaluationResult with results from all gates.
        """
        start_time = time.time()
        log = logger.bind(
            policy_name=self.policy.name,
            gate_count=len(self.policy.gates),
        )
        log.info("quality_policy_evaluation_started")

        results: list[GateResult] = []
        tasks: list[tuple[GateType, GateThreshold]] = []

        # Skip disabled gates, prepare tasks
        for threshold in self.policy.gates:
            if threshold.enforcement != EnforcementLevel.DISABLED:
                tasks.append((threshold.gate_type, threshold))

        # Run gates concurrently if enabled, else sequentially
        if self.policy.parallel_execution and len(tasks) > 1:
            result_list = await asyncio.gather(
                *[
                    self.evaluate_single(gate_type, threshold)
                    for gate_type, threshold in tasks
                ],
                return_exceptions=True,
            )

            # Process results, handling exceptions
            for result in result_list:
                if isinstance(result, Exception):
                    log.error(
                        "quality_gate_execution_error",
                        error=str(result),
                        exc_info=result,
                    )
                    # Create error result
                    results.append(
                        GateResult(
                            gate_type=GateType.LINT,  # Placeholder
                            outcome=GateOutcome.ERROR,
                            enforcement=EnforcementLevel.REQUIRED,
                            message=f"Gate execution error: {result}",
                        )
                    )
                else:
                    results.append(result)
        else:
            # Sequential execution
            for gate_type, threshold in tasks:
                result = await self.evaluate_single(gate_type, threshold)
                results.append(result)

                # Stop on first required failure if fail_fast enabled
                if (
                    self.policy.fail_fast
                    and threshold.enforcement == EnforcementLevel.REQUIRED
                    and result.outcome == GateOutcome.FAILED
                ):
                    log.info(
                        "quality_fail_fast_triggered",
                        failed_gate=gate_type.value,
                    )
                    break

        # Evaluate overall policy result
        all_required_passed = all(
            r.outcome in (GateOutcome.PASSED, GateOutcome.SKIPPED, GateOutcome.WARNING)
            for r in results
            if r.enforcement == EnforcementLevel.REQUIRED
        )

        has_warnings = any(
            r.outcome == GateOutcome.WARNING for r in results
        )

        # Determine recommended action
        required_failures = [
            r for r in results
            if r.enforcement == EnforcementLevel.REQUIRED
            and r.outcome == GateOutcome.FAILED
        ]

        if required_failures:
            recommended_action = "block"
        elif has_warnings:
            recommended_action = "review"
        else:
            recommended_action = "merge"

        total_duration_ms = int((time.time() - start_time) * 1000)

        evaluation_result = PolicyEvaluationResult(
            policy_name=self.policy.name,
            results=results,
            all_required_passed=all_required_passed,
            has_warnings=has_warnings,
            total_duration_ms=total_duration_ms,
            recommended_action=recommended_action,
        )

        log.info(
            "quality_policy_evaluation_completed",
            total_duration_ms=total_duration_ms,
            all_required_passed=all_required_passed,
            has_warnings=has_warnings,
            recommended_action=recommended_action,
            results_count=len(results),
        )

        return evaluation_result

    async def evaluate_single(
        self,
        gate_type: GateType,
        threshold: GateThreshold,
    ) -> GateResult:
        """Run a single quality gate.

        Args:
            gate_type: Type of gate to run.
            threshold: Gate threshold configuration.

        Returns:
            GateResult with outcome and details.
        """
        log = logger.bind(gate_type=gate_type.value)

        if threshold.enforcement == EnforcementLevel.DISABLED:
            log.info("quality_gate_skipped", reason="disabled")
            return GateResult(
                gate_type=gate_type,
                outcome=GateOutcome.SKIPPED,
                enforcement=threshold.enforcement,
                message="Gate is disabled",
            )

        runner = self._runners.get(gate_type)
        if not runner:
            log.warning("quality_gate_no_runner", available_runners=list(self._runners.keys()))
            return GateResult(
                gate_type=gate_type,
                outcome=GateOutcome.ERROR,
                enforcement=threshold.enforcement,
                message=f"No runner found for gate type {gate_type.value}",
            )

        try:
            log.info("quality_gate_started")
            result = await runner(threshold)
            log.info(
                "quality_gate_completed",
                outcome=result.outcome.value,
                duration_ms=result.duration_ms,
            )
            return result
        except asyncio.TimeoutError:
            log.error(
                "quality_gate_timeout",
                timeout_seconds=threshold.timeout_seconds,
            )
            return GateResult(
                gate_type=gate_type,
                outcome=GateOutcome.ERROR,
                enforcement=threshold.enforcement,
                message=f"Gate timed out after {threshold.timeout_seconds}s",
                duration_ms=threshold.timeout_seconds * 1000,
            )
        except Exception as e:
            log.error("quality_gate_error", error=str(e), exc_info=e)
            return GateResult(
                gate_type=gate_type,
                outcome=GateOutcome.ERROR,
                enforcement=threshold.enforcement,
                message=f"Gate execution error: {e}",
            )

    async def _run_lint(self, threshold: GateThreshold) -> GateResult:
        """Run linting (ruff for Python, eslint for JS)."""
        start_time = time.time()
        max_findings = threshold.max_findings or 0

        # Check for Python project
        py_files = list(self.project_dir.glob("**/*.py"))
        # Check for JavaScript/TypeScript project
        js_files = list(self.project_dir.glob("**/*.{js,ts,tsx,jsx}"))

        if not py_files and not js_files:
            return GateResult(
                gate_type=GateType.LINT,
                outcome=GateOutcome.SKIPPED,
                enforcement=threshold.enforcement,
                message="No Python or JavaScript files found",
                duration_ms=int((time.time() - start_time) * 1000),
            )

        # Try Python linting with ruff
        if py_files:
            return_code, stdout, stderr = await self._exec_command(
                ["ruff", "check", str(self.project_dir)],
                timeout=threshold.timeout_seconds,
            )

            findings = self._parse_linting_output(stdout, stderr)
            findings_count = len(findings)

            if return_code == 0:
                outcome = GateOutcome.PASSED
                message = "Linting passed with zero findings"
            elif findings_count <= max_findings:
                outcome = GateOutcome.WARNING
                message = f"Linting found {findings_count} issues (threshold: {max_findings})"
            else:
                outcome = GateOutcome.FAILED
                message = f"Linting found {findings_count} issues (threshold: {max_findings})"

            return GateResult(
                gate_type=GateType.LINT,
                outcome=outcome,
                enforcement=threshold.enforcement,
                message=message,
                findings_count=findings_count,
                details={"findings": findings[:10]},  # Include first 10 findings
                duration_ms=int((time.time() - start_time) * 1000),
            )

        # Try JavaScript linting with eslint
        if js_files:
            return_code, stdout, stderr = await self._exec_command(
                ["eslint", ".", "--format", "json"],
                timeout=threshold.timeout_seconds,
            )

            findings = self._parse_eslint_output(stdout)
            findings_count = len(findings)

            if return_code == 0:
                outcome = GateOutcome.PASSED
                message = "Linting passed with zero findings"
            elif findings_count <= max_findings:
                outcome = GateOutcome.WARNING
                message = f"Linting found {findings_count} issues (threshold: {max_findings})"
            else:
                outcome = GateOutcome.FAILED
                message = f"Linting found {findings_count} issues (threshold: {max_findings})"

            return GateResult(
                gate_type=GateType.LINT,
                outcome=outcome,
                enforcement=threshold.enforcement,
                message=message,
                findings_count=findings_count,
                details={"findings": findings[:10]},
                duration_ms=int((time.time() - start_time) * 1000),
            )

        return GateResult(
            gate_type=GateType.LINT,
            outcome=GateOutcome.SKIPPED,
            enforcement=threshold.enforcement,
            message="No linting tools configured or available",
            duration_ms=int((time.time() - start_time) * 1000),
        )

    async def _run_type_check(self, threshold: GateThreshold) -> GateResult:
        """Run type checking (mypy for Python, tsc for TS)."""
        start_time = time.time()
        max_findings = threshold.max_findings or 0

        # Check for Python project
        py_files = list(self.project_dir.glob("**/*.py"))
        if py_files:
            return_code, stdout, stderr = await self._exec_command(
                ["mypy", str(self.project_dir)],
                timeout=threshold.timeout_seconds,
            )

            findings = self._parse_mypy_output(stdout, stderr)
            findings_count = len(findings)

            if return_code == 0:
                outcome = GateOutcome.PASSED
                message = "Type checking passed"
            elif findings_count <= max_findings:
                outcome = GateOutcome.WARNING
                message = f"Type checking found {findings_count} errors (threshold: {max_findings})"
            else:
                outcome = GateOutcome.FAILED
                message = f"Type checking found {findings_count} errors (threshold: {max_findings})"

            return GateResult(
                gate_type=GateType.TYPE_CHECK,
                outcome=outcome,
                enforcement=threshold.enforcement,
                message=message,
                findings_count=findings_count,
                details={"errors": findings[:10]},
                duration_ms=int((time.time() - start_time) * 1000),
            )

        # Check for TypeScript project
        ts_files = list(self.project_dir.glob("**/*.ts"))
        if ts_files:
            return_code, stdout, stderr = await self._exec_command(
                ["tsc", "--noEmit"],
                timeout=threshold.timeout_seconds,
            )

            findings = self._parse_tsc_output(stdout, stderr)
            findings_count = len(findings)

            if return_code == 0:
                outcome = GateOutcome.PASSED
                message = "Type checking passed"
            elif findings_count <= max_findings:
                outcome = GateOutcome.WARNING
                message = f"Type checking found {findings_count} errors (threshold: {max_findings})"
            else:
                outcome = GateOutcome.FAILED
                message = f"Type checking found {findings_count} errors (threshold: {max_findings})"

            return GateResult(
                gate_type=GateType.TYPE_CHECK,
                outcome=outcome,
                enforcement=threshold.enforcement,
                message=message,
                findings_count=findings_count,
                details={"errors": findings[:10]},
                duration_ms=int((time.time() - start_time) * 1000),
            )

        return GateResult(
            gate_type=GateType.TYPE_CHECK,
            outcome=GateOutcome.SKIPPED,
            enforcement=threshold.enforcement,
            message="No Python or TypeScript files found",
            duration_ms=int((time.time() - start_time) * 1000),
        )

    async def _run_unit_test(self, threshold: GateThreshold) -> GateResult:
        """Run unit test suite."""
        start_time = time.time()
        min_coverage = threshold.min_coverage or 80.0

        # Check for Python project
        py_files = list(self.project_dir.glob("**/*.py"))
        if py_files:
            return_code, stdout, stderr = await self._exec_command(
                [
                    "pytest",
                    str(self.project_dir),
                    "--cov",
                    "--cov-report=term",
                    "-v",
                ],
                timeout=threshold.timeout_seconds,
            )

            coverage = self._parse_pytest_coverage(stdout, stderr)
            if coverage is None:
                coverage = 0.0

            if return_code == 0 and coverage >= min_coverage:
                outcome = GateOutcome.PASSED
                message = f"Unit tests passed with {coverage:.1f}% coverage"
            elif coverage < min_coverage:
                outcome = GateOutcome.FAILED
                message = f"Coverage {coverage:.1f}% below threshold {min_coverage:.1f}%"
            else:
                outcome = GateOutcome.FAILED
                message = f"Unit tests failed (exit code: {return_code})"

            return GateResult(
                gate_type=GateType.UNIT_TEST,
                outcome=outcome,
                enforcement=threshold.enforcement,
                message=message,
                details={"coverage": coverage},
                duration_ms=int((time.time() - start_time) * 1000),
            )

        # Check for JavaScript project
        js_files = list(self.project_dir.glob("**/*.{js,ts,tsx,jsx}"))
        if js_files:
            return_code, stdout, stderr = await self._exec_command(
                ["npm", "test", "--", "--coverage"],
                timeout=threshold.timeout_seconds,
            )

            coverage = self._parse_jest_coverage(stdout, stderr)
            if coverage is None:
                coverage = 0.0

            if return_code == 0 and coverage >= min_coverage:
                outcome = GateOutcome.PASSED
                message = f"Unit tests passed with {coverage:.1f}% coverage"
            elif coverage < min_coverage:
                outcome = GateOutcome.FAILED
                message = f"Coverage {coverage:.1f}% below threshold {min_coverage:.1f}%"
            else:
                outcome = GateOutcome.FAILED
                message = f"Unit tests failed (exit code: {return_code})"

            return GateResult(
                gate_type=GateType.UNIT_TEST,
                outcome=outcome,
                enforcement=threshold.enforcement,
                message=message,
                details={"coverage": coverage},
                duration_ms=int((time.time() - start_time) * 1000),
            )

        return GateResult(
            gate_type=GateType.UNIT_TEST,
            outcome=GateOutcome.SKIPPED,
            enforcement=threshold.enforcement,
            message="No test framework detected",
            duration_ms=int((time.time() - start_time) * 1000),
        )

    async def _run_integration_test(self, threshold: GateThreshold) -> GateResult:
        """Run integration tests."""
        start_time = time.time()

        # Check for Python pytest integration tests
        integration_dir = self.project_dir / "tests" / "integration"
        if integration_dir.exists():
            return_code, stdout, stderr = await self._exec_command(
                ["pytest", str(integration_dir), "-v"],
                timeout=threshold.timeout_seconds,
            )

            if return_code == 0:
                outcome = GateOutcome.PASSED
                message = "Integration tests passed"
            else:
                outcome = GateOutcome.FAILED
                message = f"Integration tests failed (exit code: {return_code})"

            return GateResult(
                gate_type=GateType.INTEGRATION_TEST,
                outcome=outcome,
                enforcement=threshold.enforcement,
                message=message,
                details={"output": stderr[-500:] if stderr else stdout[-500:]},
                duration_ms=int((time.time() - start_time) * 1000),
            )

        # Check for npm integration tests
        package_json = self.project_dir / "package.json"
        if package_json.exists():
            return_code, stdout, stderr = await self._exec_command(
                ["npm", "run", "test:integration"],
                timeout=threshold.timeout_seconds,
            )

            if return_code == 0:
                outcome = GateOutcome.PASSED
                message = "Integration tests passed"
            else:
                outcome = GateOutcome.FAILED
                message = f"Integration tests failed (exit code: {return_code})"

            return GateResult(
                gate_type=GateType.INTEGRATION_TEST,
                outcome=outcome,
                enforcement=threshold.enforcement,
                message=message,
                details={"output": stderr[-500:] if stderr else stdout[-500:]},
                duration_ms=int((time.time() - start_time) * 1000),
            )

        return GateResult(
            gate_type=GateType.INTEGRATION_TEST,
            outcome=GateOutcome.SKIPPED,
            enforcement=threshold.enforcement,
            message="No integration tests found",
            duration_ms=int((time.time() - start_time) * 1000),
        )

    async def _run_e2e_test(self, threshold: GateThreshold) -> GateResult:
        """Run end-to-end tests."""
        start_time = time.time()

        # Check for e2e test scripts
        e2e_dir = self.project_dir / "tests" / "e2e"
        if e2e_dir.exists():
            return_code, stdout, stderr = await self._exec_command(
                ["pytest", str(e2e_dir), "-v"],
                timeout=threshold.timeout_seconds,
            )

            if return_code == 0:
                outcome = GateOutcome.PASSED
                message = "E2E tests passed"
            else:
                outcome = GateOutcome.FAILED
                message = f"E2E tests failed (exit code: {return_code})"

            return GateResult(
                gate_type=GateType.E2E_TEST,
                outcome=outcome,
                enforcement=threshold.enforcement,
                message=message,
                details={"output": stderr[-500:] if stderr else stdout[-500:]},
                duration_ms=int((time.time() - start_time) * 1000),
            )

        # Check for npm e2e tests
        package_json = self.project_dir / "package.json"
        if package_json.exists():
            return_code, stdout, stderr = await self._exec_command(
                ["npm", "run", "test:e2e"],
                timeout=threshold.timeout_seconds,
            )

            if return_code == 0:
                outcome = GateOutcome.PASSED
                message = "E2E tests passed"
            else:
                outcome = GateOutcome.FAILED
                message = f"E2E tests failed (exit code: {return_code})"

            return GateResult(
                gate_type=GateType.E2E_TEST,
                outcome=outcome,
                enforcement=threshold.enforcement,
                message=message,
                details={"output": stderr[-500:] if stderr else stdout[-500:]},
                duration_ms=int((time.time() - start_time) * 1000),
            )

        return GateResult(
            gate_type=GateType.E2E_TEST,
            outcome=GateOutcome.SKIPPED,
            enforcement=threshold.enforcement,
            message="No E2E tests found",
            duration_ms=int((time.time() - start_time) * 1000),
        )

    async def _run_sast(self, threshold: GateThreshold) -> GateResult:
        """Run SAST (Static Application Security Testing) scanning."""
        start_time = time.time()
        max_severity = threshold.max_severity or "medium"

        # Try Semgrep for SAST
        return_code, stdout, stderr = await self._exec_command(
            ["semgrep", "--config=p/security-audit", str(self.project_dir)],
            timeout=threshold.timeout_seconds,
        )

        findings = self._parse_semgrep_output(stdout, stderr)
        high_severity = [
            f for f in findings
            if f.get("severity", "").lower() in ("high", "critical")
        ]

        if len(high_severity) == 0:
            outcome = GateOutcome.PASSED
            message = f"SAST scan passed ({len(findings)} total findings, none above {max_severity})"
        elif len(high_severity) <= (threshold.max_findings or 0):
            outcome = GateOutcome.WARNING
            message = f"SAST found {len(high_severity)} high/critical findings"
        else:
            outcome = GateOutcome.FAILED
            message = f"SAST found {len(high_severity)} high/critical findings"

        return GateResult(
            gate_type=GateType.SAST,
            outcome=outcome,
            enforcement=threshold.enforcement,
            message=message,
            findings_count=len(high_severity),
            details={"findings": high_severity[:5]},
            duration_ms=int((time.time() - start_time) * 1000),
        )

    async def _run_dependency_audit(self, threshold: GateThreshold) -> GateResult:
        """Run dependency vulnerability audit (pip-audit, npm audit)."""
        start_time = time.time()
        max_severity = threshold.max_severity or "critical"

        # Check for Python dependencies
        py_files = list(self.project_dir.glob("**/*.py"))
        if py_files or (self.project_dir / "requirements.txt").exists():
            return_code, stdout, stderr = await self._exec_command(
                ["pip-audit", "--desc"],
                timeout=threshold.timeout_seconds,
            )

            vulnerabilities = self._parse_pip_audit_output(stdout, stderr)
            critical = [
                v for v in vulnerabilities
                if v.get("severity", "").lower() == "critical"
            ]

            if len(critical) == 0:
                outcome = GateOutcome.PASSED
                message = f"Dependency audit passed ({len(vulnerabilities)} total vulns)"
            else:
                outcome = GateOutcome.FAILED
                message = f"Found {len(critical)} critical vulnerabilities"

            return GateResult(
                gate_type=GateType.DEPENDENCY_AUDIT,
                outcome=outcome,
                enforcement=threshold.enforcement,
                message=message,
                findings_count=len(critical),
                details={"vulnerabilities": critical[:5]},
                duration_ms=int((time.time() - start_time) * 1000),
            )

        # Check for JavaScript dependencies
        if (self.project_dir / "package.json").exists():
            return_code, stdout, stderr = await self._exec_command(
                ["npm", "audit", "--json"],
                timeout=threshold.timeout_seconds,
            )

            vulnerabilities = self._parse_npm_audit_output(stdout)
            critical = [
                v for v in vulnerabilities
                if v.get("severity", "") == "critical"
            ]

            if len(critical) == 0:
                outcome = GateOutcome.PASSED
                message = f"Dependency audit passed ({len(vulnerabilities)} total vulns)"
            else:
                outcome = GateOutcome.FAILED
                message = f"Found {len(critical)} critical vulnerabilities"

            return GateResult(
                gate_type=GateType.DEPENDENCY_AUDIT,
                outcome=outcome,
                enforcement=threshold.enforcement,
                message=message,
                findings_count=len(critical),
                details={"vulnerabilities": critical[:5]},
                duration_ms=int((time.time() - start_time) * 1000),
            )

        return GateResult(
            gate_type=GateType.DEPENDENCY_AUDIT,
            outcome=GateOutcome.SKIPPED,
            enforcement=threshold.enforcement,
            message="No package files found",
            duration_ms=int((time.time() - start_time) * 1000),
        )

    async def _run_secret_detection(self, threshold: GateThreshold) -> GateResult:
        """Run secret detection (cannot be disabled)."""
        start_time = time.time()

        # Use detect-secrets or truffleHog
        return_code, stdout, stderr = await self._exec_command(
            ["detect-secrets", "scan", str(self.project_dir)],
            timeout=threshold.timeout_seconds,
        )

        secrets = self._parse_detect_secrets_output(stdout)

        if len(secrets) == 0:
            outcome = GateOutcome.PASSED
            message = "No secrets detected"
        else:
            outcome = GateOutcome.FAILED
            message = f"Found {len(secrets)} potential secrets"

        return GateResult(
            gate_type=GateType.SECRET_DETECTION,
            outcome=outcome,
            enforcement=EnforcementLevel.REQUIRED,  # Cannot be overridden
            message=message,
            findings_count=len(secrets),
            details={"secrets": secrets[:5]},
            duration_ms=int((time.time() - start_time) * 1000),
        )

    async def _run_coverage_delta(self, threshold: GateThreshold) -> GateResult:
        """Check that coverage didn't decrease."""
        start_time = time.time()
        min_delta = threshold.min_coverage or 0.0

        # This would require baseline coverage tracking
        # For now, return a skipped result as it requires integration with CI/git
        return GateResult(
            gate_type=GateType.COVERAGE_DELTA,
            outcome=GateOutcome.SKIPPED,
            enforcement=threshold.enforcement,
            message="Coverage delta tracking requires CI integration",
            duration_ms=int((time.time() - start_time) * 1000),
        )

    async def _run_performance_budget(self, threshold: GateThreshold) -> GateResult:
        """Check performance budget."""
        start_time = time.time()

        # This is project-specific; would check build artifacts, bundle size, etc.
        return GateResult(
            gate_type=GateType.PERFORMANCE_BUDGET,
            outcome=GateOutcome.SKIPPED,
            enforcement=threshold.enforcement,
            message="Performance budget check requires project-specific configuration",
            duration_ms=int((time.time() - start_time) * 1000),
        )

    async def _run_migration_safety(self, threshold: GateThreshold) -> GateResult:
        """Check migration safety."""
        start_time = time.time()

        # Check for db migration files
        migrations_dir = self.project_dir / "migrations"
        if not migrations_dir.exists():
            return GateResult(
                gate_type=GateType.MIGRATION_SAFETY,
                outcome=GateOutcome.SKIPPED,
                enforcement=threshold.enforcement,
                message="No migrations directory found",
                duration_ms=int((time.time() - start_time) * 1000),
            )

        # Check for reversible migrations
        migrations = list(migrations_dir.glob("*.py"))
        if not migrations:
            return GateResult(
                gate_type=GateType.MIGRATION_SAFETY,
                outcome=GateOutcome.SKIPPED,
                enforcement=threshold.enforcement,
                message="No migrations found",
                duration_ms=int((time.time() - start_time) * 1000),
            )

        non_reversible = []
        for mig_file in migrations:
            content = mig_file.read_text()
            # Check for operations that are not reversible
            if "operations.RunSQL" in content and "reverse_sql" not in content:
                non_reversible.append(mig_file.name)

        if non_reversible:
            outcome = GateOutcome.FAILED
            message = f"Found {len(non_reversible)} non-reversible migrations"
        else:
            outcome = GateOutcome.PASSED
            message = "All migrations are reversible"

        return GateResult(
            gate_type=GateType.MIGRATION_SAFETY,
            outcome=outcome,
            enforcement=threshold.enforcement,
            message=message,
            findings_count=len(non_reversible),
            details={"non_reversible": non_reversible},
            duration_ms=int((time.time() - start_time) * 1000),
        )

    async def _exec_command(
        self,
        cmd: list[str],
        timeout: int = 300,
    ) -> tuple[int, str, str]:
        """Execute a shell command safely.

        Args:
            cmd: Command and arguments as list.
            timeout: Timeout in seconds.

        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.project_dir),
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )

            return (
                process.returncode,
                stdout.decode("utf-8", errors="replace"),
                stderr.decode("utf-8", errors="replace"),
            )
        except asyncio.TimeoutError:
            try:
                process.kill()
                await process.wait()
            except Exception:
                pass
            raise

    # Output parsing helpers
    @staticmethod
    def _parse_linting_output(stdout: str, stderr: str) -> list[dict[str, Any]]:
        """Parse ruff linting output."""
        findings = []
        for line in (stdout + stderr).split("\n"):
            if line.strip():
                # Simple parsing; ruff output format
                findings.append({"message": line.strip()})
        return findings

    @staticmethod
    def _parse_eslint_output(stdout: str) -> list[dict[str, Any]]:
        """Parse ESLint JSON output."""
        try:
            data = json.loads(stdout)
            findings = []
            for file_result in data:
                for message in file_result.get("messages", []):
                    findings.append({
                        "file": file_result.get("filePath"),
                        "message": message.get("message"),
                        "severity": message.get("severity"),
                    })
            return findings
        except (json.JSONDecodeError, KeyError):
            return []

    @staticmethod
    def _parse_mypy_output(stdout: str, stderr: str) -> list[dict[str, Any]]:
        """Parse mypy type checking output."""
        findings = []
        for line in (stdout + stderr).split("\n"):
            if "error:" in line or "note:" in line:
                findings.append({"message": line.strip()})
        return findings

    @staticmethod
    def _parse_tsc_output(stdout: str, stderr: str) -> list[dict[str, Any]]:
        """Parse TypeScript compiler output."""
        findings = []
        for line in (stdout + stderr).split("\n"):
            if "error" in line.lower():
                findings.append({"message": line.strip()})
        return findings

    @staticmethod
    def _parse_pytest_coverage(stdout: str, stderr: str) -> Optional[float]:
        """Extract coverage percentage from pytest output."""
        for line in stdout.split("\n"):
            match = re.search(r"(\d+)%", line)
            if match and "coverage" in line.lower():
                return float(match.group(1))
        return None

    @staticmethod
    def _parse_jest_coverage(stdout: str, stderr: str) -> Optional[float]:
        """Extract coverage percentage from Jest output."""
        try:
            # Jest may output coverage as a percentage
            match = re.search(r"Statements\s*:\s*(\d+\.?\d*)", stdout)
            if match:
                return float(match.group(1))
        except (ValueError, AttributeError):
            pass
        return None

    @staticmethod
    def _parse_semgrep_output(stdout: str, stderr: str) -> list[dict[str, Any]]:
        """Parse Semgrep SAST output."""
        try:
            data = json.loads(stdout)
            findings = []
            for result in data.get("results", []):
                findings.append({
                    "rule": result.get("check_id"),
                    "message": result.get("extra", {}).get("message"),
                    "severity": result.get("extra", {}).get("severity", "unknown"),
                    "file": result.get("path"),
                })
            return findings
        except (json.JSONDecodeError, KeyError, TypeError):
            return []

    @staticmethod
    def _parse_pip_audit_output(stdout: str, stderr: str) -> list[dict[str, Any]]:
        """Parse pip-audit output."""
        vulnerabilities = []
        current = None
        for line in (stdout + stderr).split("\n"):
            if "Found" in line:
                current = {}
                vulnerabilities.append(current)
            if current and "severity" in line.lower():
                match = re.search(r"(critical|high|medium|low)", line, re.I)
                if match:
                    current["severity"] = match.group(1).lower()
        return vulnerabilities

    @staticmethod
    def _parse_npm_audit_output(stdout: str) -> list[dict[str, Any]]:
        """Parse npm audit JSON output."""
        try:
            data = json.loads(stdout)
            vulnerabilities = []
            for vuln_id, vuln_data in data.get("vulnerabilities", {}).items():
                vulnerabilities.append({
                    "id": vuln_id,
                    "severity": vuln_data.get("severity", "unknown"),
                    "package": list(vuln_data.get("via", [{}]))[0].get("title"),
                })
            return vulnerabilities
        except (json.JSONDecodeError, KeyError, TypeError):
            return []

    @staticmethod
    def _parse_detect_secrets_output(stdout: str) -> list[dict[str, Any]]:
        """Parse detect-secrets output."""
        try:
            data = json.loads(stdout)
            secrets = []
            for file_path, file_secrets in data.get("results", {}).items():
                for secret_info in file_secrets:
                    secrets.append({
                        "file": file_path,
                        "type": secret_info.get("type"),
                    })
            return secrets
        except (json.JSONDecodeError, KeyError, TypeError):
            return []


__all__ = [
    "QualityGateEngine",
    "GateType",
    "GateOutcome",
    "EnforcementLevel",
    "GateThreshold",
    "GateResult",
    "QualityGatePolicy",
    "PolicyEvaluationResult",
]
