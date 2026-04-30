"""Main orchestrator for autonomous software development agent."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Any, Callable, Optional, Protocol

import structlog

from fastcoder.types.config import AgentConfig
from fastcoder.types.errors import ErrorContext, ErrorDetail
from fastcoder.types.iteration import Iteration
from fastcoder.types.story import (
    Priority,
    Story,
    StoryMetadata,
    StoryState,
    StorySubmission,
)
from fastcoder.types.task import ReviewReport, TestReport

from .approval_gate import ApprovalGateManager, GateStatus
from .convergence_detector import ConvergenceDetector
from .retry_policy import RetryPolicy
from .state_machine import StateMachine

logger = structlog.get_logger(__name__)


# Protocol types for loose coupling
class LLMRouter(Protocol):
    """Protocol for LLM routing component."""

    async def route(self, purpose: Any, context: Any) -> Any:
        """Route to appropriate model."""
        ...


class Analyzer(Protocol):
    """Protocol for analyzer component."""

    async def analyze(self, story: Story) -> Story:
        """Analyze story."""
        ...


class Planner(Protocol):
    """Protocol for planner component."""

    async def plan(self, story: Story) -> Story:
        """Create execution plan."""
        ...


class CodeGenerator(Protocol):
    """Protocol for code generation component."""

    async def generate(self, story: Story, error_context: Optional[ErrorContext] = None) -> Story:
        """Generate code."""
        ...


class CodeReviewer(Protocol):
    """Protocol for code review component."""

    async def review(self, story: Story) -> Story:
        """Review generated code."""
        ...


class TestGenerator(Protocol):
    """Protocol for test generation component."""

    async def generate_tests(self, story: Story) -> Story:
        """Generate tests."""
        ...


class BuildRunner(Protocol):
    """Protocol for build execution."""

    async def run_build(self, story: Story) -> Story:
        """Run build."""
        ...


class TestRunner(Protocol):
    """Protocol for test execution."""

    async def run_tests(self, story: Story) -> Story:
        """Run tests."""
        ...


class Deployer(Protocol):
    """Protocol for deployment."""

    async def deploy(self, story: Story) -> Story:
        """Deploy code."""
        ...


class ContextManager(Protocol):
    """Protocol for context management."""

    async def enrich_context(self, story: Story, iteration: int) -> None:
        """Enrich context for next attempt."""
        ...


class MemoryStore(Protocol):
    """Protocol for memory/knowledge store."""

    async def store_learning(self, story: Story, error_context: ErrorContext) -> None:
        """Store learning from error."""
        ...


class ErrorClassifier(Protocol):
    """Protocol for error classification."""

    async def classify(self, error: ErrorDetail) -> ErrorContext:
        """Classify error."""
        ...


class RecoveryManager(Protocol):
    """Protocol for recovery strategies."""

    async def suggest_recovery(self, story: Story, error_context: ErrorContext) -> str:
        """Suggest recovery strategy."""
        ...


class Orchestrator:
    """Main orchestrator managing story lifecycle."""

    def __init__(
        self,
        config: AgentConfig,
        analyzer: Optional[Analyzer] = None,
        planner: Optional[Planner] = None,
        generator: Optional[CodeGenerator] = None,
        reviewer: Optional[CodeReviewer] = None,
        test_generator: Optional[TestGenerator] = None,
        build_runner: Optional[BuildRunner] = None,
        test_runner: Optional[TestRunner] = None,
        deployer: Optional[Deployer] = None,
        context_manager: Optional[ContextManager] = None,
        memory_store: Optional[MemoryStore] = None,
        error_classifier: Optional[ErrorClassifier] = None,
        recovery_manager: Optional[RecoveryManager] = None,
        llm_router: Optional[LLMRouter] = None,
    ):
        """
        Initialize orchestrator.

        Args:
            config: Agent configuration
            analyzer: Story analyzer
            planner: Execution planner
            generator: Code generator
            reviewer: Code reviewer
            test_generator: Test generator
            build_runner: Build runner
            test_runner: Test runner
            deployer: Deployer
            context_manager: Context manager
            memory_store: Memory/knowledge store
            error_classifier: Error classifier
            recovery_manager: Recovery manager
            llm_router: LLM router
        """
        self.config = config

        # Components (stored as Any to allow for loose coupling)
        self.analyzer = analyzer
        self.planner = planner
        self.generator = generator
        self.reviewer = reviewer
        self.test_generator = test_generator
        self.build_runner = build_runner
        self.test_runner = test_runner
        self.deployer = deployer
        self.context_manager = context_manager
        self.memory_store = memory_store
        self.error_classifier = error_classifier
        self.recovery_manager = recovery_manager
        self.llm_router = llm_router

        # Orchestration components
        self.state_machine = StateMachine()
        self.retry_policy = RetryPolicy(
            max_retries_per_stage=config.safety.max_retries_per_stage
        )
        self.convergence_detector = ConvergenceDetector()
        self.approval_gate_manager = ApprovalGateManager(config.safety)

        # Lifecycle callbacks
        self._callbacks: list[Callable[[Story], None]] = []

        # Story tracking
        self._stories: dict[str, Story] = {}

        logger.info(
            "orchestrator_initialized",
            project_id=config.project.project_id,
            max_iterations=config.safety.max_iterations_per_story,
        )

    def register_callback(self, callback: Callable[[Story], None]) -> None:
        """Register lifecycle callback."""
        self._callbacks.append(callback)

    def unregister_callback(self, callback: Callable[[Story], None]) -> None:
        """Unregister callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _emit_event(self, story: Story) -> None:
        """Emit story event to all callbacks."""
        for callback in self._callbacks:
            try:
                callback(story)
            except Exception as e:
                logger.warning("callback_error", callback=str(callback), error=str(e))

    async def process_story(self, submission: StorySubmission) -> Story:
        """
        Main lifecycle: process a story from submission to completion.

        Args:
            submission: Story submission

        Returns:
            Completed or failed story
        """
        # Create story
        story_kwargs = {
            "id": str(uuid.uuid4()),
            "raw_text": submission.story,
            "project_id": submission.project_id,
            "priority": submission.priority,
        }
        if submission.constraints:
            story_kwargs["constraints"] = submission.constraints
        story = Story(**story_kwargs)

        self._stories[story.id] = story
        logger.info("story_created", story_id=story.id, project_id=story.project_id)
        self._emit_event(story)

        try:
            # Main workflow
            await self._run_analysis(story)
            if story.state == StoryState.FAILED:
                return story

            await self._run_planning(story)
            if story.state == StoryState.FAILED:
                return story

            # Coding loop with retries
            await self._run_coding_loop(story)
            if story.state == StoryState.FAILED:
                return story

            # Generate evidence bundle before deployment
            await self._generate_evidence_bundle(story)

            # Deployment and verification
            await self._run_deployment(story)
            if story.state == StoryState.FAILED:
                return story

            await self._run_verification(story)

            # Mark done
            if story.state != StoryState.FAILED:
                self.state_machine.transition(story, StoryState.DONE, reason="Successfully completed")
                story.metadata.completed_at = datetime.utcnow()

        except Exception as e:
            logger.exception("orchestrator_error", story_id=story.id, error=str(e))
            self.state_machine.transition(story, StoryState.FAILED, reason=f"Orchestrator error: {str(e)}")

        self._emit_event(story)
        return story

    async def _run_analysis(self, story: Story) -> None:
        """Analyze the story."""
        logger.info("analysis_started", story_id=story.id)
        self.state_machine.transition(story, StoryState.ANALYZING, reason="Starting analysis")
        self._emit_event(story)

        try:
            if self.analyzer:
                story = await self.analyzer.analyze(story)
            self.state_machine.transition(story, StoryState.PLANNING, reason="Analysis complete")
            logger.info("analysis_complete", story_id=story.id)
        except Exception as e:
            logger.exception("analysis_error", story_id=story.id)
            self.state_machine.transition(story, StoryState.FAILED, reason=f"Analysis failed: {str(e)}")

        self._emit_event(story)

    async def _run_planning(self, story: Story) -> None:
        """Plan execution."""
        logger.info("planning_started", story_id=story.id)
        self.state_machine.transition(story, StoryState.PLANNING, reason="Starting planning")
        self._emit_event(story)

        try:
            if self.planner:
                story = await self.planner.plan(story)
            self.state_machine.transition(story, StoryState.CODING, reason="Planning complete")
            logger.info("planning_complete", story_id=story.id)
        except Exception as e:
            logger.exception("planning_error", story_id=story.id)
            self.state_machine.transition(story, StoryState.FAILED, reason=f"Planning failed: {str(e)}")

        self._emit_event(story)

    async def _run_coding_loop(self, story: Story) -> None:
        """Run the coding-review-test loop with retries."""
        iteration_num = len(story.iterations) + 1

        while iteration_num <= self.config.safety.max_iterations_per_story:
            # Check convergence
            convergence = self.convergence_detector.check(story)
            if convergence.status != "progressing":
                logger.warning(
                    "convergence_issue",
                    story_id=story.id,
                    status=convergence.status,
                    action=convergence.recommended_action,
                )
                if convergence.recommended_action == "escalate":
                    self.state_machine.transition(
                        story,
                        StoryState.FAILED,
                        reason=f"Convergence issue: {convergence.status}",
                    )
                    self._emit_event(story)
                    return
                elif convergence.recommended_action == "replan":
                    self.state_machine.transition(
                        story,
                        StoryState.PLANNING,
                        reason="Replanning due to lack of progress",
                    )
                    self._emit_event(story)
                    await self._run_planning(story)
                    return

            # Code generation
            await self._run_coding(story, iteration_num)
            if story.state == StoryState.FAILED:
                return

            # Code review
            await self._run_review(story, iteration_num)
            if story.state == StoryState.FAILED:
                return

            # If review blocked, retry coding
            if story.state == StoryState.CODING:
                iteration_num += 1
                self._emit_event(story)
                continue

            # Testing
            await self._run_testing(story, iteration_num)
            if story.state == StoryState.FAILED:
                return

            # If tests failed, check if should retry
            if story.state == StoryState.CODING:
                iteration_num += 1
                self._emit_event(story)
                continue

            # Security scanning (if scanner attached)
            await self._run_security_scan(story, iteration_num)
            if story.state == StoryState.CODING:
                iteration_num += 1
                self._emit_event(story)
                continue

            # Quality gates (if engine attached)
            await self._run_quality_gates(story, iteration_num)
            if story.state == StoryState.CODING:
                iteration_num += 1
                self._emit_event(story)
                continue

            # Success
            break

        # Max iterations check
        if iteration_num > self.config.safety.max_iterations_per_story:
            logger.error(
                "max_iterations_exceeded",
                story_id=story.id,
                iterations=iteration_num,
            )
            self.state_machine.transition(
                story,
                StoryState.FAILED,
                reason=f"Exceeded max iterations ({self.config.safety.max_iterations_per_story})",
            )
            self._emit_event(story)

    async def _run_coding(self, story: Story, iteration_num: int) -> None:
        """Generate code with error recovery."""
        logger.info("coding_started", story_id=story.id, iteration=iteration_num)

        if story.state != StoryState.CODING:
            self.state_machine.transition(story, StoryState.CODING, reason=f"Iteration {iteration_num}")

        self._emit_event(story)

        try:
            # Determine error context for context enrichment
            error_context = None
            attempt_num = len([i for i in story.iterations if i.stage == "coding"]) + 1

            if attempt_num > 1 and story.iterations:
                # Enrich context based on attempt number
                await self._enrich_context_for_retry(story, attempt_num)
                last_iteration = story.iterations[-1]
                if last_iteration.error_context:
                    error_context = last_iteration.error_context

            # Generate code
            if self.generator:
                story = await self.generator.generate(story, error_context)

            # Check costs
            if self._check_cost_budget(story):
                self.state_machine.transition(
                    story,
                    StoryState.FAILED,
                    reason="Cost budget exceeded",
                )
                return

            logger.info("coding_complete", story_id=story.id, iteration=iteration_num)

        except Exception as e:
            logger.exception("coding_error", story_id=story.id)
            self.state_machine.transition(
                story,
                StoryState.FAILED,
                reason=f"Code generation failed: {str(e)}",
            )

        self._emit_event(story)

    async def _run_review(self, story: Story, iteration_num: int) -> None:
        """Review generated code."""
        logger.info("review_started", story_id=story.id, iteration=iteration_num)

        self.state_machine.transition(story, StoryState.REVIEWING, reason=f"Iteration {iteration_num}")
        self._emit_event(story)

        try:
            if self.reviewer:
                story = await self.reviewer.review(story)

            # Check for blocking issues
            current_iteration = story.iterations[-1] if story.iterations else None
            if current_iteration and current_iteration.review_results:
                review = current_iteration.review_results
                if not review.approved:
                    # Has blocking issues - send back to coding
                    logger.info(
                        "review_blocking_issues",
                        story_id=story.id,
                        issue_count=len(review.issues),
                    )
                    self.state_machine.transition(
                        story,
                        StoryState.CODING,
                        reason="Review found blocking issues",
                    )
                    self._emit_event(story)
                    return

            # Approved - move to testing
            self.state_machine.transition(story, StoryState.TESTING, reason="Review approved")
            logger.info("review_approved", story_id=story.id)

        except Exception as e:
            logger.exception("review_error", story_id=story.id)
            self.state_machine.transition(
                story,
                StoryState.FAILED,
                reason=f"Code review failed: {str(e)}",
            )

        self._emit_event(story)

    async def _run_testing(self, story: Story, iteration_num: int) -> None:
        """Generate and run tests."""
        logger.info("testing_started", story_id=story.id, iteration=iteration_num)

        self.state_machine.transition(story, StoryState.TESTING, reason=f"Iteration {iteration_num}")
        self._emit_event(story)

        try:
            # Generate tests
            if self.test_generator:
                story = await self.test_generator.generate_tests(story)

            # Run tests
            if self.test_runner:
                story = await self.test_runner.run_tests(story)

            # Check results
            current_iteration = story.iterations[-1] if story.iterations else None
            if current_iteration and current_iteration.test_results:
                test_report = current_iteration.test_results
                if test_report.failed > 0:
                    logger.warning(
                        "tests_failed",
                        story_id=story.id,
                        failed=test_report.failed,
                        passed=test_report.passed,
                    )

                    # Record error and retry
                    error_context = await self._create_test_error_context(story, test_report)
                    if self.retry_policy.should_retry(story.id, StoryState.TESTING, error_context):
                        self.retry_policy.record_attempt(
                            story.id,
                            StoryState.TESTING,
                            success=False,
                            error_fingerprint=error_context.classification.fingerprint,
                        )

                        if self.memory_store:
                            await self.memory_store.store_learning(story, error_context)

                        # Send back to coding
                        self.state_machine.transition(
                            story,
                            StoryState.CODING,
                            reason="Tests failed - retrying",
                        )
                    else:
                        # Give up
                        self.state_machine.transition(
                            story,
                            StoryState.FAILED,
                            reason="Tests failed and retry limit exceeded",
                        )
                    self._emit_event(story)
                    return

                # Tests passed
                logger.info(
                    "tests_passed",
                    story_id=story.id,
                    passed=test_report.passed,
                    coverage=test_report.coverage_percent,
                )

        except Exception as e:
            logger.exception("testing_error", story_id=story.id)
            self.state_machine.transition(
                story,
                StoryState.FAILED,
                reason=f"Testing failed: {str(e)}",
            )

        self._emit_event(story)

    async def _run_deployment(self, story: Story) -> None:
        """Deploy code."""
        logger.info("deployment_started", story_id=story.id)

        # Check pre-deploy gate
        gate_result = await self.approval_gate_manager.check_gate("pre_deploy", story)
        if gate_result.status == GateStatus.PENDING:
            logger.info("deployment_pending_approval", story_id=story.id)
            self._emit_event(story)
            return
        elif gate_result.status == GateStatus.BLOCKED:
            logger.error("deployment_blocked", story_id=story.id)
            self.state_machine.transition(story, StoryState.FAILED, reason="Deployment blocked by gate")
            self._emit_event(story)
            return

        self.state_machine.transition(story, StoryState.DEPLOYING, reason="Starting deployment")
        self._emit_event(story)

        try:
            if self.deployer:
                story = await self.deployer.deploy(story)

            self.state_machine.transition(story, StoryState.VERIFYING, reason="Deployment complete")
            logger.info("deployment_complete", story_id=story.id)

        except Exception as e:
            logger.exception("deployment_error", story_id=story.id)
            self.state_machine.transition(
                story,
                StoryState.FAILED,
                reason=f"Deployment failed: {str(e)}",
            )

        self._emit_event(story)

    async def _run_verification(self, story: Story) -> None:
        """Verify deployment using the Verifier adapter (if available)."""
        logger.info("verification_started", story_id=story.id)

        self.state_machine.transition(story, StoryState.VERIFYING, reason="Starting verification")
        self._emit_event(story)

        try:
            # Use the verifier adapter if attached by main.py
            verifier = getattr(self, "_verifier", None)
            if verifier:
                story = await verifier.verify(story)
                logger.info("verification_complete", story_id=story.id)
            else:
                logger.info("verification_skipped_no_verifier", story_id=story.id)

        except Exception as e:
            logger.exception("verification_error", story_id=story.id)
            self.state_machine.transition(
                story,
                StoryState.FAILED,
                reason=f"Verification failed: {str(e)}",
            )

        self._emit_event(story)

    async def _run_security_scan(self, story: Story, iteration_num: int) -> None:
        """Run SAST and secret detection on generated code."""
        scanner = getattr(self, "_security_scanner", None)
        if not scanner:
            return

        logger.info("security_scan_started", story_id=story.id, iteration=iteration_num)

        try:
            # Collect changed file paths from latest iteration
            changed_files = []
            if story.iterations:
                latest = story.iterations[-1]
                changed_files = [c.file_path for c in latest.changes if c.file_path]

            if not changed_files:
                logger.info("security_scan_skipped_no_files", story_id=story.id)
                return

            result = await scanner.scan_files(changed_files)

            # Store result on the iteration
            if story.iterations:
                story.iterations[-1].security_scan_result = result

            if not result.passed:
                logger.warning(
                    "security_scan_failed",
                    story_id=story.id,
                    critical=result.critical_count,
                    high=result.high_count,
                )
                # Record post-mortem if engine available
                post_mortem = getattr(self, "_post_mortem_engine", None)
                if post_mortem:
                    await post_mortem.analyze_gate_failure(
                        story=story,
                        gate_result={"gate_type": "sast", "findings": result.total_findings},
                        error_context=None,
                    )
                # Send back to coding for fixes
                if self.retry_policy.should_retry(story.id, StoryState.TESTING, None):
                    self.state_machine.transition(
                        story,
                        StoryState.CODING,
                        reason=f"Security scan found {result.critical_count} critical, {result.high_count} high findings",
                    )
                else:
                    self.state_machine.transition(
                        story,
                        StoryState.FAILED,
                        reason="Security scan failed and retry limit exceeded",
                    )
            else:
                logger.info("security_scan_passed", story_id=story.id)

        except Exception as e:
            logger.warning("security_scan_error", story_id=story.id, error=str(e))
            # Non-fatal — continue pipeline

        self._emit_event(story)

    async def _run_quality_gates(self, story: Story, iteration_num: int) -> None:
        """Run quality gate policy evaluation on current iteration."""
        engine = getattr(self, "_quality_gate_engine", None)
        if not engine:
            return

        logger.info("quality_gates_started", story_id=story.id, iteration=iteration_num)

        try:
            # Collect changed file paths
            changed_files = []
            if story.iterations:
                latest = story.iterations[-1]
                changed_files = [c.file_path for c in latest.changes if c.file_path]

            result = await engine.evaluate(changed_files=changed_files)

            # Store result on the iteration
            if story.iterations:
                story.iterations[-1].quality_gate_result = result

            if not result.all_required_passed:
                failed_gates = [
                    r.gate_type.value for r in result.results
                    if r.outcome.value == "failed" and r.enforcement.value == "required"
                ]
                logger.warning(
                    "quality_gates_failed",
                    story_id=story.id,
                    failed_gates=failed_gates,
                )
                # Record post-mortem
                post_mortem = getattr(self, "_post_mortem_engine", None)
                if post_mortem:
                    for gate_result in result.results:
                        if gate_result.outcome.value == "failed":
                            await post_mortem.analyze_gate_failure(
                                story=story,
                                gate_result={
                                    "gate_type": gate_result.gate_type.value,
                                    "message": gate_result.message,
                                },
                                error_context=None,
                            )
                # Send back to coding
                if self.retry_policy.should_retry(story.id, StoryState.TESTING, None):
                    self.state_machine.transition(
                        story,
                        StoryState.CODING,
                        reason=f"Quality gates failed: {', '.join(failed_gates)}",
                    )
                else:
                    self.state_machine.transition(
                        story,
                        StoryState.FAILED,
                        reason="Quality gates failed and retry limit exceeded",
                    )
            else:
                logger.info(
                    "quality_gates_passed",
                    story_id=story.id,
                    action=result.recommended_action,
                )

        except Exception as e:
            logger.warning("quality_gates_error", story_id=story.id, error=str(e))

        self._emit_event(story)

    async def _generate_evidence_bundle(self, story: Story) -> None:
        """Generate evidence bundle for the story's PR."""
        generator = getattr(self, "_evidence_generator", None)
        if not generator:
            return

        logger.info("evidence_bundle_generation_started", story_id=story.id)

        try:
            # Get security result if available
            security_result = None
            if story.iterations:
                security_result = getattr(story.iterations[-1], "security_scan_result", None)

            bundle = await generator.generate(story, security_result=security_result)

            # Store the bundle on the story for later use in PR creation
            if not hasattr(story, "_evidence_bundle"):
                object.__setattr__(story, "_evidence_bundle", None)
            story._evidence_bundle = bundle

            logger.info(
                "evidence_bundle_generated",
                story_id=story.id,
                all_gates_passed=bundle.all_gates_passed,
                action=bundle.recommended_action,
            )
        except Exception as e:
            logger.warning("evidence_bundle_error", story_id=story.id, error=str(e))

    async def _enrich_context_for_retry(self, story: Story, attempt_num: int) -> None:
        """Progressively enrich context on retries."""
        logger.info("enriching_context_for_retry", story_id=story.id, attempt=attempt_num)

        if self.context_manager:
            try:
                await self.context_manager.enrich_context(story, attempt_num)
            except Exception as e:
                logger.warning("context_enrichment_error", error=str(e))

    async def _create_test_error_context(
        self,
        story: Story,
        test_report: TestReport,
    ) -> ErrorContext:
        """Create error context from test failures."""
        error_msg = f"Tests failed: {test_report.failed} failures, {test_report.passed} passed"
        failure_details = ""
        if test_report.failures:
            failure_details = "\n".join(
                [f"  {f.suite}/{f.test}: {f.error}" for f in test_report.failures[:3]]
            )

        error_context = ErrorContext(
            attempt=len(story.iterations) + 1,
            error=ErrorDetail(
                type="TestFailure",
                message=error_msg,
                stack_trace=failure_details,
            ),
        )

        # Classify error
        if self.error_classifier:
            error_context = await self.error_classifier.classify(error_context.error)

        return error_context

    def _check_cost_budget(self, story: Story) -> bool:
        """Check if story has exceeded cost budget."""
        cost_remaining = story.constraints.cost_budget_usd - story.metadata.total_cost_usd
        if cost_remaining <= 0:
            logger.warning(
                "cost_budget_exceeded",
                story_id=story.id,
                budget=story.constraints.cost_budget_usd,
                used=story.metadata.total_cost_usd,
            )
            return True
        return False

    def get_story(self, story_id: str) -> Optional[Story]:
        """Get story by ID."""
        return self._stories.get(story_id)

    def get_all_stories(self) -> list[Story]:
        """Get all stories."""
        return list(self._stories.values())

    def get_state_history(self, story_id: str) -> list[tuple[StoryState, StoryState]]:
        """Get state transition history for a story."""
        transitions = self.state_machine.get_history(story_id)
        return [(t.from_state, t.to_state) for t in transitions]


__all__ = [
    "Orchestrator",
    "StateMachine",
    "RetryPolicy",
    "ConvergenceDetector",
    "ApprovalGateManager",
]
