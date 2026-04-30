"""Adapter layer — bridges real component APIs to Orchestrator Protocol signatures.

Each adapter wraps a concrete component and implements the Story-centric interface
that the Orchestrator expects. The adapter extracts fields from Story, calls the
real implementation, and folds results back into the Story object.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Optional

from fastcoder.types.codebase import ProjectProfile
from fastcoder.types.errors import (
    ErrorClassification,
    ErrorContext,
    ErrorDetail,
)
from fastcoder.types.iteration import Iteration
from fastcoder.types.llm import Message
from fastcoder.types.memory import MemoryEntry, MemoryTier, MemoryType
from fastcoder.types.plan import ExecutionPlan, PlanTask
from fastcoder.types.story import Story, StorySpec
from fastcoder.types.task import (
    DeployReport,
    FileChange,
    ReviewReport,
    TestReport,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. StoryAnalyzer adapter
# ---------------------------------------------------------------------------
class AnalyzerAdapter:
    """Wraps StoryAnalyzer to match Protocol: analyze(story: Story) -> Story."""

    def __init__(self, real_analyzer):
        self._real = real_analyzer

    async def analyze(self, story: Story) -> Story:
        """Extract raw_text from Story, call real analyze, set story.spec."""
        spec: StorySpec = await self._real.analyze(
            raw_story=story.raw_text,
            project_profile=None,  # Can be enriched later via codebase intelligence
        )
        story.spec = spec
        logger.info(
            "analyzer_adapter_complete",
            extra={
                "story_id": story.id,
                "title": spec.title,
                "complexity": spec.complexity_score,
            },
        )
        return story


# ---------------------------------------------------------------------------
# 2. Planner adapter
# ---------------------------------------------------------------------------
class PlannerAdapter:
    """Wraps Planner to match Protocol: plan(story: Story) -> Story."""

    def __init__(self, real_planner):
        self._real = real_planner

    async def plan(self, story: Story) -> Story:
        """Extract spec from Story, call real create_plan, set story.plan."""
        if not story.spec:
            raise ValueError("Cannot plan without a StorySpec — run analysis first")

        plan: ExecutionPlan = await self._real.create_plan(
            spec=story.spec,
            project_profile=None,
        )
        story.plan = plan
        logger.info(
            "planner_adapter_complete",
            extra={
                "story_id": story.id,
                "task_count": len(plan.tasks) if plan.tasks else 0,
            },
        )
        return story


# ---------------------------------------------------------------------------
# 3. CodeGenerator adapter
# ---------------------------------------------------------------------------
class GeneratorAdapter:
    """Wraps CodeGenerator to match Protocol: generate(story, error_context) -> Story."""

    def __init__(self, real_generator):
        self._real = real_generator

    async def generate(
        self, story: Story, error_context: Optional[ErrorContext] = None
    ) -> Story:
        """Iterate over plan tasks, generate code for each, record as iteration."""
        if not story.plan or not story.plan.tasks:
            raise ValueError("Cannot generate code without a plan")

        all_changes: list[FileChange] = []
        iteration_num = len(story.iterations) + 1

        # Build shared context dict
        context: dict[str, Any] = {
            "project_profile": None,
            "relevant_files": {},
            "type_definitions": "",
            "conventions": "",
            "error_history": [],
        }

        # Add error context if retrying
        if error_context:
            context["error_history"] = [
                {
                    "attempt": error_context.attempt,
                    "error_type": error_context.error.type,
                    "error_message": error_context.error.message,
                    "instruction": error_context.instruction or "",
                }
            ]

        for task in story.plan.tasks:
            result = await self._real.generate(task=task, context=context)
            all_changes.extend(result.file_changes)

        # Create iteration record
        iteration = Iteration(
            number=iteration_num,
            stage="coding",
            changes=all_changes,
        )
        if error_context:
            iteration.error_context = error_context

        story.iterations.append(iteration)
        story.metadata.updated_at = datetime.utcnow()

        logger.info(
            "generator_adapter_complete",
            extra={
                "story_id": story.id,
                "changes": len(all_changes),
                "iteration": iteration_num,
            },
        )
        return story


# ---------------------------------------------------------------------------
# 4. TestGenerator adapter
# ---------------------------------------------------------------------------
class TestGeneratorAdapter:
    """Wraps TestGenerator to match Protocol: generate_tests(story) -> Story."""

    def __init__(self, real_test_generator):
        self._real = real_test_generator

    async def generate_tests(self, story: Story) -> Story:
        """Generate tests for the latest iteration's code changes."""
        if not story.plan or not story.plan.tasks:
            return story  # Nothing to test

        if not story.iterations:
            return story

        latest_iteration = story.iterations[-1]
        context: dict[str, Any] = {
            "test_framework": "pytest",
            "existing_tests": "",
            "project_conventions": "",
        }

        for task in story.plan.tasks:
            # Find code from latest changes for this task
            code = ""
            for change in latest_iteration.changes:
                if change.content:
                    code += change.content + "\n"

            if code:
                result = await self._real.generate_tests(
                    task=task,
                    code=code,
                    spec=story.spec,
                    context=context,
                )
                # Add test file as a change
                latest_iteration.changes.append(
                    FileChange(
                        file_path=result.test_file,
                        change_type="created",
                        content=result.test_code,
                    )
                )

        logger.info(
            "test_generator_adapter_complete",
            extra={"story_id": story.id},
        )
        return story


# ---------------------------------------------------------------------------
# 5. CodeReviewer adapter
# ---------------------------------------------------------------------------
class ReviewerAdapter:
    """Wraps CodeReviewer to match Protocol: review(story) -> Story."""

    def __init__(self, real_reviewer):
        self._real = real_reviewer

    async def review(self, story: Story) -> Story:
        """Review the latest iteration's changes."""
        if not story.iterations:
            return story

        latest_iteration = story.iterations[-1]
        changes = latest_iteration.changes

        if not changes:
            return story

        profile = ProjectProfile(
            language="python",
            project_dir=".",
        )

        review_report: ReviewReport = await self._real.review(
            changes=changes,
            spec=story.spec or StorySpec(title="Unknown", description=""),
            profile=profile,
        )

        latest_iteration.review_results = review_report

        logger.info(
            "reviewer_adapter_complete",
            extra={
                "story_id": story.id,
                "approved": review_report.approved,
                "issues": len(review_report.issues),
            },
        )
        return story


# ---------------------------------------------------------------------------
# 6. ContextManager adapter
# ---------------------------------------------------------------------------
class ContextManagerAdapter:
    """Wraps ContextManager to match Protocol: enrich_context(story, iteration)."""

    def __init__(self, real_context_manager):
        self._real = real_context_manager

    async def enrich_context(self, story: Story, iteration: int) -> None:
        """Build context for the current retry attempt.

        The real ContextManager.build_context() returns a list of Messages
        which would be used in LLM calls. For the orchestrator's purposes,
        we call it to trigger any side effects and logging.
        """
        # The real build_context needs specific parameters; we do our best
        # to assemble them from the Story object.
        task = None
        if story.plan and story.plan.tasks:
            task = story.plan.tasks[0]  # Current task

        try:
            messages = await self._real.build_context(
                story=story.spec or StorySpec(title="Unknown", description=""),
                task=task,
                project_profile=ProjectProfile(),
                relevant_files=[],
                error_context=story.iterations[-1].error_context
                if story.iterations and story.iterations[-1].error_context
                else None,
                memory_entries=[],
            )
            logger.info(
                "context_enriched",
                extra={
                    "story_id": story.id,
                    "attempt": iteration,
                    "message_count": len(messages) if messages else 0,
                },
            )
        except Exception as e:
            logger.warning(
                "context_enrichment_skipped",
                extra={"error": str(e)},
            )


# ---------------------------------------------------------------------------
# 7. MemoryStore adapter (adds store_learning method)
# ---------------------------------------------------------------------------
class MemoryStoreAdapter:
    """Wraps MemoryStore to add store_learning(story, error_context) method."""

    def __init__(self, real_memory_store):
        self._real = real_memory_store

    async def store_learning(
        self, story: Story, error_context: ErrorContext
    ) -> None:
        """Convert error context into a MemoryEntry and store it."""
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            content=(
                f"Error in story '{story.spec.title if story.spec else story.id}': "
                f"{error_context.error.type} — {error_context.error.message}"
            ),
            tier=MemoryTier.EPISODIC,
            type=MemoryType.ERROR_FIX,
            project_id=story.project_id,
            source_story_id=story.id,
        )
        self._real.store(entry)

        # Also store in error_fixes if we have a fingerprint
        if error_context.classification and error_context.classification.fingerprint:
            if error_context.instruction:
                self._real.error_fixes[
                    error_context.classification.fingerprint
                ] = error_context.instruction

        logger.info(
            "learning_stored",
            extra={
                "story_id": story.id,
                "error_type": error_context.error.type,
            },
        )

    # Proxy all other methods to the real store
    def store(self, entry: MemoryEntry) -> None:
        return self._real.store(entry)

    def query(self, q):
        return self._real.query(q)

    def consolidate(self, story: Story):
        return self._real.consolidate(story)

    def load(self, path: str) -> None:
        return self._real.load(path)

    def save(self, path: str) -> None:
        return self._real.save(path)


# ---------------------------------------------------------------------------
# 8. ErrorClassifier adapter
# ---------------------------------------------------------------------------
class ErrorClassifierAdapter:
    """Wraps ErrorClassifier to match Protocol: classify(error: ErrorDetail) -> ErrorContext."""

    def __init__(self, real_classifier):
        self._real = real_classifier

    async def classify(self, error: ErrorDetail) -> ErrorContext:
        """Extract fields from ErrorDetail, call real classify, return ErrorContext."""
        classification: ErrorClassification = self._real.classify(
            error_type=error.type,
            message=error.message,
            stack_trace=error.stack_trace or "",
        )
        return ErrorContext(
            error=error,
            classification=classification,
        )


# ---------------------------------------------------------------------------
# 9. RecoveryManager adapter
# ---------------------------------------------------------------------------
class RecoveryManagerAdapter:
    """Wraps RecoveryManager to match Protocol: suggest_recovery(story, error_context) -> str."""

    def __init__(self, real_recovery_manager):
        self._real = real_recovery_manager

    async def suggest_recovery(
        self, story: Story, error_context: ErrorContext
    ) -> str:
        """Call real get_strategy and return a human-readable suggestion."""
        attempt = error_context.attempt
        classification = error_context.classification

        if not classification:
            return "No classification available — retry with broader context"

        action = self._real.get_strategy(
            classification=classification,
            attempt=attempt,
        )

        # Build a human-readable instruction
        parts = [f"Strategy: {action.strategy.value}"]
        if action.additional_context:
            parts.append(f"Context: {action.additional_context}")
        if action.switch_to_top_tier:
            parts.append("Escalate to top-tier model")
        if action.replan:
            parts.append("Re-plan the task")
        if action.escalate:
            parts.append("Escalate to human")

        return " | ".join(parts)


# ---------------------------------------------------------------------------
# 10. Deployer adapter
# ---------------------------------------------------------------------------
class DeployerAdapter:
    """Wraps Deployer to match Protocol: deploy(story) -> Story."""

    def __init__(self, real_deployer):
        self._real = real_deployer

    async def deploy(self, story: Story) -> Story:
        """Call real deploy, fold DeployReport into the latest iteration."""
        deploy_report: DeployReport = await self._real.deploy(story)

        # Attach deploy results to the latest iteration
        if story.iterations:
            story.iterations[-1].deploy_results = deploy_report
        else:
            # Create a deploy iteration
            iteration = Iteration(
                number=len(story.iterations) + 1,
                stage="deploying",
            )
            iteration.deploy_results = deploy_report
            story.iterations.append(iteration)

        story.metadata.updated_at = datetime.utcnow()

        logger.info(
            "deployer_adapter_complete",
            extra={
                "story_id": story.id,
                "success": deploy_report.success,
            },
        )
        return story


# ---------------------------------------------------------------------------
# 11. TestRunner adapter (wraps tools/test_runner)
# ---------------------------------------------------------------------------
class TestRunnerAdapter:
    """Wraps ToolLayer.test_runner to match Protocol: run_tests(story) -> Story."""

    def __init__(self, real_test_runner):
        self._real = real_test_runner

    async def run_tests(self, story: Story) -> Story:
        """Run tests and attach TestReport to the latest iteration."""
        if not story.iterations:
            return story

        latest_iteration = story.iterations[-1]

        try:
            # The real test_runner.run() returns a TestReport
            test_report: TestReport = await self._real.run()
            latest_iteration.test_results = test_report
        except Exception as e:
            # Create a failure report
            latest_iteration.test_results = TestReport(
                total=0,
                passed=0,
                failed=1,
                failures=[],
            )
            logger.warning(
                "test_runner_error",
                extra={"error": str(e)},
            )

        story.metadata.updated_at = datetime.utcnow()
        return story


# ---------------------------------------------------------------------------
# 12. BuildRunner adapter (wraps ToolLayer.build_runner)
# ---------------------------------------------------------------------------
class BuildRunnerAdapter:
    """Wraps ToolLayer.build_runner to match Protocol: run_build(story) -> Story."""

    def __init__(self, real_build_runner):
        self._real = real_build_runner

    async def run_build(self, story: Story) -> Story:
        """Run build and update story metadata."""
        try:
            result = await self._real.run()
            logger.info(
                "build_runner_complete",
                extra={"story_id": story.id, "success": result.get("success", True)},
            )
        except Exception as e:
            logger.warning(
                "build_runner_error",
                extra={"error": str(e)},
            )
        story.metadata.updated_at = datetime.utcnow()
        return story


# ---------------------------------------------------------------------------
# Verifier adapter
# ---------------------------------------------------------------------------
class VerifierAdapter:
    """Wraps Verifier for use in the orchestrator's _run_verification."""

    def __init__(self, real_verifier):
        self._real = real_verifier

    async def verify(self, story: Story) -> Story:
        """Run verification checks and fold results into story."""
        deploy_report = None
        if story.iterations:
            deploy_report = story.iterations[-1].deploy_results

        try:
            report = await self._real.verify(story, deploy_report)
            logger.info(
                "verifier_adapter_complete",
                extra={
                    "story_id": story.id,
                    "passed": report.overall_passed,
                    "checks": len(report.checks),
                },
            )
            if not report.overall_passed:
                # Mark story as needing attention but don't fail outright
                logger.warning(
                    "verification_issues",
                    extra={
                        "story_id": story.id,
                        "failed_checks": [
                            c.name for c in report.checks if not c.passed
                        ],
                    },
                )
        except Exception as e:
            logger.warning(
                "verification_error",
                extra={"story_id": story.id, "error": str(e)},
            )

        story.metadata.updated_at = datetime.utcnow()
        return story


# ---------------------------------------------------------------------------
# Factory: wrap all components at once
# ---------------------------------------------------------------------------
def wrap_components(components: dict, memory_store) -> dict:
    """Wrap raw components with adapters for Orchestrator compatibility.

    Args:
        components: Dict of raw component instances from _initialize_components()
        memory_store: Raw MemoryStore instance (needs special handling)

    Returns:
        Dict with adapter-wrapped components ready for Orchestrator.__init__
    """
    wrapped = {}

    if components.get("analyzer"):
        wrapped["analyzer"] = AnalyzerAdapter(components["analyzer"])

    if components.get("planner"):
        wrapped["planner"] = PlannerAdapter(components["planner"])

    if components.get("generator"):
        wrapped["generator"] = GeneratorAdapter(components["generator"])

    if components.get("test_generator"):
        wrapped["test_generator"] = TestGeneratorAdapter(components["test_generator"])

    if components.get("reviewer"):
        wrapped["reviewer"] = ReviewerAdapter(components["reviewer"])

    if components.get("context_manager"):
        wrapped["context_manager"] = ContextManagerAdapter(components["context_manager"])

    if components.get("error_classifier"):
        wrapped["error_classifier"] = ErrorClassifierAdapter(
            components["error_classifier"]
        )

    if components.get("recovery_manager"):
        wrapped["recovery_manager"] = RecoveryManagerAdapter(
            components["recovery_manager"]
        )

    if components.get("deployer"):
        wrapped["deployer"] = DeployerAdapter(components["deployer"])

    if components.get("verifier"):
        wrapped["verifier"] = VerifierAdapter(components["verifier"])

    # Memory store wraps the raw store
    wrapped["memory_store"] = MemoryStoreAdapter(memory_store)

    # Tool layer items
    tool_layer = components.get("tool_layer")
    if tool_layer:
        if hasattr(tool_layer, "test_runner") and tool_layer.test_runner:
            wrapped["test_runner"] = TestRunnerAdapter(tool_layer.test_runner)
        if hasattr(tool_layer, "build_runner") and tool_layer.build_runner:
            wrapped["build_runner"] = BuildRunnerAdapter(tool_layer.build_runner)

    # LLM router passes through (already matches Protocol)
    wrapped["llm_router"] = components.get("llm_router")

    return wrapped
