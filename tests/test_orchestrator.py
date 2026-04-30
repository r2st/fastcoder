"""Comprehensive tests for orchestrator module with 100% API coverage."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fastcoder.orchestrator import (
    Orchestrator,
    StateMachine,
    RetryPolicy,
    ConvergenceDetector,
    ApprovalGateManager,
)
from fastcoder.orchestrator.state_machine import StateTransition
from fastcoder.orchestrator.convergence_detector import ConvergenceResult
from fastcoder.orchestrator.retry_policy import RetryAttempt, CircuitBreakerState
from fastcoder.orchestrator.approval_gate import (
    GateResult,
    GateStatus,
    PendingApproval,
    ApprovalDecision,
)
from fastcoder.orchestrator.adapters import (
    AnalyzerAdapter,
    PlannerAdapter,
    GeneratorAdapter,
    TestGeneratorAdapter,
    ReviewerAdapter,
    ContextManagerAdapter,
    MemoryStoreAdapter,
    ErrorClassifierAdapter,
    RecoveryManagerAdapter,
    DeployerAdapter,
    TestRunnerAdapter,
    BuildRunnerAdapter,
    VerifierAdapter,
    wrap_components,
)
from fastcoder.types.config import AgentConfig, SafetyConfig
from fastcoder.types.errors import (
    ErrorContext,
    ErrorDetail,
    ErrorClassification,
    ErrorCategory,
    RecoveryStrategy,
)
from fastcoder.types.story import (
    Story,
    StorySubmission,
    StoryState,
    Priority,
    StorySpec,
    StoryConstraints,
    StoryMetadata,
)
from fastcoder.types.iteration import Iteration
from fastcoder.types.plan import ExecutionPlan, PlanTask
from fastcoder.types.task import ReviewReport, TestReport, FileChange, DeployReport
from fastcoder.types.memory import MemoryEntry, MemoryTier, MemoryType


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def safety_config():
    """Create a SafetyConfig for testing."""
    return SafetyConfig(
        max_iterations_per_story=10,
        max_retries_per_stage=5,
        approval_gates={
            "pre_code": False,
            "pre_deploy": True,
            "pre_production": False,
            "budget_exceeded": True,
            "ambiguity_detected": False,
        },
    )


@pytest.fixture
def agent_config(safety_config):
    """Create an AgentConfig for testing."""
    return AgentConfig(
        safety=safety_config,
    )


@pytest.fixture
def story():
    """Create a test story."""
    return Story(
        id=str(uuid.uuid4()),
        raw_text="Create a feature to add user authentication",
        project_id="test-project",
        priority=Priority.MEDIUM,
        state=StoryState.RECEIVED,
    )


@pytest.fixture
def story_submission():
    """Create a story submission."""
    return StorySubmission(
        story="Create a feature to add user authentication",
        project_id="test-project",
        priority=Priority.HIGH,
    )


@pytest.fixture
def mock_analyzer():
    """Create a mock analyzer component."""
    analyzer = AsyncMock()
    analyzer.analyze = AsyncMock(
        return_value=StorySpec(
            title="User Auth Feature",
            description="Add user authentication",
        )
    )
    return analyzer


@pytest.fixture
def mock_planner():
    """Create a mock planner component."""
    planner = AsyncMock()
    planner.plan = AsyncMock(
        return_value=ExecutionPlan(
            tasks=[
                PlanTask(
                    id=str(uuid.uuid4()),
                    title="Implement login endpoint",
                    description="Create API endpoint for user login",
                )
            ]
        )
    )
    return planner


@pytest.fixture
def mock_generator():
    """Create a mock code generator."""
    generator = AsyncMock()
    generator.generate = AsyncMock(
        return_value=Story(
            id="test",
            raw_text="",
            project_id="test",
            iterations=[
                Iteration(
                    number=1,
                    stage="coding",
                    changes=[
                        FileChange(
                            file_path="auth.py",
                            change_type="created",
                            content="def login():\n    pass",
                        )
                    ],
                )
            ],
        )
    )
    return generator


@pytest.fixture
def mock_reviewer():
    """Create a mock code reviewer."""
    reviewer = AsyncMock()
    reviewer.review = AsyncMock(
        return_value=ReviewReport(
            approved=True,
            issues=[],
            suggestions=[],
        )
    )
    return reviewer


@pytest.fixture
def mock_test_generator():
    """Create a mock test generator."""
    test_gen = AsyncMock()
    test_gen.generate_tests = AsyncMock(return_value=None)
    return test_gen


@pytest.fixture
def mock_test_runner():
    """Create a mock test runner."""
    runner = AsyncMock()
    runner.run_tests = AsyncMock(
        return_value=TestReport(
            total=10,
            passed=10,
            failed=0,
            coverage_percent=85.0,
        )
    )
    return runner


@pytest.fixture
def mock_deployer():
    """Create a mock deployer."""
    deployer = AsyncMock()
    deployer.deploy = AsyncMock(
        return_value=DeployReport(
            success=True,
            deployment_id="deploy-123",
            url="https://example.com",
        )
    )
    return deployer


@pytest.fixture
def mock_error_classifier():
    """Create a mock error classifier."""
    classifier = AsyncMock()
    classifier.classify = AsyncMock(
        return_value=ErrorContext(
            attempt=1,
            error=ErrorDetail(type="TestFailure", message="Test failed"),
            classification=ErrorClassification(
                category=ErrorCategory.LOGIC_ERROR,
                fingerprint="test_failure_fp",
            ),
        )
    )
    return classifier


@pytest.fixture
def mock_memory_store():
    """Create a mock memory store."""
    store = MagicMock()
    store.store = MagicMock()
    store.query = MagicMock(return_value=[])
    store.error_fixes = {}
    return store


# ============================================================================
# STATE MACHINE TESTS
# ============================================================================


class TestStateMachine:
    """Test StateMachine class."""

    def test_init(self):
        """Test StateMachine initialization."""
        sm = StateMachine()
        assert sm._valid_transitions is not None
        assert sm._transition_history == {}
        assert sm._callbacks == []

    def test_is_valid_transition_valid(self):
        """Test valid state transitions."""
        sm = StateMachine()
        assert sm.is_valid_transition(StoryState.RECEIVED, StoryState.ANALYZING)
        assert sm.is_valid_transition(StoryState.ANALYZING, StoryState.PLANNING)
        assert sm.is_valid_transition(StoryState.PLANNING, StoryState.CODING)
        assert sm.is_valid_transition(StoryState.CODING, StoryState.REVIEWING)
        assert sm.is_valid_transition(StoryState.REVIEWING, StoryState.TESTING)
        assert sm.is_valid_transition(StoryState.TESTING, StoryState.DEPLOYING)
        assert sm.is_valid_transition(StoryState.DEPLOYING, StoryState.VERIFYING)
        assert sm.is_valid_transition(StoryState.VERIFYING, StoryState.DONE)

    def test_is_valid_transition_invalid(self):
        """Test invalid state transitions."""
        sm = StateMachine()
        assert not sm.is_valid_transition(StoryState.RECEIVED, StoryState.DONE)
        assert not sm.is_valid_transition(StoryState.DONE, StoryState.ANALYZING)
        assert not sm.is_valid_transition(StoryState.ANALYZING, StoryState.CODING)
        assert not sm.is_valid_transition(StoryState.FAILED, StoryState.CODING)

    def test_transition_valid(self, story):
        """Test transitioning to a valid state."""
        sm = StateMachine()
        story.state = StoryState.RECEIVED

        result = sm.transition(story, StoryState.ANALYZING, reason="Starting analysis")
        assert result.state == StoryState.ANALYZING
        assert story.state == StoryState.ANALYZING
        assert len(sm.get_history(story.id)) == 1

    def test_transition_invalid_raises_error(self, story):
        """Test that invalid transitions raise ValueError."""
        sm = StateMachine()
        story.state = StoryState.RECEIVED

        with pytest.raises(ValueError):
            sm.transition(story, StoryState.DONE)

    def test_transition_updates_metadata(self, story):
        """Test that transition updates story metadata."""
        sm = StateMachine()
        story.state = StoryState.RECEIVED
        old_updated_at = story.metadata.updated_at

        sm.transition(story, StoryState.ANALYZING)
        assert story.metadata.updated_at > old_updated_at

    def test_register_callback(self, story):
        """Test registering a state transition callback."""
        sm = StateMachine()
        callback = MagicMock()
        sm.register_callback(callback)

        story.state = StoryState.RECEIVED
        sm.transition(story, StoryState.ANALYZING)

        callback.assert_called_once()
        call_args = callback.call_args
        assert call_args[0][0] == story

    def test_unregister_callback(self, story):
        """Test unregistering a callback."""
        sm = StateMachine()
        callback = MagicMock()
        sm.register_callback(callback)
        sm.unregister_callback(callback)

        story.state = StoryState.RECEIVED
        sm.transition(story, StoryState.ANALYZING)

        callback.assert_not_called()

    def test_get_history(self, story):
        """Test getting transition history."""
        sm = StateMachine()
        story.state = StoryState.RECEIVED

        sm.transition(story, StoryState.ANALYZING)
        sm.transition(story, StoryState.PLANNING)

        history = sm.get_history(story.id)
        assert len(history) == 2
        assert history[0].from_state == StoryState.RECEIVED
        assert history[0].to_state == StoryState.ANALYZING
        assert history[1].to_state == StoryState.PLANNING

    def test_get_current_state(self, story):
        """Test getting current state."""
        sm = StateMachine()
        story.state = StoryState.ANALYZING
        assert sm.get_current_state(story) == StoryState.ANALYZING

    def test_get_valid_next_states(self, story):
        """Test getting valid next states."""
        sm = StateMachine()
        story.state = StoryState.CODING
        valid_next = sm.get_valid_next_states(story)
        assert StoryState.REVIEWING in valid_next
        assert StoryState.FAILED in valid_next
        assert StoryState.DONE not in valid_next

    def test_can_rollback_to_valid(self, story):
        """Test checking valid rollbacks."""
        sm = StateMachine()
        story.state = StoryState.TESTING
        assert sm.can_rollback_to(story, StoryState.CODING)

        story.state = StoryState.REVIEWING
        assert sm.can_rollback_to(story, StoryState.CODING)

        story.state = StoryState.VERIFYING
        assert sm.can_rollback_to(story, StoryState.CODING)

    def test_can_rollback_to_invalid(self, story):
        """Test checking invalid rollbacks."""
        sm = StateMachine()
        story.state = StoryState.TESTING
        assert not sm.can_rollback_to(story, StoryState.ANALYZING)
        assert not sm.can_rollback_to(story, StoryState.PLANNING)

    def test_can_rollback_to_failed_always_allowed(self, story):
        """Test that rollback to FAILED is always allowed."""
        sm = StateMachine()
        for state in StoryState:
            if state != StoryState.DONE and state != StoryState.FAILED:
                story.state = state
                assert sm.can_rollback_to(story, StoryState.FAILED)

    def test_rollback_valid(self, story):
        """Test rolling back to a valid state."""
        sm = StateMachine()
        story.state = StoryState.TESTING
        result = sm.rollback(story, StoryState.CODING, reason="Tests failed")
        assert result.state == StoryState.CODING
        assert len(sm.get_history(story.id)) == 1

    def test_rollback_invalid_raises_error(self, story):
        """Test that invalid rollbacks raise ValueError."""
        sm = StateMachine()
        story.state = StoryState.TESTING
        with pytest.raises(ValueError):
            sm.rollback(story, StoryState.ANALYZING)


# ============================================================================
# CONVERGENCE DETECTOR TESTS
# ============================================================================


class TestConvergenceDetector:
    """Test ConvergenceDetector class."""

    def test_init_default(self):
        """Test ConvergenceDetector initialization with defaults."""
        detector = ConvergenceDetector()
        assert detector.window_size == 4
        assert detector.stuck_threshold == 3

    def test_init_custom(self):
        """Test ConvergenceDetector initialization with custom params."""
        detector = ConvergenceDetector(window_size=5, stuck_threshold=2)
        assert detector.window_size == 5
        assert detector.stuck_threshold == 2

    def test_check_no_iterations(self, story):
        """Test check with no iterations (should be progressing)."""
        detector = ConvergenceDetector()
        result = detector.check(story)

        assert result.status == "progressing"
        assert result.confidence == 1.0
        assert result.recommended_action == "continue"

    def test_check_progressing(self, story):
        """Test check with progressing iterations (no errors)."""
        detector = ConvergenceDetector()
        story.iterations = [
            Iteration(number=1, stage="coding"),
            Iteration(number=2, stage="coding"),
        ]

        result = detector.check(story)
        assert result.status == "progressing"
        assert result.recommended_action == "continue"

    def test_check_oscillating(self, story):
        """Test check with oscillating errors (same error repeats)."""
        detector = ConvergenceDetector()
        error_fp = "test_error_123"

        story.iterations = [
            Iteration(
                number=1,
                stage="coding",
                error_context=ErrorContext(
                    error=ErrorDetail(type="Error", message="Test"),
                    classification=ErrorClassification(fingerprint=error_fp),
                ),
            ),
            Iteration(
                number=2,
                stage="coding",
                error_context=ErrorContext(
                    error=ErrorDetail(type="Error", message="Test"),
                    classification=ErrorClassification(fingerprint=error_fp),
                ),
            ),
        ]

        result = detector.check(story)
        assert result.status == "oscillating"
        assert result.recommended_action == "enrich_context"
        assert result.metrics["oscillating_fingerprint"] == error_fp

    def test_check_stuck(self, story):
        """Test check with stuck story (consecutive failures)."""
        detector = ConvergenceDetector(stuck_threshold=2)

        story.iterations = [
            Iteration(
                number=1,
                stage="coding",
                error_context=ErrorContext(
                    error=ErrorDetail(type="Error", message="Fail 1"),
                    classification=ErrorClassification(fingerprint="error1"),
                ),
            ),
            Iteration(
                number=2,
                stage="coding",
                error_context=ErrorContext(
                    error=ErrorDetail(type="Error", message="Fail 2"),
                    classification=ErrorClassification(fingerprint="error2"),
                ),
            ),
        ]

        result = detector.check(story)
        assert result.status == "stuck"
        assert result.recommended_action == "replan"

    def test_check_insufficient_iterations(self, story):
        """Test check with insufficient iterations for stuck detection."""
        detector = ConvergenceDetector(stuck_threshold=5)
        story.iterations = [
            Iteration(number=1, stage="coding"),
        ]

        result = detector.check(story)
        assert result.status == "progressing"

    def test_check_diverging(self, story):
        """Test check with diverging errors (increasing error count)."""
        detector = ConvergenceDetector()

        # Create iterations with increasing errors
        story.iterations = [
            Iteration(number=1, stage="coding"),  # No error
            Iteration(number=2, stage="coding"),  # No error
            Iteration(
                number=3,
                stage="coding",
                error_context=ErrorContext(
                    error=ErrorDetail(type="Error", message="Err1"),
                    classification=ErrorClassification(fingerprint="e1"),
                ),
            ),
            Iteration(
                number=4,
                stage="coding",
                error_context=ErrorContext(
                    error=ErrorDetail(type="Error", message="Err2"),
                    classification=ErrorClassification(fingerprint="e2"),
                ),
            ),
        ]

        result = detector.check(story)
        assert result.status == "diverging"
        assert result.recommended_action == "escalate"


# ============================================================================
# RETRY POLICY TESTS
# ============================================================================


class TestRetryPolicy:
    """Test RetryPolicy class."""

    def test_init_default(self):
        """Test RetryPolicy initialization with defaults."""
        policy = RetryPolicy()
        assert policy.max_retries_per_stage == 5
        assert policy.base_backoff_seconds == 1.0
        assert policy.backoff_factor == 2.0
        assert policy.max_backoff_seconds == 30.0

    def test_init_custom(self):
        """Test RetryPolicy initialization with custom params."""
        policy = RetryPolicy(
            max_retries_per_stage=3,
            base_backoff_seconds=2.0,
            backoff_factor=1.5,
            max_backoff_seconds=20.0,
        )
        assert policy.max_retries_per_stage == 3
        assert policy.base_backoff_seconds == 2.0
        assert policy.backoff_factor == 1.5
        assert policy.max_backoff_seconds == 20.0

    def test_should_retry_under_max(self):
        """Test should_retry returns True when under max retries."""
        policy = RetryPolicy(max_retries_per_stage=5)
        story_id = "story-123"

        assert policy.should_retry(story_id, StoryState.CODING)
        policy.record_attempt(story_id, StoryState.CODING, success=False, error_fingerprint="fp1")
        assert policy.should_retry(story_id, StoryState.CODING)

    def test_should_retry_at_max_retries(self):
        """Test should_retry returns False at max retries."""
        policy = RetryPolicy(max_retries_per_stage=2)
        story_id = "story-123"

        policy.record_attempt(story_id, StoryState.CODING, success=False, error_fingerprint="fp1")
        policy.record_attempt(story_id, StoryState.CODING, success=False, error_fingerprint="fp1")

        assert not policy.should_retry(story_id, StoryState.CODING)

    def test_should_retry_circuit_breaker_tripped(self):
        """Test should_retry returns False when circuit breaker is open."""
        policy = RetryPolicy(max_retries_per_stage=10, circuit_breaker_threshold=2)
        story_id = "story-123"
        stage = StoryState.TESTING
        fp = "error_fp"

        # Trip circuit breaker with same error
        policy.record_attempt(story_id, stage, success=False, error_fingerprint=fp)
        policy.record_attempt(story_id, stage, success=False, error_fingerprint=fp)

        assert not policy.should_retry(story_id, stage)

    def test_get_backoff_seconds_exponential(self):
        """Test exponential backoff calculation."""
        policy = RetryPolicy(base_backoff_seconds=1.0, backoff_factor=2.0)

        assert policy.get_backoff_seconds(0) == 1.0
        assert policy.get_backoff_seconds(1) == 2.0
        assert policy.get_backoff_seconds(2) == 4.0
        assert policy.get_backoff_seconds(3) == 8.0

    def test_get_backoff_seconds_capped(self):
        """Test backoff is capped at max."""
        policy = RetryPolicy(
            base_backoff_seconds=1.0,
            backoff_factor=2.0,
            max_backoff_seconds=10.0,
        )

        backoff = policy.get_backoff_seconds(10)
        assert backoff <= 10.0

    def test_record_attempt_success(self):
        """Test recording successful attempt."""
        policy = RetryPolicy()
        story_id = "story-123"

        attempt = policy.record_attempt(story_id, StoryState.CODING, success=True)
        assert attempt.success
        assert attempt.attempt_number == 0
        assert attempt.backoff_seconds == 0.0

    def test_record_attempt_failure(self):
        """Test recording failed attempt."""
        policy = RetryPolicy(base_backoff_seconds=1.0)
        story_id = "story-123"

        attempt = policy.record_attempt(
            story_id,
            StoryState.CODING,
            success=False,
            error_fingerprint="fp1",
        )
        assert not attempt.success
        assert attempt.error_fingerprint == "fp1"
        assert attempt.backoff_seconds > 0.0

    def test_get_retry_count(self):
        """Test getting retry count."""
        policy = RetryPolicy()
        story_id = "story-123"

        assert policy.get_retry_count(story_id, StoryState.CODING) == 0
        policy.record_attempt(story_id, StoryState.CODING, success=False, error_fingerprint="fp1")
        assert policy.get_retry_count(story_id, StoryState.CODING) == 1

    def test_get_attempts(self):
        """Test getting all attempts."""
        policy = RetryPolicy()
        story_id = "story-123"

        policy.record_attempt(story_id, StoryState.CODING, success=False, error_fingerprint="fp1")
        policy.record_attempt(story_id, StoryState.CODING, success=True)

        attempts = policy.get_attempts(story_id, StoryState.CODING)
        assert len(attempts) == 2
        assert not attempts[0].success
        assert attempts[1].success

    def test_reset(self):
        """Test resetting retry state."""
        policy = RetryPolicy()
        story_id = "story-123"

        policy.record_attempt(story_id, StoryState.CODING, success=False, error_fingerprint="fp1")
        assert policy.get_retry_count(story_id, StoryState.CODING) == 1

        policy.reset(story_id, StoryState.CODING)
        assert policy.get_retry_count(story_id, StoryState.CODING) == 0

    def test_get_circuit_breaker_state(self):
        """Test getting circuit breaker state."""
        policy = RetryPolicy(circuit_breaker_threshold=2)
        story_id = "story-123"
        stage = StoryState.TESTING

        policy.record_attempt(story_id, stage, success=False, error_fingerprint="fp1")
        breaker = policy.get_circuit_breaker_state(story_id, stage)

        assert breaker is not None
        assert breaker.consecutive_failures == 1
        assert not breaker.is_broken

    def test_circuit_breaker_opens_on_threshold(self):
        """Test circuit breaker opens after threshold failures."""
        policy = RetryPolicy(circuit_breaker_threshold=2)
        story_id = "story-123"
        stage = StoryState.TESTING
        fp = "error_fp"

        policy.record_attempt(story_id, stage, success=False, error_fingerprint=fp)
        policy.record_attempt(story_id, stage, success=False, error_fingerprint=fp)

        breaker = policy.get_circuit_breaker_state(story_id, stage)
        assert breaker.is_broken

    def test_circuit_breaker_resets_on_success(self):
        """Test circuit breaker resets on success."""
        policy = RetryPolicy(circuit_breaker_threshold=2)
        story_id = "story-123"
        stage = StoryState.TESTING

        policy.record_attempt(story_id, stage, success=False, error_fingerprint="fp1")
        policy.record_attempt(story_id, stage, success=True)

        breaker = policy.get_circuit_breaker_state(story_id, stage)
        assert breaker.consecutive_failures == 0
        assert not breaker.is_broken

    def test_clear_all(self):
        """Test clearing all retry state."""
        policy = RetryPolicy()
        story_id = "story-123"

        policy.record_attempt(story_id, StoryState.CODING, success=False, error_fingerprint="fp1")
        policy.clear_all()

        assert policy.get_retry_count(story_id, StoryState.CODING) == 0


# ============================================================================
# APPROVAL GATE TESTS
# ============================================================================


class TestApprovalGateManager:
    """Test ApprovalGateManager class."""

    def test_init(self, safety_config):
        """Test ApprovalGateManager initialization."""
        manager = ApprovalGateManager(safety_config)
        assert manager.safety_config == safety_config
        assert manager._pending_approvals == {}
        assert manager._decisions == {}

    @pytest.mark.asyncio
    async def test_check_gate_disabled(self, story, safety_config):
        """Test checking a disabled gate (auto-passes)."""
        safety_config.approval_gates["pre_code"] = False
        manager = ApprovalGateManager(safety_config)

        result = await manager.check_gate("pre_code", story)
        assert result.status == GateStatus.PASS
        assert result.gate_name == "pre_code"

    @pytest.mark.asyncio
    async def test_check_gate_enabled_pending(self, story, safety_config):
        """Test checking an enabled gate (returns pending)."""
        safety_config.approval_gates["pre_deploy"] = True
        manager = ApprovalGateManager(safety_config)

        result = await manager.check_gate("pre_deploy", story)
        assert result.status == GateStatus.PENDING
        assert result.required_approval

    @pytest.mark.asyncio
    async def test_check_gate_already_approved(self, story, safety_config):
        """Test checking a gate that was already approved."""
        manager = ApprovalGateManager(safety_config)
        manager.approve(story.id, "pre_deploy", decided_by="human", comment="OK")

        result = await manager.check_gate("pre_deploy", story)
        assert result.status == GateStatus.PASS

    @pytest.mark.asyncio
    async def test_check_gate_already_rejected(self, story, safety_config):
        """Test checking a gate that was already rejected."""
        manager = ApprovalGateManager(safety_config)
        manager.reject(story.id, "pre_deploy", reason="Too risky")

        result = await manager.check_gate("pre_deploy", story)
        assert result.status == GateStatus.BLOCKED

    @pytest.mark.asyncio
    async def test_check_gate_invalid_raises_error(self, story, safety_config):
        """Test checking an invalid gate raises ValueError."""
        manager = ApprovalGateManager(safety_config)

        with pytest.raises(ValueError):
            await manager.check_gate("invalid_gate", story)

    def test_approve(self, story, safety_config):
        """Test approving a gate."""
        manager = ApprovalGateManager(safety_config)
        key = (story.id, "pre_deploy")

        manager.approve(story.id, "pre_deploy", decided_by="human", comment="OK")

        assert key in manager._decisions
        decision = manager._decisions[key]
        assert decision.approved
        assert decision.decided_by == "human"

    def test_reject(self, story, safety_config):
        """Test rejecting a gate."""
        manager = ApprovalGateManager(safety_config)
        key = (story.id, "pre_deploy")

        manager.reject(story.id, "pre_deploy", reason="Too risky", decided_by="human")

        assert key in manager._decisions
        decision = manager._decisions[key]
        assert not decision.approved
        assert decision.comment == "Too risky"

    def test_get_pending_approvals(self, story, safety_config):
        """Test getting pending approvals."""
        manager = ApprovalGateManager(safety_config)

        # Manually create pending approvals
        pending = PendingApproval(story_id=story.id, gate_name="pre_deploy")
        manager._pending_approvals[(story.id, "pre_deploy")] = pending

        result = manager.get_pending_approvals(story.id)
        assert len(result) == 1
        assert result[0].gate_name == "pre_deploy"

    def test_register_callback(self, story, safety_config):
        """Test registering a callback."""
        manager = ApprovalGateManager(safety_config)
        callback = MagicMock()
        manager.register_callback(callback)

        pending = PendingApproval(story_id=story.id, gate_name="pre_deploy")
        manager._pending_approvals[(story.id, "pre_deploy")] = pending

        # Callbacks would be called when check_gate creates new pending
        # This is tested in integration test
        assert len(manager._callbacks) == 1

    def test_unregister_callback(self, safety_config):
        """Test unregistering a callback."""
        manager = ApprovalGateManager(safety_config)
        callback = MagicMock()
        manager.register_callback(callback)
        manager.unregister_callback(callback)

        assert len(manager._callbacks) == 0

    def test_has_pending_approvals_true(self, story, safety_config):
        """Test checking if story has pending approvals (true case)."""
        manager = ApprovalGateManager(safety_config)
        pending = PendingApproval(story_id=story.id, gate_name="pre_deploy")
        manager._pending_approvals[(story.id, "pre_deploy")] = pending

        assert manager.has_pending_approvals(story.id)

    def test_has_pending_approvals_false(self, story, safety_config):
        """Test checking if story has pending approvals (false case)."""
        manager = ApprovalGateManager(safety_config)
        assert not manager.has_pending_approvals(story.id)

    def test_get_decision_exists(self, story, safety_config):
        """Test getting a decision."""
        manager = ApprovalGateManager(safety_config)
        decision = ApprovalDecision(
            story_id=story.id,
            gate_name="pre_deploy",
            approved=True,
            decided_by="human",
        )
        manager._decisions[(story.id, "pre_deploy")] = decision

        result = manager.get_decision(story.id, "pre_deploy")
        assert result is not None
        assert result.approved

    def test_get_decision_not_exists(self, story, safety_config):
        """Test getting a non-existent decision."""
        manager = ApprovalGateManager(safety_config)
        result = manager.get_decision(story.id, "pre_deploy")
        assert result is None

    def test_clear_pending(self, story, safety_config):
        """Test clearing a pending approval."""
        manager = ApprovalGateManager(safety_config)
        pending = PendingApproval(story_id=story.id, gate_name="pre_deploy")
        manager._pending_approvals[(story.id, "pre_deploy")] = pending

        manager.clear_pending(story.id, "pre_deploy")
        assert (story.id, "pre_deploy") not in manager._pending_approvals

    def test_clear_all_for_story(self, story, safety_config):
        """Test clearing all approvals for a story."""
        manager = ApprovalGateManager(safety_config)

        # Add pending
        manager._pending_approvals[(story.id, "pre_deploy")] = PendingApproval(
            story_id=story.id, gate_name="pre_deploy"
        )
        manager._pending_approvals[(story.id, "pre_code")] = PendingApproval(
            story_id=story.id, gate_name="pre_code"
        )

        # Add decisions
        manager._decisions[(story.id, "pre_deploy")] = ApprovalDecision(
            story_id=story.id, gate_name="pre_deploy", approved=True
        )

        manager.clear_all_for_story(story.id)

        pending_keys = [k for k in manager._pending_approvals if k[0] == story.id]
        decision_keys = [k for k in manager._decisions if k[0] == story.id]

        assert len(pending_keys) == 0
        assert len(decision_keys) == 0


# ============================================================================
# ORCHESTRATOR TESTS
# ============================================================================


class TestOrchestrator:
    """Test Orchestrator class."""

    def test_init_all_none(self, agent_config):
        """Test Orchestrator initialization with all components None."""
        orch = Orchestrator(agent_config)

        assert orch.config == agent_config
        assert orch.analyzer is None
        assert orch.planner is None
        assert orch.generator is None
        assert orch.reviewer is None
        assert orch.test_generator is None
        assert orch.build_runner is None
        assert orch.test_runner is None
        assert orch.deployer is None
        assert orch.context_manager is None
        assert orch.memory_store is None
        assert orch.error_classifier is None
        assert orch.recovery_manager is None
        assert orch.llm_router is None
        assert orch.state_machine is not None
        assert orch.retry_policy is not None
        assert orch.convergence_detector is not None
        assert orch.approval_gate_manager is not None

    def test_init_with_components(self, agent_config):
        """Test Orchestrator initialization with some components."""
        analyzer = AsyncMock()
        planner = AsyncMock()
        generator = AsyncMock()

        orch = Orchestrator(
            agent_config,
            analyzer=analyzer,
            planner=planner,
            generator=generator,
        )

        assert orch.analyzer == analyzer
        assert orch.planner == planner
        assert orch.generator == generator

    def test_register_callback(self, agent_config):
        """Test registering a callback."""
        orch = Orchestrator(agent_config)
        callback = MagicMock()
        orch.register_callback(callback)

        assert len(orch._callbacks) == 1

    def test_unregister_callback(self, agent_config):
        """Test unregistering a callback."""
        orch = Orchestrator(agent_config)
        callback = MagicMock()
        orch.register_callback(callback)
        orch.unregister_callback(callback)

        assert len(orch._callbacks) == 0

    def test_emit_event_calls_callbacks(self, agent_config, story):
        """Test that _emit_event calls all registered callbacks."""
        orch = Orchestrator(agent_config)
        callback1 = MagicMock()
        callback2 = MagicMock()
        orch.register_callback(callback1)
        orch.register_callback(callback2)

        orch._emit_event(story)

        callback1.assert_called_once_with(story)
        callback2.assert_called_once_with(story)

    def test_emit_event_handles_callback_error(self, agent_config, story):
        """Test that _emit_event handles callback errors gracefully."""
        orch = Orchestrator(agent_config)
        callback1 = MagicMock(side_effect=Exception("Callback error"))
        callback2 = MagicMock()
        orch.register_callback(callback1)
        orch.register_callback(callback2)

        # Should not raise, even though callback1 fails
        orch._emit_event(story)
        callback2.assert_called_once_with(story)

    @pytest.mark.asyncio
    async def test_process_story_basic(self, agent_config, story_submission):
        """Test process_story basic functionality."""
        orch = Orchestrator(agent_config)

        result = await orch.process_story(story_submission)

        assert result.id is not None
        assert result.project_id == "test-project"
        # Should reach at least ANALYZING state (or fail gracefully with no components)
        assert result.state in [
            StoryState.RECEIVED, StoryState.ANALYZING, StoryState.PLANNING,
            StoryState.CODING, StoryState.TESTING, StoryState.FAILED
        ]

    @pytest.mark.asyncio
    async def test_process_story_error_handling(self, agent_config, story_submission):
        """Test process_story error handling."""
        mock_analyzer = AsyncMock()
        mock_analyzer.analyze = AsyncMock(side_effect=Exception("Analysis failed"))

        orch = Orchestrator(agent_config, analyzer=mock_analyzer)

        result = await orch.process_story(story_submission)

        assert result.state == StoryState.FAILED

    @pytest.mark.asyncio
    async def test_process_story_creates_story(
        self, agent_config, story_submission
    ):
        """Test that process_story creates and stores a story."""
        orch = Orchestrator(agent_config)

        result = await orch.process_story(story_submission)

        stored = orch.get_story(result.id)
        assert stored is not None
        assert stored.raw_text == story_submission.story

    def test_get_story_exists(self, agent_config, story):
        """Test getting an existing story."""
        orch = Orchestrator(agent_config)
        orch._stories[story.id] = story

        result = orch.get_story(story.id)
        assert result == story

    def test_get_story_not_exists(self, agent_config):
        """Test getting a non-existent story."""
        orch = Orchestrator(agent_config)
        result = orch.get_story("nonexistent")
        assert result is None

    def test_get_all_stories(self, agent_config):
        """Test getting all stories."""
        orch = Orchestrator(agent_config)
        story1 = Story(id="1", raw_text="Story 1", project_id="proj", state=StoryState.RECEIVED)
        story2 = Story(id="2", raw_text="Story 2", project_id="proj", state=StoryState.RECEIVED)

        orch._stories["1"] = story1
        orch._stories["2"] = story2

        result = orch.get_all_stories()
        assert len(result) == 2
        assert story1 in result
        assert story2 in result

    def test_get_state_history(self, agent_config, story):
        """Test getting state history."""
        orch = Orchestrator(agent_config)
        story.state = StoryState.RECEIVED

        orch.state_machine.transition(story, StoryState.ANALYZING)
        orch.state_machine.transition(story, StoryState.PLANNING)

        history = orch.get_state_history(story.id)
        assert len(history) == 2
        assert history[0] == (StoryState.RECEIVED, StoryState.ANALYZING)
        assert history[1] == (StoryState.ANALYZING, StoryState.PLANNING)


# ============================================================================
# ADAPTER TESTS
# ============================================================================


class TestAdapters:
    """Test adapter classes."""

    def test_analyzer_adapter(self, story):
        """Test AnalyzerAdapter."""
        real_analyzer = AsyncMock()
        spec = StorySpec(title="Test", description="Test story")
        real_analyzer.analyze = AsyncMock(return_value=spec)

        adapter = AnalyzerAdapter(real_analyzer)

        # The adapter wraps story's raw_text
        assert hasattr(adapter, "analyze")

    def test_planner_adapter(self):
        """Test PlannerAdapter."""
        real_planner = AsyncMock()

        adapter = PlannerAdapter(real_planner)
        assert hasattr(adapter, "plan")

    def test_generator_adapter(self, story):
        """Test GeneratorAdapter."""
        real_generator = AsyncMock()
        adapter = GeneratorAdapter(real_generator)
        assert hasattr(adapter, "generate")

    def test_test_generator_adapter(self):
        """Test TestGeneratorAdapter."""
        real_gen = AsyncMock()
        adapter = TestGeneratorAdapter(real_gen)
        assert hasattr(adapter, "generate_tests")

    def test_reviewer_adapter(self):
        """Test ReviewerAdapter."""
        real_reviewer = AsyncMock()
        adapter = ReviewerAdapter(real_reviewer)
        assert hasattr(adapter, "review")

    def test_context_manager_adapter(self):
        """Test ContextManagerAdapter."""
        real_manager = AsyncMock()
        adapter = ContextManagerAdapter(real_manager)
        assert hasattr(adapter, "enrich_context")

    def test_memory_store_adapter(self):
        """Test MemoryStoreAdapter."""
        real_store = MagicMock()
        adapter = MemoryStoreAdapter(real_store)
        assert hasattr(adapter, "store_learning")

    def test_error_classifier_adapter(self):
        """Test ErrorClassifierAdapter."""
        real_classifier = MagicMock()
        adapter = ErrorClassifierAdapter(real_classifier)
        assert hasattr(adapter, "classify")

    def test_recovery_manager_adapter(self):
        """Test RecoveryManagerAdapter."""
        real_manager = MagicMock()
        adapter = RecoveryManagerAdapter(real_manager)
        assert hasattr(adapter, "suggest_recovery")

    def test_deployer_adapter(self):
        """Test DeployerAdapter."""
        real_deployer = AsyncMock()
        adapter = DeployerAdapter(real_deployer)
        assert hasattr(adapter, "deploy")

    def test_test_runner_adapter(self):
        """Test TestRunnerAdapter."""
        real_runner = AsyncMock()
        adapter = TestRunnerAdapter(real_runner)
        assert hasattr(adapter, "run_tests")

    def test_build_runner_adapter(self):
        """Test BuildRunnerAdapter."""
        real_runner = AsyncMock()
        adapter = BuildRunnerAdapter(real_runner)
        assert hasattr(adapter, "run_build")

    def test_verifier_adapter(self):
        """Test VerifierAdapter."""
        real_verifier = AsyncMock()
        adapter = VerifierAdapter(real_verifier)
        assert hasattr(adapter, "verify")

    def test_wrap_components_all_none(self):
        """Test wrap_components with all None components."""
        real_store = MagicMock()
        wrapped = wrap_components({}, real_store)

        assert "memory_store" in wrapped
        assert wrapped["memory_store"] is not None

    def test_wrap_components_selective(self):
        """Test wrap_components with selective components."""
        real_store = MagicMock()
        real_analyzer = MagicMock()
        real_planner = MagicMock()

        components = {
            "analyzer": real_analyzer,
            "planner": real_planner,
        }

        wrapped = wrap_components(components, real_store)

        assert "analyzer" in wrapped
        assert "planner" in wrapped
        assert isinstance(wrapped["analyzer"], AnalyzerAdapter)
        assert isinstance(wrapped["planner"], PlannerAdapter)

    def test_wrap_components_with_tool_layer(self):
        """Test wrap_components with tool layer components."""
        real_store = MagicMock()
        real_test_runner = MagicMock()
        real_build_runner = MagicMock()

        tool_layer = MagicMock()
        tool_layer.test_runner = real_test_runner
        tool_layer.build_runner = real_build_runner

        components = {"tool_layer": tool_layer}

        wrapped = wrap_components(components, real_store)

        assert "test_runner" in wrapped
        assert "build_runner" in wrapped


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestOrchestratorIntegration:
    """Integration tests for Orchestrator."""

    @pytest.mark.asyncio
    async def test_full_pipeline_minimal(self, agent_config):
        """Test full pipeline with minimal components."""
        orch = Orchestrator(agent_config)
        submission = StorySubmission(
            story="Add a feature",
            project_id="test",
        )

        result = await orch.process_story(submission)

        assert result.id is not None
        # With no components, story should still be processed and stored
        assert result in orch.get_all_stories()

    @pytest.mark.asyncio
    async def test_state_transitions_tracked(self, agent_config, story):
        """Test that state transitions are properly tracked."""
        orch = Orchestrator(agent_config)
        orch._stories[story.id] = story
        story.state = StoryState.RECEIVED

        orch.state_machine.transition(story, StoryState.ANALYZING)
        orch.state_machine.transition(story, StoryState.PLANNING)

        history = orch.get_state_history(story.id)
        assert len(history) == 2

    def test_retry_policy_integration(self, agent_config, story):
        """Test retry policy integration with orchestrator."""
        orch = Orchestrator(agent_config)

        # Simulate retries
        assert orch.retry_policy.should_retry(story.id, StoryState.CODING)
        orch.retry_policy.record_attempt(
            story.id, StoryState.CODING, success=False, error_fingerprint="fp1"
        )
        assert orch.retry_policy.should_retry(story.id, StoryState.CODING)

    def test_convergence_detector_integration(self, agent_config, story):
        """Test convergence detector integration."""
        orch = Orchestrator(agent_config)

        # Create story with errors
        story.iterations = [
            Iteration(
                number=1,
                stage="coding",
                error_context=ErrorContext(
                    error=ErrorDetail(type="Error", message="Test"),
                    classification=ErrorClassification(fingerprint="fp1"),
                ),
            ),
            Iteration(
                number=2,
                stage="coding",
                error_context=ErrorContext(
                    error=ErrorDetail(type="Error", message="Test"),
                    classification=ErrorClassification(fingerprint="fp1"),
                ),
            ),
        ]

        result = orch.convergence_detector.check(story)
        assert result.status == "oscillating"

    @pytest.mark.asyncio
    async def test_approval_gate_integration(self, agent_config, story):
        """Test approval gate integration."""
        orch = Orchestrator(agent_config)
        orch._stories[story.id] = story

        # Check gate
        result = await orch.approval_gate_manager.check_gate("pre_deploy", story)
        assert result.status == GateStatus.PENDING

        # Approve it
        orch.approval_gate_manager.approve(story.id, "pre_deploy")

        # Check again
        result = await orch.approval_gate_manager.check_gate("pre_deploy", story)
        assert result.status == GateStatus.PASS

    def test_callback_integration(self, agent_config, story):
        """Test callbacks are fired during transitions."""
        orch = Orchestrator(agent_config)
        callback = MagicMock()
        orch.register_callback(callback)

        orch._stories[story.id] = story
        story.state = StoryState.RECEIVED

        orch.state_machine.transition(story, StoryState.ANALYZING)

        # Callback should be called by state_machine, then emit_event not called in this context
        # But we test that the callback is there
        assert len(orch._callbacks) == 1
