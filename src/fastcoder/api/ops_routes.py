"""Operational API routes for the workspace frontend — instructions, approvals, activity, and detailed story views."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field

from fastcoder.types.story import (
    Priority,
    Story,
    StoryConstraints,
    StoryState,
    StorySubmission,
    AcceptanceCriterion,
)


# ── Enums ──
class InstructionType(str, Enum):
    USER_STORY = "user_story"
    CODE_CHANGE = "code_change"
    BUG_FIX = "bug_fix"
    REFACTOR = "refactor"
    FEATURE = "feature"
    CUSTOM = "custom"


class ActivityType(str, Enum):
    INSTRUCTION_SUBMITTED = "instruction_submitted"
    STATE_CHANGED = "state_changed"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_DECIDED = "approval_decided"
    ITERATION_STARTED = "iteration_started"
    ITERATION_COMPLETED = "iteration_completed"
    ERROR_OCCURRED = "error_occurred"
    STORY_COMPLETED = "story_completed"
    STORY_FAILED = "story_failed"
    FILE_CHANGED = "file_changed"
    TEST_RUN = "test_run"
    CODE_REVIEWED = "code_reviewed"
    HUMAN_FEEDBACK = "human_feedback"


# ── Request/Response Models ──
class InstructionRequest(BaseModel):
    """Rich instruction submission model with input size limits."""

    type: InstructionType = InstructionType.USER_STORY
    title: str = Field(..., min_length=1, max_length=500)
    description: str = Field(..., min_length=1, max_length=50_000)
    acceptance_criteria: list[dict] = Field(default_factory=list, max_length=50)
    # each dict has: description, given (optional), when (optional), then (optional)
    priority: Priority = Priority.MEDIUM
    project_id: str = Field(default="default", max_length=200)
    target_files: list[str] = Field(default_factory=list, max_length=100)
    constraints: Optional[dict] = None
    # constraints dict can have: max_iterations, cost_budget_usd, target_branch, deploy_target, approval_gates
    context: Optional[str] = Field(default=None, max_length=50_000)
    tags: list[str] = Field(default_factory=list, max_length=20)


class InstructionResponse(BaseModel):
    instruction_id: str
    story_id: str
    type: InstructionType
    title: str
    state: StoryState
    submitted_at: str
    tracking_url: str


class ApprovalRequest(BaseModel):
    decision: str = Field(..., pattern=r"^(approve|reject)$")  # only "approve" or "reject"
    comment: str = Field(default="", max_length=5000)
    decided_by: str = Field(default="human", max_length=200)


class ApprovalResponse(BaseModel):
    story_id: str
    gate_name: str
    decision: str
    comment: str
    decided_at: str
    decided_by: str


class PendingApprovalResponse(BaseModel):
    story_id: str
    gate_name: str
    requested_at: str
    story_title: str
    story_state: str
    context: dict = Field(default_factory=dict)


class StoryDetailResponse(BaseModel):
    """Full detailed view of a story for the frontend."""

    story_id: str
    raw_text: str
    state: StoryState
    priority: str
    project_id: str
    spec: Optional[dict] = None  # StorySpec as dict
    plan: Optional[dict] = None  # ExecutionPlan as dict
    iterations: list[dict] = Field(default_factory=list)
    constraints: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)
    pending_approvals: list[dict] = Field(default_factory=list)
    file_changes: list[dict] = Field(default_factory=list)
    test_results: Optional[dict] = None
    review_results: Optional[dict] = None


class ActivityEntry(BaseModel):
    id: str
    type: ActivityType
    story_id: Optional[str] = None
    title: str
    detail: str = ""
    timestamp: str
    metadata: dict = Field(default_factory=dict)


class HumanFeedbackRequest(BaseModel):
    """Human feedback on a running story."""

    story_id: str
    feedback_type: str  # "guidance", "correction", "approval", "abort"
    message: str
    target_stage: Optional[str] = None  # which stage this applies to


class StoryTimelineEntry(BaseModel):
    timestamp: str
    event: str
    detail: str = ""
    stage: Optional[str] = None


class DashboardStats(BaseModel):
    total_instructions: int = 0
    active_stories: int = 0
    pending_approvals: int = 0
    completed_today: int = 0
    failed_today: int = 0
    total_cost_today_usd: float = 0.0
    avg_completion_time_mins: float = 0.0
    stories_by_state: dict = Field(default_factory=dict)


# ── Project Models ──
class ProjectCreateRequest(BaseModel):
    """Request model for creating a new project."""
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    color: str = Field(default="#6366f1", max_length=20)


class ProjectUpdateRequest(BaseModel):
    """Request model for updating a project."""
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    description: Optional[str] = Field(default=None, max_length=2000)
    color: Optional[str] = Field(default=None, max_length=20)


class ProjectResponse(BaseModel):
    """Response model for a project."""
    id: str
    name: str
    description: str = ""
    color: str = "#6366f1"
    created_at: str
    request_count: int = 0


def create_ops_router(
    orchestrator, story_store: dict, activity_log: list, approval_manager=None
) -> APIRouter:
    """
    Create and configure operational API routes for the workspace frontend.

    Args:
        orchestrator: The orchestrator instance (may be None)
        story_store: Dictionary mapping story_id to Story objects
        activity_log: Shared list for activity log entries
        approval_manager: Optional approval manager instance

    Returns:
        Configured APIRouter with all operational endpoints
    """
    router = APIRouter(prefix="/api/v1/ops", tags=["operations"])

    # instruction_store maps instruction_id -> {instruction data + story_id}
    instruction_store: dict[str, dict] = {}
    # feedback_store maps story_id -> list of feedback entries
    feedback_store: dict[str, list] = {}
    # project_store maps project_id -> {project data}
    project_store: dict[str, dict] = {}

    def short_uuid():
        """Generate a short 8-character UUID string."""
        return str(uuid.uuid4())[:8]

    def add_activity(
        type: ActivityType,
        title: str,
        detail: str = "",
        story_id: str = None,
        metadata: dict = None,
    ):
        """Add an activity log entry."""
        entry = {
            "id": f"ACT-{short_uuid()}",
            "type": type.value,
            "story_id": story_id,
            "title": title,
            "detail": detail,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }
        activity_log.append(entry)
        # Keep only last 1000 entries
        if len(activity_log) > 1000:
            activity_log[:] = activity_log[-1000:]

    def get_story_or_404(story_id: str) -> Story:
        """Retrieve a story or raise 404."""
        if story_id not in story_store:
            raise HTTPException(status_code=404, detail=f"Story {story_id} not found")
        return story_store[story_id]

    def story_to_dict(story: Story) -> dict:
        """Convert Story object to dictionary, handling nested objects."""
        result = {
            "story_id": story.story_id,
            "raw_text": story.raw_text,
            "state": story.state.value if hasattr(story.state, "value") else str(story.state),
            "priority": story.priority.value if hasattr(story.priority, "value") else str(story.priority),
            "project_id": getattr(story, "project_id", "default"),
            "spec": None,
            "plan": None,
            "iterations": [],
            "constraints": {},
            "metadata": {},
            "pending_approvals": [],
            "file_changes": [],
            "test_results": None,
            "review_results": None,
        }

        # Serialize spec as dict
        if hasattr(story, "spec") and story.spec:
            spec = story.spec
            result["spec"] = {
                "type": getattr(spec, "type", None),
                "description": getattr(spec, "description", None),
                "acceptance_criteria": getattr(spec, "acceptance_criteria", []),
                "constraints": getattr(spec, "constraints", None),
            }

        # Serialize plan as dict
        if hasattr(story, "plan") and story.plan:
            plan = story.plan
            result["plan"] = {
                "name": getattr(plan, "name", None),
                "steps": getattr(plan, "steps", []),
                "estimated_cost": getattr(plan, "estimated_cost", None),
            }

        # Serialize iterations
        if hasattr(story, "iterations") and story.iterations:
            for it in story.iterations:
                iteration_dict = {
                    "iteration_number": getattr(it, "iteration_number", None),
                    "started_at": getattr(it, "started_at", None),
                    "ended_at": getattr(it, "ended_at", None),
                    "status": getattr(it, "status", None),
                    "changes": getattr(it, "changes", []),
                    "errors": getattr(it, "errors", []),
                }
                result["iterations"].append(iteration_dict)

        # Add constraints
        if hasattr(story, "constraints") and story.constraints:
            constraints = story.constraints
            result["constraints"] = {
                "max_iterations": getattr(constraints, "max_iterations", None),
                "cost_budget_usd": getattr(constraints, "cost_budget_usd", None),
                "target_branch": getattr(constraints, "target_branch", None),
                "deploy_target": getattr(constraints, "deploy_target", None),
                "approval_gates": getattr(constraints, "approval_gates", []),
            }

        # Add metadata
        if hasattr(story, "metadata") and story.metadata:
            result["metadata"] = story.metadata

        # Collect file changes from all iterations
        for iteration in getattr(story, "iterations", []):
            for change in getattr(iteration, "changes", []):
                change_dict = {
                    "file": getattr(change, "file", None),
                    "type": getattr(change, "type", None),
                    "lines_added": getattr(change, "lines_added", 0),
                    "lines_removed": getattr(change, "lines_removed", 0),
                    "timestamp": getattr(change, "timestamp", None),
                }
                result["file_changes"].append(change_dict)

        # Get test results from latest iteration
        if hasattr(story, "iterations") and story.iterations:
            latest_it = story.iterations[-1]
            if hasattr(latest_it, "test_results"):
                result["test_results"] = latest_it.test_results
            if hasattr(latest_it, "review_results"):
                result["review_results"] = latest_it.review_results

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # ENDPOINT 1: POST /instructions — Submit a rich instruction
    # ═══════════════════════════════════════════════════════════════════════════
    @router.post("/instructions", response_model=InstructionResponse)
    async def submit_instruction(req: InstructionRequest, background_tasks: BackgroundTasks):
        """
        Submit a rich instruction which creates a story internally.

        Converts InstructionRequest into StorySubmission, adds to story_store,
        logs activity, and starts background processing if orchestrator is available.

        Args:
            req: The instruction request
            background_tasks: FastAPI background task queue

        Returns:
            InstructionResponse with story tracking info
        """
        instruction_id = f"INSTR-{short_uuid()}"
        story_id = f"STORY-{short_uuid()}"

        # Build acceptance criteria
        acceptance_criteria = []
        for ac_dict in req.acceptance_criteria:
            acceptance_criteria.append(
                {
                    "description": ac_dict.get("description", ""),
                    "given": ac_dict.get("given"),
                    "when": ac_dict.get("when"),
                    "then": ac_dict.get("then"),
                }
            )

        # Build submission
        submission = {
            "story_id": story_id,
            "title": req.title,
            "description": req.description,
            "raw_text": f"{req.title}\n\n{req.description}",
            "priority": req.priority,
            "project_id": req.project_id,
            "target_files": req.target_files,
            "acceptance_criteria": acceptance_criteria,
            "context": req.context,
            "tags": req.tags,
        }

        # Create story object
        try:
            story = Story(
                story_id=story_id,
                raw_text=submission["raw_text"],
                priority=req.priority,
                state=StoryState.CREATED,
                metadata={
                    "instruction_id": instruction_id,
                    "instruction_type": req.type.value,
                    "project_id": req.project_id,
                    "created_at": datetime.utcnow().isoformat(),
                    "tags": req.tags,
                },
            )

            if req.constraints:
                story.constraints = StoryConstraints(**req.constraints)

            story_store[story_id] = story
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to create story: {str(e)}")

        # Store instruction mapping
        instruction_store[instruction_id] = {
            "instruction_id": instruction_id,
            "story_id": story_id,
            "type": req.type.value,
            "title": req.title,
            "submission": submission,
            "submitted_at": datetime.utcnow().isoformat(),
        }

        # Log activity
        add_activity(
            ActivityType.INSTRUCTION_SUBMITTED,
            f"Instruction '{req.title}' submitted",
            detail=f"Type: {req.type.value}, Priority: {req.priority.value}",
            story_id=story_id,
            metadata={"instruction_id": instruction_id, "type": req.type.value},
        )

        # Start background processing if orchestrator is available
        if orchestrator:

            async def process_instruction():
                try:
                    # Update story state
                    if story_id in story_store:
                        story_store[story_id].state = StoryState.SUBMITTED
                        add_activity(
                            ActivityType.STATE_CHANGED,
                            "Story submitted for processing",
                            story_id=story_id,
                        )
                    # Orchestrator processing would happen here
                except Exception as e:
                    if story_id in story_store:
                        story_store[story_id].state = StoryState.FAILED
                    add_activity(
                        ActivityType.ERROR_OCCURRED,
                        f"Error processing instruction: {str(e)}",
                        story_id=story_id,
                    )

            background_tasks.add_task(process_instruction)

        return InstructionResponse(
            instruction_id=instruction_id,
            story_id=story_id,
            type=req.type,
            title=req.title,
            state=story.state,
            submitted_at=instruction_store[instruction_id]["submitted_at"],
            tracking_url=f"/api/v1/ops/instructions/{instruction_id}",
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # ENDPOINT 2: GET /instructions — List all instructions with filters
    # ═══════════════════════════════════════════════════════════════════════════
    @router.get("/instructions", response_model=list[InstructionResponse])
    async def list_instructions(
        type: Optional[str] = Query(None),
        state: Optional[str] = Query(None),
        project_id: Optional[str] = Query(None),
        limit: int = Query(50, ge=1, le=500),
    ):
        """
        List all submitted instructions with optional filtering.

        Args:
            type: Filter by instruction type
            state: Filter by story state
            project_id: Filter by project ID
            limit: Maximum number of results

        Returns:
            List of InstructionResponse objects
        """
        results = []

        for instr_id, instr_data in list(instruction_store.items()):
            story_id = instr_data["story_id"]

            # Apply filters
            if type and instr_data["type"] != type:
                continue
            if project_id and instr_data["submission"].get("project_id") != project_id:
                continue

            # Get story state
            story = story_store.get(story_id)
            if state and story and story.state.value != state:
                continue

            # Build response
            results.append(
                InstructionResponse(
                    instruction_id=instr_id,
                    story_id=story_id,
                    type=InstructionType(instr_data["type"]),
                    title=instr_data["title"],
                    state=story.state if story else StoryState.CREATED,
                    submitted_at=instr_data["submitted_at"],
                    tracking_url=f"/api/v1/ops/instructions/{instr_id}",
                )
            )

        # Apply limit
        return results[:limit]

    # ═══════════════════════════════════════════════════════════════════════════
    # ENDPOINT 3: GET /instructions/{instruction_id} — Get instruction detail
    # ═══════════════════════════════════════════════════════════════════════════
    @router.get("/instructions/{instruction_id}", response_model=InstructionResponse)
    async def get_instruction(instruction_id: str):
        """
        Get detailed information about a specific instruction.

        Args:
            instruction_id: The instruction ID

        Returns:
            InstructionResponse with full instruction details
        """
        if instruction_id not in instruction_store:
            raise HTTPException(status_code=404, detail=f"Instruction {instruction_id} not found")

        instr_data = instruction_store[instruction_id]
        story_id = instr_data["story_id"]
        story = story_store.get(story_id)

        return InstructionResponse(
            instruction_id=instruction_id,
            story_id=story_id,
            type=InstructionType(instr_data["type"]),
            title=instr_data["title"],
            state=story.state if story else StoryState.CREATED,
            submitted_at=instr_data["submitted_at"],
            tracking_url=f"/api/v1/ops/instructions/{instruction_id}",
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # ENDPOINT 4: POST /instructions/{instruction_id}/cancel — Cancel instruction
    # ═══════════════════════════════════════════════════════════════════════════
    @router.post("/instructions/{instruction_id}/cancel", response_model=dict)
    async def cancel_instruction(instruction_id: str):
        """
        Cancel/abort a running instruction's story.

        Sets the story state to FAILED and logs the cancellation.

        Args:
            instruction_id: The instruction ID to cancel

        Returns:
            Status response
        """
        if instruction_id not in instruction_store:
            raise HTTPException(status_code=404, detail=f"Instruction {instruction_id} not found")

        instr_data = instruction_store[instruction_id]
        story_id = instr_data["story_id"]
        story = get_story_or_404(story_id)

        # Cancel the story
        story.state = StoryState.FAILED

        add_activity(
            ActivityType.STORY_FAILED,
            f"Instruction cancelled: {instr_data['title']}",
            detail="Cancelled by user",
            story_id=story_id,
            metadata={"instruction_id": instruction_id},
        )

        return {
            "instruction_id": instruction_id,
            "story_id": story_id,
            "status": "cancelled",
            "cancelled_at": datetime.utcnow().isoformat(),
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # ENDPOINT 5: POST /instructions/{instruction_id}/retry — Retry instruction
    # ═══════════════════════════════════════════════════════════════════════════
    @router.post("/instructions/{instruction_id}/retry", response_model=dict)
    async def retry_instruction(instruction_id: str, background_tasks: BackgroundTasks):
        """
        Retry a failed instruction by re-submitting its story for processing.

        Only works if the story is in FAILED state.

        Args:
            instruction_id: The instruction ID to retry
            background_tasks: FastAPI background task queue

        Returns:
            Status response
        """
        if instruction_id not in instruction_store:
            raise HTTPException(status_code=404, detail=f"Instruction {instruction_id} not found")

        instr_data = instruction_store[instruction_id]
        story_id = instr_data["story_id"]
        story = get_story_or_404(story_id)

        # Check if story is in a retryable state
        if story.state != StoryState.FAILED:
            raise HTTPException(
                status_code=400,
                detail=f"Can only retry FAILED stories. Current state: {story.state.value}",
            )

        # Reset story state
        story.state = StoryState.CREATED

        add_activity(
            ActivityType.INSTRUCTION_SUBMITTED,
            f"Instruction retry initiated: {instr_data['title']}",
            story_id=story_id,
            metadata={"instruction_id": instruction_id},
        )

        # Schedule background processing
        if orchestrator:

            async def retry_processing():
                try:
                    if story_id in story_store:
                        story_store[story_id].state = StoryState.SUBMITTED
                except Exception as e:
                    add_activity(
                        ActivityType.ERROR_OCCURRED,
                        f"Error during retry: {str(e)}",
                        story_id=story_id,
                    )

            background_tasks.add_task(retry_processing)

        return {
            "instruction_id": instruction_id,
            "story_id": story_id,
            "status": "retry_initiated",
            "retried_at": datetime.utcnow().isoformat(),
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # ENDPOINT 6: GET /stories/{story_id}/detail — Return full story detail
    # ═══════════════════════════════════════════════════════════════════════════
    @router.get("/stories/{story_id}/detail", response_model=StoryDetailResponse)
    async def get_story_detail(story_id: str):
        """
        Get comprehensive detail view of a story.

        Returns full iteration data, file changes, test results, review results,
        pending approvals, plan, spec, and metadata.

        Args:
            story_id: The story ID

        Returns:
            StoryDetailResponse with complete story information
        """
        story = get_story_or_404(story_id)
        story_dict = story_to_dict(story)

        # Build pending approvals list from constraints
        if story.constraints and hasattr(story.constraints, "approval_gates"):
            gates = story.constraints.approval_gates or []
            for gate in gates:
                gate_name = gate if isinstance(gate, str) else getattr(gate, "name", str(gate))
                story_dict["pending_approvals"].append(
                    {
                        "gate_name": gate_name,
                        "requested_at": datetime.utcnow().isoformat(),
                        "status": "pending",
                    }
                )

        return StoryDetailResponse(**story_dict)

    # ═══════════════════════════════════════════════════════════════════════════
    # ENDPOINT 7: GET /stories/{story_id}/timeline — Return story timeline
    # ═══════════════════════════════════════════════════════════════════════════
    @router.get("/stories/{story_id}/timeline", response_model=list[StoryTimelineEntry])
    async def get_story_timeline(story_id: str):
        """
        Build timeline of story events from creation through completion.

        Includes state changes, iteration starts/completions, approvals, etc.

        Args:
            story_id: The story ID

        Returns:
            List of timeline entries sorted by timestamp
        """
        story = get_story_or_404(story_id)
        timeline = []

        # Story creation
        created_at = story.metadata.get("created_at") if hasattr(story, "metadata") else None
        if created_at:
            timeline.append(
                StoryTimelineEntry(
                    timestamp=created_at,
                    event="Story created",
                    detail=f"Priority: {story.priority.value}",
                )
            )

        # Iteration events
        if hasattr(story, "iterations"):
            for it in story.iterations:
                if hasattr(it, "started_at") and it.started_at:
                    timeline.append(
                        StoryTimelineEntry(
                            timestamp=it.started_at,
                            event="Iteration started",
                            detail=f"Iteration {getattr(it, 'iteration_number', '?')}",
                            stage="execution",
                        )
                    )
                if hasattr(it, "ended_at") and it.ended_at:
                    timeline.append(
                        StoryTimelineEntry(
                            timestamp=it.ended_at,
                            event="Iteration completed",
                            detail=f"Iteration {getattr(it, 'iteration_number', '?')}",
                            stage="execution",
                        )
                    )

        # Sort by timestamp
        timeline.sort(key=lambda x: x.timestamp)

        return timeline

    # ═══════════════════════════════════════════════════════════════════════════
    # ENDPOINT 8: GET /stories/{story_id}/changes — Return file changes
    # ═══════════════════════════════════════════════════════════════════════════
    @router.get("/stories/{story_id}/changes", response_model=list[dict])
    async def get_story_changes(story_id: str):
        """
        Get all file changes aggregated from all iterations of the story.

        Args:
            story_id: The story ID

        Returns:
            List of file change entries
        """
        story = get_story_or_404(story_id)
        changes = []

        if hasattr(story, "iterations"):
            for iteration in story.iterations:
                if hasattr(iteration, "changes"):
                    for change in iteration.changes:
                        change_dict = {
                            "file": getattr(change, "file", None),
                            "type": getattr(change, "type", None),
                            "lines_added": getattr(change, "lines_added", 0),
                            "lines_removed": getattr(change, "lines_removed", 0),
                            "timestamp": getattr(change, "timestamp", None),
                            "iteration": getattr(iteration, "iteration_number", None),
                        }
                        changes.append(change_dict)

        # Sort by timestamp
        changes.sort(key=lambda x: x.get("timestamp") or "", reverse=True)

        return changes

    # ═══════════════════════════════════════════════════════════════════════════
    # ENDPOINT 9: GET /approvals/pending — List all pending approvals
    # ═══════════════════════════════════════════════════════════════════════════
    @router.get("/approvals/pending", response_model=list[PendingApprovalResponse])
    async def get_pending_approvals():
        """
        List all pending approvals across all stories.

        Checks approval_manager if available and story states for PENDING gates.

        Returns:
            List of pending approval entries
        """
        pending = []

        # Check all stories for pending approval gates
        for story_id, story in story_store.items():
            if not hasattr(story, "constraints") or not story.constraints:
                continue

            gates = getattr(story.constraints, "approval_gates", []) or []
            for gate in gates:
                gate_name = gate if isinstance(gate, str) else getattr(gate, "name", str(gate))
                pending.append(
                    PendingApprovalResponse(
                        story_id=story_id,
                        gate_name=gate_name,
                        requested_at=datetime.utcnow().isoformat(),
                        story_title=story.raw_text.split("\n")[0][:100],
                        story_state=story.state.value,
                        context={
                            "priority": story.priority.value,
                            "project_id": getattr(story, "project_id", "default"),
                        },
                    )
                )

        return pending

    # ═══════════════════════════════════════════════════════════════════════════
    # ENDPOINT 10: POST /approvals/{story_id}/{gate_name} — Submit approval
    # ═══════════════════════════════════════════════════════════════════════════
    @router.post("/approvals/{story_id}/{gate_name}", response_model=ApprovalResponse)
    async def submit_approval(story_id: str, gate_name: str, req: ApprovalRequest):
        """
        Submit an approval decision for a story gate.

        Uses approval_manager if available, records decision in activity log.

        Args:
            story_id: The story ID
            gate_name: The approval gate name
            req: The approval decision request

        Returns:
            ApprovalResponse with decision details
        """
        story = get_story_or_404(story_id)

        # Validate decision
        if req.decision not in ["approve", "reject"]:
            raise HTTPException(status_code=400, detail="Decision must be 'approve' or 'reject'")

        decided_at = datetime.utcnow().isoformat()

        # Call approval manager if available
        if approval_manager and hasattr(approval_manager, "submit_decision"):
            try:
                approval_manager.submit_decision(
                    story_id=story_id,
                    gate_name=gate_name,
                    decision=req.decision,
                    comment=req.comment,
                    decided_by=req.decided_by,
                )
            except Exception as e:
                # Log but don't fail
                pass

        # Log activity
        add_activity(
            ActivityType.APPROVAL_DECIDED,
            f"Approval gate '{gate_name}' {req.decision}",
            detail=f"Comment: {req.comment}",
            story_id=story_id,
            metadata={
                "gate_name": gate_name,
                "decision": req.decision,
                "decided_by": req.decided_by,
            },
        )

        return ApprovalResponse(
            story_id=story_id,
            gate_name=gate_name,
            decision=req.decision,
            comment=req.comment,
            decided_at=decided_at,
            decided_by=req.decided_by,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # ENDPOINT 11: GET /activity — Return activity log with filters
    # ═══════════════════════════════════════════════════════════════════════════
    @router.get("/activity", response_model=list[ActivityEntry])
    async def get_activity(
        type: Optional[str] = Query(None),
        story_id: Optional[str] = Query(None),
        limit: int = Query(50, ge=1, le=500),
    ):
        """
        Get activity log entries with optional filtering.

        Args:
            type: Filter by activity type
            story_id: Filter by story ID
            limit: Maximum number of entries (default 50)

        Returns:
            List of activity entries, most recent first
        """
        results = []

        for entry in reversed(activity_log):
            # Apply filters
            if type and entry["type"] != type:
                continue
            if story_id and entry.get("story_id") != story_id:
                continue

            results.append(
                ActivityEntry(
                    id=entry["id"],
                    type=ActivityType(entry["type"]),
                    story_id=entry.get("story_id"),
                    title=entry["title"],
                    detail=entry.get("detail", ""),
                    timestamp=entry["timestamp"],
                    metadata=entry.get("metadata", {}),
                )
            )

            if len(results) >= limit:
                break

        return results

    # ═══════════════════════════════════════════════════════════════════════════
    # ENDPOINT 12: POST /feedback — Submit human feedback on a story
    # ═══════════════════════════════════════════════════════════════════════════
    @router.post("/feedback", response_model=dict)
    async def submit_feedback(req: HumanFeedbackRequest):
        """
        Submit human feedback on a running story.

        Stores feedback and logs activity.

        Args:
            req: The feedback request

        Returns:
            Feedback submission response
        """
        story = get_story_or_404(req.story_id)

        feedback_id = f"FB-{short_uuid()}"
        feedback_entry = {
            "feedback_id": feedback_id,
            "story_id": req.story_id,
            "feedback_type": req.feedback_type,
            "message": req.message,
            "target_stage": req.target_stage,
            "submitted_at": datetime.utcnow().isoformat(),
        }

        if req.story_id not in feedback_store:
            feedback_store[req.story_id] = []

        feedback_store[req.story_id].append(feedback_entry)

        add_activity(
            ActivityType.HUMAN_FEEDBACK,
            f"Feedback submitted: {req.feedback_type}",
            detail=req.message[:200],
            story_id=req.story_id,
            metadata={
                "feedback_id": feedback_id,
                "feedback_type": req.feedback_type,
                "target_stage": req.target_stage,
            },
        )

        return {
            "feedback_id": feedback_id,
            "story_id": req.story_id,
            "status": "submitted",
            "submitted_at": feedback_entry["submitted_at"],
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # ENDPOINT 13: GET /feedback/{story_id} — Get all feedback for a story
    # ═══════════════════════════════════════════════════════════════════════════
    @router.get("/feedback/{story_id}", response_model=list[dict])
    async def get_story_feedback(story_id: str):
        """
        Get all human feedback entries for a specific story.

        Args:
            story_id: The story ID

        Returns:
            List of feedback entries for the story
        """
        story = get_story_or_404(story_id)

        return feedback_store.get(story_id, [])

    # ═══════════════════════════════════════════════════════════════════════════
    # ENDPOINT 14: GET /dashboard — Return dashboard statistics
    # ═══════════════════════════════════════════════════════════════════════════
    @router.get("/dashboard", response_model=DashboardStats)
    async def get_dashboard_stats():
        """
        Get dashboard statistics aggregated from story_store.

        Calculates active stories, pending approvals, completion rates, costs, etc.

        Returns:
            DashboardStats with comprehensive metrics
        """
        now = datetime.utcnow()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        total_instructions = len(instruction_store)
        active_stories = 0
        pending_approvals = 0
        completed_today = 0
        failed_today = 0
        total_cost_today_usd = 0.0
        completion_times = []
        stories_by_state = {}

        for story_id, story in story_store.items():
            state = story.state.value

            # Count by state
            stories_by_state[state] = stories_by_state.get(state, 0) + 1

            # Check if active
            if state not in ["DONE", "FAILED"]:
                active_stories += 1

            # Check for pending approvals
            if hasattr(story, "constraints") and story.constraints:
                gates = getattr(story.constraints, "approval_gates", []) or []
                pending_approvals += len(gates)

            # Check completion time
            metadata = getattr(story, "metadata", {})
            created_at_str = metadata.get("created_at")
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(created_at_str)
                    if state == "DONE":
                        if created_at >= today_start:
                            completed_today += 1
                        completion_time = (now - created_at).total_seconds() / 60
                        completion_times.append(completion_time)
                    elif state == "FAILED" and created_at >= today_start:
                        failed_today += 1
                except Exception:
                    pass

            # Aggregate cost
            if hasattr(story, "constraints") and story.constraints:
                cost = getattr(story.constraints, "cost_budget_usd", 0.0) or 0.0
                total_cost_today_usd += cost

        # Calculate average completion time
        avg_completion_time_mins = 0.0
        if completion_times:
            avg_completion_time_mins = sum(completion_times) / len(completion_times)

        return DashboardStats(
            total_instructions=total_instructions,
            active_stories=active_stories,
            pending_approvals=pending_approvals,
            completed_today=completed_today,
            failed_today=failed_today,
            total_cost_today_usd=total_cost_today_usd,
            avg_completion_time_mins=avg_completion_time_mins,
            stories_by_state=stories_by_state,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # ENDPOINT 15: POST /stories/{story_id}/pause — Pause a running story
    # ═══════════════════════════════════════════════════════════════════════════
    @router.post("/stories/{story_id}/pause", response_model=dict)
    async def pause_story(story_id: str):
        """
        Pause a running story if the orchestrator supports it.

        Args:
            story_id: The story ID to pause

        Returns:
            Status response
        """
        story = get_story_or_404(story_id)

        # Check if story can be paused
        if story.state in [StoryState.DONE, StoryState.FAILED]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot pause story in {story.state.value} state",
            )

        # Attempt to pause via orchestrator
        paused = False
        if orchestrator and hasattr(orchestrator, "pause_story"):
            try:
                orchestrator.pause_story(story_id)
                paused = True
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to pause story: {str(e)}")

        # Store pause state in metadata
        if not hasattr(story, "metadata"):
            story.metadata = {}
        story.metadata["paused_at"] = datetime.utcnow().isoformat()
        story.metadata["paused"] = True

        add_activity(
            ActivityType.STATE_CHANGED,
            "Story paused",
            story_id=story_id,
            metadata={"paused": True},
        )

        return {
            "story_id": story_id,
            "status": "paused",
            "paused_at": story.metadata["paused_at"],
            "orchestrator_support": paused,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # ENDPOINT 16: POST /stories/{story_id}/resume — Resume a paused story
    # ═══════════════════════════════════════════════════════════════════════════
    @router.post("/stories/{story_id}/resume", response_model=dict)
    async def resume_story(story_id: str, background_tasks: BackgroundTasks):
        """
        Resume a paused story.

        Args:
            story_id: The story ID to resume
            background_tasks: FastAPI background task queue

        Returns:
            Status response
        """
        story = get_story_or_404(story_id)

        # Check if story is paused
        metadata = getattr(story, "metadata", {})
        if not metadata.get("paused"):
            raise HTTPException(status_code=400, detail="Story is not paused")

        # Attempt to resume via orchestrator
        resumed = False
        if orchestrator and hasattr(orchestrator, "resume_story"):
            try:
                orchestrator.resume_story(story_id)
                resumed = True
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to resume story: {str(e)}")

        # Update metadata
        story.metadata["paused"] = False
        story.metadata["resumed_at"] = datetime.utcnow().isoformat()

        add_activity(
            ActivityType.STATE_CHANGED,
            "Story resumed",
            story_id=story_id,
            metadata={"resumed": True},
        )

        return {
            "story_id": story_id,
            "status": "resumed",
            "resumed_at": story.metadata["resumed_at"],
            "orchestrator_support": resumed,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # PROJECT MANAGEMENT ENDPOINTS
    # ═══════════════════════════════════════════════════════════════════════════

    def _count_project_requests(project_id: str) -> int:
        """Count how many instructions belong to a project."""
        count = 0
        for instr_data in instruction_store.values():
            if instr_data.get("submission", {}).get("project_id") == project_id:
                count += 1
        return count

    @router.get("/projects", response_model=list[ProjectResponse])
    async def list_projects():
        """List all projects with their request counts."""
        results = []
        for proj_id, proj in project_store.items():
            results.append(ProjectResponse(
                id=proj_id,
                name=proj["name"],
                description=proj.get("description", ""),
                color=proj.get("color", "#6366f1"),
                created_at=proj["created_at"],
                request_count=_count_project_requests(proj_id),
            ))
        # Sort by created_at descending (newest first)
        results.sort(key=lambda p: p.created_at, reverse=True)
        return results

    @router.post("/projects", response_model=ProjectResponse)
    async def create_project(req: ProjectCreateRequest):
        """Create a new project."""
        project_id = f"PROJ-{short_uuid()}"
        now = datetime.utcnow().isoformat()
        project_store[project_id] = {
            "name": req.name,
            "description": req.description,
            "color": req.color,
            "created_at": now,
        }
        add_activity(
            ActivityType.INSTRUCTION_SUBMITTED,
            f"Project created: {req.name}",
            detail=f"Project {project_id} created",
            metadata={"project_id": project_id},
        )
        return ProjectResponse(
            id=project_id,
            name=req.name,
            description=req.description,
            color=req.color,
            created_at=now,
            request_count=0,
        )

    @router.put("/projects/{project_id}", response_model=ProjectResponse)
    async def update_project(project_id: str, req: ProjectUpdateRequest):
        """Update an existing project."""
        if project_id not in project_store:
            raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
        proj = project_store[project_id]
        if req.name is not None:
            proj["name"] = req.name
        if req.description is not None:
            proj["description"] = req.description
        if req.color is not None:
            proj["color"] = req.color
        return ProjectResponse(
            id=project_id,
            name=proj["name"],
            description=proj.get("description", ""),
            color=proj.get("color", "#6366f1"),
            created_at=proj["created_at"],
            request_count=_count_project_requests(project_id),
        )

    @router.delete("/projects/{project_id}")
    async def delete_project(project_id: str):
        """Delete a project (requests remain but become unassigned)."""
        if project_id not in project_store:
            raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
        name = project_store[project_id]["name"]
        del project_store[project_id]
        add_activity(
            ActivityType.STATE_CHANGED,
            f"Project deleted: {name}",
            detail=f"Project {project_id} removed",
            metadata={"project_id": project_id},
        )
        return {"status": "deleted", "project_id": project_id}

    return router
