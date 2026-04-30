"""API route definitions for the Autonomous Software Development Agent."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field

from fastcoder.types.story import (
    Priority,
    Story,
    StoryConstraints,
    StoryState,
    StorySubmission,
)


# Response models
class StoryStatusResponse(BaseModel):
    """Response for GET /api/v1/stories/{story_id}."""

    story_id: str
    state: StoryState
    iteration_count: int = 0
    plan_summary: Optional[str] = None
    progress: dict = Field(default_factory=dict)
    timeline: dict = Field(default_factory=dict)
    cost: dict = Field(default_factory=dict)


class StorySubmissionResponse(BaseModel):
    """Response for POST /api/v1/stories."""

    story_id: str
    state: StoryState
    status: str
    tracking_url: str


class ApprovalGateRequest(BaseModel):
    """Request for POST /api/v1/stories/{story_id}/approve."""

    gate: str
    decision: str  # approved or rejected
    comment: Optional[str] = None


class BatchSubmissionRequest(BaseModel):
    """Request for POST /api/v1/stories/batch."""

    stories: list[StorySubmission]


class MetricsResponse(BaseModel):
    """Response for GET /api/v1/metrics."""

    completion_rate: float = 0.0
    avg_iterations: float = 0.0
    avg_cost_per_story: float = 0.0
    total_stories: int = 0
    completed_stories: int = 0
    failed_stories: int = 0


def create_router(orchestrator, story_store: dict) -> APIRouter:
    """Create and return the API router with all endpoints.

    Args:
        orchestrator: The Orchestrator instance.
        story_store: Dict to store Story objects keyed by story_id.

    Returns:
        Configured APIRouter.
    """
    router = APIRouter(prefix="/api/v1", tags=["stories"])

    # Helper function to generate short UUID
    def short_uuid() -> str:
        return str(uuid.uuid4())[:8]

    @router.post("/stories", response_model=StorySubmissionResponse, status_code=202)
    async def submit_story(
        submission: StorySubmission,
        background_tasks: BackgroundTasks,
    ) -> StorySubmissionResponse:
        """Submit a new story for processing.

        Accepts StorySubmission body, creates Story with STORY-{uuid_short} ID,
        and launches orchestrator.process_story() as a background task.

        Args:
            submission: StorySubmission object with story text and metadata.
            background_tasks: FastAPI background tasks.

        Returns:
            StorySubmissionResponse with story_id, state, status, and tracking_url.
        """
        story_id = f"STORY-{short_uuid()}"
        story = Story(
            id=story_id,
            raw_text=submission.story,
            project_id=submission.project_id,
            priority=submission.priority,
            constraints=submission.constraints or StoryConstraints(),
            state=StoryState.RECEIVED,
        )
        story_store[story_id] = story

        # Launch processing in background
        background_tasks.add_task(orchestrator.process_story, story)

        return StorySubmissionResponse(
            story_id=story_id,
            state=story.state,
            status="submitted",
            tracking_url=f"/api/v1/stories/{story_id}",
        )

    @router.get("/stories/{story_id}", response_model=StoryStatusResponse)
    async def get_story_status(story_id: str) -> StoryStatusResponse:
        """Get the current status of a story.

        Returns state, iteration count, plan summary, progress, timeline, and cost.

        Args:
            story_id: The story ID.

        Returns:
            StoryStatusResponse with full status details.

        Raises:
            HTTPException: 404 if story not found.
        """
        story = story_store.get(story_id)
        if not story:
            raise HTTPException(status_code=404, detail="Story not found")

        plan_summary = None
        if story.plan:
            plan_summary = f"{len(story.plan.tasks)} tasks planned"

        progress = {
            "iteration": len(story.iterations),
            "total_iterations": story.constraints.max_iterations,
        }

        timeline = {
            "created_at": story.metadata.created_at.isoformat(),
            "updated_at": story.metadata.updated_at.isoformat(),
            "completed_at": (
                story.metadata.completed_at.isoformat()
                if story.metadata.completed_at
                else None
            ),
        }

        cost = {
            "total_cost_usd": story.metadata.total_cost_usd,
            "budget_usd": story.constraints.cost_budget_usd,
            "remaining_budget_usd": (
                story.constraints.cost_budget_usd
                - story.metadata.total_cost_usd
            ),
        }

        return StoryStatusResponse(
            story_id=story_id,
            state=story.state,
            iteration_count=len(story.iterations),
            plan_summary=plan_summary,
            progress=progress,
            timeline=timeline,
            cost=cost,
        )

    @router.post("/stories/{story_id}/approve")
    async def approve_story(
        story_id: str,
        request: ApprovalGateRequest,
    ) -> dict:
        """Submit approval decision for a story at a gate.

        Args:
            story_id: The story ID.
            request: ApprovalGateRequest with gate, decision, and optional comment.

        Returns:
            Dict with decision, gate, and timestamp.

        Raises:
            HTTPException: 404 if story not found.
        """
        story = story_store.get(story_id)
        if not story:
            raise HTTPException(status_code=404, detail="Story not found")

        # Would integrate with approval workflow in orchestrator
        return {
            "story_id": story_id,
            "gate": request.gate,
            "decision": request.decision,
            "comment": request.comment,
            "timestamp": datetime.utcnow().isoformat(),
        }

    @router.post("/stories/batch", response_model=list[StorySubmissionResponse])
    async def batch_submit_stories(
        request: BatchSubmissionRequest,
        background_tasks: BackgroundTasks,
    ) -> list[StorySubmissionResponse]:
        """Submit multiple stories for batch processing.

        Args:
            request: BatchSubmissionRequest with list of stories.
            background_tasks: FastAPI background tasks.

        Returns:
            List of StorySubmissionResponse objects.
        """
        responses = []
        for submission in request.stories:
            story_id = f"STORY-{short_uuid()}"
            story = Story(
                id=story_id,
                raw_text=submission.story,
                project_id=submission.project_id,
                priority=submission.priority,
                constraints=submission.constraints or StoryConstraints(),
                state=StoryState.RECEIVED,
            )
            story_store[story_id] = story
            background_tasks.add_task(orchestrator.process_story, story)

            responses.append(
                StorySubmissionResponse(
                    story_id=story_id,
                    state=story.state,
                    status="submitted",
                    tracking_url=f"/api/v1/stories/{story_id}",
                )
            )
        return responses

    @router.get("/stories", response_model=list[StoryStatusResponse])
    async def list_stories(
        state: Optional[str] = Query(None),
        project_id: Optional[str] = Query(None),
    ) -> list[StoryStatusResponse]:
        """List all stories with optional filters.

        Args:
            state: Optional StoryState filter.
            project_id: Optional project_id filter.

        Returns:
            List of StoryStatusResponse objects matching filters.
        """
        results = []
        for story_id, story in story_store.items():
            if state and story.state.value != state:
                continue
            if project_id and story.project_id != project_id:
                continue

            plan_summary = None
            if story.plan:
                plan_summary = f"{len(story.plan.tasks)} tasks planned"

            progress = {
                "iteration": len(story.iterations),
                "total_iterations": story.constraints.max_iterations,
            }

            timeline = {
                "created_at": story.metadata.created_at.isoformat(),
                "updated_at": story.metadata.updated_at.isoformat(),
                "completed_at": (
                    story.metadata.completed_at.isoformat()
                    if story.metadata.completed_at
                    else None
                ),
            }

            cost = {
                "total_cost_usd": story.metadata.total_cost_usd,
                "budget_usd": story.constraints.cost_budget_usd,
                "remaining_budget_usd": (
                    story.constraints.cost_budget_usd
                    - story.metadata.total_cost_usd
                ),
            }

            results.append(
                StoryStatusResponse(
                    story_id=story_id,
                    state=story.state,
                    iteration_count=len(story.iterations),
                    plan_summary=plan_summary,
                    progress=progress,
                    timeline=timeline,
                    cost=cost,
                )
            )
        return results

    @router.get("/metrics", response_model=MetricsResponse)
    async def get_metrics() -> MetricsResponse:
        """Get system-wide metrics.

        Returns:
            MetricsResponse with completion rate, avg iterations, avg cost.
        """
        total = len(story_store)
        if total == 0:
            return MetricsResponse()

        completed = sum(
            1
            for s in story_store.values()
            if s.state == StoryState.DONE
        )
        failed = sum(
            1
            for s in story_store.values()
            if s.state == StoryState.FAILED
        )

        avg_iterations = (
            sum(len(s.iterations) for s in story_store.values()) / total
            if total > 0
            else 0.0
        )

        avg_cost = (
            sum(s.metadata.total_cost_usd for s in story_store.values())
            / total
            if total > 0
            else 0.0
        )

        return MetricsResponse(
            completion_rate=completed / total if total > 0 else 0.0,
            avg_iterations=avg_iterations,
            avg_cost_per_story=avg_cost,
            total_stories=total,
            completed_stories=completed,
            failed_stories=failed,
        )

    return router
