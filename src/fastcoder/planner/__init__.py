"""Planner — converts story specifications into actionable execution plans."""

from __future__ import annotations

import json
import re
import structlog
from collections import deque, defaultdict
from typing import Callable, Optional

from fastcoder.types.codebase import ProjectProfile
from fastcoder.types.errors import ErrorContext
from fastcoder.types.llm import CompletionRequest, Message
from fastcoder.types.plan import (
    DeployStrategy,
    ExecutionPlan,
    PlanTask,
    TaskAction,
    TestingStrategy,
)
from fastcoder.types.story import StorySpec, StoryType

logger = structlog.get_logger(__name__)


class Planner:
    """Converts story specifications into ordered execution plans.

    Creates:
    - Ordered task list with dependencies
    - Testing strategy based on story type/complexity
    - Deploy strategy based on story type
    - Estimated token budgets for each task
    - Validates: no circular deps, reasonable estimates
    """

    def __init__(
        self,
        llm_complete: Callable,
        codebase_query: Optional[Callable] = None,
    ):
        """Initialize the planner.

        Args:
            llm_complete: Async function for LLM completion
            codebase_query: Optional callable to query codebase structure
        """
        self.llm_complete = llm_complete
        self.codebase_query = codebase_query
        self.logger = structlog.get_logger(f"{__name__}.{self.__class__.__name__}")

    async def create_plan(
        self,
        spec: StorySpec,
        project_profile: Optional[ProjectProfile] = None,
    ) -> ExecutionPlan:
        """Create an execution plan from a story specification.

        Args:
            spec: StorySpec from the analyzer
            project_profile: Optional project context

        Returns:
            ExecutionPlan: Ordered, validated plan with testing/deploy strategies

        Raises:
            ValueError: If plan creation fails or has circular dependencies
        """
        self.logger.info(f"Creating plan for story: {spec.title}")

        try:
            # Get LLM to generate initial task list
            tasks = await self._generate_tasks(spec, project_profile)

            # Topologically sort tasks by dependencies
            tasks = self._topological_sort(tasks)

            # Detect and validate circular dependencies
            cycles = self._detect_circular_deps(tasks)
            if cycles:
                raise ValueError(f"Circular dependencies detected: {cycles}")

            # Determine testing strategy
            testing_strategy = self._determine_testing_strategy(spec)

            # Determine deploy strategy
            deploy_strategy = self._determine_deploy_strategy(spec)

            # Estimate total tokens
            estimated_total_tokens = sum(task.estimated_tokens for task in tasks)

            plan = ExecutionPlan(
                story_id="",  # Will be filled by orchestrator
                tasks=tasks,
                testing_strategy=testing_strategy,
                deploy_strategy=deploy_strategy,
                estimated_total_tokens=estimated_total_tokens,
            )

            self.logger.info(
                f"Plan created: {len(tasks)} tasks, testing={testing_strategy.value}, "
                f"deploy={deploy_strategy.value}, tokens={estimated_total_tokens}"
            )
            return plan

        except Exception as e:
            self.logger.error(f"Plan creation failed: {e}")
            # Return minimal plan
            return self._create_fallback_plan(spec)

    async def revise_plan(
        self,
        plan: ExecutionPlan,
        error_context: ErrorContext,
    ) -> ExecutionPlan:
        """Revise a plan based on error feedback.

        Args:
            plan: Original ExecutionPlan
            error_context: Error details and classification

        Returns:
            ExecutionPlan: Revised plan with potential decomposition
        """
        self.logger.info(
            f"Revising plan after error: {error_context.error.message}"
        )

        # Extract the failed task (if identifiable)
        failed_task_id = error_context.error.file or "unknown"

        # Build revision prompt
        prompt = self._build_revision_prompt(
            plan, error_context, failed_task_id
        )

        try:
            response = await self.llm_complete(prompt)
            revised_tasks = self._parse_task_response(response.content)

            # Update plan with revised tasks
            plan.tasks = revised_tasks
            plan.revision += 1

            # Validate revised plan
            cycles = self._detect_circular_deps(revised_tasks)
            if cycles:
                self.logger.warning(f"Cycles in revised plan: {cycles}")

            # Re-estimate tokens
            plan.estimated_total_tokens = sum(
                task.estimated_tokens for task in revised_tasks
            )

            self.logger.info(f"Plan revised (revision {plan.revision})")
            return plan

        except Exception as e:
            self.logger.error(f"Plan revision failed: {e}")
            return plan  # Return original plan on failure

    def _topological_sort(self, tasks: list[PlanTask]) -> list[PlanTask]:
        """Topologically sort tasks by dependencies using Kahn's algorithm.

        Args:
            tasks: Unsorted list of PlanTask objects

        Returns:
            list[PlanTask]: Sorted tasks where dependencies come first

        Raises:
            ValueError: If circular dependencies exist
        """
        # Build dependency graph
        task_by_id = {task.id: task for task in tasks}
        in_degree = {task.id: len(task.depends_on) for task in tasks}
        graph = defaultdict(list)

        for task in tasks:
            for dep_id in task.depends_on:
                if dep_id in task_by_id:
                    graph[dep_id].append(task.id)

        # Kahn's algorithm
        queue = deque([task_id for task_id in in_degree if in_degree[task_id] == 0])
        sorted_tasks = []

        while queue:
            current_id = queue.popleft()
            sorted_tasks.append(task_by_id[current_id])

            for neighbor_id in graph[current_id]:
                in_degree[neighbor_id] -= 1
                if in_degree[neighbor_id] == 0:
                    queue.append(neighbor_id)

        if len(sorted_tasks) != len(tasks):
            raise ValueError("Circular dependencies detected in task graph")

        return sorted_tasks

    def _detect_circular_deps(self, tasks: list[PlanTask]) -> list[list[str]]:
        """Detect circular dependencies using DFS.

        Args:
            tasks: List of tasks to check

        Returns:
            list[list[str]]: List of cycles found (empty if none)
        """
        task_by_id = {task.id: task for task in tasks}
        visited = set()
        rec_stack = set()
        cycles = []

        def dfs(node_id: str, path: list[str]) -> None:
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)

            if node_id in task_by_id:
                for dep_id in task_by_id[node_id].depends_on:
                    if dep_id not in visited:
                        dfs(dep_id, path[:])
                    elif dep_id in rec_stack:
                        # Found cycle
                        cycle_start = path.index(dep_id)
                        cycle = path[cycle_start:] + [dep_id]
                        cycles.append(cycle)

            rec_stack.remove(node_id)

        for task in tasks:
            if task.id not in visited:
                dfs(task.id, [])

        return cycles

    async def _generate_tasks(
        self,
        spec: StorySpec,
        project_profile: Optional[ProjectProfile] = None,
    ) -> list[PlanTask]:
        """Generate initial task list from story spec using LLM.

        Args:
            spec: StorySpec with acceptance criteria and dependencies
            project_profile: Optional project context

        Returns:
            list[PlanTask]: Generated tasks (unsorted)
        """
        prompt = self._build_planning_prompt(spec, project_profile)

        try:
            response = await self.llm_complete(prompt)
            tasks = self._parse_task_response(response.content)
            return tasks
        except Exception as e:
            self.logger.warning(f"LLM task generation failed: {e}")
            return self._generate_fallback_tasks(spec)

    def _build_planning_prompt(
        self,
        spec: StorySpec,
        project_profile: Optional[ProjectProfile] = None,
    ) -> CompletionRequest:
        """Build the LLM prompt for plan generation."""
        system_prompt = """You are an expert software development planner.
Given a story specification, generate an ordered task list to implement it.

For each task, specify:
- id: unique identifier (e.g., "task-1", "task-create-model")
- action: create_file, modify_file, delete_file, or run_command
- target: file path or command name
- description: what the task accomplishes
- depends_on: list of task IDs this depends on (empty if no deps)
- estimated_tokens: rough token estimate for LLM to complete this task (typical: 1000-4000)

Guidelines:
- Break work into atomic, independent tasks where possible
- Order tasks to minimize rework (types before implementations, tests after code)
- Be specific about file paths and actions
- Estimate conservatively on tokens

Respond with ONLY valid JSON, no markdown or extra text:
{
  "tasks": [
    {
      "id": "string",
      "action": "create_file|modify_file|delete_file|run_command",
      "target": "string",
      "description": "string",
      "depends_on": ["string"],
      "estimated_tokens": 2000
    }
  ]
}"""

        criteria_text = "\n".join(
            f"  - {c.id}: {c.description}" for c in spec.acceptance_criteria
        )

        deps_text = "\n".join(
            f"  - {d.file_path} ({d.relationship})" for d in spec.dependencies
        )

        project_context = ""
        if project_profile:
            project_context = f"""

Project Context:
- Language: {project_profile.language}
- Framework: {project_profile.framework or 'None'}
- Test Framework: {project_profile.test_framework}
- Naming: {project_profile.naming_conventions}"""

        user_message = f"""Story: {spec.title}
Type: {spec.story_type.value}
Complexity: {spec.complexity_score}/10

Description:
{spec.description}

Acceptance Criteria:
{criteria_text}

Dependencies:
{deps_text if deps_text else "  (none identified)"}

Generate a task list to implement this story.{project_context}"""

        return CompletionRequest(
            model="",  # Router will fill this
            messages=[
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_message),
            ],
            max_tokens=2000,
            temperature=0.3,
        )

    def _build_revision_prompt(
        self,
        plan: ExecutionPlan,
        error_context: ErrorContext,
        failed_task_id: str,
    ) -> CompletionRequest:
        """Build prompt for plan revision after error."""
        system_prompt = """You are an expert at debugging failed software tasks.
Given a failed task and error details, revise the execution plan to fix the issue.

You may:
- Decompose the failed task into smaller steps
- Add preliminary setup/fix tasks
- Reorder dependencies
- Adjust token estimates

Respond with revised task list in JSON format (same structure as planning prompt)."""

        tasks_text = "\n".join(
            f"  - {t.id}: {t.action.value} {t.target} (depends: {t.depends_on})"
            for t in plan.tasks
        )

        error_msg = error_context.error.message or "Unknown error"
        stack_trace = error_context.error.stack_trace or ""

        user_message = f"""Failed Task: {failed_task_id}
Error: {error_msg}
Stack Trace: {stack_trace[:500]}

Current Plan:
{tasks_text}

Revise the plan to fix this error."""

        return CompletionRequest(
            model="",
            messages=[
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_message),
            ],
            max_tokens=2000,
            temperature=0.3,
        )

    def _parse_task_response(self, response_content: str) -> list[PlanTask]:
        """Parse LLM response into PlanTask objects.

        Extracts JSON and validates task structure.
        """
        # Extract JSON from response
        json_match = re.search(r"\{.*\}", response_content, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found in response")

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

        tasks = []
        for i, task_data in enumerate(data.get("tasks", [])):
            try:
                action = TaskAction(task_data.get("action", "create_file"))
            except ValueError:
                action = TaskAction.CREATE_FILE

            task = PlanTask(
                id=task_data.get("id", f"task-{i}"),
                action=action,
                target=task_data.get("target", ""),
                description=task_data.get("description", ""),
                depends_on=task_data.get("depends_on", []),
                estimated_tokens=max(
                    1000, int(task_data.get("estimated_tokens", 2000))
                ),
            )
            tasks.append(task)

        return tasks

    def _determine_testing_strategy(self, spec: StorySpec) -> TestingStrategy:
        """Determine testing strategy based on story type and complexity.

        - Bugfix: UNIT_INTEGRATION (focus on fix validation)
        - Simple feature: UNIT (basic unit tests)
        - Complex feature: UNIT_INTEGRATION_E2E (full coverage)
        - Refactor: UNIT (ensure behavior preserved)
        - Infra: INTEGRATION (system integration tests)
        """
        if spec.story_type == StoryType.BUGFIX:
            return TestingStrategy.UNIT_INTEGRATION
        elif spec.story_type == StoryType.REFACTOR:
            return TestingStrategy.UNIT
        elif spec.story_type == StoryType.INFRA:
            return TestingStrategy.INTEGRATION
        elif spec.complexity_score >= 7:
            return TestingStrategy.UNIT_INTEGRATION_E2E
        elif spec.complexity_score >= 5:
            return TestingStrategy.UNIT_INTEGRATION
        else:
            return TestingStrategy.UNIT

    def _determine_deploy_strategy(self, spec: StorySpec) -> DeployStrategy:
        """Determine deploy strategy based on story type and complexity.

        - Bugfix: STAGING_FIRST (validate fix in staging)
        - High-risk (infra, complex): STAGING_FIRST (conservative)
        - Feature: PR_ONLY (code review before deploy)
        - Refactor: STAGING_FIRST (ensure stability)
        """
        if spec.story_type in (StoryType.INFRA, StoryType.REFACTOR):
            return DeployStrategy.STAGING_FIRST
        elif spec.story_type == StoryType.BUGFIX:
            if spec.complexity_score >= 6:
                return DeployStrategy.STAGING_FIRST
            return DeployStrategy.PR_ONLY
        elif spec.complexity_score >= 8:
            return DeployStrategy.STAGING_FIRST
        else:
            return DeployStrategy.PR_ONLY

    def _generate_fallback_tasks(self, spec: StorySpec) -> list[PlanTask]:
        """Generate minimal task list when LLM fails.

        Creates: code -> test -> review -> deploy structure
        """
        task_id_counter = 0
        tasks = []

        # Implementation task
        task_id_counter += 1
        impl_task = PlanTask(
            id=f"task-{task_id_counter}",
            action=TaskAction.CREATE_FILE,
            target="implementation",
            description=f"Implement: {spec.title}",
            depends_on=[],
            estimated_tokens=3000,
        )
        tasks.append(impl_task)

        # Test task
        task_id_counter += 1
        test_task = PlanTask(
            id=f"task-{task_id_counter}",
            action=TaskAction.RUN_COMMAND,
            target="test",
            description="Run tests to verify implementation",
            depends_on=[impl_task.id],
            estimated_tokens=1500,
        )
        tasks.append(test_task)

        # Review task (if applicable)
        if len(spec.acceptance_criteria) > 1 or spec.complexity_score >= 5:
            task_id_counter += 1
            review_task = PlanTask(
                id=f"task-{task_id_counter}",
                action=TaskAction.RUN_COMMAND,
                target="review",
                description="Code review and quality checks",
                depends_on=[test_task.id],
                estimated_tokens=1000,
            )
            tasks.append(review_task)

        return tasks

    def _create_fallback_plan(self, spec: StorySpec) -> ExecutionPlan:
        """Create minimal plan when planning fails."""
        tasks = self._generate_fallback_tasks(spec)

        return ExecutionPlan(
            story_id="",
            tasks=tasks,
            testing_strategy=self._determine_testing_strategy(spec),
            deploy_strategy=self._determine_deploy_strategy(spec),
            estimated_total_tokens=sum(t.estimated_tokens for t in tasks),
        )
