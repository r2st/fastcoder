"""Code Generator — produces production-ready code from tasks and LLM."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from fastcoder.types.codebase import ProjectProfile
from fastcoder.types.errors import ErrorContext
from fastcoder.types.llm import CompletionResponse, Message
from fastcoder.types.plan import PlanTask
from fastcoder.types.story import StorySpec
from fastcoder.types.task import FileChange


@dataclass
class GenerationResult:
    """Result from code generation."""

    code: str
    file_changes: list[FileChange] = field(default_factory=list)
    reasoning: str = ""
    confidence: float = 1.0
    reflection_issues: list[dict[str, Any]] = field(default_factory=list)


class CodeGenerator:
    """Generates code from tasks using an LLM."""

    def __init__(self, llm_complete: Callable[[list[Message], dict], CompletionResponse]) -> None:
        """Initialize the code generator.

        Args:
            llm_complete: Async callable that takes messages and metadata, returns CompletionResponse
        """
        self.llm_complete = llm_complete

    async def generate(
        self,
        task: PlanTask,
        context: dict[str, Any],
    ) -> GenerationResult:
        """Generate code for a task.

        Args:
            task: The PlanTask to generate code for
            context: Dict with: project_profile, relevant_files, type_definitions,
                     conventions, error_history

        Returns:
            GenerationResult with generated code and file changes
        """
        project_profile: ProjectProfile = context.get("project_profile")
        relevant_files: dict[str, str] = context.get("relevant_files", {})
        type_definitions: str = context.get("type_definitions", "")
        conventions: str = context.get("conventions", "")
        error_history: list[dict] = context.get("error_history", [])

        # Build the prompt
        prompt = self._build_generation_prompt(
            task=task,
            project_profile=project_profile,
            relevant_files=relevant_files,
            type_definitions=type_definitions,
            conventions=conventions,
            error_history=error_history,
        )

        # Call LLM
        messages = [Message(role="user", content=prompt)]
        response = await self.llm_complete(messages, {"purpose": "code_generation"})

        # Extract code from response
        code = self._extract_code(response.content)
        reasoning = response.content if not code else ""

        # Parse file changes from task target
        file_changes = self._parse_file_changes(task, code)

        # Self-reflection on generated code
        confidence, reflection_issues = await self._self_reflect(code, task, context)

        # Apply fixes if confidence is low
        if confidence < 0.7 and reflection_issues:
            code, file_changes = await self._apply_fixes(
                code, reflection_issues, file_changes, task, context
            )
            confidence = min(confidence + 0.15, 1.0)  # Modest improvement

        return GenerationResult(
            code=code,
            file_changes=file_changes,
            reasoning=reasoning,
            confidence=confidence,
            reflection_issues=reflection_issues,
        )

    async def fix(
        self,
        task: PlanTask,
        error_context: ErrorContext,
        context: dict[str, Any],
    ) -> GenerationResult:
        """Fix code based on an error.

        Args:
            task: The PlanTask to fix code for
            error_context: Details about the error, including previous code
            context: Generation context (same as generate)

        Returns:
            GenerationResult with fixed code
        """
        attempt = error_context.attempt

        # Build error-aware prompt
        prompt = self._build_fix_prompt(
            task=task,
            error_context=error_context,
            previous_code=error_context.previous_code or "",
            attempt_number=attempt,
            context=context,
        )

        # Call LLM with progressively more context
        metadata = {
            "purpose": "error_recovery",
            "attempt": attempt,
            "category": error_context.classification.category.value,
        }

        messages = [Message(role="user", content=prompt)]
        response = await self.llm_complete(messages, metadata)

        code = self._extract_code(response.content)
        reasoning = response.content if not code else ""

        file_changes = self._parse_file_changes(task, code)

        # Self-reflect with error context
        confidence, reflection_issues = await self._self_reflect(code, task, context)

        return GenerationResult(
            code=code,
            file_changes=file_changes,
            reasoning=reasoning,
            confidence=confidence,
            reflection_issues=reflection_issues,
        )

    async def _self_reflect(
        self,
        code: str,
        task: PlanTask,
        context: dict[str, Any],
    ) -> tuple[float, list[dict[str, Any]]]:
        """Self-reflect on generated code across 6 axes.

        Args:
            code: The generated code to evaluate
            task: The task context
            context: Generation context

        Returns:
            Tuple of (confidence_score: 0-1, issues: list of dicts)
        """
        prompt = f"""Analyze this code for a task: {task.description}

CODE:
```
{code}
```

Rate the code on these 6 axes (0-10 each):
1. Correctness: Does it correctly implement the task?
2. Edge cases: Are null, empty, and boundary cases handled?
3. Security: Any vulnerabilities, injection risks, or unsafe patterns?
4. Consistency: Does it follow the project conventions?
5. Testability: Is it easy to test? Good separation of concerns?
6. Performance: Any obvious inefficiencies or N+1 queries?

Respond in JSON with:
{{
  "correctness": <0-10>,
  "edge_cases": <0-10>,
  "security": <0-10>,
  "consistency": <0-10>,
  "testability": <0-10>,
  "performance": <0-10>,
  "issues": [
    {{"axis": "...", "issue": "...", "severity": "blocking|major|minor"}},
    ...
  ]
}}"""

        messages = [Message(role="user", content=prompt)]
        response = await self.llm_complete(messages, {"purpose": "self_reflection"})

        # Parse JSON response
        json_str = response.content.strip()
        if json_str.startswith("```"):
            json_str = json_str.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
            json_str = json_str.strip()

        try:
            evaluation = json.loads(json_str)
        except json.JSONDecodeError:
            # Fallback if parsing fails
            return 0.85, []

        # Calculate confidence as average of all axes
        scores = [
            evaluation.get("correctness", 7),
            evaluation.get("edge_cases", 7),
            evaluation.get("security", 7),
            evaluation.get("consistency", 7),
            evaluation.get("testability", 7),
            evaluation.get("performance", 7),
        ]
        confidence = sum(scores) / len(scores) / 10.0

        issues = evaluation.get("issues", [])

        return confidence, issues

    async def _apply_fixes(
        self,
        code: str,
        issues: list[dict[str, Any]],
        file_changes: list[FileChange],
        task: PlanTask,
        context: dict[str, Any],
    ) -> tuple[str, list[FileChange]]:
        """Apply fixes to code based on reflection issues.

        Args:
            code: Original code
            issues: Issues found during reflection
            file_changes: Original file changes
            task: Task context
            context: Generation context

        Returns:
            Tuple of (fixed_code, fixed_file_changes)
        """
        if not issues:
            return code, file_changes

        # Build fix prompt
        issues_str = "\n".join(
            [f"- {issue.get('severity', 'warning').upper()}: {issue.get('issue', '')}" for issue in issues]
        )

        prompt = f"""Fix these issues in the code:

ISSUES:
{issues_str}

ORIGINAL CODE:
```
{code}
```

Provide the corrected code that addresses all issues. Return only the code block."""

        messages = [Message(role="user", content=prompt)]
        response = await self.llm_complete(messages, {"purpose": "code_refinement"})

        fixed_code = self._extract_code(response.content)
        if not fixed_code:
            fixed_code = code

        # Update file changes with fixed code
        updated_changes = []
        for change in file_changes:
            updated_change = change.model_copy()
            if change.content:
                updated_change.content = fixed_code
            updated_changes.append(updated_change)

        return fixed_code, updated_changes

    def _build_generation_prompt(
        self,
        task: PlanTask,
        project_profile: ProjectProfile,
        relevant_files: dict[str, str],
        type_definitions: str,
        conventions: str,
        error_history: list[dict],
    ) -> str:
        """Build a comprehensive generation prompt."""
        context_sections = []

        # Error history context (if any)
        if error_history:
            context_sections.append("## Recent Errors\n" + "\n".join([str(e) for e in error_history[-3:]]))

        # Project conventions
        if conventions:
            context_sections.append(f"## Project Conventions\n{conventions}")

        # Type definitions
        if type_definitions:
            context_sections.append(f"## Type Definitions\n{type_definitions}")

        # Relevant files
        if relevant_files:
            context_sections.append("## Relevant Codebase Files")
            for path, content in list(relevant_files.items())[:5]:  # Limit to 5 files
                context_sections.append(f"\n### {path}\n```\n{content[:500]}\n```")

        context_str = "\n\n".join(context_sections)

        prompt = f"""You are a code generation expert. Generate high-quality, production-ready code.

## Task
{task.description}

Target: {task.target}
Action: {task.action.value}

{context_str}

## Instructions
1. Generate clean, well-structured code following project conventions
2. Include proper error handling and edge case coverage
3. Add clear comments for complex logic
4. Ensure the code is testable and maintainable
5. For file creation/modification, provide complete file content

Respond with the generated code in a markdown code block (```python or ```).
Include the full file content if creating/modifying files."""

        return prompt

    def _build_fix_prompt(
        self,
        task: PlanTask,
        error_context: ErrorContext,
        previous_code: str,
        attempt_number: int,
        context: dict[str, Any],
    ) -> str:
        """Build an error-aware fix prompt."""
        strategy = error_context.classification.recovery_strategy.value
        category = error_context.classification.category.value

        sections = [
            f"## Error Context\nAttempt: {attempt_number}",
            f"Category: {category}",
            f"Recovery Strategy: {strategy}",
            f"Error: {error_context.error.message}",
        ]

        if error_context.error.stack_trace:
            sections.append(f"Stack Trace:\n```\n{error_context.error.stack_trace}\n```")

        if previous_code:
            sections.append(f"## Previous Code\n```\n{previous_code}\n```")

        # Enrichment based on attempt number
        if attempt_number >= 2:
            type_defs = context.get("type_definitions", "")
            if type_defs:
                sections.append(f"## Type Definitions for Reference\n{type_defs}")

        if attempt_number >= 3:
            relevant = context.get("relevant_files", {})
            if relevant:
                sections.append("## Similar Patterns in Codebase")
                for path, content in list(relevant.items())[:2]:
                    sections.append(f"From {path}:\n```\n{content[:300]}\n```")

        context_str = "\n\n".join(sections)

        prompt = f"""Fix the code to resolve this error. You are on attempt {attempt_number}.

## Task
{task.description}

{context_str}

## Instructions for {strategy}
- Provide the corrected code in a markdown code block
- Explain the root cause briefly
- Address the specific error category: {category}

Respond with fixed code and a brief explanation."""

        return prompt

    def _extract_code(self, response_text: str) -> str:
        """Extract code from markdown code blocks."""
        # Try to find ```python or ``` code blocks
        patterns = [
            r"```(?:python|py)?\n(.*?)\n```",  # With language specifier
            r"```\n(.*?)\n```",  # Without language
        ]

        for pattern in patterns:
            match = re.search(pattern, response_text, re.DOTALL)
            if match:
                return match.group(1).strip()

        # If no code blocks found, return empty string
        return ""

    def _parse_file_changes(self, task: PlanTask, code: str) -> list[FileChange]:
        """Parse file changes from task and generated code."""
        changes = []

        if not code:
            return changes

        change = FileChange(
            file_path=task.target,
            change_type="created" if "create" in task.action.value else "modified",
            content=code,
        )
        changes.append(change)

        return changes
