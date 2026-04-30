"""Code Reviewer — performs multi-dimensional code review using LLM."""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from fastcoder.types.codebase import ProjectProfile
from fastcoder.types.llm import CompletionResponse, Message
from fastcoder.types.story import StorySpec
from fastcoder.types.task import FileChange, ReviewIssue, ReviewReport


class CodeReviewer:
    """Reviews code for security, performance, correctness, and style."""

    def __init__(self, llm_complete: Callable[[list[Message], dict], CompletionResponse]) -> None:
        """Initialize the code reviewer.

        Args:
            llm_complete: Async callable that takes messages and metadata (should be different
                         provider from generator for fresh perspective)
        """
        self.llm_complete = llm_complete

    async def review(
        self,
        changes: list[FileChange],
        spec: StorySpec,
        profile: ProjectProfile,
    ) -> ReviewReport:
        """Review code changes for quality and compliance.

        Args:
            changes: List of FileChange objects to review
            spec: Story specification including acceptance criteria
            profile: Project profile with conventions and standards

        Returns:
            ReviewReport with issues found and approval decision
        """
        # Build review prompt
        prompt = self._build_review_prompt(changes, spec, profile)

        # Call LLM
        messages = [Message(role="user", content=prompt)]
        response = await self.llm_complete(messages, {"purpose": "code_review"})

        # Parse review findings
        issues, summary = self._parse_review_response(response.content)

        # Approve only if no blocking issues
        approved = not any(issue.severity == "blocking" for issue in issues)

        return ReviewReport(
            approved=approved,
            issues=issues,
            summary=summary,
            reviewer_model=response.model,
        )

    def _build_review_prompt(
        self,
        changes: list[FileChange],
        spec: StorySpec,
        profile: ProjectProfile,
    ) -> str:
        """Build a comprehensive code review prompt."""
        # Format file changes as diffs
        changes_section = "## Code Changes\n"
        for change in changes:
            changes_section += f"\n### {change.file_path}\n"
            changes_section += f"Type: {change.change_type}\n"

            if change.content:
                changes_section += f"```python\n{change.content}\n```"
            elif change.diff:
                changes_section += f"```diff\n{change.diff}\n```"

        # Format acceptance criteria
        criteria_section = ""
        if spec.acceptance_criteria:
            criteria_section = "\n## Acceptance Criteria\n"
            for criterion in spec.acceptance_criteria:
                criteria_section += f"- **AC-{criterion.id}**: {criterion.description}\n"

        # Format coding standards
        standards_section = "\n## Coding Standards\n"
        if profile.naming_conventions:
            standards_section += "**Naming Conventions:**\n"
            for scope, pattern in profile.naming_conventions.items():
                standards_section += f"- {scope}: {pattern}\n"

        if profile.error_handling_pattern:
            standards_section += f"\n**Error Handling:** {profile.error_handling_pattern}\n"

        if profile.import_style:
            standards_section += f"**Import Style:** {profile.import_style}\n"

        prompt = f"""Perform a comprehensive code review of the following changes.

## Story
{spec.title}

{spec.description}

{changes_section}

{criteria_section}

{standards_section}

## Review Checklist
Review the code for:

1. **Correctness** - Does it correctly implement the requirements?
   - Logic is sound
   - Edge cases handled
   - No obvious bugs

2. **Security** - Are there security vulnerabilities?
   - Input validation
   - Injection risks (SQL, command, etc.)
   - Proper authentication/authorization
   - Secure defaults

3. **Performance** - Are there performance issues?
   - Algorithm efficiency
   - Database query optimization (N+1, etc.)
   - Unnecessary allocations
   - Caching opportunities

4. **Maintainability** - Is the code maintainable?
   - Code clarity and readability
   - Proper comments for complex logic
   - DRY principle (Don't Repeat Yourself)
   - Proper abstraction levels

5. **Compliance** - Does it follow standards?
   - Naming conventions
   - Error handling patterns
   - Import style
   - Acceptance criteria met

## Response Format
Respond with a JSON object:
{{
  "issues": [
    {{
      "severity": "blocking|suggestion|nit",
      "category": "security|performance|correctness|style|maintainability",
      "file": "<filename>",
      "line": <line number or null>,
      "description": "<issue description>",
      "suggested_fix": "<optional fix>"
    }},
    ...
  ],
  "summary": "<overall assessment>",
  "approval": <true|false>
}}

IMPORTANT:
- Set severity="blocking" ONLY for issues that prevent deployment
- Use "suggestion" for improvements
- Use "nit" for minor style/formatting
- Include line numbers when possible
- Be constructive and specific"""

        return prompt

    def _parse_review_response(self, response_text: str) -> tuple[list[ReviewIssue], str]:
        """Parse review response and extract issues."""
        issues: list[ReviewIssue] = []
        summary = ""

        # Try to extract JSON
        json_str = response_text.strip()

        # Find JSON block if wrapped in code fence
        if "```" in json_str:
            match = re.search(r"```(?:json)?\s*\n(.*?)\n```", json_str, re.DOTALL)
            if match:
                json_str = match.group(1).strip()

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Fallback: create a generic issue
            return [
                ReviewIssue(
                    severity="suggestion",
                    category="correctness",
                    file="unknown",
                    description="Review completed. Check summary for details.",
                    suggested_fix=None,
                )
            ], response_text[:500]

        # Parse issues
        if "issues" in data:
            for issue_data in data.get("issues", []):
                issue = ReviewIssue(
                    severity=issue_data.get("severity", "suggestion"),
                    category=issue_data.get("category", "correctness"),
                    file=issue_data.get("file", "unknown"),
                    line=issue_data.get("line"),
                    description=issue_data.get("description", ""),
                    suggested_fix=issue_data.get("suggested_fix"),
                )
                issues.append(issue)

        # Extract summary
        summary = data.get("summary", response_text[:200])

        return issues, summary
