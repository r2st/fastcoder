"""Context Manager — assembles layered context for LLM interactions.

Manages context across multiple layers with token budgets:
- System (2000): Agent instructions, tool definitions
- Project (1500): Profile, directory tree, conventions
- Story (2000): Current story, acceptance criteria, plan summary
- Task (1000): Current task description, targets
- Code (16000): Target files, dependencies, type definitions
- Error (3000): Previous attempts, stack traces (on retries)
- Memory (1500): RAG-retrieved past lessons
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass
from typing import Optional

from fastcoder.types.codebase import ProjectProfile
from fastcoder.types.errors import ErrorContext
from fastcoder.types.llm import Message
from fastcoder.types.memory import MemoryEntry
from fastcoder.types.plan import ExecutionPlan, PlanTask
from fastcoder.types.story import StorySpec

logger = logging.getLogger(__name__)


@dataclass
class TokenBudget:
    """Token budget allocation per context layer."""

    system: int = 2000
    project: int = 1500
    story: int = 2000
    task: int = 1000
    code: int = 16000
    error: int = 3000
    memory: int = 1500

    @property
    def total(self) -> int:
        """Total available tokens."""
        return (
            self.system
            + self.project
            + self.story
            + self.task
            + self.code
            + self.error
            + self.memory
        )


class ContextManager:
    """Assembles and manages layered context for LLM interactions.

    Handles:
    - Context assembly from multiple layers
    - Smart file selection for code context
    - Token budgeting and overflow handling
    - Token estimation and diff generation
    - Memory integration (RAG)
    """

    def __init__(
        self,
        budget: Optional[TokenBudget] = None,
        tokens_per_char: float = 0.25,
    ):
        """Initialize context manager.

        Args:
            budget: Token budget allocation (uses defaults if None)
            tokens_per_char: Estimation factor (~4 chars per token)
        """
        self.budget = budget or TokenBudget()
        self.tokens_per_char = tokens_per_char
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def build_context(
        self,
        story: StorySpec,
        task: PlanTask,
        project_profile: ProjectProfile,
        relevant_files: list[str],
        error_context: Optional[ErrorContext] = None,
        memory_entries: Optional[list[MemoryEntry]] = None,
    ) -> list[Message]:
        """Assemble complete context for LLM interaction.

        Args:
            story: Current StorySpec
            task: Current PlanTask
            project_profile: ProjectProfile with conventions
            relevant_files: List of file paths to include
            error_context: Optional error context (on retries)
            memory_entries: Optional RAG-retrieved memory entries

        Returns:
            list[Message]: Ordered messages ready for LLM
        """
        messages = []
        token_usage = {"tokens": 0}

        # 1. System layer (always included)
        system_msg = self._build_system_message()
        messages.append(system_msg)
        token_usage["tokens"] += self.estimate_tokens(system_msg.content)

        # 2. Project layer
        project_msg = self._build_project_message(project_profile)
        if token_usage["tokens"] + self.estimate_tokens(project_msg.content) <= self.budget.system + self.budget.project:
            messages.append(project_msg)
            token_usage["tokens"] += self.estimate_tokens(project_msg.content)

        # 3. Story layer
        story_msg = self._build_story_message(story)
        if token_usage["tokens"] + self.estimate_tokens(story_msg.content) <= self.budget.system + self.budget.project + self.budget.story:
            messages.append(story_msg)
            token_usage["tokens"] += self.estimate_tokens(story_msg.content)

        # 4. Task layer
        task_msg = self._build_task_message(task)
        if token_usage["tokens"] + self.estimate_tokens(task_msg.content) <= self.budget.system + self.budget.project + self.budget.story + self.budget.task:
            messages.append(task_msg)
            token_usage["tokens"] += self.estimate_tokens(task_msg.content)

        # 5. Code layer (may be truncated)
        code_msg = await self._build_code_message(
            relevant_files,
            self.budget.code,
        )
        if code_msg and token_usage["tokens"] + self.estimate_tokens(code_msg.content) <= self.budget.total:
            messages.append(code_msg)
            token_usage["tokens"] += self.estimate_tokens(code_msg.content)

        # 6. Error layer (only on retries)
        if error_context:
            error_msg = self._build_error_message(error_context)
            if token_usage["tokens"] + self.estimate_tokens(error_msg.content) <= self.budget.total - self.budget.memory:
                messages.append(error_msg)
                token_usage["tokens"] += self.estimate_tokens(error_msg.content)

        # 7. Memory layer (RAG-retrieved lessons)
        if memory_entries:
            memory_msg = self._build_memory_message(memory_entries)
            if token_usage["tokens"] + self.estimate_tokens(memory_msg.content) <= self.budget.total:
                messages.append(memory_msg)
                token_usage["tokens"] += self.estimate_tokens(memory_msg.content)

        self.logger.info(f"Context assembled: {len(messages)} messages, ~{token_usage['tokens']} tokens")
        return messages

    def _build_system_message(self) -> Message:
        """Build system/instructions layer."""
        content = """You are an autonomous software development agent.
Your role:
1. Read and understand the current task
2. Analyze relevant code context
3. Generate correct, complete implementations
4. Write comprehensive tests
5. Handle errors gracefully

Constraints:
- Write real, working code
- Follow project conventions
- Handle edge cases
- Validate inputs
- Add appropriate error handling
- Include docstrings and type hints

Output format:
- Provide code blocks with proper syntax highlighting
- Explain your reasoning
- Note any assumptions or limitations
- Suggest improvements or follow-up work if applicable"""

        return Message(role="system", content=content)

    def _build_project_message(self, profile: ProjectProfile) -> Message:
        """Build project context layer."""
        naming_str = "\n".join(
            f"  {k}: {v}" for k, v in profile.naming_conventions.items()
        )

        content = f"""Project Configuration:
Language: {profile.language}
Framework: {profile.framework or "None"}
Package Manager: {profile.package_manager}
Test Framework: {profile.test_framework}

Naming Conventions:
{naming_str}

Import Style: {profile.import_style}
Error Handling: {profile.error_handling_pattern}

Directory Structure:
{profile.directory_structure or "Standard layout"}"""

        return Message(role="user", content=content)

    def _build_story_message(self, story: StorySpec) -> Message:
        """Build story context layer."""
        criteria_str = "\n".join(
            f"  {c.id}: {c.description}" for c in story.acceptance_criteria
        )

        ambiguities_str = ""
        if story.ambiguities:
            ambiguities_str = "\nNote - Ambiguities:\n"
            ambiguities_str += "\n".join(f"  - {a}" for a in story.ambiguities)

        content = f"""Story: {story.title}
Type: {story.story_type.value}
Complexity: {story.complexity_score}/10

Description:
{story.description}

Acceptance Criteria:
{criteria_str}{ambiguities_str}"""

        return Message(role="user", content=content)

    def _build_task_message(self, task: PlanTask) -> Message:
        """Build task context layer."""
        deps_str = ""
        if task.depends_on:
            deps_str = f"\nDependencies: {', '.join(task.depends_on)}"

        content = f"""Current Task: {task.id}
Action: {task.action.value}
Target: {task.target}

Description:
{task.description}{deps_str}

Estimated Tokens: {task.estimated_tokens}"""

        return Message(role="user", content=content)

    async def _build_code_message(
        self,
        relevant_files: list[str],
        budget: int,
    ) -> Optional[Message]:
        """Build code context layer with smart file selection."""
        if not relevant_files:
            return None

        code_snippets = []
        remaining_budget = budget
        processed_files = set()

        for file_path in relevant_files:
            if remaining_budget <= 100:  # Reserve space
                break

            if file_path in processed_files:
                continue

            try:
                # In real implementation, would read file from disk
                # For now, create a placeholder
                content = f"# File: {file_path}\n# [Content would be loaded from disk]\n"
                file_tokens = self.estimate_tokens(content)

                if file_tokens <= remaining_budget:
                    code_snippets.append(content)
                    remaining_budget -= file_tokens
                    processed_files.add(file_path)
                else:
                    # Include skeleton (signatures only)
                    skeleton = self.extract_skeleton(content)
                    skeleton_tokens = self.estimate_tokens(skeleton)
                    if skeleton_tokens <= remaining_budget:
                        code_snippets.append(skeleton)
                        remaining_budget -= skeleton_tokens
                        processed_files.add(file_path)

            except Exception as e:
                self.logger.debug(f"Could not load file {file_path}: {e}")

        if not code_snippets:
            return None

        content = "Relevant Code Context:\n\n" + "\n\n".join(code_snippets)
        return Message(role="user", content=content)

    def _build_error_message(self, error_context: ErrorContext) -> Message:
        """Build error context layer (on retries)."""
        error_msg = error_context.error.message or "Unknown error"
        error_type = error_context.error.type or "Unknown"

        attempts_str = ""
        if error_context.previous_attempts:
            attempts_str = "\nPrevious Attempts:\n"
            for attempt in error_context.previous_attempts[-3:]:  # Last 3 attempts
                attempts_str += f"  Attempt {attempt.attempt_number}: {attempt.error_message}\n"

        stack_trace_str = ""
        if error_context.error.stack_trace:
            stack_trace_str = f"\nStack Trace:\n{error_context.error.stack_trace[:500]}"

        instruction_str = ""
        if error_context.instruction:
            instruction_str = f"\nRecovery Instructions:\n{error_context.instruction}"

        content = f"""Previous Error (Attempt {error_context.attempt}):
Error Type: {error_type}
Message: {error_msg}{attempts_str}{stack_trace_str}{instruction_str}

Please fix this error and try again."""

        return Message(role="user", content=content)

    def _build_memory_message(self, memory_entries: list[MemoryEntry]) -> Message:
        """Build memory context layer (RAG)."""
        memories_str = "\n".join(
            f"  [{e.type.value}] {e.content}" for e in memory_entries[:5]
        )

        content = f"""Relevant Knowledge from Previous Tasks:
{memories_str}

Consider these patterns when implementing this task."""

        return Message(role="user", content=content)

    def select_files(
        self,
        task: PlanTask,
        dependency_graph: Optional[dict] = None,
        symbol_table: Optional[dict] = None,
    ) -> list[str]:
        """Smart file selection for code context.

        Selects files by:
        - Direct dependencies (imports)
        - Type definitions needed
        - Sibling modules (same package)
        - Recently modified files (recency)

        Args:
            task: Current PlanTask
            dependency_graph: Optional {file: [dependencies]}
            symbol_table: Optional {symbol: [files_defining_it]}

        Returns:
            list[str]: Ordered list of relevant file paths
        """
        selected = set()

        # Start with task target
        if task.target and "/" in task.target:
            selected.add(task.target)

        # Add dependencies
        if dependency_graph and task.target in dependency_graph:
            selected.update(dependency_graph[task.target][:3])

        # Add sibling modules (same directory)
        if task.target:
            dir_path = task.target.rsplit("/", 1)[0] if "/" in task.target else ""
            if dir_path and symbol_table:
                for symbol, files in symbol_table.items():
                    for f in files:
                        if f.startswith(dir_path):
                            selected.add(f)
                            if len(selected) >= 10:
                                break
                    if len(selected) >= 10:
                        break

        return sorted(list(selected))[:20]  # Limit to 20 files

    def extract_skeleton(self, content: str) -> str:
        """Extract function/class signatures from code content.

        Strips implementation details, keeping only:
        - Class definitions
        - Function/method signatures
        - Type hints
        - Docstrings (first line only)

        Args:
            content: Full file content

        Returns:
            str: Skeleton with implementations removed
        """
        lines = []
        skip_block = False
        in_function = False
        indent_level = 0

        for line in content.split("\n"):
            stripped = line.lstrip()

            # Detect function/class definitions
            if stripped.startswith(("def ", "class ", "async def ")):
                in_function = True
                indent_level = len(line) - len(stripped)
                lines.append(line)
                continue

            # Skip function/class bodies
            if in_function:
                current_indent = len(line) - len(stripped)
                if line.strip() and current_indent <= indent_level:
                    in_function = False
                elif "\"\"\"" in line or "'''" in line:
                    # Include docstrings
                    lines.append(line)
                    continue
                else:
                    continue

            # Include imports and comments
            if stripped.startswith(("import ", "from ")) or stripped.startswith("#"):
                lines.append(line)

        return "\n".join(lines)

    def create_diff_context(self, old: str, new: str) -> str:
        """Create unified diff context between two code versions.

        Args:
            old: Previous code version
            new: New code version

        Returns:
            str: Unified diff in context format
        """
        old_lines = old.split("\n")
        new_lines = new.split("\n")

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile="previous",
            tofile="current",
            lineterm="",
        )

        return "\n".join(diff)

    def handle_overflow(
        self,
        messages: list[Message],
        max_tokens: int,
    ) -> list[Message]:
        """Handle context overflow by priority-based eviction.

        Eviction priority (first to last):
        1. Memory entries
        2. Error context
        3. Code context (keep essentials)
        4. (Never evict: System, Story, Task)

        Args:
            messages: Original message list
            max_tokens: Maximum allowed tokens

        Returns:
            list[Message]: Trimmed message list
        """
        current_tokens = sum(
            self.estimate_tokens(m.content) for m in messages
        )

        if current_tokens <= max_tokens:
            return messages

        self.logger.warning(
            f"Context overflow: {current_tokens} > {max_tokens} tokens, evicting"
        )

        # Find priority roles to evict
        evict_order = ["memory", "error", "code"]
        remaining = messages[:]

        for role in evict_order:
            current_tokens = sum(
                self.estimate_tokens(m.content) for m in remaining
            )
            if current_tokens <= max_tokens:
                return remaining

            # Remove first message with this role
            for i, msg in enumerate(remaining):
                if role in msg.content.lower()[:50]:
                    remaining.pop(i)
                    break

        return remaining

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Uses simple heuristic: ~4 characters per token.

        Args:
            text: Text to estimate

        Returns:
            int: Estimated token count
        """
        return max(1, int(len(text) * self.tokens_per_char))
