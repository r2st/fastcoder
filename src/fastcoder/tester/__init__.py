"""Test Generator — generates comprehensive test suites from code and specs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

from fastcoder.types.codebase import ProjectProfile
from fastcoder.types.llm import CompletionResponse, Message
from fastcoder.types.plan import PlanTask
from fastcoder.types.story import StorySpec, AcceptanceCriterion
from fastcoder.types.task import FileChange, TestFailure


@dataclass
class TestGenResult:
    """Result from test generation."""

    test_code: str
    test_file: str
    criteria_mapping: dict[str, list[str]] = field(default_factory=dict)
    coverage_estimate: float = 0.0
    edge_cases_covered: list[str] = field(default_factory=list)


class TestGenerator:
    """Generates comprehensive test suites using an LLM."""

    def __init__(self, llm_complete: Callable[[list[Message], dict], CompletionResponse]) -> None:
        """Initialize the test generator.

        Args:
            llm_complete: Async callable that takes messages and metadata, returns CompletionResponse
        """
        self.llm_complete = llm_complete

    async def generate_tests(
        self,
        task: PlanTask,
        code: str,
        spec: StorySpec,
        context: dict[str, Any],
    ) -> TestGenResult:
        """Generate comprehensive tests for code.

        Args:
            task: The PlanTask being tested
            code: The generated code to test
            spec: Story specification with acceptance criteria
            context: Dict with: test_framework, existing_tests, project_conventions

        Returns:
            TestGenResult with test code and criteria mapping
        """
        test_framework = context.get("test_framework", "pytest")
        existing_tests = context.get("existing_tests", "")
        conventions = context.get("project_conventions", "")

        # Build test generation prompt
        prompt = self._build_test_prompt(
            task=task,
            code=code,
            spec=spec,
            test_framework=test_framework,
            existing_tests=existing_tests,
            conventions=conventions,
        )

        # Call LLM
        messages = [Message(role="user", content=prompt)]
        response = await self.llm_complete(messages, {"purpose": "test_generation"})

        # Extract test code
        test_code = self._extract_code(response.content)

        # Generate test file path
        test_file = self._generate_test_file_path(task.target)

        # Build criteria mapping
        criteria_mapping = self._extract_criteria_mapping(test_code, spec)

        # Identify edge cases
        edge_cases = self._extract_edge_cases(test_code)

        # Estimate coverage
        coverage = self._estimate_coverage(test_code, code)

        return TestGenResult(
            test_code=test_code,
            test_file=test_file,
            criteria_mapping=criteria_mapping,
            coverage_estimate=coverage,
            edge_cases_covered=edge_cases,
        )

    async def generate_regression_test(
        self,
        failure: TestFailure,
        fixed_code: str,
    ) -> str:
        """Generate a regression test for a bug fix.

        Args:
            failure: The test failure that was fixed
            fixed_code: The code that fixes the issue

        Returns:
            Test code that locks in the fix
        """
        prompt = f"""Create a regression test that ensures this bug does not resurface.

## Original Bug
Test: {failure.test}
Suite: {failure.suite}
Error: {failure.error}

Expected: {failure.expected}
Actual: {failure.actual}

## Fixed Code
```python
{fixed_code}
```

## Requirements
1. Test the specific scenario that caused the original failure
2. Use @pytest.mark.regression decorator
3. Include docstring explaining what was fixed
4. Test both the fix and edge cases around it

Return the regression test code in a pytest code block."""

        messages = [Message(role="user", content=prompt)]
        response = await self.llm_complete(messages, {"purpose": "regression_testing"})

        test_code = self._extract_code(response.content)
        return test_code

    def _build_test_prompt(
        self,
        task: PlanTask,
        code: str,
        spec: StorySpec,
        test_framework: str,
        existing_tests: str,
        conventions: str,
    ) -> str:
        """Build a comprehensive test generation prompt."""
        # Build acceptance criteria section
        criteria_section = ""
        if spec.acceptance_criteria:
            criteria_lines = []
            for criterion in spec.acceptance_criteria:
                criteria_lines.append(f"  - AC-{criterion.id}: {criterion.description}")
            criteria_section = "\n## Acceptance Criteria\n" + "\n".join(criteria_lines)

        # Build existing tests reference
        existing_section = ""
        if existing_tests:
            existing_section = f"\n## Existing Tests (for patterns)\n```\n{existing_tests[:500]}\n```"

        # Build conventions section
        conventions_section = ""
        if conventions:
            conventions_section = f"\n## Project Conventions\n{conventions}"

        prompt = f"""Generate comprehensive tests for the following code using {test_framework}.

## Task
{task.description}

## Code to Test
```python
{code}
```

{criteria_section}
{existing_section}
{conventions_section}

## Requirements
1. Generate unit tests covering:
   - Happy path / normal behavior
   - Edge cases (null, empty, boundary values)
   - Error conditions and exception handling
   - Input validation

2. Link tests to acceptance criteria using @pytest.mark.criterion(criterion_id="AC-X") decorators

3. Include docstrings explaining what each test verifies

4. Cover these edge cases:
   - Null/None inputs
   - Empty collections
   - Boundary values
   - Invalid types
   - Exception paths
   - Concurrent access (if applicable)

5. Use clear, descriptive test names following pattern: test_<function>_<scenario>_<expected_result>

Respond with the complete test code in a pytest code block (```python)."""

        return prompt

    def _extract_code(self, response_text: str) -> str:
        """Extract code from markdown code blocks."""
        patterns = [
            r"```(?:python|py)?\n(.*?)\n```",
            r"```\n(.*?)\n```",
        ]

        for pattern in patterns:
            match = re.search(pattern, response_text, re.DOTALL)
            if match:
                return match.group(1).strip()

        return ""

    def _generate_test_file_path(self, source_file: str) -> str:
        """Generate test file path from source file path.

        Examples:
            src/auth/login.py -> tests/auth/test_login.py
            lib/utils.py -> tests/test_utils.py
        """
        # Replace src/ with tests/, lib/ with tests/, add test_ prefix
        if "/src/" in source_file:
            test_file = source_file.replace("/src/", "/tests/")
        elif "/lib/" in source_file:
            test_file = source_file.replace("/lib/", "/tests/")
        else:
            # Default: put in tests/ folder
            parts = source_file.split("/")
            test_file = "tests/" + "/".join(parts)

        # Ensure test_ prefix on file name
        if not test_file.endswith("test_"):
            dirname = "/".join(test_file.split("/")[:-1])
            basename = test_file.split("/")[-1]
            if not basename.startswith("test_"):
                basename = "test_" + basename
            test_file = f"{dirname}/{basename}" if dirname else f"tests/{basename}"

        return test_file

    def _extract_criteria_mapping(self, test_code: str, spec: StorySpec) -> dict[str, list[str]]:
        """Extract mapping of acceptance criteria to test functions.

        Returns:
            Dict mapping criterion ID to list of test function names
        """
        mapping: dict[str, list[str]] = {}

        # Find all @pytest.mark.criterion decorators
        criterion_pattern = r'@pytest\.mark\.criterion\(criterion_id="([^"]+)"\)'
        test_pattern = r'def (test_\w+)\('

        # Find all test functions and their preceding decorators
        lines = test_code.split("\n")
        current_criterion = None

        for i, line in enumerate(lines):
            criterion_match = re.search(criterion_pattern, line)
            if criterion_match:
                current_criterion = criterion_match.group(1)

            test_match = re.search(test_pattern, line)
            if test_match and current_criterion:
                test_func = test_match.group(1)
                if current_criterion not in mapping:
                    mapping[current_criterion] = []
                mapping[current_criterion].append(test_func)
                current_criterion = None

        return mapping

    def _extract_edge_cases(self, test_code: str) -> list[str]:
        """Extract detected edge cases from test code."""
        edge_cases = []

        patterns = {
            "null/None handling": r"(None|null|empty)",
            "boundary values": r"(boundary|edge|limit|max|min|zero)",
            "error conditions": r"(raises|exception|error|invalid)",
            "empty collections": r"(\[\]|{}|empty)",
            "large inputs": r"(large|performance|scale|many)",
        }

        for case_type, pattern in patterns.items():
            if re.search(pattern, test_code, re.IGNORECASE):
                edge_cases.append(case_type)

        return edge_cases

    def _estimate_coverage(self, test_code: str, source_code: str) -> float:
        """Estimate test coverage based on test count and source complexity."""
        # Count test functions
        test_count = len(re.findall(r"def test_", test_code))

        # Count functions/classes in source
        source_functions = len(re.findall(r"^\s*def ", source_code, re.MULTILINE))
        source_classes = len(re.findall(r"^\s*class ", source_code, re.MULTILINE))
        source_complexity = source_functions + source_classes

        if source_complexity == 0:
            return 0.85  # Assume high coverage for simple code

        # Simple heuristic: tests per unit, capped at 0.95
        coverage = min(test_count / max(source_complexity, 1) * 0.5 + 0.35, 0.95)
        return round(coverage, 2)
