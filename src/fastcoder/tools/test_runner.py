"""Test runner tool with auto-detection."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

from fastcoder.tools.output_parser import OutputParser, TestReport
from fastcoder.tools.shell_executor import ShellExecutor
from fastcoder.types.tools import ToolName, ToolResult


class TestRunnerTool:
    """Run tests with auto-detection of framework."""

    def __init__(self, project_dir: str, shell_executor: ShellExecutor):
        """Initialize test runner."""
        self.project_dir = Path(project_dir).resolve()
        self.shell = shell_executor
        self.test_framework = self._detect_test_framework()

    def _detect_test_framework(self) -> str:
        """Auto-detect test framework."""
        package_json = self.project_dir / "package.json"
        if package_json.exists():
            import json

            try:
                with open(package_json) as f:
                    data = json.load(f)
                    deps = {**data.get("devDependencies", {}), **data.get("dependencies", {})}
                    if "vitest" in deps:
                        return "vitest"
                    if "jest" in deps:
                        return "jest"
                    if "mocha" in deps:
                        return "mocha"
            except Exception:
                pass

        pyproject = self.project_dir / "pyproject.toml"
        if pyproject.exists():
            return "pytest"

        if (self.project_dir / "pytest.ini").exists():
            return "pytest"

        if (self.project_dir / "setup.cfg").exists():
            return "pytest"

        return "pytest"

    async def run_all(self) -> ToolResult:
        """Run all tests."""
        if self.test_framework == "pytest":
            cmd = "pytest --tb=short -v"
        elif self.test_framework == "jest":
            cmd = "npm test"
        elif self.test_framework == "vitest":
            cmd = "vitest run"
        elif self.test_framework == "mocha":
            cmd = "mocha"
        else:
            cmd = "pytest"

        result = await self.shell.execute(cmd)

        if self.test_framework == "pytest":
            report = OutputParser.parse_pytest_output(result.stdout + result.stderr)
        elif self.test_framework == "jest":
            report = OutputParser.parse_jest_output(result.stdout)
        elif self.test_framework in ["vitest", "mocha"]:
            report = OutputParser.parse_jest_output(result.stdout)
        else:
            report = TestReport()

        result.parsed = OutputParser.create_test_report(report)
        return result

    def _validate_file_path(self, file: str) -> str:
        """Validate and sanitize a test file path to prevent injection.

        Args:
            file: File path to validate

        Returns:
            Sanitized file path

        Raises:
            ValueError: If file path contains dangerous characters
        """
        # Resolve the path and ensure it's within the project directory
        resolved = (self.project_dir / file).resolve()
        if not str(resolved).startswith(str(self.project_dir)):
            raise ValueError(f"Path traversal attempt: {file}")
        # Only allow safe characters in file paths
        if re.search(r'[;&|`$(){}!#~<>\\\'"*?\n\r\t]', file):
            raise ValueError(f"Unsafe characters in file path: {file}")
        return shlex.quote(file)

    def _validate_test_name(self, test_name: str) -> str:
        """Validate and sanitize a test name to prevent injection.

        Args:
            test_name: Test name/pattern to validate

        Returns:
            Sanitized test name

        Raises:
            ValueError: If test name contains dangerous characters
        """
        # Only allow word characters, dots, brackets, colons, spaces, hyphens
        if re.search(r'[;&|`${}!#~<>\\\n\r\t]', test_name):
            raise ValueError(f"Unsafe characters in test name: {test_name}")
        return shlex.quote(test_name)

    async def run_file(self, file: str) -> ToolResult:
        """Run tests in specific file."""
        safe_file = self._validate_file_path(file)
        if self.test_framework == "pytest":
            cmd = f"pytest {safe_file} -v"
        elif self.test_framework == "jest":
            cmd = f"jest {safe_file}"
        elif self.test_framework == "vitest":
            cmd = f"vitest run {safe_file}"
        elif self.test_framework == "mocha":
            cmd = f"mocha {safe_file}"
        else:
            cmd = f"pytest {safe_file}"

        result = await self.shell.execute(cmd)

        if self.test_framework == "pytest":
            report = OutputParser.parse_pytest_output(result.stdout + result.stderr)
        else:
            report = OutputParser.parse_jest_output(result.stdout)

        result.parsed = OutputParser.create_test_report(report)
        return result

    async def run_single(self, file: str, test_name: str) -> ToolResult:
        """Run single test."""
        safe_file = self._validate_file_path(file)
        safe_name = self._validate_test_name(test_name)
        if self.test_framework == "pytest":
            cmd = f"pytest {safe_file}::{safe_name} -v"
        elif self.test_framework == "jest":
            cmd = f"jest {safe_file} -t {safe_name}"
        elif self.test_framework == "vitest":
            cmd = f"vitest run {safe_file} -t {safe_name}"
        else:
            cmd = f"mocha {safe_file} --grep {safe_name}"

        result = await self.shell.execute(cmd)

        if self.test_framework == "pytest":
            report = OutputParser.parse_pytest_output(result.stdout + result.stderr)
        else:
            report = OutputParser.parse_jest_output(result.stdout)

        result.parsed = OutputParser.create_test_report(report)
        return result

    async def get_coverage(self) -> ToolResult:
        """Get test coverage."""
        if self.test_framework == "pytest":
            cmd = "pytest --cov --cov-report=term-missing"
        elif self.test_framework == "jest":
            cmd = "jest --coverage"
        elif self.test_framework == "vitest":
            cmd = "vitest run --coverage"
        else:
            cmd = "pytest --cov"

        result = await self.shell.execute(cmd)

        coverage_match = None
        if self.test_framework == "pytest":
            report = OutputParser.parse_pytest_output(result.stdout)
            coverage_match = report.coverage
        else:
            import re

            match = re.search(r"Statements\s*:\s*(\d+(?:\.\d+)?)", result.stdout)
            if match:
                coverage_match = float(match.group(1))

        if coverage_match:
            result.parsed = OutputParser.create_test_report(
                TestReport(coverage=coverage_match)
            )

        return result
