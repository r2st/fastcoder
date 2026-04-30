"""Build and quality tools runner."""

from __future__ import annotations

import json
from pathlib import Path

from fastcoder.tools.output_parser import OutputParser
from fastcoder.tools.shell_executor import ShellExecutor
from fastcoder.types.tools import ToolName, ToolResult


class BuildRunner:
    """Run build, lint, format, and type-checking operations."""

    def __init__(self, project_dir: str, shell_executor: ShellExecutor):
        """Initialize build runner."""
        self.project_dir = Path(project_dir).resolve()
        self.shell = shell_executor
        self.build_cmd = self._detect_build_command()
        self.lint_cmd = self._detect_lint_command()
        self.format_cmd = self._detect_format_command()
        self.type_check_cmd = self._detect_type_check_command()

    def _detect_build_command(self) -> str:
        """Detect build command from package.json or pyproject.toml."""
        package_json = self.project_dir / "package.json"
        if package_json.exists():
            try:
                with open(package_json) as f:
                    data = json.load(f)
                    scripts = data.get("scripts", {})
                    if "build" in scripts:
                        return "npm run build"
            except Exception:
                pass

        pyproject = self.project_dir / "pyproject.toml"
        if pyproject.exists():
            return "python -m build"

        return ""

    def _detect_lint_command(self) -> str:
        """Detect lint command."""
        package_json = self.project_dir / "package.json"
        if package_json.exists():
            try:
                with open(package_json) as f:
                    data = json.load(f)
                    scripts = data.get("scripts", {})
                    if "lint" in scripts:
                        return "npm run lint"
                    deps = {**data.get("devDependencies", {}), **data.get("dependencies", {})}
                    if "eslint" in deps:
                        return "eslint ."
            except Exception:
                pass

        if (self.project_dir / "pyproject.toml").exists():
            return "ruff check ."

        if (self.project_dir / ".ruff.toml").exists():
            return "ruff check ."

        return ""

    def _detect_format_command(self) -> str:
        """Detect format command."""
        package_json = self.project_dir / "package.json"
        if package_json.exists():
            try:
                with open(package_json) as f:
                    data = json.load(f)
                    scripts = data.get("scripts", {})
                    if "format" in scripts:
                        return "npm run format"
                    deps = {**data.get("devDependencies", {}), **data.get("dependencies", {})}
                    if "prettier" in deps:
                        return "prettier --write ."
            except Exception:
                pass

        if (self.project_dir / "pyproject.toml").exists():
            return "ruff format ."

        return ""

    def _detect_type_check_command(self) -> str:
        """Detect type check command."""
        package_json = self.project_dir / "package.json"
        if package_json.exists():
            try:
                with open(package_json) as f:
                    data = json.load(f)
                    scripts = data.get("scripts", {})
                    if "type-check" in scripts:
                        return "npm run type-check"
                    deps = {**data.get("devDependencies", {}), **data.get("dependencies", {})}
                    if "typescript" in deps:
                        return "tsc --noEmit"
            except Exception:
                pass

        if (self.project_dir / "pyproject.toml").exists():
            return "mypy ."

        return ""

    async def build(self) -> ToolResult:
        """Build the project."""
        if not self.build_cmd:
            return ToolResult(
                tool=ToolName.BUILD_TOOLS,
                operation="build",
                exit_code=1,
                stderr="No build command detected",
            )

        result = await self.shell.execute(self.build_cmd)
        return result

    async def lint(self) -> ToolResult:
        """Run linter."""
        if not self.lint_cmd:
            return ToolResult(
                tool=ToolName.BUILD_TOOLS,
                operation="lint",
                exit_code=1,
                stderr="No lint command detected",
            )

        result = await self.shell.execute(self.lint_cmd)

        if "eslint" in self.lint_cmd:
            errors = OutputParser.parse_eslint_output(result.stdout)
            result.parsed = OutputParser.create_lint_report(errors)
        elif "ruff" in self.lint_cmd:
            errors = OutputParser.parse_ruff_output(result.stdout)
            result.parsed = OutputParser.create_lint_report(errors)

        return result

    async def format_code(self) -> ToolResult:
        """Format code."""
        if not self.format_cmd:
            return ToolResult(
                tool=ToolName.BUILD_TOOLS,
                operation="format_code",
                exit_code=1,
                stderr="No format command detected",
            )

        result = await self.shell.execute(self.format_cmd)
        return result

    async def type_check(self) -> ToolResult:
        """Run type checker."""
        if not self.type_check_cmd:
            return ToolResult(
                tool=ToolName.BUILD_TOOLS,
                operation="type_check",
                exit_code=1,
                stderr="No type check command detected",
            )

        result = await self.shell.execute(self.type_check_cmd)

        if "tsc" in self.type_check_cmd:
            errors = OutputParser.parse_tsc_output(result.stderr)
            result.parsed = OutputParser.create_type_check_report(errors)
        elif "mypy" in self.type_check_cmd:
            errors = OutputParser.parse_mypy_output(result.stdout)
            result.parsed = OutputParser.create_type_check_report(errors)

        return result
