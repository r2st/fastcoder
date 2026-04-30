"""Parse tool output into structured formats."""

from __future__ import annotations

import json
import re
from typing import Optional

from fastcoder.types.tools import LintError, ParsedToolOutput, TypeCheckError


class TestReport:
    """Structured test report."""

    def __init__(
        self,
        passed: int = 0,
        failed: int = 0,
        skipped: int = 0,
        duration_sec: float = 0.0,
        coverage: Optional[float] = None,
    ):
        self.passed = passed
        self.failed = failed
        self.skipped = skipped
        self.duration_sec = duration_sec
        self.coverage = coverage

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "duration_sec": self.duration_sec,
            "coverage": self.coverage,
        }


class OutputParser:
    """Parse stdout/stderr from various tools."""

    @staticmethod
    def parse_pytest_output(stdout: str) -> TestReport:
        """Parse pytest output."""
        report = TestReport()

        passed_match = re.search(r"(\d+) passed", stdout)
        if passed_match:
            report.passed = int(passed_match.group(1))

        failed_match = re.search(r"(\d+) failed", stdout)
        if failed_match:
            report.failed = int(failed_match.group(1))

        skipped_match = re.search(r"(\d+) skipped", stdout)
        if skipped_match:
            report.skipped = int(skipped_match.group(1))

        time_match = re.search(r"(\d+\.\d+)s", stdout)
        if time_match:
            report.duration_sec = float(time_match.group(1))

        coverage_match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", stdout)
        if coverage_match:
            report.coverage = float(coverage_match.group(1))

        return report

    @staticmethod
    def parse_jest_output(stdout: str) -> TestReport:
        """Parse Jest output."""
        report = TestReport()

        try:
            json_match = re.search(r"\{.*\"numPassedTests\".*\}", stdout, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                report.passed = data.get("numPassedTests", 0)
                report.failed = data.get("numFailedTests", 0)
                report.skipped = data.get("numPendingTests", 0)
                report.duration_sec = data.get("testResults", [{}])[0].get(
                    "perfStats", {}
                ).get("end", 0) / 1000.0
        except (json.JSONDecodeError, IndexError):
            pass

        if not report.passed and not report.failed:
            passed_match = re.search(r"(\d+) passed", stdout)
            if passed_match:
                report.passed = int(passed_match.group(1))

            failed_match = re.search(r"(\d+) failed", stdout)
            if failed_match:
                report.failed = int(failed_match.group(1))

        return report

    @staticmethod
    def parse_eslint_output(stdout: str) -> list[LintError]:
        """Parse ESLint output."""
        errors = []

        lines = stdout.split("\n")
        for line in lines:
            match = re.match(
                r"(.+?):(\d+):(\d+):\s+(\w+)\s+(.*?)\s+\(([^)]+)\)", line
            )
            if match:
                errors.append(
                    LintError(
                        file=match.group(1),
                        line=int(match.group(2)),
                        column=int(match.group(3)),
                        severity=match.group(4).lower(),
                        message=match.group(5),
                        rule=match.group(6),
                    )
                )

        return errors

    @staticmethod
    def parse_ruff_output(stdout: str) -> list[LintError]:
        """Parse ruff output."""
        errors = []

        lines = stdout.split("\n")
        for line in lines:
            match = re.match(r"(.+?):(\d+):(\d+):\s+([A-Z]\d+)\s+(.*)", line)
            if match:
                errors.append(
                    LintError(
                        file=match.group(1),
                        line=int(match.group(2)),
                        column=int(match.group(3)),
                        rule=match.group(4),
                        message=match.group(5),
                        severity="error",
                    )
                )

        return errors

    @staticmethod
    def parse_tsc_output(stderr: str) -> list[TypeCheckError]:
        """Parse TypeScript compiler output."""
        errors = []

        lines = stderr.split("\n")
        for line in lines:
            match = re.match(
                r"(.+?)\((\d+),(\d+)\):\s+error\s+TS(\d+):\s+(.*)", line
            )
            if match:
                errors.append(
                    TypeCheckError(
                        file=match.group(1),
                        line=int(match.group(2)),
                        column=int(match.group(3)),
                        code=f"TS{match.group(4)}",
                        message=match.group(5),
                    )
                )

        return errors

    @staticmethod
    def parse_mypy_output(stdout: str) -> list[TypeCheckError]:
        """Parse mypy output."""
        errors = []

        lines = stdout.split("\n")
        for line in lines:
            match = re.match(r"(.+?):(\d+):(\d+):\s+error:\s+(.*)", line)
            if match:
                errors.append(
                    TypeCheckError(
                        file=match.group(1),
                        line=int(match.group(2)),
                        column=int(match.group(3)),
                        message=match.group(4),
                    )
                )

        return errors

    @staticmethod
    def create_test_report(report: TestReport) -> ParsedToolOutput:
        """Create parsed tool output for test report."""
        return ParsedToolOutput(type="test_report", data=report.to_dict())

    @staticmethod
    def create_lint_report(errors: list[LintError]) -> ParsedToolOutput:
        """Create parsed tool output for lint report."""
        return ParsedToolOutput(
            type="lint_report",
            data={
                "errors": [e.dict() for e in errors],
                "total": len(errors),
            },
        )

    @staticmethod
    def create_type_check_report(
        errors: list[TypeCheckError],
    ) -> ParsedToolOutput:
        """Create parsed tool output for type check."""
        return ParsedToolOutput(
            type="type_check",
            data={
                "errors": [e.dict() for e in errors],
                "total": len(errors),
            },
        )
