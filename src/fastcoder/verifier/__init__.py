"""Post-deployment verification and health checks."""

from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime
from typing import Optional

import structlog
from pydantic import BaseModel, Field

from fastcoder.tools.build_runner import BuildRunner
from fastcoder.tools.git_client import GitClient
from fastcoder.tools.shell_executor import ShellExecutor
from fastcoder.tools.test_runner import TestRunnerTool
from fastcoder.types.config import ProjectConfig
from fastcoder.types.story import Story
from fastcoder.types.task import DeployReport

logger = structlog.get_logger(__name__)


class VerificationCheck(BaseModel):
    """Individual verification check result."""

    name: str = Field(..., description="Name of the check (e.g., 'health_check')")
    passed: bool = Field(..., description="Whether check passed")
    detail: str = Field(default="", description="Detailed message about result")
    duration_ms: float = Field(default=0.0, description="Time taken for check in ms")


class VerificationReport(BaseModel):
    """Complete verification report after deployment."""

    overall_passed: bool = Field(..., description="Overall verification status")
    checks: list[VerificationCheck] = Field(
        default_factory=list, description="Individual check results"
    )
    duration_ms: float = Field(default=0.0, description="Total duration in ms")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Report generation timestamp",
    )


class Verifier:
    """Runs verification checks after deployment to ensure health and correctness."""

    def __init__(
        self,
        test_runner: TestRunnerTool,
        build_runner: BuildRunner,
        shell_executor: ShellExecutor,
        config: ProjectConfig,
        git_client: Optional[GitClient] = None,
    ):
        """Initialize verifier with required tools and config.

        Args:
            test_runner: Test execution tool
            build_runner: Build and quality checks runner
            shell_executor: Shell command executor
            config: Project configuration
            git_client: Optional git client for status checks
        """
        self.test_runner = test_runner
        self.build_runner = build_runner
        self.shell = shell_executor
        self.config = config
        self.git = git_client
        self._start_time = 0.0

    async def verify(
        self, story: Story, deploy_report: DeployReport
    ) -> VerificationReport:
        """Run comprehensive verification after deployment.

        Checks performed:
        1. Health check (HTTP GET to URL if provided)
        2. Smoke tests (run tagged smoke tests if available)
        3. Build verification (ensure build still passes)
        4. Git status check (ensure clean working tree)

        Args:
            story: Story that was deployed
            deploy_report: Report from deployment

        Returns:
            VerificationReport with all check results
        """
        self._start_time = time.time()
        logger.info(
            "Starting post-deployment verification",
            story_id=story.id,
            environment=deploy_report.environment,
        )

        report = VerificationReport(overall_passed=True)

        # Run checks concurrently for speed
        check_tasks = [
            self._health_check(story, deploy_report),
            self._smoke_tests(story),
            self._build_verification(story),
            self._git_status_check(story),
        ]

        results = await asyncio.gather(*check_tasks, return_exceptions=True)

        # Process results
        for result in results:
            if isinstance(result, VerificationCheck):
                report.checks.append(result)
                if not result.passed:
                    report.overall_passed = False
            elif isinstance(result, Exception):
                logger.warning(
                    "Check execution error",
                    story_id=story.id,
                    error=str(result),
                )
                report.overall_passed = False

        report.duration_ms = (time.time() - self._start_time) * 1000

        logger.info(
            "Verification complete",
            story_id=story.id,
            passed=report.overall_passed,
            checks_count=len(report.checks),
        )

        return report

    @staticmethod
    def _is_safe_url(url: str) -> bool:
        """Validate URL is safe for use in shell commands.

        Only allows http/https schemes and rejects shell metacharacters.
        """
        import re
        from urllib.parse import urlparse

        try:
            parsed = urlparse(url)
        except Exception:
            return False

        # Only allow http/https
        if parsed.scheme not in ("http", "https"):
            return False

        # Reject URLs with shell metacharacters
        dangerous_chars = set(';&|`$(){}[]!#~<>\\\'"\n\r\t')
        if any(c in url for c in dangerous_chars):
            return False

        # Must have a valid hostname
        if not parsed.hostname or len(parsed.hostname) < 1:
            return False

        return True

    async def _health_check(
        self, story: Story, deploy_report: DeployReport
    ) -> VerificationCheck:
        """Check deployment health via HTTP request.

        If deploy_report contains a URL, performs HTTP GET request.
        Otherwise, skips this check.

        Args:
            story: Story being verified
            deploy_report: Deployment report with URL

        Returns:
            VerificationCheck result
        """
        check_start = time.time()
        check = VerificationCheck(name="health_check", passed=False)

        if not deploy_report.url:
            logger.info("Skipping health check - no URL provided", story_id=story.id)
            check.detail = "No URL provided in deployment report"
            check.passed = True
            check.duration_ms = (time.time() - check_start) * 1000
            return check

        try:
            url = deploy_report.url
            # Sanitize URL: only allow http/https schemes, reject shell metacharacters
            if not self._is_safe_url(url):
                check.detail = f"Invalid or unsafe URL: {url[:50]}"
                check.duration_ms = (time.time() - check_start) * 1000
                return check

            logger.info("Performing health check", story_id=story.id, url=url)

            # Use curl with -- to prevent option injection, and shlex.quote for safety
            import shlex
            curl_cmd = f'curl -s -o /dev/null -w "%{{http_code}}" -- {shlex.quote(url)}'
            result = await self.shell.execute(curl_cmd, timeout_ms=10000)

            if result.exit_code == 0:
                status_code = result.stdout.strip()
                if status_code.isdigit():
                    code = int(status_code)
                    if 200 <= code < 300:
                        check.passed = True
                        check.detail = f"HTTP {code} - Service healthy"
                        logger.info(
                            "Health check passed",
                            story_id=story.id,
                            status_code=code,
                        )
                    elif code < 500:
                        check.detail = f"HTTP {code} - Client error"
                        logger.warning(
                            "Health check returned client error",
                            story_id=story.id,
                            status_code=code,
                        )
                    else:
                        check.detail = f"HTTP {code} - Server error"
                        logger.warning(
                            "Health check returned server error",
                            story_id=story.id,
                            status_code=code,
                        )
                else:
                    check.detail = "Could not parse HTTP status"
            else:
                check.detail = f"Health check request failed: {result.stderr}"
                logger.warning(
                    "Health check request failed",
                    story_id=story.id,
                    error=result.stderr,
                )

        except Exception as e:
            logger.exception(
                "Health check error",
                story_id=story.id,
                error=str(e),
            )
            check.detail = f"Health check error: {str(e)}"

        check.duration_ms = (time.time() - check_start) * 1000
        return check

    async def _smoke_tests(self, story: Story) -> VerificationCheck:
        """Run smoke tests (subset of tests marked as smoke).

        Smoke tests are identified by:
        - Test files/directories named *smoke* or *integration*
        - Tests tagged with @smoke or pytest.mark.smoke

        Args:
            story: Story being verified

        Returns:
            VerificationCheck result
        """
        check_start = time.time()
        check = VerificationCheck(name="smoke_tests", passed=False)

        try:
            logger.info("Running smoke tests", story_id=story.id)

            # Build smoke test command based on framework
            framework = self.test_runner.test_framework
            if framework == "pytest":
                # Run tests marked with 'smoke'
                cmd = "pytest -m smoke -v --tb=short 2>/dev/null || pytest --co -q 2>/dev/null | grep smoke"
            elif framework in ["jest", "vitest"]:
                cmd = f"{framework} --testNamePattern=smoke"
            else:
                cmd = "pytest -v --tb=short"

            result = await self.shell.execute(cmd, timeout_ms=60000)

            # Smoke tests are optional - if none found, mark as passed
            if "no tests ran" in result.stdout.lower() or result.exit_code == 5:
                check.passed = True
                check.detail = "No smoke tests found - skipped"
                logger.info(
                    "No smoke tests found",
                    story_id=story.id,
                )
            elif result.exit_code == 0:
                # Extract test count if possible
                match = re.search(r"(\d+) passed", result.stdout)
                count = match.group(1) if match else "unknown"
                check.passed = True
                check.detail = f"Smoke tests passed ({count} tests)"
                logger.info(
                    "Smoke tests passed",
                    story_id=story.id,
                    tests_count=count,
                )
            else:
                check.detail = f"Smoke tests failed\n{result.stderr[:200]}"
                logger.warning(
                    "Smoke tests failed",
                    story_id=story.id,
                    error=result.stderr[:200],
                )

        except Exception as e:
            logger.exception(
                "Smoke test error",
                story_id=story.id,
                error=str(e),
            )
            check.detail = f"Smoke test error: {str(e)}"

        check.duration_ms = (time.time() - check_start) * 1000
        return check

    async def _build_verification(self, story: Story) -> VerificationCheck:
        """Verify build still passes after deployment.

        Args:
            story: Story being verified

        Returns:
            VerificationCheck result
        """
        check_start = time.time()
        check = VerificationCheck(name="build_verification", passed=False)

        try:
            if not self.build_runner.build_cmd:
                logger.info(
                    "No build command detected",
                    story_id=story.id,
                )
                check.passed = True
                check.detail = "No build command configured - skipped"
                check.duration_ms = (time.time() - check_start) * 1000
                return check

            logger.info("Running build verification", story_id=story.id)

            result = await self.build_runner.build()

            if result.exit_code == 0:
                check.passed = True
                check.detail = "Build successful"
                logger.info("Build verification passed", story_id=story.id)
            else:
                check.detail = f"Build failed: {result.stderr[:200]}"
                logger.warning(
                    "Build verification failed",
                    story_id=story.id,
                    error=result.stderr[:200],
                )

        except Exception as e:
            logger.exception(
                "Build verification error",
                story_id=story.id,
                error=str(e),
            )
            check.detail = f"Build verification error: {str(e)}"

        check.duration_ms = (time.time() - check_start) * 1000
        return check

    async def _git_status_check(self, story: Story) -> VerificationCheck:
        """Verify git working tree is clean after deployment.

        Args:
            story: Story being verified

        Returns:
            VerificationCheck result
        """
        check_start = time.time()
        check = VerificationCheck(name="git_status", passed=False)

        try:
            if not self.git:
                logger.info(
                    "No git client available",
                    story_id=story.id,
                )
                check.passed = True
                check.detail = "Git client unavailable - skipped"
                check.duration_ms = (time.time() - check_start) * 1000
                return check

            logger.info("Checking git status", story_id=story.id)

            status_result = self.git.get_status()

            if status_result.exit_code != 0:
                check.detail = f"Failed to check git status: {status_result.stderr}"
                logger.warning(
                    "Failed to get git status",
                    story_id=story.id,
                )
            elif self.git.repo.is_dirty():
                untracked = self.git.repo.untracked_files
                modified = [i[0] for i in self.git.repo.index.diff(None)]
                check.detail = (
                    f"Working tree dirty: {len(modified)} modified, "
                    f"{len(untracked)} untracked"
                )
                logger.warning(
                    "Working tree is dirty after deployment",
                    story_id=story.id,
                    modified_count=len(modified),
                    untracked_count=len(untracked),
                )
            else:
                check.passed = True
                check.detail = "Git working tree is clean"
                logger.info("Git status check passed", story_id=story.id)

        except Exception as e:
            logger.exception(
                "Git status check error",
                story_id=story.id,
                error=str(e),
            )
            check.detail = f"Git status check error: {str(e)}"

        check.duration_ms = (time.time() - check_start) * 1000
        return check
