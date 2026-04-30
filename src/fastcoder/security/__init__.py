"""SAST (Static Application Security Testing) integration module.

Runs security scanners (Bandit, Semgrep) and secret detection on project files.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import structlog

from fastcoder.types.security import (
    FindingCategory,
    SASTReport,
    SecretFinding,
    SecurityFinding,
    SecurityScanResult,
    Severity,
)

logger = structlog.get_logger(__name__)

# Timeout for individual scanner commands (ms)
SCANNER_TIMEOUT_MS = 120000  # 2 minutes


class SecurityScanner:
    """Runs SAST tools and secret detection on project files."""

    def __init__(self, project_dir: str):
        """Initialize the security scanner.

        Args:
            project_dir: Root directory of the project to scan.
        """
        self.project_dir = Path(project_dir).resolve()
        if not self.project_dir.exists():
            raise ValueError(f"Project directory does not exist: {project_dir}")

    async def scan_files(self, file_paths: list[str]) -> SecurityScanResult:
        """Scan specific files (typically changed files in current iteration).

        Args:
            file_paths: List of file paths to scan (absolute or relative to project_dir).

        Returns:
            SecurityScanResult containing all findings.
        """
        logger.info("Starting security scan for specific files", file_count=len(file_paths))
        start_time = time.time()

        # Normalize paths to be absolute
        normalized_paths = []
        for path in file_paths:
            abs_path = Path(path)
            if not abs_path.is_absolute():
                abs_path = self.project_dir / path
            abs_path = abs_path.resolve()

            # Verify path is within project directory
            if not abs_path.is_relative_to(self.project_dir):
                logger.warning("File path outside project directory, skipping", path=path)
                continue

            if abs_path.exists():
                normalized_paths.append(str(abs_path))
            else:
                logger.warning("File does not exist, skipping", path=path)

        if not normalized_paths:
            logger.info("No valid files to scan")
            return SecurityScanResult()

        return await self._run_scanners(normalized_paths)

    async def full_scan(self) -> SecurityScanResult:
        """Scan the entire project directory.

        Returns:
            SecurityScanResult containing all findings.
        """
        logger.info("Starting full project security scan", project_dir=str(self.project_dir))
        start_time = time.time()

        # Get all relevant files in the project
        file_paths = self._get_project_files()
        logger.info("Found files to scan", file_count=len(file_paths))

        if not file_paths:
            logger.info("No files found to scan")
            return SecurityScanResult()

        return await self._run_scanners(file_paths)

    async def _run_scanners(self, file_paths: list[str]) -> SecurityScanResult:
        """Run all configured scanners on the given files.

        Args:
            file_paths: List of absolute file paths to scan.

        Returns:
            SecurityScanResult with findings from all scanners.
        """
        result = SecurityScanResult()
        start_time = time.time()

        # Run scanners concurrently
        tasks = [
            self._run_bandit(file_paths),
            self._run_semgrep(file_paths),
            self._detect_secrets(file_paths),
        ]

        try:
            bandit_report, semgrep_report, secret_findings = await asyncio.gather(*tasks)

            if bandit_report:
                result.sast_reports.append(bandit_report)

            if semgrep_report:
                result.sast_reports.append(semgrep_report)

            if secret_findings:
                result.secret_findings.extend(secret_findings)

        except Exception as e:
            logger.error("Error running scanners", error=str(e), exc_info=True)

        # Aggregate results
        self._aggregate_results(result)
        scan_duration_ms = int((time.time() - start_time) * 1000)

        for report in result.sast_reports:
            if report.scan_duration_ms == 0:
                report.scan_duration_ms = scan_duration_ms

        logger.info(
            "Security scan completed",
            duration_ms=scan_duration_ms,
            total_findings=result.total_findings,
            critical=result.critical_count,
            high=result.high_count,
            passed=result.passed,
        )

        return result

    async def _run_bandit(self, file_paths: list[str]) -> Optional[SASTReport]:
        """Run Bandit on Python files.

        Args:
            file_paths: List of files to scan.

        Returns:
            SASTReport with findings, or None if Bandit is not available.
        """
        logger.info("Running Bandit security scan", file_count=len(file_paths))
        start_time = time.time()

        # Filter for Python files
        python_files = [f for f in file_paths if f.endswith(".py")]
        if not python_files:
            logger.debug("No Python files to scan with Bandit")
            return None

        # Build Bandit command
        cmd = [
            "bandit",
            "-f",
            "json",
            "-r",
            *python_files,
        ]

        try:
            stdout = await self._run_tool(cmd, SCANNER_TIMEOUT_MS)

            report = SASTReport(
                scanner="bandit",
                scanned_files=len(python_files),
                scan_duration_ms=int((time.time() - start_time) * 1000),
            )

            if not stdout:
                logger.debug("No output from Bandit")
                return report

            try:
                data = json.loads(stdout)
            except json.JSONDecodeError:
                logger.warning("Failed to parse Bandit JSON output")
                return report

            # Parse Bandit results
            for result in data.get("results", []):
                severity = self._map_bandit_severity(result.get("severity", "MEDIUM"))
                category = self._map_bandit_test_id(result.get("test_id", ""))

                finding = SecurityFinding(
                    id=f"bandit_{result.get('test_id', 'unknown')}_{result.get('line_number', 0)}",
                    severity=severity,
                    category=category,
                    title=result.get("test", ""),
                    description=result.get("issue_text", ""),
                    file_path=result.get("filename", ""),
                    line_number=result.get("line_number"),
                    scanner="bandit",
                    rule_id=result.get("test_id", ""),
                    snippet=result.get("code", ""),
                )
                report.findings.append(finding)

            report.passed = all(f.severity not in [Severity.CRITICAL, Severity.HIGH] for f in report.findings)
            logger.info("Bandit scan complete", finding_count=len(report.findings), passed=report.passed)
            return report

        except FileNotFoundError:
            logger.warning("Bandit not installed, skipping Bandit scan")
            return None
        except Exception as e:
            logger.error("Error running Bandit", error=str(e), exc_info=True)
            return None

    async def _run_semgrep(self, file_paths: list[str]) -> Optional[SASTReport]:
        """Run Semgrep on all files.

        Args:
            file_paths: List of files to scan.

        Returns:
            SASTReport with findings, or None if Semgrep is not available.
        """
        logger.info("Running Semgrep security scan", file_count=len(file_paths))
        start_time = time.time()

        if not file_paths:
            return None

        # Build Semgrep command
        cmd = [
            "semgrep",
            "--json",
            "--no-error",
            "--quiet",
            *file_paths,
        ]

        try:
            stdout = await self._run_tool(cmd, SCANNER_TIMEOUT_MS)

            report = SASTReport(
                scanner="semgrep",
                scanned_files=len(file_paths),
                scan_duration_ms=int((time.time() - start_time) * 1000),
            )

            if not stdout:
                logger.debug("No output from Semgrep")
                return report

            try:
                data = json.loads(stdout)
            except json.JSONDecodeError:
                logger.warning("Failed to parse Semgrep JSON output")
                return report

            # Parse Semgrep results
            for result in data.get("results", []):
                severity = self._map_semgrep_severity(result)
                category = FindingCategory.OTHER

                finding = SecurityFinding(
                    id=f"semgrep_{result.get('check_id', 'unknown')}_{result.get('start', {}).get('line', 0)}",
                    severity=severity,
                    category=category,
                    title=result.get("check_id", ""),
                    description=result.get("message", ""),
                    file_path=result.get("path", ""),
                    line_number=result.get("start", {}).get("line"),
                    column=result.get("start", {}).get("col"),
                    scanner="semgrep",
                    rule_id=result.get("check_id", ""),
                    snippet=result.get("extra", {}).get("lines", ""),
                )
                report.findings.append(finding)

            report.passed = all(f.severity not in [Severity.CRITICAL, Severity.HIGH] for f in report.findings)
            logger.info("Semgrep scan complete", finding_count=len(report.findings), passed=report.passed)
            return report

        except FileNotFoundError:
            logger.warning("Semgrep not installed, skipping Semgrep scan")
            return None
        except Exception as e:
            logger.error("Error running Semgrep", error=str(e), exc_info=True)
            return None

    async def _detect_secrets(self, file_paths: list[str]) -> list[SecretFinding]:
        """Run regex + entropy-based secret detection.

        Args:
            file_paths: List of files to scan.

        Returns:
            List of SecretFinding objects.
        """
        logger.info("Running secret detection", file_count=len(file_paths))
        findings = []

        for file_path in file_paths:
            try:
                path = Path(file_path)
                if not path.exists() or not path.is_file():
                    continue

                # Skip large files (>10MB)
                if path.stat().st_size > 10 * 1024 * 1024:
                    logger.debug("Skipping large file", path=file_path)
                    continue

                # Skip binary files
                if self._is_binary_file(file_path):
                    continue

                content = path.read_text(errors="ignore")
                file_findings = self._detect_secrets_in_content(content, file_path)
                findings.extend(file_findings)

            except Exception as e:
                logger.warning("Error reading file for secret detection", path=file_path, error=str(e))
                continue

        logger.info("Secret detection complete", finding_count=len(findings))
        return findings

    def _detect_secrets_in_content(self, content: str, file_path: str) -> list[SecretFinding]:
        """Detect secrets in file content using regex patterns and entropy analysis.

        Args:
            content: File content.
            file_path: Path to the file (for reporting).

        Returns:
            List of SecretFinding objects.
        """
        findings = []
        lines = content.split("\n")

        # AWS Access Key patterns
        aws_pattern = re.compile(r"AKIA[0-9A-Z]{16}")
        for line_num, line in enumerate(lines, 1):
            for match in aws_pattern.finditer(line):
                finding = SecretFinding(
                    id=f"secret_aws_{line_num}_{match.start()}",
                    severity=Severity.CRITICAL,
                    category=FindingCategory.SECRET_LEAK,
                    title="AWS Access Key Exposed",
                    description="AWS access key found in code",
                    file_path=file_path,
                    line_number=line_num,
                    column=match.start(),
                    scanner="secret-detection",
                    rule_id="aws_access_key",
                    secret_type="api_key",
                    snippet=line.strip()[:100],
                )
                findings.append(finding)

        # GitHub token patterns
        github_pattern = re.compile(r"(ghp_|gho_|ghs_|github_pat_)[a-zA-Z0-9_]{36,255}")
        for line_num, line in enumerate(lines, 1):
            for match in github_pattern.finditer(line):
                finding = SecretFinding(
                    id=f"secret_github_{line_num}_{match.start()}",
                    severity=Severity.CRITICAL,
                    category=FindingCategory.SECRET_LEAK,
                    title="GitHub Token Exposed",
                    description="GitHub personal access token found in code",
                    file_path=file_path,
                    line_number=line_num,
                    column=match.start(),
                    scanner="secret-detection",
                    rule_id="github_token",
                    secret_type="token",
                    snippet=line.strip()[:100],
                )
                findings.append(finding)

        # Generic API key patterns
        api_key_pattern = re.compile(r"(?:api_key|apikey|api-key)\s*[=:]\s*['\"]?([a-zA-Z0-9_\-]{20,})['\"]?", re.IGNORECASE)
        for line_num, line in enumerate(lines, 1):
            for match in api_key_pattern.finditer(line):
                key_value = match.group(1) if match.groups() else match.group(0)
                entropy = self._calculate_shannon_entropy(key_value)

                if entropy > 4.5:
                    finding = SecretFinding(
                        id=f"secret_api_{line_num}_{match.start()}",
                        severity=Severity.HIGH,
                        category=FindingCategory.SECRET_LEAK,
                        title="API Key Exposed",
                        description="High entropy string matching API key pattern",
                        file_path=file_path,
                        line_number=line_num,
                        column=match.start(),
                        scanner="secret-detection",
                        rule_id="generic_api_key",
                        secret_type="api_key",
                        entropy=entropy,
                        snippet=line.strip()[:100],
                    )
                    findings.append(finding)

        # Private key patterns
        private_key_pattern = re.compile(r"-----BEGIN\s+(?:RSA|DSA|EC|OPENSSH|PGP|ENCRYPTED)?\s*PRIVATE\s+KEY.*?-----END", re.DOTALL)
        for match in private_key_pattern.finditer(content):
            lines_before = content[:match.start()].count("\n") + 1
            finding = SecretFinding(
                id=f"secret_privkey_{lines_before}",
                severity=Severity.CRITICAL,
                category=FindingCategory.SECRET_LEAK,
                title="Private Key Exposed",
                description="Private key material found in code",
                file_path=file_path,
                line_number=lines_before,
                scanner="secret-detection",
                rule_id="private_key",
                secret_type="private_key",
                snippet="<REDACTED>",
            )
            findings.append(finding)

        # Password in config patterns
        password_pattern = re.compile(
            r"(?:password|passwd|pwd)\s*[=:]\s*['\"]?([^\s'\"]+)['\"]?",
            re.IGNORECASE,
        )
        for line_num, line in enumerate(lines, 1):
            # Skip common test/fake passwords
            if any(fake in line.lower() for fake in ["test", "example", "changeme", "temp"]):
                continue

            for match in password_pattern.finditer(line):
                password_value = match.group(1) if match.groups() else match.group(0)
                entropy = self._calculate_shannon_entropy(password_value)

                # Only flag if it looks like a real password (length + entropy)
                if len(password_value) >= 8 and entropy > 3.5:
                    finding = SecretFinding(
                        id=f"secret_password_{line_num}_{match.start()}",
                        severity=Severity.HIGH,
                        category=FindingCategory.HARDCODED_CREDENTIALS,
                        title="Hardcoded Password",
                        description="Password hardcoded in configuration",
                        file_path=file_path,
                        line_number=line_num,
                        column=match.start(),
                        scanner="secret-detection",
                        rule_id="hardcoded_password",
                        secret_type="password",
                        entropy=entropy,
                        snippet=line.strip()[:100],
                    )
                    findings.append(finding)

        # JWT token patterns
        jwt_pattern = re.compile(r"eyJ[a-zA-Z0-9_\-]+\.eyJ[a-zA-Z0-9_\-]+\.[a-zA-Z0-9_\-]+")
        for line_num, line in enumerate(lines, 1):
            for match in jwt_pattern.finditer(line):
                finding = SecretFinding(
                    id=f"secret_jwt_{line_num}_{match.start()}",
                    severity=Severity.HIGH,
                    category=FindingCategory.SECRET_LEAK,
                    title="JWT Token Exposed",
                    description="JWT token found in code",
                    file_path=file_path,
                    line_number=line_num,
                    column=match.start(),
                    scanner="secret-detection",
                    rule_id="jwt_token",
                    secret_type="token",
                    snippet=line.strip()[:100],
                )
                findings.append(finding)

        # Database connection strings
        db_pattern = re.compile(
            r"(?:database|db)_?(?:url|uri|connection)\s*[=:]\s*['\"]?([^'\";\s]+:[^'\";\s]+@[^'\";\s]+)['\"]?",
            re.IGNORECASE,
        )
        for line_num, line in enumerate(lines, 1):
            for match in db_pattern.finditer(line):
                finding = SecretFinding(
                    id=f"secret_db_{line_num}_{match.start()}",
                    severity=Severity.HIGH,
                    category=FindingCategory.SECRET_LEAK,
                    title="Database Connection String",
                    description="Database credentials embedded in connection string",
                    file_path=file_path,
                    line_number=line_num,
                    column=match.start(),
                    scanner="secret-detection",
                    rule_id="database_connection",
                    secret_type="database_url",
                    snippet=line.strip()[:100],
                )
                findings.append(finding)

        # Slack tokens
        slack_pattern = re.compile(r"(xoxb-[a-zA-Z0-9\-]{10,32}|xoxp-[a-zA-Z0-9\-]{10,32}|xoxs-[a-zA-Z0-9\-]{10,32})")
        for line_num, line in enumerate(lines, 1):
            for match in slack_pattern.finditer(line):
                finding = SecretFinding(
                    id=f"secret_slack_{line_num}_{match.start()}",
                    severity=Severity.HIGH,
                    category=FindingCategory.SECRET_LEAK,
                    title="Slack Token Exposed",
                    description="Slack API token found in code",
                    file_path=file_path,
                    line_number=line_num,
                    column=match.start(),
                    scanner="secret-detection",
                    rule_id="slack_token",
                    secret_type="token",
                    snippet=line.strip()[:100],
                )
                findings.append(finding)

        # High entropy base64 strings (potential secrets)
        base64_pattern = re.compile(r"(?:[A-Za-z0-9+/]{40,}={0,2})")
        for line_num, line in enumerate(lines, 1):
            for match in base64_pattern.finditer(line):
                b64_str = match.group(0)

                # Skip if it looks like code or common non-secret base64
                if any(skip in line.lower() for skip in ["import", "require", "data:image", "charset"]):
                    continue

                entropy = self._calculate_shannon_entropy(b64_str)
                if entropy > 4.5 and len(b64_str) >= 40:
                    finding = SecretFinding(
                        id=f"secret_b64_{line_num}_{match.start()}",
                        severity=Severity.LOW,
                        category=FindingCategory.SECRET_LEAK,
                        title="High Entropy Base64 String",
                        description="Base64-encoded string with high entropy (possible secret)",
                        file_path=file_path,
                        line_number=line_num,
                        column=match.start(),
                        scanner="secret-detection",
                        rule_id="high_entropy_base64",
                        secret_type="base64_string",
                        entropy=entropy,
                        snippet=line.strip()[:100],
                    )
                    findings.append(finding)

        return findings

    @staticmethod
    def _calculate_shannon_entropy(data: str) -> float:
        """Calculate Shannon entropy of a string.

        Used to detect high-entropy strings that are likely secrets.

        Args:
            data: String to analyze.

        Returns:
            Shannon entropy value (0-8 typical range).
        """
        if not data:
            return 0.0

        # Count character frequencies
        char_counts = Counter(data)
        entropy = 0.0

        for count in char_counts.values():
            p = count / len(data)
            entropy -= p * math.log2(p)

        return entropy

    async def _run_tool(self, cmd: list[str], timeout_ms: int) -> str:
        """Run a CLI tool and return stdout.

        Args:
            cmd: Command to run (as list).
            timeout_ms: Timeout in milliseconds.

        Raises:
            FileNotFoundError: If the tool is not found.
            asyncio.TimeoutError: If the tool times out.
            Exception: For other errors.

        Returns:
            Standard output from the command.
        """
        timeout_sec = timeout_ms / 1000.0

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.project_dir),
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout_sec,
                )
                return stdout.decode("utf-8", errors="replace")
            except asyncio.TimeoutError:
                process.kill()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    process.kill()
                raise TimeoutError(f"Tool timed out after {timeout_ms}ms")

        except FileNotFoundError:
            raise FileNotFoundError(f"Command not found: {cmd[0]}")

    def _get_project_files(self) -> list[str]:
        """Get all relevant files in the project directory.

        Returns:
            List of absolute file paths.
        """
        files = []
        extensions = {
            ".py",  # Python
            ".js",
            ".jsx",
            ".ts",
            ".tsx",  # JavaScript/TypeScript
            ".go",  # Go
            ".rs",  # Rust
            ".java",  # Java
            ".c",
            ".cpp",
            ".h",  # C/C++
            ".rb",  # Ruby
            ".php",  # PHP
            ".sh",  # Shell
            ".yml",
            ".yaml",  # YAML
            ".json",  # JSON
            ".xml",  # XML
            ".dockerfile",
            ".gradle",  # Docker/Build
        }

        # Common files to exclude
        exclude_dirs = {
            ".git",
            ".venv",
            "venv",
            "node_modules",
            ".pytest_cache",
            "__pycache__",
            ".mypy_cache",
            "dist",
            "build",
            ".tox",
            ".egg-info",
        }

        try:
            for root, dirs, files_in_dir in Path(self.project_dir).walk():
                # Remove excluded directories from dirs to prevent descent
                dirs[:] = [d for d in dirs if d not in exclude_dirs]

                for file_name in files_in_dir:
                    file_path = Path(root) / file_name
                    if file_path.suffix.lower() in extensions:
                        files.append(str(file_path))
        except AttributeError:
            # Fallback for Python < 3.12 (Path.walk not available)
            import os
            for root, dirs, files_in_dir in os.walk(str(self.project_dir)):
                # Remove excluded directories from dirs to prevent descent
                dirs[:] = [d for d in dirs if d not in exclude_dirs]

                for file_name in files_in_dir:
                    file_path = Path(root) / file_name
                    if file_path.suffix.lower() in extensions:
                        files.append(str(file_path))
        except Exception as e:
            logger.error("Error walking project directory", error=str(e), exc_info=True)

        return files

    @staticmethod
    def _is_binary_file(file_path: str) -> bool:
        """Check if a file is binary.

        Args:
            file_path: Path to the file.

        Returns:
            True if the file appears to be binary.
        """
        try:
            with open(file_path, "rb") as f:
                chunk = f.read(8192)
                return b"\x00" in chunk
        except Exception:
            return True

    @staticmethod
    def _map_bandit_severity(severity: str) -> Severity:
        """Map Bandit severity to our Severity enum.

        Args:
            severity: Bandit severity string.

        Returns:
            Severity enum value.
        """
        severity_lower = severity.lower()
        if severity_lower == "high":
            return Severity.HIGH
        elif severity_lower == "medium":
            return Severity.MEDIUM
        elif severity_lower == "low":
            return Severity.LOW
        return Severity.MEDIUM

    @staticmethod
    def _map_bandit_test_id(test_id: str) -> FindingCategory:
        """Map Bandit test ID to FindingCategory.

        Args:
            test_id: Bandit test ID.

        Returns:
            FindingCategory enum value.
        """
        test_id_lower = test_id.lower()

        if "assert" in test_id_lower or "binding" in test_id_lower:
            return FindingCategory.OTHER
        elif "crypto" in test_id_lower or "hash" in test_id_lower:
            return FindingCategory.INSECURE_CRYPTO
        elif "pickle" in test_id_lower or "deserialization" in test_id_lower:
            return FindingCategory.INSECURE_DESERIALIZATION
        elif "exec" in test_id_lower or "eval" in test_id_lower or "injection" in test_id_lower:
            return FindingCategory.INJECTION
        elif "hardcoded" in test_id_lower:
            return FindingCategory.HARDCODED_CREDENTIALS
        elif "sql" in test_id_lower or "command" in test_id_lower:
            return FindingCategory.INJECTION
        else:
            return FindingCategory.OTHER

    @staticmethod
    def _map_semgrep_severity(result: dict) -> Severity:
        """Map Semgrep severity to our Severity enum.

        Args:
            result: Semgrep result dict.

        Returns:
            Severity enum value.
        """
        # Semgrep uses different severity keys depending on configuration
        severity = result.get("severity", "").lower()
        if not severity:
            severity = result.get("extra", {}).get("severity", "").lower()

        if severity in ["critical", "c"]:
            return Severity.CRITICAL
        elif severity in ["high", "h"]:
            return Severity.HIGH
        elif severity in ["medium", "m"]:
            return Severity.MEDIUM
        elif severity in ["low", "l"]:
            return Severity.LOW
        else:
            return Severity.MEDIUM

    def _aggregate_results(self, result: SecurityScanResult) -> None:
        """Aggregate and count findings in a SecurityScanResult.

        Args:
            result: SecurityScanResult to aggregate.
        """
        all_findings = []
        all_findings.extend(result.secret_findings)

        for report in result.sast_reports:
            all_findings.extend(report.findings)

        result.total_findings = len(all_findings)
        result.critical_count = sum(1 for f in all_findings if f.severity == Severity.CRITICAL)
        result.high_count = sum(1 for f in all_findings if f.severity == Severity.HIGH)
        result.medium_count = sum(1 for f in all_findings if f.severity == Severity.MEDIUM)
        result.low_count = sum(1 for f in all_findings if f.severity == Severity.LOW)

        result.passed = result.critical_count == 0 and result.high_count == 0
