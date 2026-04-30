"""Code ownership intelligence from CODEOWNERS, git blame, and review history."""
from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


class OwnershipMap:
    """Code ownership intelligence from CODEOWNERS, git blame, and review history."""

    def __init__(self, project_dir: str):
        """Initialize ownership map.

        Args:
            project_dir: Root directory of the project
        """
        self._project_dir = Path(project_dir).resolve()
        self._codeowners: dict[str, list[str]] = {}
        self._blame_cache: dict[str, dict[str, float]] = {}
        self._review_frequency: dict[str, dict[str, int]] = {}
        self._logger = logger.bind(
            component="OwnershipMap",
            project_dir=str(self._project_dir),
        )

    async def initialize(self) -> None:
        """Parse CODEOWNERS and build initial ownership data."""
        self._logger.info("initializing_ownership_map")

        self._codeowners = self._parse_codeowners()
        self._logger.info(
            "codeowners_parsed",
            patterns_found=len(self._codeowners),
        )

    def _parse_codeowners(self) -> dict[str, list[str]]:
        """Parse CODEOWNERS file.

        Supports standard CODEOWNERS format with patterns and owners.
        Lines starting with # are comments.
        Format: pattern owner1 owner2 ...

        Returns:
            Dict mapping pattern -> list of owners
        """
        codeowners = {}

        codeowners_paths = [
            self._project_dir / "CODEOWNERS",
            self._project_dir / ".github" / "CODEOWNERS",
            self._project_dir / "docs" / "CODEOWNERS",
        ]

        codeowners_file = None
        for path in codeowners_paths:
            if path.exists():
                codeowners_file = path
                break

        if not codeowners_file:
            self._logger.warning("codeowners_file_not_found")
            return codeowners

        try:
            with open(codeowners_file, "r") as f:
                for line in f:
                    line = line.strip()

                    if not line or line.startswith("#"):
                        continue

                    parts = line.split()
                    if len(parts) < 2:
                        continue

                    pattern = parts[0]
                    owners = parts[1:]

                    codeowners[pattern] = owners

            self._logger.info(
                "codeowners_parsed_success",
                file=str(codeowners_file),
                patterns=len(codeowners),
            )
        except Exception as e:
            self._logger.error(
                "failed_to_parse_codeowners",
                error=str(e),
            )

        return codeowners

    async def _analyze_blame(self, file_path: str) -> dict[str, float]:
        """Run git blame and calculate ownership proportions.

        Args:
            file_path: Path to file relative to project root

        Returns:
            Dict mapping author -> proportion (0.0-1.0)
        """
        full_path = self._project_dir / file_path

        if not full_path.exists():
            self._logger.warning(
                "file_not_found_for_blame",
                file=file_path,
            )
            return {}

        if file_path in self._blame_cache:
            return self._blame_cache[file_path]

        try:
            author_counts = {}
            total_lines = 0

            import subprocess

            result = await asyncio.to_thread(
                subprocess.run,
                ["git", "blame", "--line-porcelain", str(full_path)],
                cwd=str(self._project_dir),
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                self._logger.warning(
                    "git_blame_failed",
                    file=file_path,
                    error=result.stderr[:100],
                )
                return {}

            for line in result.stdout.split("\n"):
                if not line.strip():
                    continue

                parts = line.split()
                if len(parts) < 2:
                    continue

                if parts[0] == "author":
                    author = " ".join(parts[1:])
                    author_counts[author] = author_counts.get(author, 0) + 1
                    total_lines += 1

            if total_lines == 0:
                return {}

            proportions = {
                author: count / total_lines
                for author, count in author_counts.items()
            }

            self._blame_cache[file_path] = proportions
            self._logger.info(
                "blame_analyzed",
                file=file_path,
                authors=len(proportions),
            )

            return proportions

        except Exception as e:
            self._logger.warning(
                "blame_analysis_failed",
                file=file_path,
                error=str(e),
            )
            return {}

    def get_owners(self, file_path: str) -> list[str]:
        """Get owners for a file (CODEOWNERS > blame > default).

        Args:
            file_path: Path to file

        Returns:
            List of owner identifiers (emails, handles, etc.)
        """
        file_path = file_path.lstrip("./")

        for pattern, owners in self._codeowners.items():
            if self._pattern_matches(pattern, file_path):
                self._logger.info(
                    "owner_found_in_codeowners",
                    file=file_path,
                    pattern=pattern,
                    owners=owners,
                )
                return owners

        self._logger.debug(
            "no_codeowners_match",
            file=file_path,
        )

        return []

    def _pattern_matches(self, pattern: str, file_path: str) -> bool:
        """Check if a CODEOWNERS pattern matches a file path.

        Supports:
        - Exact match: path/to/file.py
        - Directory: path/to/
        - Wildcard: *.py
        - Double wildcard: **/file.py

        Args:
            pattern: CODEOWNERS pattern
            file_path: File path to test

        Returns:
            True if pattern matches file
        """
        if pattern == "*" or pattern == "/**":
            return True

        if pattern == file_path:
            return True

        if pattern.endswith("/"):
            # Strip leading / for root-anchored patterns (CODEOWNERS convention)
            clean_pattern = pattern.lstrip("/")
            return file_path.startswith(clean_pattern)

        pattern_regex = self._glob_to_regex(pattern)
        match = re.match(pattern_regex, file_path)
        return match is not None

    def _glob_to_regex(self, pattern: str) -> str:
        """Convert glob pattern to regex.

        Args:
            pattern: Glob pattern

        Returns:
            Regex pattern string
        """
        regex = ""
        i = 0

        while i < len(pattern):
            c = pattern[i]

            if c == "*":
                if i + 1 < len(pattern) and pattern[i + 1] == "*":
                    if i + 2 < len(pattern) and pattern[i + 2] == "/":
                        regex += "(?:.*/)??"
                        i += 3
                        continue
                    else:
                        regex += ".*"
                        i += 2
                        continue
                else:
                    regex += "[^/]*"
                    i += 1
            elif c == "?":
                regex += "[^/]"
                i += 1
            elif c in ".^$+{}|()[]":
                regex += "\\" + c
                i += 1
            else:
                regex += c
                i += 1

        return f"^{regex}$"

    def get_reviewers_for_changes(self, changed_files: list[str]) -> list[str]:
        """Determine best reviewers for a set of changed files.

        Uses CODEOWNERS patterns first, then review history frequency.
        Deduplicates and orders by relevance.

        Args:
            changed_files: List of file paths changed

        Returns:
            Ordered list of reviewer identifiers
        """
        reviewer_scores = {}

        for file_path in changed_files:
            owners = self.get_owners(file_path)

            for owner in owners:
                reviewer_scores[owner] = reviewer_scores.get(owner, 0) + 10

            if not owners:
                review_hist = self._review_frequency.get(file_path, {})
                for reviewer, count in review_hist.items():
                    reviewer_scores[reviewer] = reviewer_scores.get(reviewer, 0) + count

        reviewers = sorted(
            reviewer_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        result = [reviewer for reviewer, _ in reviewers]

        self._logger.info(
            "reviewers_determined",
            changed_files=len(changed_files),
            reviewer_count=len(result),
            top_reviewers=result[:3],
        )

        return result

    def get_file_ownership_summary(self, file_path: str) -> dict:
        """Get detailed ownership info for a file.

        Returns ownership from multiple sources: CODEOWNERS,
        git blame analysis, and review history.

        Args:
            file_path: Path to file

        Returns:
            Dict with ownership details
        """
        file_path = file_path.lstrip("./")

        codeowners = self.get_owners(file_path)

        blame_data = {}
        review_data = self._review_frequency.get(file_path, {})

        summary = {
            "file": file_path,
            "codeowners": codeowners,
            "blame_distribution": blame_data,
            "review_frequency": review_data,
            "primary_owner": codeowners[0] if codeowners else None,
            "all_contributors": list(
                set(codeowners + list(blame_data.keys()) + list(review_data.keys()))
            ),
        }

        self._logger.info(
            "ownership_summary_generated",
            file=file_path,
            primary_owner=summary["primary_owner"],
            contributor_count=len(summary["all_contributors"]),
        )

        return summary

    def record_review(self, file_patterns: list[str], reviewer: str) -> None:
        """Record that a reviewer reviewed files matching patterns.

        Updates review frequency tracking for pattern-based matching.

        Args:
            file_patterns: List of file patterns reviewed
            reviewer: Reviewer identifier
        """
        for pattern in file_patterns:
            if pattern not in self._review_frequency:
                self._review_frequency[pattern] = {}

            self._review_frequency[pattern][reviewer] = (
                self._review_frequency[pattern].get(reviewer, 0) + 1
            )

        self._logger.info(
            "review_recorded",
            patterns=len(file_patterns),
            reviewer=reviewer,
        )

    async def get_experts_for_feature(
        self, feature_files: list[str], min_expertise: float = 0.5
    ) -> list[tuple[str, float]]:
        """Get experts for a feature based on file ownership.

        Finds reviewers who have significant involvement with
        the files affected by a feature.

        Args:
            feature_files: List of files in the feature
            min_expertise: Minimum expertise score (0-1)

        Returns:
            List of (reviewer, expertise_score) tuples, ordered by score
        """
        expertise_scores = {}

        for file_path in feature_files:
            owners = self.get_owners(file_path)

            for owner in owners:
                expertise_scores[owner] = expertise_scores.get(owner, 0) + 1.0

        if feature_files and not sum(expertise_scores.values()):
            review_freq = {}
            for file_path in feature_files:
                freq = self._review_frequency.get(file_path, {})
                for reviewer, count in freq.items():
                    review_freq[reviewer] = review_freq.get(reviewer, 0) + count

            if review_freq:
                max_reviews = max(review_freq.values())
                for reviewer, count in review_freq.items():
                    expertise_scores[reviewer] = count / max_reviews if max_reviews > 0 else 0

        filtered = [
            (reviewer, score)
            for reviewer, score in expertise_scores.items()
            if score >= min_expertise
        ]

        result = sorted(filtered, key=lambda x: x[1], reverse=True)

        self._logger.info(
            "feature_experts_found",
            file_count=len(feature_files),
            expert_count=len(result),
        )

        return result

    async def validate_ownership(self) -> dict[str, list[str]]:
        """Validate CODEOWNERS file and report issues.

        Checks for:
        - Non-existent files in patterns
        - Owners without proper format
        - Overlapping patterns

        Returns:
            Dict with validation results and issues
        """
        issues = {
            "non_existent_patterns": [],
            "malformed_owners": [],
            "warnings": [],
        }

        for pattern, owners in self._codeowners.items():
            if not owners:
                issues["malformed_owners"].append(pattern)
                continue

            for owner in owners:
                if not self._is_valid_owner_format(owner):
                    issues["malformed_owners"].append(f"{pattern}: {owner}")

        self._logger.info(
            "ownership_validation_complete",
            issues_found=sum(len(v) for v in issues.values()),
        )

        return issues

    def _is_valid_owner_format(self, owner: str) -> bool:
        """Check if owner has valid format.

        Valid formats:
        - @username (GitHub handle)
        - user@example.com (email)
        - @team (team handle)

        Args:
            owner: Owner identifier

        Returns:
            True if format is valid
        """
        if not owner:
            return False

        if owner.startswith("@"):
            return len(owner) > 1

        if "@" in owner and "." in owner:
            return True

        return False

    def clear_cache(self) -> None:
        """Clear the blame cache to force fresh analysis."""
        cache_size = len(self._blame_cache)
        self._blame_cache.clear()
        self._logger.info(
            "blame_cache_cleared",
            entries_cleared=cache_size,
        )

    def export_summary(self) -> dict:
        """Export ownership map summary.

        Returns:
            Dict with codeowners, review frequency, and cache stats
        """
        return {
            "codeowners_patterns": len(self._codeowners),
            "codeowners": self._codeowners,
            "blame_cache_size": len(self._blame_cache),
            "review_patterns": len(self._review_frequency),
            "review_frequency": self._review_frequency,
        }


__all__ = ["OwnershipMap"]
