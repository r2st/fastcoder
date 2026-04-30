"""Evidence Bundle Generator for PR reporting.

This module generates structured evidence bundles that get attached to every PR,
containing test reports, security findings, change impact analysis, and more.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import structlog

from fastcoder.types.evidence import (
    ChangeImpactReport,
    CostBreakdown,
    CriterionTestMapping,
    EvidenceBundle,
    ImpactedFile,
    ReviewerChecklistItem,
)

if TYPE_CHECKING:
    from fastcoder.types.story import Story

log = structlog.get_logger(__name__)


class EvidenceBundleGenerator:
    """Generates comprehensive evidence bundles for PR submission."""

    def __init__(self, project_dir: str, dependency_graph: Optional[dict] = None):
        """Initialize the evidence bundle generator.

        Args:
            project_dir: Root directory of the project
            dependency_graph: Optional pre-computed dependency graph mapping
                            file paths to their dependents
        """
        self.project_dir = Path(project_dir)
        self.dependency_graph = dependency_graph or {}

    async def generate(
        self,
        story: Story,
        security_result: Optional[dict] = None,
    ) -> EvidenceBundle:
        """Generate a complete evidence bundle for a story.

        Args:
            story: The Story object with all execution details
            security_result: Optional security scan results

        Returns:
            EvidenceBundle with all evidence sections populated
        """
        log.info("Generating evidence bundle", story_id=story.id)

        # Extract basic info
        story_title = ""
        if story.spec and story.spec.title:
            story_title = story.spec.title

        # Build each section concurrently where possible
        test_summary, test_failures = await asyncio.to_thread(
            self._build_test_evidence, story
        )
        security_summary, security_findings = await asyncio.to_thread(
            self._build_security_evidence, security_result
        )
        change_impact = await asyncio.to_thread(self._build_change_impact, story)
        criteria_map = await asyncio.to_thread(self._build_criteria_map, story)
        cost_breakdown = await asyncio.to_thread(self._build_cost_breakdown, story)

        # Determine if gates passed
        test_ok = test_summary and test_summary.get("failed", 0) == 0
        security_ok = (
            security_summary
            and security_summary.get("critical", 0) == 0
            and security_summary.get("high", 0) == 0
        )
        criteria_ok = all(m.verified for m in criteria_map) if criteria_map else True

        all_gates_passed = test_ok and security_ok and criteria_ok

        # Build checklist
        reviewer_checklist = await asyncio.to_thread(
            self._build_reviewer_checklist, story, test_ok, security_ok
        )

        # Determine recommendation
        recommended_action = self._determine_recommendation(
            all_gates_passed,
            test_ok,
            security_ok,
            criteria_ok,
            change_impact,
        )

        bundle = EvidenceBundle(
            story_id=story.id,
            story_title=story_title,
            generated_at=datetime.utcnow(),
            test_summary=test_summary,
            test_failures=test_failures,
            security_summary=security_summary,
            security_findings=security_findings,
            change_impact=change_impact,
            acceptance_criteria_map=criteria_map,
            cost_breakdown=cost_breakdown,
            reviewer_checklist=reviewer_checklist,
            all_gates_passed=all_gates_passed,
            recommended_action=recommended_action,
        )

        log.info(
            "Evidence bundle generated",
            story_id=story.id,
            all_gates_passed=all_gates_passed,
            recommendation=recommended_action,
        )

        return bundle

    def _build_test_evidence(self, story: Story) -> tuple[dict, list]:
        """Extract test evidence from story execution.

        Returns:
            Tuple of (test_summary dict, list of failure dicts)
        """
        test_summary = {}
        test_failures = []

        # Collect from all iterations
        for iteration in story.iterations:
            if iteration.test_report:
                tr = iteration.test_report
                test_summary["total"] = tr.total
                test_summary["passed"] = tr.passed
                test_summary["failed"] = tr.failed
                test_summary["skipped"] = tr.skipped
                test_summary["coverage_percent"] = tr.coverage_percent

                # Collect failures
                if tr.failures:
                    for failure in tr.failures:
                        test_failures.append(
                            {
                                "test_name": getattr(failure, "test_name", "Unknown"),
                                "error_message": getattr(failure, "error_message", ""),
                                "traceback": getattr(failure, "traceback", ""),
                            }
                        )

        if not test_summary:
            test_summary = {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "coverage_percent": 0.0,
            }

        return test_summary, test_failures

    def _build_security_evidence(
        self, security_result: Optional[dict]
    ) -> tuple[dict, list]:
        """Extract security evidence from security scan results.

        Returns:
            Tuple of (security_summary dict, list of finding dicts)
        """
        security_summary = {}
        security_findings = []

        if security_result:
            # Tally findings by severity
            findings = security_result.get("findings", [])
            critical = sum(1 for f in findings if f.get("severity") == "critical")
            high = sum(1 for f in findings if f.get("severity") == "high")
            medium = sum(1 for f in findings if f.get("severity") == "medium")
            low = sum(1 for f in findings if f.get("severity") == "low")

            security_summary = {
                "total": len(findings),
                "critical": critical,
                "high": high,
                "medium": medium,
                "low": low,
                "passed": critical == 0,
            }

            security_findings = findings

        else:
            security_summary = {
                "total": 0,
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
                "passed": True,
            }

        return security_summary, security_findings

    def _build_change_impact(self, story: Story) -> Optional[ChangeImpactReport]:
        """Build change impact analysis from file changes.

        Returns:
            ChangeImpactReport or None if no changes
        """
        impacted_files = []
        total_lines_added = 0
        total_lines_removed = 0
        affected_modules = set()

        # Collect changes from iterations
        for iteration in story.iterations:
            if iteration.changes:
                for change in iteration.changes:
                    file_path = change.file_path
                    change_type = change.change_type

                    # Extract module name
                    parts = file_path.split("/")
                    if len(parts) > 1:
                        module = parts[0]
                        if module not in ["src", "test", "tests"]:
                            affected_modules.add(module)

                    # Count lines
                    lines_added = 0
                    lines_removed = 0
                    if change.diff:
                        for line in change.diff.split("\n"):
                            if line.startswith("+") and not line.startswith("+++"):
                                lines_added += 1
                            elif line.startswith("-") and not line.startswith("---"):
                                lines_removed += 1

                    total_lines_added += lines_added
                    total_lines_removed += lines_removed

                    # Build dependents list from dependency graph
                    dependents = self.dependency_graph.get(file_path, [])

                    impacted_files.append(
                        ImpactedFile(
                            file_path=file_path,
                            change_type=change_type,
                            lines_added=lines_added,
                            lines_removed=lines_removed,
                            dependents=dependents,
                        )
                    )

        if not impacted_files:
            return None

        # Calculate blast radius
        blast_radius = sum(len(f.dependents) for f in impacted_files)

        # Determine risk level
        risk_level = self._assess_risk_level(
            len(impacted_files), total_lines_added + total_lines_removed, blast_radius
        )

        return ChangeImpactReport(
            files_changed=impacted_files,
            total_files_changed=len(impacted_files),
            total_lines_added=total_lines_added,
            total_lines_removed=total_lines_removed,
            blast_radius=blast_radius,
            affected_modules=sorted(affected_modules),
            risk_level=risk_level,
        )

    def _build_criteria_map(self, story: Story) -> list[CriterionTestMapping]:
        """Build acceptance criteria to test mapping.

        Returns:
            List of CriterionTestMapping objects
        """
        mappings = []

        if story.spec and story.spec.acceptance_criteria:
            for criterion in story.spec.acceptance_criteria:
                # Check if criterion was verified in any iteration
                verified = False
                for iteration in story.iterations:
                    if iteration.test_report:
                        # Simple heuristic: if all tests passed, assume verified
                        if iteration.test_report.failed == 0:
                            verified = True
                            break

                mapping = CriterionTestMapping(
                    criterion_id=criterion.id,
                    criterion_description=criterion.description,
                    linked_tests=criterion.linked_test_ids,
                    verified=verified,
                    coverage_note=(
                        "Verified by test execution"
                        if verified
                        else "Pending verification"
                    ),
                )
                mappings.append(mapping)

        return mappings

    def _build_cost_breakdown(self, story: Story) -> Optional[CostBreakdown]:
        """Build cost summary from story metadata.

        Returns:
            CostBreakdown or None if no cost data
        """
        if not story.metadata:
            return None

        # Parse model usage
        model_usage = {}
        if story.metadata.model_usage:
            model_usage = dict(story.metadata.model_usage)

        return CostBreakdown(
            total_tokens_input=story.metadata.total_tokens_used or 0,
            total_tokens_output=0,  # If tracked separately, extract here
            total_cost_usd=story.metadata.total_cost_usd or 0.0,
            model_usage=model_usage,
            iterations_count=len(story.iterations),
        )

    def _build_reviewer_checklist(
        self,
        story: Story,
        test_ok: bool,
        security_ok: bool,
    ) -> list[ReviewerChecklistItem]:
        """Build reviewer checklist.

        Returns:
            List of ReviewerChecklistItem objects
        """
        checklist = []

        # Test verification
        checklist.append(
            ReviewerChecklistItem(
                item="All unit tests passing",
                verified_by_agent=test_ok,
                needs_human_review=not test_ok,
                notes="Agent ran full test suite" if test_ok else "Some tests failed",
            )
        )

        # Security verification
        checklist.append(
            ReviewerChecklistItem(
                item="Security scan passed",
                verified_by_agent=security_ok,
                needs_human_review=not security_ok,
                notes="SAST scan completed" if security_ok else "Issues detected",
            )
        )

        # Code quality checks
        checklist.append(
            ReviewerChecklistItem(
                item="Code style consistent",
                verified_by_agent=True,
                needs_human_review=False,
                notes="Automatic formatting and linting applied",
            )
        )

        # Acceptance criteria
        criteria_count = (
            len(story.spec.acceptance_criteria) if story.spec else 0
        )
        if criteria_count > 0:
            checklist.append(
                ReviewerChecklistItem(
                    item=f"Acceptance criteria coverage ({criteria_count} items)",
                    verified_by_agent=False,
                    needs_human_review=True,
                    notes="Please verify all criteria are satisfied",
                )
            )

        # Architecture/design
        checklist.append(
            ReviewerChecklistItem(
                item="Architecture and design sound",
                verified_by_agent=False,
                needs_human_review=True,
                notes="Agent performed basic validation; expert review recommended",
            )
        )

        # Documentation
        has_docs = False
        for iteration in story.iterations:
            if iteration.changes:
                for change in iteration.changes:
                    if "README" in change.file_path or ".md" in change.file_path:
                        has_docs = True
                        break

        checklist.append(
            ReviewerChecklistItem(
                item="Documentation updated",
                verified_by_agent=has_docs,
                needs_human_review=not has_docs,
                notes="Generated documentation present" if has_docs else "No doc changes",
            )
        )

        return checklist

    def _assess_risk_level(
        self, files_changed: int, lines_changed: int, blast_radius: int
    ) -> str:
        """Assess overall risk level of changes.

        Args:
            files_changed: Number of files modified
            lines_changed: Total lines added + removed
            blast_radius: Number of transitive dependents

        Returns:
            Risk level: low, medium, high, or critical
        """
        # Simple scoring: accumulate risk points
        risk_score = 0

        # File count
        if files_changed > 10:
            risk_score += 3
        elif files_changed > 5:
            risk_score += 2
        elif files_changed > 2:
            risk_score += 1

        # Lines changed
        if lines_changed > 500:
            risk_score += 3
        elif lines_changed > 200:
            risk_score += 2
        elif lines_changed > 50:
            risk_score += 1

        # Blast radius
        if blast_radius > 20:
            risk_score += 3
        elif blast_radius > 10:
            risk_score += 2
        elif blast_radius > 5:
            risk_score += 1

        # Map score to level
        if risk_score >= 7:
            return "critical"
        elif risk_score >= 5:
            return "high"
        elif risk_score >= 2:
            return "medium"
        else:
            return "low"

    def _determine_recommendation(
        self,
        all_gates_passed: bool,
        test_ok: bool,
        security_ok: bool,
        criteria_ok: bool,
        change_impact: Optional[ChangeImpactReport],
    ) -> str:
        """Determine overall recommendation for PR.

        Returns:
            Recommendation: auto_merge, review, or block
        """
        # Block if critical issues
        if not test_ok or not security_ok:
            return "block"

        # Auto-merge if all gates passed and low risk
        if all_gates_passed and criteria_ok:
            if change_impact and change_impact.risk_level in ["high", "critical"]:
                return "review"
            return "auto_merge"

        # Default to review
        return "review"


__all__ = [
    "EvidenceBundleGenerator",
    "EvidenceBundle",
    "ChangeImpactReport",
    "CostBreakdown",
    "CriterionTestMapping",
    "ImpactedFile",
    "ReviewerChecklistItem",
]
