"""Evidence bundle types for PR reporting."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ImpactedFile(BaseModel):
    """Represents a file affected by changes."""

    file_path: str
    change_type: str  # created, modified, deleted
    lines_added: int = 0
    lines_removed: int = 0
    dependents: list[str] = Field(default_factory=list)  # files that import this


class ChangeImpactReport(BaseModel):
    """Analysis of how changes impact the codebase."""

    files_changed: list[ImpactedFile] = Field(default_factory=list)
    total_files_changed: int = 0
    total_lines_added: int = 0
    total_lines_removed: int = 0
    blast_radius: int = 0  # number of transitive dependents affected
    affected_modules: list[str] = Field(default_factory=list)
    risk_level: str = "low"  # low, medium, high, critical


class CriterionTestMapping(BaseModel):
    """Maps an acceptance criterion to its verifying tests."""

    criterion_id: str
    criterion_description: str
    linked_tests: list[str] = Field(default_factory=list)
    verified: bool = False
    coverage_note: str = ""


class CostBreakdown(BaseModel):
    """Breakdown of costs for generating this story."""

    total_tokens_input: int = 0
    total_tokens_output: int = 0
    total_cost_usd: float = 0.0
    model_usage: dict[str, int] = Field(default_factory=dict)  # model -> tokens
    iterations_count: int = 0


class ReviewerChecklistItem(BaseModel):
    """Individual item in the reviewer checklist."""

    item: str
    verified_by_agent: bool = False
    needs_human_review: bool = False
    notes: str = ""


class EvidenceBundle(BaseModel):
    """Complete evidence bundle for a story/PR."""

    story_id: str
    story_title: str = ""
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    # Test evidence
    test_summary: Optional[dict] = None  # {total, passed, failed, skipped, coverage}
    test_failures: list[dict] = Field(default_factory=list)

    # Security evidence
    security_summary: Optional[dict] = None  # {total, critical, high, medium, low, passed}
    security_findings: list[dict] = Field(default_factory=list)

    # Impact analysis
    change_impact: Optional[ChangeImpactReport] = None

    # Criteria mapping
    acceptance_criteria_map: list[CriterionTestMapping] = Field(default_factory=list)

    # Cost
    cost_breakdown: Optional[CostBreakdown] = None

    # Reviewer checklist
    reviewer_checklist: list[ReviewerChecklistItem] = Field(default_factory=list)

    # Overall status
    all_gates_passed: bool = False
    recommended_action: str = "review"  # auto_merge, review, block

    def to_markdown(self) -> str:
        """Generate a Markdown report suitable for PR description."""
        lines = []

        # Header
        lines.append("## Evidence Bundle Report")
        lines.append(f"Generated: {self.generated_at.isoformat()}")
        lines.append("")

        # Test Evidence Section
        if self.test_summary:
            lines.append(self._format_test_section())

        # Security Evidence Section
        if self.security_summary:
            lines.append(self._format_security_section())

        # Change Impact Section
        if self.change_impact:
            lines.append(self._format_impact_section())

        # Acceptance Criteria Section
        if self.acceptance_criteria_map:
            lines.append(self._format_criteria_section())

        # Cost Summary Section
        if self.cost_breakdown:
            lines.append(self._format_cost_section())

        # Reviewer Checklist Section
        if self.reviewer_checklist:
            lines.append(self._format_checklist_section())

        # Summary
        lines.append(self._format_summary_section())

        return "\n".join(lines)

    def to_pr_comment(self) -> str:
        """Generate a condensed PR comment."""
        lines = []

        # Status indicator
        if self.all_gates_passed:
            status_emoji = "✅"
            status_text = "All gates passed"
        else:
            status_emoji = "⚠️"
            status_text = "Issues detected"

        lines.append(f"{status_emoji} **{status_text}**")
        lines.append("")

        # Quick summary
        if self.test_summary:
            passed = self.test_summary.get("passed", 0)
            total = self.test_summary.get("total", 0)
            lines.append(f"Tests: {passed}/{total} passed")

        if self.security_summary:
            critical = self.security_summary.get("critical", 0)
            high = self.security_summary.get("high", 0)
            if critical > 0 or high > 0:
                lines.append(f"Security: {critical} critical, {high} high severity issues")

        if self.change_impact:
            lines.append(f"Impact: {self.change_impact.total_files_changed} files changed")

        lines.append("")
        lines.append(f"Recommendation: **{self.recommended_action.upper()}**")

        return "\n".join(lines)

    def _format_test_section(self) -> str:
        """Format test evidence as Markdown."""
        lines = ["### Test Report"]

        ts = self.test_summary
        passed = ts.get("passed", 0)
        failed = ts.get("failed", 0)
        total = ts.get("total", 0)
        coverage = ts.get("coverage_percent", 0)

        # Status indicator
        if failed == 0:
            lines.append("✅ All tests passing")
        else:
            lines.append(f"❌ {failed} test(s) failing")

        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total | {total} |")
        lines.append(f"| Passed | {passed} |")
        lines.append(f"| Failed | {failed} |")
        lines.append(f"| Skipped | {ts.get('skipped', 0)} |")
        lines.append(f"| Coverage | {coverage:.1f}% |")

        if self.test_failures:
            lines.append("")
            lines.append("<details><summary>Failures</summary>")
            lines.append("")
            for failure in self.test_failures[:5]:  # Show first 5
                test_name = failure.get("test_name", "Unknown")
                error = failure.get("error_message", "No error message")
                lines.append(f"**{test_name}**")
                lines.append(f"```\n{error}\n```")
                lines.append("")

            if len(self.test_failures) > 5:
                lines.append(f"... and {len(self.test_failures) - 5} more failures")

            lines.append("</details>")

        lines.append("")
        return "\n".join(lines)

    def _format_security_section(self) -> str:
        """Format security evidence as Markdown."""
        lines = ["### Security Report"]

        ss = self.security_summary
        critical = ss.get("critical", 0)
        high = ss.get("high", 0)
        medium = ss.get("medium", 0)
        low = ss.get("low", 0)
        passed = ss.get("passed", False)

        # Status indicator
        if critical == 0 and high == 0:
            lines.append("✅ No critical or high-severity issues")
        else:
            lines.append(f"❌ {critical} critical, {high} high-severity issue(s)")

        lines.append("")
        lines.append("| Severity | Count |")
        lines.append("|----------|-------|")
        lines.append(f"| Critical | {critical} |")
        lines.append(f"| High | {high} |")
        lines.append(f"| Medium | {medium} |")
        lines.append(f"| Low | {low} |")

        if self.security_findings:
            lines.append("")
            lines.append("<details><summary>Findings</summary>")
            lines.append("")
            for finding in self.security_findings[:5]:  # Show first 5
                title = finding.get("title", "Unknown")
                severity = finding.get("severity", "unknown")
                description = finding.get("description", "")
                lines.append(f"**[{severity.upper()}] {title}**")
                if description:
                    lines.append(description)
                lines.append("")

            if len(self.security_findings) > 5:
                lines.append(f"... and {len(self.security_findings) - 5} more findings")

            lines.append("</details>")

        lines.append("")
        return "\n".join(lines)

    def _format_impact_section(self) -> str:
        """Format change impact as Markdown."""
        lines = ["### Change Impact Analysis"]

        ci = self.change_impact
        lines.append(f"Risk Level: **{ci.risk_level.upper()}**")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Files Changed | {ci.total_files_changed} |")
        lines.append(f"| Lines Added | {ci.total_lines_added} |")
        lines.append(f"| Lines Removed | {ci.total_lines_removed} |")
        lines.append(f"| Blast Radius | {ci.blast_radius} |")

        if ci.affected_modules:
            lines.append("")
            lines.append(f"**Affected Modules:** {', '.join(ci.affected_modules)}")

        if ci.files_changed:
            lines.append("")
            lines.append("<details><summary>Changed Files</summary>")
            lines.append("")
            for file in ci.files_changed:
                change_type = file.change_type
                emoji = "📝" if change_type == "modified" else "✨" if change_type == "created" else "🗑️"
                lines.append(f"{emoji} `{file.file_path}` ({change_type})")
                if file.lines_added > 0 or file.lines_removed > 0:
                    lines.append(f"   +{file.lines_added} -{file.lines_removed}")
                if file.dependents:
                    lines.append(f"   Dependents: {', '.join(file.dependents)}")

            lines.append("</details>")

        lines.append("")
        return "\n".join(lines)

    def _format_criteria_section(self) -> str:
        """Format acceptance criteria mapping as Markdown."""
        lines = ["### Acceptance Criteria Coverage"]

        verified_count = sum(1 for m in self.acceptance_criteria_map if m.verified)
        total_count = len(self.acceptance_criteria_map)

        lines.append(f"**{verified_count}/{total_count}** criteria verified")
        lines.append("")

        if self.acceptance_criteria_map:
            lines.append("<details><summary>Criteria Details</summary>")
            lines.append("")

            for mapping in self.acceptance_criteria_map:
                status_emoji = "✅" if mapping.verified else "❌"
                lines.append(f"{status_emoji} **{mapping.criterion_id}**")
                lines.append(f"{mapping.criterion_description}")

                if mapping.linked_tests:
                    lines.append(f"Tests: {', '.join(f'`{t}`' for t in mapping.linked_tests)}")

                if mapping.coverage_note:
                    lines.append(f"Note: {mapping.coverage_note}")

                lines.append("")

            lines.append("</details>")

        lines.append("")
        return "\n".join(lines)

    def _format_cost_section(self) -> str:
        """Format cost breakdown as Markdown."""
        lines = ["### Cost Summary"]

        cb = self.cost_breakdown
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Input Tokens | {cb.total_tokens_input:,} |")
        lines.append(f"| Output Tokens | {cb.total_tokens_output:,} |")
        lines.append(f"| Total Cost | ${cb.total_cost_usd:.4f} |")
        lines.append(f"| Iterations | {cb.iterations_count} |")

        if cb.model_usage:
            lines.append("")
            lines.append("**Model Usage:**")
            for model, tokens in sorted(cb.model_usage.items()):
                lines.append(f"- {model}: {tokens:,} tokens")

        lines.append("")
        return "\n".join(lines)

    def _format_checklist_section(self) -> str:
        """Format reviewer checklist as Markdown."""
        lines = ["### Reviewer Checklist"]

        lines.append("")
        for item in self.reviewer_checklist:
            if item.verified_by_agent:
                emoji = "✅"
                status = "Verified by agent"
            elif item.needs_human_review:
                emoji = "👀"
                status = "Needs human review"
            else:
                emoji = "⚪"
                status = "Not required"

            lines.append(f"{emoji} {item.item} - {status}")

            if item.notes:
                lines.append(f"   > {item.notes}")

        lines.append("")
        return "\n".join(lines)

    def _format_summary_section(self) -> str:
        """Format overall summary as Markdown."""
        lines = ["---"]
        lines.append("")

        if self.all_gates_passed:
            lines.append("### ✅ All Gates Passed")
            recommendation = "AUTO_MERGE" if self.recommended_action == "auto_merge" else "APPROVED FOR REVIEW"
        else:
            lines.append("### ⚠️ Issues Detected")
            if self.recommended_action == "block":
                recommendation = "BLOCKED - Fix required"
            else:
                recommendation = "NEEDS REVIEW"

        lines.append(f"**Recommendation:** {recommendation}")
        lines.append("")

        return "\n".join(lines)
