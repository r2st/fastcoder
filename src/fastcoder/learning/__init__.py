"""Post-mortem learning engine for autonomous dev agent."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import structlog

from fastcoder.types.errors import ErrorCategory, ErrorClassification
from fastcoder.types.learning import (
    FailureClass,
    HeuristicRule,
    LearningStats,
    PostMortemEntry,
)
from fastcoder.types.memory import MemoryEntry, MemoryTier, MemoryType

logger = structlog.get_logger(__name__)


class PostMortemEngine:
    """Automated post-mortem analysis and learning loop."""

    def __init__(self, memory_store=None, project_id: str = "default"):
        """Initialize the post-mortem engine.

        Args:
            memory_store: Optional memory store for persisting learnings
            project_id: Project identifier for scoping learnings
        """
        self._post_mortems: list[PostMortemEntry] = []
        self._heuristics: list[HeuristicRule] = []
        self._memory_store = memory_store
        self._project_id = project_id
        self._logger = logger.bind(component="PostMortemEngine", project=project_id)

    async def analyze_gate_failure(
        self,
        story,
        gate_result,
        error_context: Optional[dict[str, Any]] = None,
    ) -> PostMortemEntry:
        """Analyze a quality gate failure and create a post-mortem.

        Args:
            story: Story object with story_id and context
            gate_result: Gate result object with failure details
            error_context: Optional error context dict

        Returns:
            PostMortemEntry with analysis
        """
        # Accept both dict and object for gate_result
        if isinstance(gate_result, dict):
            _gate_type = gate_result.get("gate_type", "unknown")
            _details = gate_result.get("message", gate_result.get("details", ""))
        else:
            _gate_type = getattr(gate_result, "gate_type", "unknown")
            _details = getattr(gate_result, "details", getattr(gate_result, "message", ""))

        self._logger.info(
            "analyzing_gate_failure",
            story_id=story.id,
            gate_type=_gate_type,
        )

        if error_context and isinstance(error_context, dict):
            failure_details = error_context.get("error_message", str(_details))
        elif error_context and isinstance(error_context, str):
            failure_details = error_context
        else:
            failure_details = str(_details)
        gate_type = str(_gate_type)

        failure_class = self._classify_failure(failure_details, gate_type)
        root_cause = self._extract_root_cause(failure_details, failure_class)

        fingerprint = self._generate_fingerprint(
            failure_class, root_cause, gate_type
        )

        post_mortem = PostMortemEntry(
            id=str(uuid4()),
            story_id=story.id,
            project_id=self._project_id,
            failure_class=failure_class,
            failure_details=failure_details,
            root_cause=root_cause,
            error_fingerprint=fingerprint,
            gate_type=gate_type,
            iteration_count=1,
        )

        self._post_mortems.append(post_mortem)

        heuristic = await self._generate_heuristic(post_mortem)
        if heuristic:
            self._heuristics.append(heuristic)
            post_mortem.heuristic_update = heuristic.id
            self._logger.info(
                "heuristic_generated",
                post_mortem_id=post_mortem.id,
                heuristic_id=heuristic.id,
            )

        memory_id = await self._store_as_memory(post_mortem)
        if memory_id:
            post_mortem.memory_entry_id = memory_id

        self._logger.info(
            "post_mortem_created",
            post_mortem_id=post_mortem.id,
            failure_class=failure_class.value,
        )

        return post_mortem

    async def analyze_reviewer_rejection(
        self,
        story,
        reviewer_comment: str,
        rejection_details: Optional[dict[str, Any]] = None,
    ) -> PostMortemEntry:
        """Analyze a human reviewer rejection and extract learnings.

        Args:
            story: Story object
            reviewer_comment: Feedback from reviewer
            rejection_details: Optional dict with rejection info

        Returns:
            PostMortemEntry with analysis
        """
        self._logger.info(
            "analyzing_reviewer_rejection",
            story_id=story.id,
            comment_length=len(reviewer_comment),
        )

        rejection_details = rejection_details or {}
        failure_details = f"Reviewer: {reviewer_comment}"

        root_cause = self._extract_root_cause_from_comment(reviewer_comment)

        fingerprint = self._generate_fingerprint(
            FailureClass.REVIEWER_REJECTION, root_cause
        )

        post_mortem = PostMortemEntry(
            id=str(uuid4()),
            story_id=story.id,
            project_id=self._project_id,
            failure_class=FailureClass.REVIEWER_REJECTION,
            failure_details=failure_details,
            reviewer_comment=reviewer_comment,
            root_cause=root_cause,
            error_fingerprint=fingerprint,
            iteration_count=rejection_details.get("iteration_count", 1),
        )

        self._post_mortems.append(post_mortem)

        heuristic = await self._generate_heuristic(post_mortem)
        if heuristic:
            self._heuristics.append(heuristic)
            post_mortem.heuristic_update = heuristic.id

        memory_id = await self._store_as_memory(post_mortem)
        if memory_id:
            post_mortem.memory_entry_id = memory_id

        self._logger.info(
            "reviewer_rejection_recorded",
            post_mortem_id=post_mortem.id,
            root_cause=root_cause,
        )

        return post_mortem

    async def record_resolution(
        self, post_mortem_id: str, resolution: str
    ) -> None:
        """Record what eventually fixed the issue.

        Args:
            post_mortem_id: ID of the post-mortem entry
            resolution: Description of the fix
        """
        for pm in self._post_mortems:
            if pm.id == post_mortem_id:
                pm.resolution = resolution
                self._logger.info(
                    "resolution_recorded",
                    post_mortem_id=post_mortem_id,
                    resolution_length=len(resolution),
                )
                return

        self._logger.warning(
            "post_mortem_not_found",
            post_mortem_id=post_mortem_id,
        )

    def _classify_failure(
        self, failure_details: str, gate_type: Optional[str] = None
    ) -> FailureClass:
        """Classify the failure into a category.

        Args:
            failure_details: Error message or description
            gate_type: Type of gate that failed

        Returns:
            FailureClass enum
        """
        failure_lower = failure_details.lower()

        if gate_type == "security":
            return FailureClass.SECURITY_FINDING

        if gate_type == "performance":
            return FailureClass.PERFORMANCE_REGRESSION

        if any(
            word in failure_lower
            for word in [
                "syntax",
                "parse",
                "invalid token",
                "unexpected token",
            ]
        ):
            return FailureClass.GATE_FAILURE

        if any(
            word in failure_lower
            for word in ["type", "typing", "type annotation", "mypy"]
        ):
            return FailureClass.GATE_FAILURE

        if any(
            word in failure_lower
            for word in ["import", "module not found", "no such file"]
        ):
            return FailureClass.GATE_FAILURE

        if any(
            word in failure_lower
            for word in ["convention", "style", "format", "lint", "pep"]
        ):
            return FailureClass.CONVENTION_VIOLATION

        if any(
            word in failure_lower
            for word in ["test", "failed", "assertion", "regression"]
        ):
            return FailureClass.TEST_REGRESSION

        if any(
            word in failure_lower
            for word in ["build", "compile", "link", "cmake"]
        ):
            return FailureClass.BUILD_FAILURE

        if any(
            word in failure_lower
            for word in [
                "deploy",
                "rollout",
                "kubernetes",
                "health check",
            ]
        ):
            return FailureClass.DEPLOYMENT_FAILURE

        return FailureClass.GATE_FAILURE

    def _extract_root_cause(
        self, failure_details: str, failure_class: FailureClass
    ) -> str:
        """Extract the root cause from failure details.

        Args:
            failure_details: Error message
            failure_class: Classification of failure

        Returns:
            Root cause description
        """
        lines = failure_details.split("\n")

        if failure_class == FailureClass.TEST_REGRESSION:
            for line in lines:
                if "AssertionError" in line or "assert" in line.lower():
                    return line.strip()

        if failure_class == FailureClass.CONVENTION_VIOLATION:
            for line in lines:
                if "expected" in line.lower() or "found" in line.lower():
                    return line.strip()

        if len(lines) > 0:
            return lines[0].strip()

        return "Unknown root cause"

    def _extract_root_cause_from_comment(self, comment: str) -> str:
        """Extract root cause from reviewer comment.

        Args:
            comment: Reviewer feedback

        Returns:
            Root cause summary
        """
        lines = comment.split("\n")
        for line in lines:
            line = line.strip()
            if line and not line.startswith("http"):
                return line[:200]

        return "Reviewer feedback"

    def _generate_fingerprint(
        self,
        failure_class: FailureClass,
        root_cause: str,
        gate_type: Optional[str] = None,
    ) -> str:
        """Generate a fingerprint for the failure.

        Args:
            failure_class: Classification
            root_cause: Root cause
            gate_type: Gate type

        Returns:
            Fingerprint hash
        """
        components = [
            failure_class.value,
            root_cause[:100],
            gate_type or "none",
        ]
        combined = "|".join(components)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    async def _generate_heuristic(
        self, post_mortem: PostMortemEntry
    ) -> Optional[HeuristicRule]:
        """Generate a heuristic rule from a post-mortem entry.

        Args:
            post_mortem: PostMortemEntry to analyze

        Returns:
            HeuristicRule if one can be generated, None otherwise
        """
        rule_type = self._map_failure_to_rule_type(post_mortem.failure_class)

        trigger_pattern = self._extract_trigger_pattern(
            post_mortem.failure_class, post_mortem.root_cause
        )

        if not trigger_pattern:
            return None

        action = self._generate_action(
            post_mortem.failure_class, post_mortem.root_cause
        )

        heuristic = HeuristicRule(
            id=str(uuid4()),
            project_id=self._project_id,
            rule_type=rule_type,
            trigger_pattern=trigger_pattern,
            action=action,
            source_post_mortem_id=post_mortem.id,
        )

        self._logger.info(
            "heuristic_rule_created",
            heuristic_id=heuristic.id,
            rule_type=rule_type,
        )

        return heuristic

    def _map_failure_to_rule_type(self, failure_class: FailureClass) -> str:
        """Map failure class to heuristic rule type.

        Args:
            failure_class: Failure classification

        Returns:
            Rule type string
        """
        mapping = {
            FailureClass.GATE_FAILURE: "pre_check",
            FailureClass.REVIEWER_REJECTION: "convention",
            FailureClass.TEST_REGRESSION: "pre_check",
            FailureClass.CONVENTION_VIOLATION: "convention",
            FailureClass.SECURITY_FINDING: "avoidance",
            FailureClass.PERFORMANCE_REGRESSION: "pre_check",
            FailureClass.BUILD_FAILURE: "pre_check",
            FailureClass.DEPLOYMENT_FAILURE: "pre_check",
        }
        return mapping.get(failure_class, "pre_check")

    def _extract_trigger_pattern(
        self, failure_class: FailureClass, root_cause: str
    ) -> str:
        """Extract a trigger pattern from failure info.

        Args:
            failure_class: Failure classification
            root_cause: Root cause

        Returns:
            Pattern string
        """
        if failure_class == FailureClass.CONVENTION_VIOLATION:
            match = re.search(r"(format|style|naming|pattern)", root_cause.lower())
            if match:
                return match.group(1)

        if failure_class == FailureClass.SECURITY_FINDING:
            match = re.search(
                r"(injection|xss|sql|authentication|authorization)",
                root_cause.lower(),
            )
            if match:
                return match.group(1)

        if failure_class == FailureClass.TEST_REGRESSION:
            match = re.search(r"test_(\w+)", root_cause.lower())
            if match:
                return f"test_{match.group(1)}"

        return ""

    def _generate_action(
        self, failure_class: FailureClass, root_cause: str
    ) -> str:
        """Generate an action for the heuristic rule.

        Args:
            failure_class: Failure classification
            root_cause: Root cause

        Returns:
            Action description
        """
        actions = {
            FailureClass.CONVENTION_VIOLATION: "Run linter before submission",
            FailureClass.SECURITY_FINDING: "Add security pre-check",
            FailureClass.TEST_REGRESSION: "Run full test suite",
            FailureClass.PERFORMANCE_REGRESSION: "Profile before submission",
            FailureClass.BUILD_FAILURE: "Validate build in pre-check",
            FailureClass.DEPLOYMENT_FAILURE: "Add deployment dry-run",
            FailureClass.REVIEWER_REJECTION: "Review code against patterns",
            FailureClass.GATE_FAILURE: "Enhanced validation",
        }
        return actions.get(failure_class, "Review and validate")

    async def _store_as_memory(self, post_mortem: PostMortemEntry) -> Optional[str]:
        """Store the post-mortem as a memory entry for RAG retrieval.

        Args:
            post_mortem: PostMortemEntry to store

        Returns:
            Memory entry ID if stored, None otherwise
        """
        if not self._memory_store:
            return None

        content = self._format_post_mortem_for_memory(post_mortem)

        memory_entry = MemoryEntry(
            id=str(uuid4()),
            type=MemoryType.PATTERN,
            tier=MemoryTier.PROJECT,
            context=post_mortem.failure_class.value,
            content=content,
            source_story_id=post_mortem.story_id,
            effectiveness_score=0.5,
            project_id=self._project_id,
        )

        try:
            await self._memory_store.store(memory_entry)
            self._logger.info(
                "memory_entry_stored",
                memory_id=memory_entry.id,
                post_mortem_id=post_mortem.id,
            )
            return memory_entry.id
        except Exception as e:
            self._logger.warning(
                "failed_to_store_memory",
                error=str(e),
                post_mortem_id=post_mortem.id,
            )
            return None

    def _format_post_mortem_for_memory(self, post_mortem: PostMortemEntry) -> str:
        """Format a post-mortem for storage as memory.

        Args:
            post_mortem: PostMortemEntry

        Returns:
            Formatted content string
        """
        lines = [
            f"Failure Class: {post_mortem.failure_class.value}",
            f"Root Cause: {post_mortem.root_cause}",
            f"Details: {post_mortem.failure_details[:200]}",
        ]

        if post_mortem.reviewer_comment:
            lines.append(f"Reviewer: {post_mortem.reviewer_comment[:200]}")

        if post_mortem.resolution:
            lines.append(f"Resolution: {post_mortem.resolution[:200]}")

        return "\n".join(lines)

    def get_applicable_heuristics(self, context: dict[str, Any]) -> list[HeuristicRule]:
        """Get heuristics that apply to the current context.

        Args:
            context: Context dict with failure info

        Returns:
            List of applicable HeuristicRules
        """
        applicable = []

        failure_class_str = context.get("failure_class", "")
        for heuristic in self._heuristics:
            if not heuristic.active:
                continue

            if (
                failure_class_str
                and heuristic.source_post_mortem_id
            ):
                pm = self._get_post_mortem(heuristic.source_post_mortem_id)
                if pm and pm.failure_class.value == failure_class_str:
                    applicable.append(heuristic)
                    continue

            if heuristic.trigger_pattern:
                for key, value in context.items():
                    if isinstance(value, str):
                        if re.search(heuristic.trigger_pattern, value, re.IGNORECASE):
                            applicable.append(heuristic)
                            break

        return applicable

    def _get_post_mortem(self, post_mortem_id: str) -> Optional[PostMortemEntry]:
        """Get a post-mortem by ID.

        Args:
            post_mortem_id: ID to look up

        Returns:
            PostMortemEntry or None
        """
        for pm in self._post_mortems:
            if pm.id == post_mortem_id:
                return pm
        return None

    def get_stats(self) -> LearningStats:
        """Get learning statistics.

        Returns:
            LearningStats with aggregated metrics
        """
        failure_counts = {}
        for pm in self._post_mortems:
            key = pm.failure_class.value
            failure_counts[key] = failure_counts.get(key, 0) + 1

        active_heuristics = sum(1 for h in self._heuristics if h.active)

        return LearningStats(
            total_post_mortems=len(self._post_mortems),
            total_heuristics=len(self._heuristics),
            active_heuristics=active_heuristics,
            failure_reduction_rate=self._calculate_failure_reduction(),
            top_failure_classes=failure_counts,
        )

    def _calculate_failure_reduction(self) -> float:
        """Calculate the failure reduction rate.

        Returns:
            Float between 0 and 1
        """
        if len(self._post_mortems) < 2:
            return 0.0

        effective_heuristics = sum(
            h.times_effective for h in self._heuristics if h.active
        )
        total_heuristic_applications = sum(
            h.times_applied for h in self._heuristics if h.active
        )

        if total_heuristic_applications == 0:
            return 0.0

        return effective_heuristics / total_heuristic_applications

    def save(self, path: str) -> None:
        """Persist post-mortems and heuristics to disk.

        Args:
            path: Directory path to save to
        """
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)

        post_mortems_file = path_obj / "post_mortems.json"
        heuristics_file = path_obj / "heuristics.json"

        try:
            with open(post_mortems_file, "w") as f:
                data = [pm.model_dump(mode="json") for pm in self._post_mortems]
                json.dump(data, f, indent=2, default=str)

            with open(heuristics_file, "w") as f:
                data = [h.model_dump(mode="json") for h in self._heuristics]
                json.dump(data, f, indent=2, default=str)

            self._logger.info(
                "learning_saved",
                path=path,
                post_mortems=len(self._post_mortems),
                heuristics=len(self._heuristics),
            )
        except Exception as e:
            self._logger.error(
                "failed_to_save_learning",
                path=path,
                error=str(e),
            )

    def load(self, path: str) -> None:
        """Load post-mortems and heuristics from disk.

        Args:
            path: Directory path to load from
        """
        path_obj = Path(path)

        post_mortems_file = path_obj / "post_mortems.json"
        heuristics_file = path_obj / "heuristics.json"

        try:
            if post_mortems_file.exists():
                with open(post_mortems_file, "r") as f:
                    data = json.load(f)
                    self._post_mortems = [
                        PostMortemEntry(**item) for item in data
                    ]

            if heuristics_file.exists():
                with open(heuristics_file, "r") as f:
                    data = json.load(f)
                    self._heuristics = [HeuristicRule(**item) for item in data]

            self._logger.info(
                "learning_loaded",
                path=path,
                post_mortems=len(self._post_mortems),
                heuristics=len(self._heuristics),
            )
        except Exception as e:
            self._logger.error(
                "failed_to_load_learning",
                path=path,
                error=str(e),
            )


__all__ = [
    "PostMortemEngine",
    "PostMortemEntry",
    "HeuristicRule",
    "FailureClass",
    "LearningStats",
]
