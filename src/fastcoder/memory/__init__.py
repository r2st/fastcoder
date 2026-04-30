"""Memory Store — multi-tier in-memory storage for self-improvement via RAG."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastcoder.types.memory import (
    MemoryConsolidationResult,
    MemoryEntry,
    MemoryQuery,
    MemoryTier,
    MemoryType,
)
from fastcoder.types.story import Story


class MemoryStore:
    """Multi-tier in-memory store for learned patterns, fixes, conventions."""

    def __init__(self, max_entries_per_tier: int = 500):
        """Initialize MemoryStore with empty tier dictionaries.

        Args:
            max_entries_per_tier: Maximum entries per tier before eviction.
        """
        self.max_entries_per_tier = max_entries_per_tier
        self.memories: dict[MemoryTier, list[MemoryEntry]] = {
            tier: [] for tier in MemoryTier
        }
        self.error_fixes: dict[str, str] = {}  # fingerprint -> fix

    def store(self, entry: MemoryEntry) -> None:
        """Add a memory entry to the appropriate tier.

        Args:
            entry: MemoryEntry to store.
        """
        if entry.tier not in self.memories:
            self.memories[entry.tier] = []
        self.memories[entry.tier].append(entry)
        if len(self.memories[entry.tier]) > self.max_entries_per_tier:
            self.evict(entry.tier, self.max_entries_per_tier)

    def query(self, q: MemoryQuery) -> list[MemoryEntry]:
        """Retrieve memories by text similarity using Jaccard distance on tokens.

        Args:
            q: MemoryQuery with query text and optional filters.

        Returns:
            List of matching MemoryEntry objects sorted by similarity.
        """
        candidates: list[tuple[MemoryEntry, float]] = []

        tiers = [q.tier] if q.tier else list(MemoryTier)
        for tier in tiers:
            for entry in self.memories.get(tier, []):
                if q.type and entry.type != q.type:
                    continue
                if q.project_id and entry.project_id != q.project_id:
                    continue
                if entry.effectiveness_score < q.min_effectiveness:
                    continue

                similarity = self._jaccard_similarity(q.query, entry.content)
                if similarity > 0.0:
                    candidates.append((entry, similarity))

        # Sort by similarity descending, then by effectiveness_score
        candidates.sort(
            key=lambda x: (x[1], x[0].effectiveness_score), reverse=True
        )
        return [entry for entry, _ in candidates[: q.max_results]]

    def consolidate(self, story: Story) -> MemoryConsolidationResult:
        """After story completes, extract lessons and merge similar memories.

        Extracts:
        - Error patterns fixed (from failed iterations)
        - Conventions discovered (from successful code generation)
        - Effective strategies (high-scoring patterns)

        Merges similar memories with content similarity > 0.8.

        Args:
            story: Completed Story.

        Returns:
            MemoryConsolidationResult with new, updated, and evicted entries.
        """
        result = MemoryConsolidationResult()

        # Extract error fixes from iterations
        for iteration in story.iterations:
            if iteration.error_fingerprint and iteration.error_fix:
                self.record_error_fix(
                    iteration.error_fingerprint, iteration.error_fix, story.id
                )

        # Create memories from story success
        if story.state == story.state.DONE:
            strategy_entry = MemoryEntry(
                id=str(uuid.uuid4()),
                type=MemoryType.PATTERN,
                tier=MemoryTier.SEMANTIC,
                context=story.spec.description if story.spec else "",
                content=f"Successfully completed {story.spec.title if story.spec else 'story'} "
                f"in {len(story.iterations)} iterations.",
                source_story_id=story.id,
                effectiveness_score=0.9,
                project_id=story.project_id,
            )
            result.new_memories.append(strategy_entry)
            self.store(strategy_entry)

        # Merge similar memories
        merged_ids = set()
        for tier in MemoryTier:
            entries = self.memories[tier]
            for i, entry1 in enumerate(entries):
                if entry1.id in merged_ids:
                    continue
                for entry2 in entries[i + 1 :]:
                    if entry2.id in merged_ids:
                        continue
                    similarity = self._jaccard_similarity(
                        entry1.content, entry2.content
                    )
                    if similarity > 0.8:
                        entry1.effectiveness_score = max(
                            entry1.effectiveness_score,
                            entry2.effectiveness_score,
                        )
                        entry1.use_count += entry2.use_count
                        merged_ids.add(entry2.id)
                        result.merged_count += 1

        for mid in merged_ids:
            result.evicted_memory_ids.append(mid)
            for tier in MemoryTier:
                self.memories[tier] = [
                    e for e in self.memories[tier] if e.id != mid
                ]

        return result

    def decay(self, decay_rate: float = 0.95) -> None:
        """Reduce effectiveness score for entries not used recently.

        Args:
            decay_rate: Multiplier for entries not used in the last 7 days.
        """
        cutoff = datetime.utcnow() - timedelta(days=7)
        for tier in MemoryTier:
            for entry in self.memories[tier]:
                if entry.last_used_at < cutoff:
                    entry.effectiveness_score *= decay_rate

    def evict(self, tier: MemoryTier, max_entries: int = 500) -> None:
        """Remove lowest-scoring entries from a tier.

        Args:
            tier: MemoryTier to evict from.
            max_entries: Maximum entries to keep.
        """
        if tier not in self.memories:
            return
        entries = self.memories[tier]
        if len(entries) <= max_entries:
            return
        entries.sort(
            key=lambda e: (e.effectiveness_score, e.use_count), reverse=True
        )
        self.memories[tier] = entries[:max_entries]

    def get_error_fix(self, fingerprint: str) -> str | None:
        """Retrieve a procedural memory fix by error fingerprint.

        Args:
            fingerprint: Error fingerprint hash.

        Returns:
            Fix string if found, None otherwise.
        """
        return self.error_fixes.get(fingerprint)

    def record_error_fix(
        self, fingerprint: str, fix: str, story_id: str
    ) -> None:
        """Record an error fix in procedural memory.

        Args:
            fingerprint: Error fingerprint hash.
            fix: Fix description/code.
            story_id: Story ID where fix was applied.
        """
        self.error_fixes[fingerprint] = fix
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            type=MemoryType.ERROR_FIX,
            tier=MemoryTier.PROCEDURAL,
            context=f"Error fingerprint: {fingerprint}",
            content=fix,
            source_story_id=story_id,
            effectiveness_score=0.85,
        )
        self.store(entry)

    def save(self, file_path: str) -> None:
        """Persist memory store to JSON using Pydantic serialization.

        Args:
            file_path: Path to save JSON file.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "memories": {
                tier.value: [e.model_dump() for e in entries]
                for tier, entries in self.memories.items()
            },
            "error_fixes": self.error_fixes,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load(self, file_path: str) -> None:
        """Load memory store from JSON file.

        Args:
            file_path: Path to load JSON file from.
        """
        path = Path(file_path)
        if not path.exists():
            return

        try:
            with open(path) as f:
                data = json.load(f)

            # Restore memories
            for tier_name, entries_data in data.get("memories", {}).items():
                try:
                    tier = MemoryTier(tier_name)
                    self.memories[tier] = [
                        MemoryEntry(**e) for e in entries_data
                    ]
                except ValueError:
                    pass

            # Restore error fixes
            self.error_fixes = data.get("error_fixes", {})
        except (json.JSONDecodeError, IOError):
            pass

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity between two texts on tokens.

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            Jaccard similarity score (0 to 1).
        """
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        return intersection / union if union > 0 else 0.0
