"""Story Analyzer — parses and structures user stories into actionable specifications."""

from __future__ import annotations

import json
import re
import structlog
from typing import Callable, Optional

from pydantic import ValidationError

from fastcoder.types.codebase import ProjectProfile
from fastcoder.types.llm import CompletionRequest, Message
from fastcoder.types.story import (
    AcceptanceCriterion,
    FileDependency,
    StorySpec,
    StoryType,
)

logger = structlog.get_logger(__name__)


class StoryAnalyzer:
    """Analyzes raw user stories and converts them into structured specifications.

    Uses LLM to extract:
    - Story title and description
    - Acceptance criteria with BDD Given/When/Then structure
    - Story type classification (feature/bugfix/refactor/infra)
    - Complexity score (1-10) based on criteria count, dependencies, story type
    - Ambiguities and detected dependencies
    """

    def __init__(self, llm_complete: Callable):
        """Initialize the analyzer with an LLM completion function.

        Args:
            llm_complete: Async function matching ModelRouter.complete signature
                         (CompletionRequest) -> CompletionResponse
        """
        self.llm_complete = llm_complete
        self.logger = structlog.get_logger(f"{__name__}.{self.__class__.__name__}")

    async def analyze(
        self,
        raw_story: str,
        project_profile: Optional[ProjectProfile] = None,
    ) -> StorySpec:
        """Parse a raw story into a structured specification.

        Args:
            raw_story: Raw user story text
            project_profile: Optional project context (language, framework, etc.)

        Returns:
            StorySpec: Structured story specification with all fields populated

        Raises:
            ValueError: If story analysis fails after retries
        """
        self.logger.info(f"Analyzing story ({len(raw_story)} chars)")

        # Build the analysis prompt
        prompt = self._build_analysis_prompt(raw_story, project_profile)

        # Call LLM to parse the story
        try:
            response = await self.llm_complete(prompt)
            self.logger.debug(f"LLM response: {response.content[:200]}...")

            # Parse the JSON response
            spec = self._parse_llm_response(response.content, raw_story)

            # Calculate complexity score if not provided by LLM
            if spec.complexity_score is None or spec.complexity_score == 0:
                spec.complexity_score = self._calculate_complexity(spec)

            # Validate the spec
            spec = StorySpec.model_validate(spec)

            self.logger.info(
                f"Story analyzed: type={spec.story_type}, complexity={spec.complexity_score}, "
                f"criteria={len(spec.acceptance_criteria)}, ambiguities={len(spec.ambiguities)}"
            )
            return spec

        except Exception as e:
            self.logger.error(f"Story analysis failed: {e}")
            # Fall back to basic parsing
            return self._fallback_parse(raw_story, project_profile)

    def _build_analysis_prompt(
        self,
        raw_story: str,
        project_profile: Optional[ProjectProfile] = None,
    ) -> CompletionRequest:
        """Build the LLM prompt for story analysis.

        Returns a prompt that asks the LLM to:
        - Extract title and description
        - Identify acceptance criteria with BDD structure
        - Classify story type
        - Calculate complexity
        - Detect ambiguities
        """
        system_prompt = """You are an expert software engineer analyzing user stories.
Parse the given story and extract:
1. A clear title (concise, actionable)
2. A description (what needs to be built)
3. Acceptance criteria with BDD structure (Given/When/Then) where possible
4. Story type: 'feature', 'bugfix', 'refactor', or 'infra'
5. Complexity score (1-10): 1=trivial, 5=moderate, 10=very complex
   - Count criteria: 1-2 criteria = -2, 3-5 = 0, 6+ = +2
   - Bugfix = -1, Feature = 0, Refactor = +1, Infra = +2
   - Unknown dependencies = +1
6. File dependencies if mentioned
7. Ambiguities or unclear requirements

Respond with ONLY valid JSON, no markdown or extra text:
{
  "title": "string",
  "description": "string",
  "story_type": "feature|bugfix|refactor|infra",
  "acceptance_criteria": [
    {
      "id": "AC-1",
      "description": "string",
      "given": "string or null",
      "when": "string or null",
      "then": "string or null",
      "testable": true
    }
  ],
  "complexity_score": 1-10,
  "dependencies": [
    {"file_path": "path", "relationship": "imports|modifies|etc", "confidence": 0.0-1.0}
  ],
  "ambiguities": ["string"]
}"""

        project_context = ""
        if project_profile:
            project_context = f"""

Project Context:
- Language: {project_profile.language}
- Framework: {project_profile.framework or 'None'}
- Test Framework: {project_profile.test_framework}
- Naming: {project_profile.naming_conventions}"""

        user_message = f"""Analyze this story:

{raw_story}{project_context}"""

        return CompletionRequest(
            model="",  # Router will fill this
            messages=[
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_message),
            ],
            max_tokens=2000,
            temperature=0.3,
        )

    def _parse_llm_response(self, response_content: str, raw_story: str) -> StorySpec:
        """Parse the LLM's JSON response into a StorySpec.

        Extracts JSON and validates structure. Falls back to basic parsing
        if JSON extraction fails.
        """
        # Try to extract JSON from response
        json_match = re.search(r"\{.*\}", response_content, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found in LLM response")

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse LLM JSON: {e}")
            raise ValueError(f"Invalid JSON in LLM response: {e}")

        # Extract and validate story type
        story_type_str = data.get("story_type", "feature").lower()
        try:
            story_type = StoryType(story_type_str)
        except ValueError:
            self.logger.warning(f"Invalid story type '{story_type_str}', defaulting to feature")
            story_type = StoryType.FEATURE

        # Parse acceptance criteria
        acceptance_criteria = []
        for i, criterion_data in enumerate(data.get("acceptance_criteria", [])):
            criterion = AcceptanceCriterion(
                id=criterion_data.get("id", f"AC-{i+1}"),
                description=criterion_data.get("description", ""),
                given=criterion_data.get("given"),
                when=criterion_data.get("when"),
                then=criterion_data.get("then"),
                testable=criterion_data.get("testable", True),
            )
            acceptance_criteria.append(criterion)

        # Parse dependencies
        dependencies = []
        for dep_data in data.get("dependencies", []):
            dep = FileDependency(
                file_path=dep_data.get("file_path", ""),
                relationship=dep_data.get("relationship", "imports"),
                confidence=float(dep_data.get("confidence", 1.0)),
            )
            if dep.file_path:
                dependencies.append(dep)

        # Extract complexity score
        complexity_score = data.get("complexity_score", 5)
        try:
            complexity_score = max(1, min(10, int(complexity_score)))
        except (TypeError, ValueError):
            complexity_score = 5

        spec = StorySpec(
            title=data.get("title", "Untitled Story"),
            description=data.get("description", raw_story),
            story_type=story_type,
            acceptance_criteria=acceptance_criteria,
            complexity_score=complexity_score,
            dependencies=dependencies,
            ambiguities=data.get("ambiguities", []),
        )

        return spec

    def _calculate_complexity(self, spec: StorySpec) -> int:
        """Calculate complexity score based on story characteristics.

        Scoring:
        - Base: 5
        - 1-2 criteria: -2
        - 3-5 criteria: 0
        - 6+ criteria: +2
        - Bugfix: -1
        - Feature: 0
        - Refactor: +1
        - Infra: +2
        - Dependencies: +1 per unknown dependency
        """
        score = 5

        # Criteria count
        criteria_count = len(spec.acceptance_criteria)
        if criteria_count <= 2:
            score -= 2
        elif criteria_count >= 6:
            score += 2

        # Story type
        type_adjustments = {
            StoryType.BUGFIX: -1,
            StoryType.FEATURE: 0,
            StoryType.REFACTOR: 1,
            StoryType.INFRA: 2,
        }
        score += type_adjustments.get(spec.story_type, 0)

        # Unknown dependencies (low confidence)
        low_confidence_deps = sum(
            1 for dep in spec.dependencies if dep.confidence < 0.7
        )
        score += low_confidence_deps

        # Ambiguities
        score += min(2, len(spec.ambiguities))

        return max(1, min(10, score))

    def _fallback_parse(
        self,
        raw_story: str,
        project_profile: Optional[ProjectProfile] = None,
    ) -> StorySpec:
        """Fallback basic parsing when LLM fails.

        Extracts what we can from the raw text without LLM assistance.
        """
        self.logger.warning("Using fallback story parsing (LLM unavailable)")

        # Try to extract title (first line or first sentence)
        lines = raw_story.strip().split("\n")
        title = lines[0][:100] if lines else "Untitled Story"

        # Detect story type from keywords
        story_type = StoryType.FEATURE
        story_lower = raw_story.lower()
        if "bug" in story_lower or "fix" in story_lower:
            story_type = StoryType.BUGFIX
        elif "refactor" in story_lower or "improve" in story_lower:
            story_type = StoryType.REFACTOR
        elif "infra" in story_lower or "setup" in story_lower:
            story_type = StoryType.INFRA

        # Try to extract "Given/When/Then" patterns
        acceptance_criteria = []
        given_when_then_blocks = re.findall(
            r"(?:^|\n)\s*(?:Given|When|Then)[\s:].*?(?=(?:Given|When|Then)|$)",
            raw_story,
            re.IGNORECASE | re.MULTILINE,
        )

        if given_when_then_blocks:
            # Parse BDD format
            current_given = None
            current_when = None
            current_then = None

            for block in given_when_then_blocks:
                if block.lower().startswith("given"):
                    current_given = block[5:].strip()
                elif block.lower().startswith("when"):
                    current_when = block[4:].strip()
                elif block.lower().startswith("then"):
                    current_then = block[4:].strip()
                    if current_given or current_when or current_then:
                        acceptance_criteria.append(
                            AcceptanceCriterion(
                                id=f"AC-{len(acceptance_criteria)+1}",
                                description=f"{current_given or ''} {current_when or ''} {current_then or ''}".strip(),
                                given=current_given,
                                when=current_when,
                                then=current_then,
                                testable=True,
                            )
                        )
                        current_given = None
                        current_when = None
                        current_then = None

        # If no structured criteria found, create one from description
        if not acceptance_criteria:
            acceptance_criteria = [
                AcceptanceCriterion(
                    id="AC-1",
                    description=raw_story,
                    testable=True,
                )
            ]

        # Estimate complexity
        complexity_score = self._estimate_complexity_from_text(
            raw_story, story_type, len(acceptance_criteria)
        )

        return StorySpec(
            title=title,
            description=raw_story,
            story_type=story_type,
            acceptance_criteria=acceptance_criteria,
            complexity_score=complexity_score,
            dependencies=[],
            ambiguities=self._detect_ambiguities(raw_story),
        )

    def _estimate_complexity_from_text(
        self,
        text: str,
        story_type: StoryType,
        criteria_count: int,
    ) -> int:
        """Quick complexity estimate from text patterns."""
        score = 5

        # Criteria count
        if criteria_count <= 2:
            score -= 2
        elif criteria_count >= 6:
            score += 2

        # Story type
        type_adjustments = {
            StoryType.BUGFIX: -1,
            StoryType.FEATURE: 0,
            StoryType.REFACTOR: 1,
            StoryType.INFRA: 2,
        }
        score += type_adjustments.get(story_type, 0)

        # Word count heuristic
        word_count = len(text.split())
        if word_count > 200:
            score += 1

        return max(1, min(10, score))

    def _detect_ambiguities(self, text: str) -> list[str]:
        """Detect common ambiguity markers in story text."""
        ambiguities = []

        # Look for vague terms
        vague_terms = {
            r"\b(might|maybe|could|should|seems|appears)\b": "Vague or uncertain language",
            r"\b(etc|and so on|etc\.)\b": "Incomplete list",
            r"\b(some|several|many|few)\b": "Unspecified quantity",
            r"\b(soon|later|quickly|eventually)\b": "Vague timing",
            r"\b(relevant|appropriate|suitable)\b": "Subjective criteria",
        }

        text_lower = text.lower()
        for pattern, meaning in vague_terms.items():
            if re.search(pattern, text_lower):
                if meaning not in ambiguities:
                    ambiguities.append(meaning)

        return ambiguities
