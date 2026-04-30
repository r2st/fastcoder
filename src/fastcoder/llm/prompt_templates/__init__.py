"""Prompt template management and rendering."""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from fastcoder.types.llm import Message, PromptSection, PromptTemplate, ReasoningMode

logger = logging.getLogger(__name__)


class PromptRegistry:
    """Registry for versioned prompt templates."""

    def __init__(self):
        """Initialize empty registry."""
        self._templates: Dict[str, Dict[str, PromptTemplate]] = {}
        self._default_versions: Dict[str, str] = {}
        self._initialize_built_in_templates()

    def register(self, template: PromptTemplate) -> None:
        """Register a prompt template.

        Args:
            template: PromptTemplate to register
        """
        if template.id not in self._templates:
            self._templates[template.id] = {}

        self._templates[template.id][template.version] = template

        # Set as default if first version
        if template.id not in self._default_versions:
            self._default_versions[template.id] = template.version

        logger.info(f"Registered template: {template.id} v{template.version}")

    def get(
        self, template_id: str, version: Optional[str] = None
    ) -> PromptTemplate:
        """Get a prompt template.

        Args:
            template_id: Template identifier
            version: Template version (uses default if not specified)

        Returns:
            PromptTemplate

        Raises:
            KeyError: If template not found
        """
        if template_id not in self._templates:
            raise KeyError(f"Template not found: {template_id}")

        version = version or self._default_versions.get(template_id)
        if version not in self._templates[template_id]:
            raise KeyError(f"Template version not found: {template_id}@{version}")

        return self._templates[template_id][version]

    def render(
        self, template_id: str, variables: Dict[str, str], version: Optional[str] = None
    ) -> List[Message]:
        """Render a template with variables.

        Args:
            template_id: Template identifier
            variables: Variable substitutions
            version: Template version

        Returns:
            List of rendered Message objects

        Raises:
            KeyError: If template not found
        """
        template = self.get(template_id, version)
        messages = []

        # Sort sections by priority
        sorted_sections = sorted(template.sections, key=lambda s: s.priority)

        for section in sorted_sections:
            if not section.required:
                # Skip optional sections if variables missing
                if not all(
                    f"{{{{{var}}}}}" in section.template or var in variables
                    for var in self._extract_variables(section.template)
                ):
                    continue

            # Render section template
            content = self._render_template(section.template, variables)

            messages.append(
                Message(
                    role=section.role,
                    content=content,
                )
            )

        return messages

    @staticmethod
    def _extract_variables(template: str) -> List[str]:
        """Extract variable names from template.

        Args:
            template: Template string with {{variable}} placeholders

        Returns:
            List of variable names
        """
        return re.findall(r"\{\{(\w+)\}\}", template)

    @staticmethod
    def _render_template(template: str, variables: Dict[str, str]) -> str:
        """Render template string with variables.

        Args:
            template: Template string with {{variable}} placeholders
            variables: Variable substitutions

        Returns:
            Rendered string
        """
        result = template
        for key, value in variables.items():
            result = result.replace(f"{{{{{key}}}}}", str(value))
        return result

    def _initialize_built_in_templates(self) -> None:
        """Initialize built-in prompt templates."""

        # Story Analysis template
        story_analysis = PromptTemplate(
            id="story_analysis",
            version="1.0",
            reasoning_mode=ReasoningMode.CHAIN_OF_THOUGHT,
            sections=[
                PromptSection(
                    role="system",
                    template="""You are an expert software engineer analyzing user stories for implementation.

Your task is to:
1. Understand the user story requirements
2. Identify any ambiguities or edge cases
3. Assess complexity and effort
4. Identify potential risks
5. List required skills and knowledge

Provide structured analysis focusing on clarity, completeness, and implementability.""",
                    priority=1,
                    required=True,
                ),
                PromptSection(
                    role="user",
                    template="""Please analyze this user story:

{{story_content}}

Provide:
- Summary of requirements
- Identified ambiguities (if any)
- Complexity score (1-10)
- Risk assessment
- Required implementation areas""",
                    priority=2,
                    required=True,
                ),
            ],
            max_output_tokens=2048,
        )
        self.register(story_analysis)

        # Planning template
        planning = PromptTemplate(
            id="planning",
            version="1.0",
            reasoning_mode=ReasoningMode.CHAIN_OF_THOUGHT,
            sections=[
                PromptSection(
                    role="system",
                    template="""You are an expert software architect creating detailed implementation plans.

Your task is to:
1. Break down the story into concrete tasks
2. Define clear acceptance criteria
3. Identify dependencies between tasks
4. Estimate effort for each task
5. Plan for testing and integration

Create actionable, well-sequenced steps that minimize rework.""",
                    priority=1,
                    required=True,
                ),
                PromptSection(
                    role="user",
                    template="""Create an implementation plan for this story:

{{story_content}}

Context:
- Project Type: {{project_type}}
- Current Tech Stack: {{tech_stack}}

Provide:
- Numbered task list with clear acceptance criteria
- Task dependencies
- Estimated effort per task
- Integration points
- Testing strategy""",
                    priority=2,
                    required=True,
                ),
            ],
            max_output_tokens=4096,
        )
        self.register(planning)

        # Code Generation template
        code_gen = PromptTemplate(
            id="code_generation",
            version="1.0",
            reasoning_mode=ReasoningMode.CHAIN_OF_THOUGHT,
            sections=[
                PromptSection(
                    role="system",
                    template="""You are an expert software developer writing production-quality code.

Your task is to:
1. Write clean, well-structured code
2. Follow best practices and conventions
3. Include comprehensive error handling
4. Add meaningful comments for complex logic
5. Ensure type safety and proper logging

Code must be complete, tested, and ready for production.""",
                    priority=1,
                    required=True,
                ),
                PromptSection(
                    role="user",
                    template="""Implement the following:

{{task_description}}

Requirements:
- Language: {{language}}
- Framework: {{framework}}
- Code style: {{code_style}}
- Error handling: Comprehensive
- Logging: Include appropriate logging

Provide complete, production-ready code with:
- All necessary imports
- Type hints
- Error handling
- Unit test cases
- Documentation""",
                    priority=2,
                    required=True,
                ),
            ],
            max_output_tokens=4096,
        )
        self.register(code_gen)

        # Test Generation template
        test_gen = PromptTemplate(
            id="test_generation",
            version="1.0",
            reasoning_mode=ReasoningMode.CHAIN_OF_THOUGHT,
            sections=[
                PromptSection(
                    role="system",
                    template="""You are an expert QA engineer creating comprehensive test suites.

Your task is to:
1. Identify all code paths and edge cases
2. Write unit tests for core functionality
3. Include integration tests
4. Add performance tests if needed
5. Ensure high coverage (80%+)

Tests must be clear, maintainable, and catch real bugs.""",
                    priority=1,
                    required=True,
                ),
                PromptSection(
                    role="user",
                    template="""Create comprehensive tests for this code:

{{code_content}}

Test Framework: {{test_framework}}
Coverage Target: {{coverage_target}}%

Provide:
- Unit tests for all functions
- Edge case tests
- Integration tests
- Mocking/fixtures as needed
- Clear test names and assertions""",
                    priority=2,
                    required=True,
                ),
            ],
            max_output_tokens=3072,
        )
        self.register(test_gen)

        # Code Review template
        code_review = PromptTemplate(
            id="code_review",
            version="1.0",
            reasoning_mode=ReasoningMode.CHAIN_OF_THOUGHT,
            sections=[
                PromptSection(
                    role="system",
                    template="""You are a senior code reviewer providing constructive feedback.

Your task is to:
1. Check code quality and style
2. Identify potential bugs or issues
3. Review performance implications
4. Check security concerns
5. Suggest improvements

Provide specific, actionable feedback that helps developers improve.""",
                    priority=1,
                    required=True,
                ),
                PromptSection(
                    role="user",
                    template="""Review this code:

{{code_content}}

Context:
- Project: {{project_name}}
- Purpose: {{code_purpose}}

Provide detailed review covering:
- Code quality and maintainability
- Potential bugs or edge cases
- Performance concerns
- Security issues
- Suggested improvements
- Positive aspects to keep""",
                    priority=2,
                    required=True,
                ),
            ],
            max_output_tokens=2048,
        )
        self.register(code_review)

        # Error Analysis template
        error_analysis = PromptTemplate(
            id="error_analysis",
            version="1.0",
            reasoning_mode=ReasoningMode.CHAIN_OF_THOUGHT,
            sections=[
                PromptSection(
                    role="system",
                    template="""You are an expert debugger analyzing error conditions.

Your task is to:
1. Understand the error and its root cause
2. Identify contributing factors
3. Trace error propagation
4. Suggest fixes with explanations
5. Recommend prevention strategies

Provide clear, logical analysis and actionable solutions.""",
                    priority=1,
                    required=True,
                ),
                PromptSection(
                    role="user",
                    template="""Analyze this error:

Error Message: {{error_message}}
Stack Trace:
{{stack_trace}}

Context:
- Code being run: {{code_snippet}}
- Recent changes: {{recent_changes}}

Provide:
- Root cause analysis
- Contributing factors
- Step-by-step fix
- Code changes needed
- Tests to prevent regression
- Prevention strategies""",
                    priority=2,
                    required=True,
                ),
            ],
            max_output_tokens=2048,
        )
        self.register(error_analysis)

        # Documentation template
        docs = PromptTemplate(
            id="documentation",
            version="1.0",
            reasoning_mode=ReasoningMode.DIRECT,
            sections=[
                PromptSection(
                    role="system",
                    template="""You are a technical writer creating clear, comprehensive documentation.

Your task is to:
1. Explain functionality clearly
2. Provide usage examples
3. Document parameters and return values
4. Include common pitfalls and tips
5. Keep documentation maintainable

Documentation should be helpful to developers of all levels.""",
                    priority=1,
                    required=True,
                ),
                PromptSection(
                    role="user",
                    template="""Create documentation for:

{{component_description}}

Component Type: {{component_type}}
Language: {{language}}

Include:
- What it does (1-2 sentences)
- How to use it with examples
- Parameters and return values
- Common use cases
- Known limitations
- Links to related components""",
                    priority=2,
                    required=True,
                ),
            ],
            max_output_tokens=2048,
        )
        self.register(docs)

        # Commit Message template
        commit_msg = PromptTemplate(
            id="commit_message",
            version="1.0",
            reasoning_mode=ReasoningMode.DIRECT,
            sections=[
                PromptSection(
                    role="system",
                    template="""You are a developer writing clear, concise commit messages following conventional commit format.

Format: type(scope): subject
Additional details in body if needed.

Types: feat, fix, refactor, perf, test, docs, chore
Keep subject under 50 chars, body lines under 72 chars.""",
                    priority=1,
                    required=True,
                ),
                PromptSection(
                    role="user",
                    template="""Write a commit message for these changes:

{{changes_summary}}

Provide the commit message in conventional commit format.""",
                    priority=2,
                    required=True,
                ),
            ],
            max_output_tokens=512,
        )
        self.register(commit_msg)

        # Self Reflection template
        reflection = PromptTemplate(
            id="self_reflection",
            version="1.0",
            reasoning_mode=ReasoningMode.SELF_REFLECT,
            sections=[
                PromptSection(
                    role="system",
                    template="""You are a reflective agent reviewing your work and decisions.

Your task is to:
1. Assess what you did and why
2. Identify what went well
3. Identify areas for improvement
4. Learn from mistakes
5. Plan better approaches

Honest self-assessment leads to continuous improvement.""",
                    priority=1,
                    required=True,
                ),
                PromptSection(
                    role="user",
                    template="""Reflect on this work:

What was done: {{work_description}}
Results: {{results}}
Challenges: {{challenges}}

Provide:
- Assessment of approach
- What went well
- What could improve
- Lessons learned
- Better approaches for next time""",
                    priority=2,
                    required=True,
                ),
            ],
            max_output_tokens=1024,
        )
        self.register(reflection)


# Global registry instance
_global_registry: Optional[PromptRegistry] = None


def get_prompt_registry() -> PromptRegistry:
    """Get or create global prompt registry.

    Returns:
        Global PromptRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = PromptRegistry()
    return _global_registry
