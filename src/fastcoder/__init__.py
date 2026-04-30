"""Autonomous Software Development Agent — intelligent code generation and development.

Main exports:
- start_agent: Async entry point for starting the agent server
- AgentConfig: Configuration class for the agent
- Story: Primary input entity for the agent
- StorySubmission: User submission schema
"""

from __future__ import annotations

__version__ = "3.1.0"

from fastcoder.main import start_agent
from fastcoder.types.config import AgentConfig
from fastcoder.types.story import Story, StorySubmission

__all__ = [
    "__version__",
    "start_agent",
    "AgentConfig",
    "Story",
    "StorySubmission",
]
