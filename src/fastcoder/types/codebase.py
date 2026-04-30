"""Codebase Intelligence Engine types."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ProjectProfile(BaseModel):
    language: str = "python"
    framework: str = ""
    package_manager: str = "pip"
    test_framework: str = "pytest"
    naming_conventions: dict[str, str] = Field(
        default_factory=lambda: {
            "files": "snake_case",
            "functions": "snake_case",
            "classes": "PascalCase",
            "constants": "UPPER_SNAKE_CASE",
        }
    )
    directory_structure: str = ""
    error_handling_pattern: str = "try-catch"
    import_style: str = "named"
    lint_config: Optional[dict[str, Any]] = None
    formatting_config: Optional[dict[str, Any]] = None


class SymbolInfo(BaseModel):
    name: str
    kind: str  # function, class, interface, type, variable, enum, constant
    file: str
    line: int = 0
    exported: bool = True
    type_signature: Optional[str] = None
    docstring: Optional[str] = None
    usage_count: int = 0


class DependencyNode(BaseModel):
    file: str
    imports: list[str] = Field(default_factory=list)
    imported_by: list[str] = Field(default_factory=list)
    is_circular: bool = False


class ASTNode(BaseModel):
    file: str
    type: str  # function, class, method, interface, type_alias, enum
    name: Optional[str] = None
    start_line: int = 0
    end_line: int = 0
    children: list["ASTNode"] = Field(default_factory=list)
    signature: Optional[str] = None
    exported: bool = False


ASTNode.model_rebuild()


class CodeChunk(BaseModel):
    file: str
    start_line: int = 0
    end_line: int = 0
    content: str = ""
    type: str = "block"  # function, class, method, block, module
    name: Optional[str] = None
    embedding: Optional[list[float]] = None


class SearchResult(BaseModel):
    chunk: CodeChunk
    score: float = 0.0
    match_type: str = "text"  # semantic, structural, text


class APIEndpoint(BaseModel):
    method: str
    path: str
    handler_file: str
    handler_function: str = ""
    request_schema: Optional[dict[str, Any]] = None
    response_schema: Optional[dict[str, Any]] = None


class APISurface(BaseModel):
    endpoints: list[APIEndpoint] = Field(default_factory=list)
    event_handlers: list[str] = Field(default_factory=list)
    cli_commands: list[str] = Field(default_factory=list)


class DetectedPattern(BaseModel):
    category: str
    pattern: str
    examples: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class ConventionScanResult(BaseModel):
    profile: ProjectProfile
    confidence: float = 0.0
    detected_patterns: list[DetectedPattern] = Field(default_factory=list)
