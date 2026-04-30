"""Codebase Intelligence Engine - combines AST indexing, dependency analysis, and semantic search."""

import re
from pathlib import Path
from typing import Optional

from fastcoder.types.codebase import (
    APISurface,
    APIEndpoint,
    ASTNode,
    CodeChunk,
    ConventionScanResult,
    ProjectProfile,
    SearchResult,
    SymbolInfo,
)

from .ast_indexer import ASTIndexer
from .convention_detector import ConventionDetector
from .dependency_graph import DependencyGraph
from .semantic_search import SemanticSearch
from .symbol_table import SymbolTable


class CodebaseIntelligence:
    """Facade combining all codebase analysis components."""

    def __init__(self) -> None:
        """Initialize the codebase intelligence engine."""
        self.ast_indexer = ASTIndexer()
        self.dependency_graph = DependencyGraph()
        self.convention_detector = ConventionDetector()
        self.symbol_table = SymbolTable()
        self.semantic_search = SemanticSearch()
        self.project_profile: Optional[ProjectProfile] = None
        self.ast_index: dict[str, list[ASTNode]] = {}

    async def initialize(self, project_dir: str) -> ConventionScanResult:
        """
        Perform full initial indexing of the project.

        Args:
            project_dir: Root directory of the project

        Returns:
            ConventionScanResult with detected profile and patterns
        """
        # Index AST
        self.ast_index = await self.ast_indexer.index_project(project_dir)

        # Build dependency graph
        await self.dependency_graph.build(project_dir)

        # Detect conventions
        convention_result = await self.convention_detector.detect(project_dir)
        self.project_profile = convention_result.profile

        # Build symbol table
        self.symbol_table.build(self.ast_index)

        # Build semantic search index
        chunks = await self._build_code_chunks(project_dir)
        self.semantic_search.index(chunks)

        return convention_result

    async def reindex(self, changed_files: list[str]) -> list[str]:
        """
        Perform incremental re-indexing for changed files.

        Args:
            changed_files: List of changed file paths

        Returns:
            List of all impacted files
        """
        project_path = Path(changed_files[0]).parent if changed_files else Path.cwd()

        # Re-index changed files
        for file_path in changed_files:
            try:
                content = Path(file_path).read_text(encoding="utf-8")
                nodes = await self.ast_indexer.index_file(file_path, content)
                self.ast_index[file_path] = nodes
                self.dependency_graph.add_file(file_path, content)
                self.symbol_table.remove_symbols_for_file(file_path)
                self.symbol_table.build(self.ast_index)
                self.semantic_search.remove_chunks_for_file(file_path)
            except (UnicodeDecodeError, OSError):
                continue

        # Get impacted files
        impacted = self.dependency_graph.get_impacted_files(changed_files)

        return impacted

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """
        Search for code matching the query.

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of SearchResult objects
        """
        return self.semantic_search.search(query, top_k)

    def get_project_profile(self) -> Optional[ProjectProfile]:
        """
        Get the detected project profile.

        Returns:
            ProjectProfile object or None
        """
        return self.project_profile

    def get_impacted_files(self, changed_files: list[str]) -> list[str]:
        """
        Get all files impacted by changes to the given files.

        Args:
            changed_files: List of changed file paths

        Returns:
            List of impacted file paths
        """
        return self.dependency_graph.get_impacted_files(changed_files)

    def resolve_symbol(self, name: str) -> list[SymbolInfo]:
        """
        Resolve a symbol by name.

        Args:
            name: Symbol name

        Returns:
            List of matching SymbolInfo objects
        """
        return self.symbol_table.lookup(name)

    def get_file_skeleton(self, file_path: str) -> str:
        """
        Get code skeleton (signatures without bodies) for a file.

        Args:
            file_path: Path to the file

        Returns:
            Skeleton code
        """
        try:
            content = Path(file_path).read_text(encoding="utf-8")
            suffix = Path(file_path).suffix.lower()
            language = "python" if suffix == ".py" else "typescript"
            return self.ast_indexer.extract_skeleton(content, language)
        except (UnicodeDecodeError, OSError):
            return ""

    def get_api_surface(self) -> APISurface:
        """
        Extract API surface (endpoints) from Flask/FastAPI/Express patterns.

        Returns:
            APISurface object with endpoints
        """
        endpoints: list[APIEndpoint] = []

        # Check each indexed file for API patterns
        for file_path, nodes in self.ast_index.items():
            try:
                content = Path(file_path).read_text(encoding="utf-8")

                # Flask/FastAPI patterns
                flask_pattern = r"@(?:app|blueprint)\.route\(['\"]([^'\"]+)['\"](?:,\s*methods=\[([^]]+)\])?\)"
                for match in re.finditer(flask_pattern, content):
                    path = match.group(1)
                    methods = match.group(2) if match.group(2) else "GET"
                    methods = [m.strip().strip("'\"") for m in methods.split(",")]
                    for method in methods:
                        endpoints.append(
                            APIEndpoint(
                                method=method,
                                path=path,
                                handler_file=file_path,
                            )
                        )

                # FastAPI patterns
                fastapi_pattern = r"@app\.(?:get|post|put|delete|patch)\(['\"]([^'\"]+)['\"]\)"
                for match in re.finditer(fastapi_pattern, content):
                    path = match.group(1)
                    method = match.group(0).split(".")[1].split("(")[0].upper()
                    endpoints.append(
                        APIEndpoint(
                            method=method,
                            path=path,
                            handler_file=file_path,
                        )
                    )

                # Express patterns
                express_pattern = r"app\.(?:get|post|put|delete|patch)\(['\"]([^'\"]+)['\"]\)"
                for match in re.finditer(express_pattern, content):
                    path = match.group(1)
                    method = match.group(0).split(".")[1].split("(")[0].upper()
                    endpoints.append(
                        APIEndpoint(
                            method=method,
                            path=path,
                            handler_file=file_path,
                        )
                    )

            except (UnicodeDecodeError, OSError):
                continue

        # Detect event handlers
        event_handlers: list[str] = []
        for file_path, nodes in self.ast_index.items():
            try:
                content = Path(file_path).read_text(encoding="utf-8")
                event_matches = re.findall(r"@(?:on|listen|event)\(['\"]([^'\"]+)['\"]\)", content)
                event_handlers.extend(event_matches)
            except (UnicodeDecodeError, OSError):
                continue

        # Detect CLI commands
        cli_commands: list[str] = []
        for file_path, nodes in self.ast_index.items():
            try:
                content = Path(file_path).read_text(encoding="utf-8")
                cli_matches = re.findall(r"@(?:click|typer|argparse)\.command\(['\"]?([^'\")\s]+)?['\"]?\)", content)
                cli_commands.extend([m for m in cli_matches if m])
            except (UnicodeDecodeError, OSError):
                continue

        return APISurface(
            endpoints=endpoints,
            event_handlers=event_handlers,
            cli_commands=cli_commands,
        )

    def get_statistics(self) -> dict:
        """
        Get comprehensive statistics about the codebase.

        Returns:
            Dict with various codebase metrics
        """
        return {
            "ast": {
                "total_files": len(self.ast_index),
                "total_nodes": sum(len(nodes) for nodes in self.ast_index.values()),
            },
            "symbols": self.symbol_table.get_stats(),
            "dependencies": {
                "total_nodes": len(self.dependency_graph.nodes),
                "circular_deps": sum(1 for n in self.dependency_graph.nodes.values() if n.is_circular),
            },
            "search": self.semantic_search.get_index_stats(),
        }

    async def _build_code_chunks(self, project_dir: str) -> list[CodeChunk]:
        """
        Build code chunks for semantic search indexing.

        Args:
            project_dir: Root directory of the project

        Returns:
            List of CodeChunk objects
        """
        chunks: list[CodeChunk] = []
        project_path = Path(project_dir)

        patterns = ["**/*.py", "**/*.ts", "**/*.tsx", "**/*.js", "**/*.jsx"]

        for pattern in patterns:
            for file_path in project_path.glob(pattern):
                if any(part in file_path.parts for part in {"node_modules", "venv", "dist", ".git", "__pycache__"}):
                    continue

                try:
                    content = file_path.read_text(encoding="utf-8")
                    lines = content.split("\n")

                    # Create chunks per function/class
                    chunk_start = 0
                    in_definition = False
                    definition_type = ""

                    for line_idx, line in enumerate(lines):
                        stripped = line.strip()

                        # Detect function/class definitions
                        if stripped.startswith(("def ", "class ", "function ", "export function", "export class")):
                            if in_definition:
                                # Save previous chunk
                                chunk_content = "\n".join(lines[chunk_start : line_idx + 1])
                                chunks.append(
                                    CodeChunk(
                                        file=str(file_path),
                                        start_line=chunk_start,
                                        end_line=line_idx + 1,
                                        content=chunk_content,
                                        type=definition_type,
                                    )
                                )

                            chunk_start = line_idx
                            in_definition = True
                            definition_type = "function" if "function" in stripped or "def" in stripped else "class"

                    # Save final chunk if any
                    if in_definition:
                        chunk_content = "\n".join(lines[chunk_start:])
                        chunks.append(
                            CodeChunk(
                                file=str(file_path),
                                start_line=chunk_start,
                                end_line=len(lines),
                                content=chunk_content,
                                type=definition_type,
                            )
                        )

                except (UnicodeDecodeError, OSError):
                    continue

        return chunks

    def get_circular_dependencies(self) -> list[list[str]]:
        """
        Detect and return circular dependencies.

        Returns:
            List of cycles, each cycle is a list of file paths
        """
        return self.dependency_graph.detect_circular_dependencies()

    def search_symbols(self, query: str) -> list[SymbolInfo]:
        """
        Search for symbols matching the query.

        Args:
            query: Search query

        Returns:
            List of matching SymbolInfo objects
        """
        return self.symbol_table.search_symbols(query)

    def get_file_dependencies(self, file_path: str) -> dict[str, list[str]]:
        """
        Get dependency information for a file.

        Args:
            file_path: Path to the file

        Returns:
            Dict with imports and imported_by lists
        """
        return {
            "imports": self.dependency_graph.get_dependencies(file_path),
            "imported_by": self.dependency_graph.get_dependents(file_path),
        }


__all__ = ["CodebaseIntelligence"]
