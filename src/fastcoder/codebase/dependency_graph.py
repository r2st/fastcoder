"""Dependency graph building and analysis for codebases."""

import re
from pathlib import Path
from typing import Optional

from fastcoder.types.codebase import DependencyNode


class DependencyGraph:
    """Builds and analyzes import relationships between files."""

    def __init__(self) -> None:
        """Initialize the dependency graph."""
        self.nodes: dict[str, DependencyNode] = {}
        self.file_to_path: dict[str, Path] = {}

    async def build(self, project_dir: str) -> dict[str, DependencyNode]:
        """
        Build dependency graph by scanning all source files.

        Args:
            project_dir: Root directory of the project

        Returns:
            Dict of DependencyNode objects keyed by file path
        """
        self.nodes.clear()
        self.file_to_path.clear()
        project_path = Path(project_dir)

        # First pass: create nodes for all source files
        patterns = ["**/*.py", "**/*.ts", "**/*.tsx", "**/*.js", "**/*.jsx"]
        file_list: list[Path] = []

        for pattern in patterns:
            for file_path in project_path.glob(pattern):
                if any(part in file_path.parts for part in {"node_modules", "venv", "dist", ".git", "__pycache__"}):
                    continue
                file_list.append(file_path)
                file_path_str = str(file_path)
                self.nodes[file_path_str] = DependencyNode(file=file_path_str)
                self.file_to_path[file_path_str] = file_path

        # Second pass: extract imports
        for file_path in file_list:
            try:
                content = file_path.read_text(encoding="utf-8")
                self.add_file(str(file_path), content)
            except (UnicodeDecodeError, OSError):
                continue

        # Third pass: detect circular dependencies
        self._detect_circular_dependencies_marking()

        return self.nodes

    def add_file(self, file_path: str, content: str) -> None:
        """
        Update dependency information for a single file.

        Args:
            file_path: Path to the file
            content: File content
        """
        if file_path not in self.nodes:
            self.nodes[file_path] = DependencyNode(file=file_path)

        file_path_obj = Path(file_path)
        suffix = file_path_obj.suffix.lower()

        imports: list[str] = []
        if suffix == ".py":
            imports = self._extract_python_imports(file_path, content)
        elif suffix in {".ts", ".tsx", ".js", ".jsx"}:
            imports = self._extract_typescript_imports(file_path, content)

        # Resolve relative imports to absolute paths
        resolved_imports: list[str] = []
        for imp in imports:
            resolved = self._resolve_import(file_path, imp, file_path_obj.suffix.lower())
            if resolved:
                resolved_imports.append(resolved)

        self.nodes[file_path].imports = resolved_imports

        # Update backward references
        for other_file, node in self.nodes.items():
            if file_path in node.imports and file_path not in node.imported_by:
                node.imported_by.append(file_path)

    def remove_file(self, file_path: str) -> None:
        """
        Remove a file from the dependency graph.

        Args:
            file_path: Path to the file
        """
        if file_path not in self.nodes:
            return

        # Remove from other nodes' import/imported_by lists
        for node in self.nodes.values():
            if file_path in node.imports:
                node.imports.remove(file_path)
            if file_path in node.imported_by:
                node.imported_by.remove(file_path)

        del self.nodes[file_path]
        if file_path in self.file_to_path:
            del self.file_to_path[file_path]

    def get_dependencies(self, file_path: str) -> list[str]:
        """
        Get files that this file directly imports.

        Args:
            file_path: Path to the file

        Returns:
            List of imported file paths
        """
        if file_path in self.nodes:
            return self.nodes[file_path].imports
        return []

    def get_dependents(self, file_path: str) -> list[str]:
        """
        Get files that directly import this file.

        Args:
            file_path: Path to the file

        Returns:
            List of dependent file paths
        """
        if file_path in self.nodes:
            return self.nodes[file_path].imported_by
        return []

    def get_impacted_files(self, changed_files: list[str]) -> list[str]:
        """
        Get all files transitively impacted by changes to the given files.

        Args:
            changed_files: List of changed file paths

        Returns:
            List of all impacted file paths (including the changed files)
        """
        impacted: set[str] = set(changed_files)
        to_process = list(changed_files)

        while to_process:
            current = to_process.pop(0)
            for dependent in self.get_dependents(current):
                if dependent not in impacted:
                    impacted.add(dependent)
                    to_process.append(dependent)

        return sorted(list(impacted))

    def detect_circular_dependencies(self) -> list[list[str]]:
        """
        Detect circular dependencies using DFS.

        Returns:
            List of cycles, where each cycle is a list of file paths
        """
        cycles: list[list[str]] = []
        visited: set[str] = set()
        rec_stack: set[str] = set()
        path: list[str] = []

        def dfs(node_path: str) -> None:
            visited.add(node_path)
            rec_stack.add(node_path)
            path.append(node_path)

            for neighbor in self.nodes[node_path].imports:
                if neighbor not in self.nodes:
                    continue

                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start_idx = path.index(neighbor)
                    cycle = path[cycle_start_idx:] + [neighbor]
                    cycles.append(cycle)

            path.pop()
            rec_stack.remove(node_path)

        for node_file in self.nodes:
            if node_file not in visited:
                dfs(node_file)

        return cycles

    def _detect_circular_dependencies_marking(self) -> None:
        """Mark nodes that are part of circular dependencies."""
        cycles = self.detect_circular_dependencies()
        for cycle in cycles:
            for file_path in cycle:
                if file_path in self.nodes:
                    self.nodes[file_path].is_circular = True

    def _extract_python_imports(self, file_path: str, content: str) -> list[str]:
        """Extract import statements from Python code."""
        imports: list[str] = []

        # Pattern: import x or import x as y
        import_pattern = r"^import\s+([\w.]+)(?:\s+as\s+\w+)?"
        for match in re.finditer(import_pattern, content, re.MULTILINE):
            imports.append(match.group(1))

        # Pattern: from x import y
        from_pattern = r"^from\s+([\w.]+)\s+import"
        for match in re.finditer(from_pattern, content, re.MULTILINE):
            imports.append(match.group(1))

        return imports

    def _extract_typescript_imports(self, file_path: str, content: str) -> list[str]:
        """Extract import statements from TypeScript/JavaScript code."""
        imports: list[str] = []

        # Pattern: import ... from '...' or import ... from "..."
        import_pattern = r'import\s+(?:\{[^}]*\}|[\w*]+)\s+from\s+["\']([^"\']+)["\']'
        for match in re.finditer(import_pattern, content):
            imports.append(match.group(1))

        # Pattern: require('...')
        require_pattern = r"require\(['\"]([^'\"]+)['\"]\)"
        for match in re.finditer(require_pattern, content):
            imports.append(match.group(1))

        return imports

    def _resolve_import(self, file_path: str, import_str: str, suffix: str) -> Optional[str]:
        """
        Resolve an import statement to an actual file path.

        Args:
            file_path: Path of the importing file
            import_str: Import statement (module name or relative path)
            suffix: File extension

        Returns:
            Absolute path of the imported file or None if not found
        """
        file_path_obj = Path(file_path)

        # Handle relative imports (TypeScript/JavaScript)
        if import_str.startswith("."):
            base_dir = file_path_obj.parent
            if suffix in {".ts", ".tsx", ".js", ".jsx"}:
                # Try with various extensions
                for ext in [".ts", ".tsx", ".js", ".jsx", ""]:
                    candidate = (base_dir / import_str.replace("./", "").replace("../", "../")).with_suffix(ext)
                    if candidate.exists():
                        return str(candidate)
            return None

        # Handle Python imports
        if suffix == ".py":
            # Standard library or third-party - skip for now
            if import_str in {"os", "sys", "re", "json", "pathlib"}:
                return None

            # Try to find in project
            base_dir = file_path_obj.parent
            parts = import_str.split(".")
            candidate = base_dir / "/".join(parts)

            for ext in [".py", "/__init__.py"]:
                full_path = candidate.with_suffix(ext) if ext == ".py" else Path(str(candidate) + ext)
                if full_path.exists():
                    return str(full_path)

        return None
