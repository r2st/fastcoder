"""AST-based code indexing for Python, TypeScript, and JavaScript."""

import ast
import hashlib
import re
from pathlib import Path
from typing import Optional

from fastcoder.types.codebase import ASTNode


class ASTIndexer:
    """Indexes Abstract Syntax Trees for Python and regex-based for TS/JS files."""

    def __init__(self) -> None:
        """Initialize the AST indexer with content hash tracking."""
        self.file_hashes: dict[str, str] = {}
        self.ast_cache: dict[str, list[ASTNode]] = {}

    def _hash_content(self, content: str) -> str:
        """Generate SHA256 hash of file content."""
        return hashlib.sha256(content.encode()).hexdigest()

    async def index_file(self, file_path: str, content: str) -> list[ASTNode]:
        """
        Index a single file and extract AST nodes.

        Args:
            file_path: Path to the file
            content: File content

        Returns:
            List of ASTNode objects extracted from the file
        """
        file_path_obj = Path(file_path)
        suffix = file_path_obj.suffix.lower()

        # Check if content has changed
        content_hash = self._hash_content(content)
        if file_path in self.file_hashes and self.file_hashes[file_path] == content_hash:
            return self.ast_cache.get(file_path, [])

        # Index based on file type
        nodes: list[ASTNode] = []
        if suffix == ".py":
            nodes = self._index_python_file(file_path, content)
        elif suffix in {".ts", ".tsx", ".js", ".jsx"}:
            nodes = self._index_typescript_file(file_path, content)

        # Cache the results
        self.file_hashes[file_path] = content_hash
        self.ast_cache[file_path] = nodes

        return nodes

    def _index_python_file(self, file_path: str, content: str) -> list[ASTNode]:
        """Extract Python AST nodes using ast module."""
        nodes: list[ASTNode] = []
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return nodes

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                nodes.append(
                    ASTNode(
                        file=file_path,
                        type="function",
                        name=node.name,
                        start_line=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                        signature=self._get_python_function_signature(node),
                        exported=self._is_python_exported(node),
                    )
                )

            elif isinstance(node, ast.ClassDef):
                class_node = ASTNode(
                    file=file_path,
                    type="class",
                    name=node.name,
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    exported=self._is_python_exported(node),
                )
                # Add methods as children
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        class_node.children.append(
                            ASTNode(
                                file=file_path,
                                type="method",
                                name=item.name,
                                start_line=item.lineno,
                                end_line=item.end_lineno or item.lineno,
                                signature=self._get_python_function_signature(item),
                                exported=not item.name.startswith("_"),
                            )
                        )
                nodes.append(class_node)

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    nodes.append(
                        ASTNode(
                            file=file_path,
                            type="import",
                            name=alias.asname or alias.name,
                            start_line=node.lineno,
                            end_line=node.end_lineno or node.lineno,
                        )
                    )

            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                for alias in node.names:
                    nodes.append(
                        ASTNode(
                            file=file_path,
                            type="import",
                            name=f"{module_name}.{alias.name}",
                            start_line=node.lineno,
                            end_line=node.end_lineno or node.lineno,
                        )
                    )

        return nodes

    def _get_python_function_signature(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        """Extract function signature from AST node."""
        prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
        args = node.args
        arg_names = [arg.arg for arg in args.args]
        return f"{prefix}def {node.name}({', '.join(arg_names)})"

    def _is_python_exported(self, node: ast.stmt) -> bool:
        """Check if a Python node is exported (not private)."""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return not node.name.startswith("_")
        return False

    def _index_typescript_file(self, file_path: str, content: str) -> list[ASTNode]:
        """Extract TypeScript/JavaScript nodes using regex patterns."""
        nodes: list[ASTNode] = []

        # Pattern for export declarations
        export_pattern = r"export\s+(?:default\s+)?(?:const|let|var|function|class|interface|type|enum)\s+(\w+)"
        for match in re.finditer(export_pattern, content):
            line_num = content[:match.start()].count("\n") + 1
            nodes.append(
                ASTNode(
                    file=file_path,
                    type="export",
                    name=match.group(1),
                    start_line=line_num,
                    exported=True,
                )
            )

        # Pattern for function declarations
        func_pattern = r"(?:async\s+)?function\s+(\w+)\s*\((.*?)\)"
        for match in re.finditer(func_pattern, content):
            line_num = content[:match.start()].count("\n") + 1
            nodes.append(
                ASTNode(
                    file=file_path,
                    type="function",
                    name=match.group(1),
                    start_line=line_num,
                    signature=f"function {match.group(1)}({match.group(2)})",
                )
            )

        # Pattern for class declarations
        class_pattern = r"class\s+(\w+)(?:\s+extends\s+(\w+))?"
        for match in re.finditer(class_pattern, content):
            line_num = content[:match.start()].count("\n") + 1
            nodes.append(
                ASTNode(
                    file=file_path,
                    type="class",
                    name=match.group(1),
                    start_line=line_num,
                )
            )

        # Pattern for interface declarations
        interface_pattern = r"interface\s+(\w+)"
        for match in re.finditer(interface_pattern, content):
            line_num = content[:match.start()].count("\n") + 1
            nodes.append(
                ASTNode(
                    file=file_path,
                    type="interface",
                    name=match.group(1),
                    start_line=line_num,
                )
            )

        # Pattern for type declarations
        type_pattern = r"type\s+(\w+)\s*="
        for match in re.finditer(type_pattern, content):
            line_num = content[:match.start()].count("\n") + 1
            nodes.append(
                ASTNode(
                    file=file_path,
                    type="type_alias",
                    name=match.group(1),
                    start_line=line_num,
                )
            )

        # Pattern for enum declarations
        enum_pattern = r"enum\s+(\w+)"
        for match in re.finditer(enum_pattern, content):
            line_num = content[:match.start()].count("\n") + 1
            nodes.append(
                ASTNode(
                    file=file_path,
                    type="enum",
                    name=match.group(1),
                    start_line=line_num,
                )
            )

        return nodes

    async def index_project(self, project_dir: str) -> dict[str, list[ASTNode]]:
        """
        Index all source files in a project.

        Args:
            project_dir: Root directory of the project

        Returns:
            Dict mapping file paths to lists of ASTNode objects
        """
        result: dict[str, list[ASTNode]] = {}
        project_path = Path(project_dir)

        # Define source file patterns
        patterns = ["**/*.py", "**/*.ts", "**/*.tsx", "**/*.js", "**/*.jsx"]

        for pattern in patterns:
            for file_path in project_path.glob(pattern):
                # Skip node_modules, venv, dist, etc.
                if any(part in file_path.parts for part in {"node_modules", "venv", "dist", ".git", "__pycache__"}):
                    continue

                try:
                    content = file_path.read_text(encoding="utf-8")
                    nodes = await self.index_file(str(file_path), content)
                    result[str(file_path)] = nodes
                except (UnicodeDecodeError, OSError):
                    continue

        return result

    def get_exports(self, file_path: str) -> list[ASTNode]:
        """
        Get all exported symbols from a file.

        Args:
            file_path: Path to the file

        Returns:
            List of exported ASTNode objects
        """
        nodes = self.ast_cache.get(file_path, [])
        return [node for node in nodes if node.exported]

    def get_function_signature(self, file_path: str, func_name: str) -> Optional[str]:
        """
        Get the signature of a function.

        Args:
            file_path: Path to the file
            func_name: Name of the function

        Returns:
            Function signature string or None if not found
        """
        nodes = self.ast_cache.get(file_path, [])
        for node in nodes:
            if node.type == "function" and node.name == func_name:
                return node.signature
        return None

    def extract_skeleton(self, content: str, language: str = "python") -> str:
        """
        Extract code skeleton (signatures without bodies).

        Args:
            content: File content
            language: Programming language (python, typescript, javascript)

        Returns:
            Skeleton code with function/class signatures only
        """
        if language == "python":
            return self._extract_python_skeleton(content)
        else:
            return self._extract_typescript_skeleton(content)

    def _extract_python_skeleton(self, content: str) -> str:
        """Extract Python code skeleton."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return ""

        skeleton_lines: list[str] = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
                args = node.args
                arg_names = [arg.arg for arg in args.args]
                skeleton_lines.append(f"{prefix}def {node.name}({', '.join(arg_names)}): ...")

            elif isinstance(node, ast.ClassDef):
                skeleton_lines.append(f"class {node.name}: ...")

        return "\n".join(skeleton_lines)

    def _extract_typescript_skeleton(self, content: str) -> str:
        """Extract TypeScript/JavaScript code skeleton."""
        skeleton_lines: list[str] = []

        # Extract function signatures
        func_pattern = r"(?:async\s+)?function\s+(\w+)\s*\((.*?)\)"
        for match in re.finditer(func_pattern, content):
            skeleton_lines.append(f"function {match.group(1)}({match.group(2)}) {{ ... }}")

        # Extract class declarations
        class_pattern = r"class\s+(\w+)"
        for match in re.finditer(class_pattern, content):
            skeleton_lines.append(f"class {match.group(1)} {{ ... }}")

        # Extract interface declarations
        interface_pattern = r"interface\s+(\w+)"
        for match in re.finditer(interface_pattern, content):
            skeleton_lines.append(f"interface {match.group(1)} {{ ... }}")

        return "\n".join(skeleton_lines)
