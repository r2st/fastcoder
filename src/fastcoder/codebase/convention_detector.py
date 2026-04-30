"""Detects coding conventions, frameworks, and patterns in projects."""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional

from fastcoder.types.codebase import ConventionScanResult, DetectedPattern, ProjectProfile


class ConventionDetector:
    """Detects project conventions, frameworks, and patterns."""

    # Framework detection mappings
    FRAMEWORK_DEPENDENCIES = {
        "django": ["django"],
        "flask": ["flask"],
        "fastapi": ["fastapi"],
        "react": ["react"],
        "vue": ["vue"],
        "angular": ["@angular/core"],
        "express": ["express"],
        "next": ["next"],
        "nest": ["@nestjs/core"],
    }

    FRAMEWORK_PATTERNS = {
        "django": [r"from django", r"import django"],
        "flask": [r"from flask import", r"Flask(__name__)"],
        "fastapi": [r"from fastapi import", r"FastAPI()"],
        "react": [r"import React", r"jsx", r"tsx"],
        "express": [r"express\(\)", r"app\.get\(", r"app\.post\("],
    }

    async def detect(self, project_dir: str) -> ConventionScanResult:
        """
        Scan project and detect conventions.

        Args:
            project_dir: Root directory of the project

        Returns:
            ConventionScanResult with detected profile and patterns
        """
        project_path = Path(project_dir)

        profile = ProjectProfile()

        # Detect primary language
        profile.language = self._detect_language(project_path)

        # Detect framework
        profile.framework = self._detect_framework(project_path)

        # Detect package manager
        profile.package_manager = self._detect_package_manager(project_path)

        # Detect test framework
        profile.test_framework = self._detect_test_framework(project_path)

        # Detect naming conventions
        naming = self._detect_naming_conventions(project_path)
        profile.naming_conventions = naming

        # Detect import style
        profile.import_style = self._detect_import_style(project_path)

        # Detect error handling pattern
        profile.error_handling_pattern = self._detect_error_handling(project_path)

        # Read lint configuration
        profile.lint_config = self._read_lint_config(project_path)
        profile.formatting_config = self._read_formatting_config(project_path)

        # Detect patterns
        patterns = self._detect_patterns(project_path)

        # Calculate confidence
        confidence = min(1.0, len([x for x in [profile.language, profile.framework] if x]) / 2.0 * 0.8 + 0.2)

        return ConventionScanResult(
            profile=profile,
            confidence=confidence,
            detected_patterns=patterns,
        )

    def _detect_language(self, project_path: Path) -> str:
        """Detect primary programming language."""
        file_counts = Counter()

        for suffix in [".py", ".ts", ".tsx", ".js", ".jsx"]:
            files = list(project_path.glob(f"**/*{suffix}"))
            files = [f for f in files if not any(p in f.parts for p in {"node_modules", "venv", ".git", "__pycache__"})]
            if files:
                file_counts[suffix] = len(files)

        if not file_counts:
            return "unknown"

        most_common_suffix = file_counts.most_common(1)[0][0]
        if most_common_suffix == ".py":
            return "python"
        elif most_common_suffix in {".ts", ".tsx"}:
            return "typescript"
        elif most_common_suffix in {".js", ".jsx"}:
            return "javascript"

        return "unknown"

    def _detect_framework(self, project_path: Path) -> str:
        """Detect framework from dependencies and code patterns."""
        # Check package.json
        package_json_path = project_path / "package.json"
        if package_json_path.exists():
            try:
                content = json.loads(package_json_path.read_text())
                deps = {**content.get("dependencies", {}), **content.get("devDependencies", {})}
                for framework, patterns in self.FRAMEWORK_DEPENDENCIES.items():
                    if any(pattern in deps for pattern in patterns):
                        return framework
            except (json.JSONDecodeError, OSError):
                pass

        # Check requirements.txt or pyproject.toml
        requirements_path = project_path / "requirements.txt"
        if requirements_path.exists():
            content = requirements_path.read_text()
            for framework, patterns in self.FRAMEWORK_DEPENDENCIES.items():
                if any(pattern in content for pattern in patterns):
                    return framework

        pyproject_path = project_path / "pyproject.toml"
        if pyproject_path.exists():
            content = pyproject_path.read_text()
            for framework, patterns in self.FRAMEWORK_DEPENDENCIES.items():
                if any(pattern in content for pattern in patterns):
                    return framework

        # Check code patterns
        all_files = list(project_path.glob("**/*.py")) + list(project_path.glob("**/*.{ts,tsx,js,jsx}"))
        all_files = [f for f in all_files if not any(p in f.parts for p in {"node_modules", "venv", ".git", "__pycache__"})]

        for file_path in all_files[:20]:  # Sample first 20 files
            try:
                content = file_path.read_text(encoding="utf-8")
                for framework, patterns in self.FRAMEWORK_PATTERNS.items():
                    if any(re.search(pattern, content) for pattern in patterns):
                        return framework
            except (UnicodeDecodeError, OSError):
                continue

        return ""

    def _detect_package_manager(self, project_path: Path) -> str:
        """Detect package manager from lock files."""
        if (project_path / "package-lock.json").exists():
            return "npm"
        elif (project_path / "yarn.lock").exists():
            return "yarn"
        elif (project_path / "pnpm-lock.yaml").exists():
            return "pnpm"
        elif (project_path / "Pipfile.lock").exists():
            return "pipenv"
        elif (project_path / "poetry.lock").exists():
            return "poetry"
        elif (project_path / "requirements.txt").exists() or (project_path / "pyproject.toml").exists():
            return "pip"

        return "unknown"

    def _detect_test_framework(self, project_path: Path) -> str:
        """Detect test framework."""
        # Check package.json
        package_json_path = project_path / "package.json"
        if package_json_path.exists():
            try:
                content = json.loads(package_json_path.read_text())
                deps = {**content.get("dependencies", {}), **content.get("devDependencies", {})}
                if "jest" in deps:
                    return "jest"
                elif "mocha" in deps:
                    return "mocha"
                elif "vitest" in deps:
                    return "vitest"
            except (json.JSONDecodeError, OSError):
                pass

        # Check pyproject.toml
        pyproject_path = project_path / "pyproject.toml"
        if pyproject_path.exists():
            content = pyproject_path.read_text()
            if "pytest" in content:
                return "pytest"
            elif "unittest" in content:
                return "unittest"

        # Check for test files
        test_files = list(project_path.glob("**/*test*.py")) + list(project_path.glob("**/*.test.{ts,tsx,js,jsx}"))
        if test_files:
            # Try to infer from test file content
            for file_path in test_files[:5]:
                try:
                    content = file_path.read_text(encoding="utf-8")
                    if "import unittest" in content or "from unittest" in content:
                        return "unittest"
                    elif "import pytest" in content or "from pytest" in content:
                        return "pytest"
                    elif "describe(" in content or "it(" in content:
                        return "jest"
                except (UnicodeDecodeError, OSError):
                    continue

        return "pytest" if Path(project_path / "pyproject.toml").exists() else "jest"

    def _detect_naming_conventions(self, project_path: Path) -> dict[str, str]:
        """Detect naming conventions by sampling code."""
        conventions: dict[str, str] = {
            "files": self._detect_naming_style(self._sample_filenames(project_path)),
            "functions": self._detect_naming_style(self._sample_function_names(project_path)),
            "classes": self._detect_naming_style(self._sample_class_names(project_path)),
            "constants": "UPPER_SNAKE_CASE",
        }
        return conventions

    def _sample_filenames(self, project_path: Path) -> list[str]:
        """Sample filenames from project."""
        files = list(project_path.glob("**/*.py"))
        files += list(project_path.glob("**/*.ts"))
        files += list(project_path.glob("**/*.js"))
        files = [f for f in files if not any(p in f.parts for p in {"node_modules", "venv", ".git", "__pycache__"})]
        return [f.stem for f in files[:50]]

    def _sample_function_names(self, project_path: Path) -> list[str]:
        """Sample function names from project."""
        names: list[str] = []
        for file_path in list(project_path.glob("**/*.py"))[:10]:
            try:
                content = file_path.read_text(encoding="utf-8")
                matches = re.findall(r"def\s+(\w+)\s*\(", content)
                names.extend(matches)
            except (UnicodeDecodeError, OSError):
                continue

        for file_path in list(project_path.glob("**/*.{ts,js}"))[:10]:
            try:
                content = file_path.read_text(encoding="utf-8")
                matches = re.findall(r"function\s+(\w+)\s*\(", content)
                names.extend(matches)
            except (UnicodeDecodeError, OSError):
                continue

        return names[:50]

    def _sample_class_names(self, project_path: Path) -> list[str]:
        """Sample class names from project."""
        names: list[str] = []
        for file_path in list(project_path.glob("**/*.py"))[:10]:
            try:
                content = file_path.read_text(encoding="utf-8")
                matches = re.findall(r"class\s+(\w+)", content)
                names.extend(matches)
            except (UnicodeDecodeError, OSError):
                continue

        for file_path in list(project_path.glob("**/*.{ts,js}"))[:10]:
            try:
                content = file_path.read_text(encoding="utf-8")
                matches = re.findall(r"class\s+(\w+)", content)
                names.extend(matches)
            except (UnicodeDecodeError, OSError):
                continue

        return names[:50]

    def _detect_naming_style(self, names: list[str]) -> str:
        """Detect naming convention style."""
        if not names:
            return "unknown"

        snake_count = sum(1 for n in names if re.match(r"^[a-z_]+$", n) or "_" in n)
        camel_count = sum(1 for n in names if re.match(r"^[a-z][a-zA-Z0-9]*$", n) and not "_" in n)
        pascal_count = sum(1 for n in names if re.match(r"^[A-Z][a-zA-Z0-9]*$", n))
        kebab_count = sum(1 for n in names if re.match(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$", n))

        counts = {
            "snake_case": snake_count,
            "camelCase": camel_count,
            "PascalCase": pascal_count,
            "kebab-case": kebab_count,
        }

        if not any(counts.values()):
            return "unknown"

        return max(counts, key=counts.get)

    def _detect_import_style(self, project_path: Path) -> str:
        """Detect import statement style."""
        py_files = list(project_path.glob("**/*.py"))[:5]
        named_imports = 0
        default_imports = 0

        for file_path in py_files:
            try:
                content = file_path.read_text(encoding="utf-8")
                named_imports += len(re.findall(r"from\s+\w+\s+import", content))
                default_imports += len(re.findall(r"^import\s+\w+", content, re.MULTILINE))
            except (UnicodeDecodeError, OSError):
                continue

        if named_imports > default_imports:
            return "named"
        return "default"

    def _detect_error_handling(self, project_path: Path) -> str:
        """Detect error handling pattern."""
        py_files = list(project_path.glob("**/*.py"))[:5]
        try_count = 0
        result_count = 0

        for file_path in py_files:
            try:
                content = file_path.read_text(encoding="utf-8")
                try_count += len(re.findall(r"try:", content))
                result_count += len(re.findall(r"Result\[", content))
            except (UnicodeDecodeError, OSError):
                continue

        if try_count > result_count:
            return "try-catch"
        return "result-type"

    def _read_lint_config(self, project_path: Path) -> Optional[dict]:
        """Read linting configuration."""
        for config_file in [".eslintrc.js", ".eslintrc.json", ".eslintrc.yml"]:
            config_path = project_path / config_file
            if config_path.exists():
                try:
                    if config_file.endswith(".json"):
                        return json.loads(config_path.read_text())
                    else:
                        # For JS or YAML, just return True to indicate presence
                        return {"enabled": True}
                except (json.JSONDecodeError, OSError):
                    continue

        # Check for ruff config
        pyproject_path = project_path / "pyproject.toml"
        if pyproject_path.exists():
            try:
                content = pyproject_path.read_text()
                if "[tool.ruff]" in content:
                    return {"enabled": True, "type": "ruff"}
            except OSError:
                pass

        return None

    def _read_formatting_config(self, project_path: Path) -> Optional[dict]:
        """Read formatting configuration."""
        prettier_path = project_path / ".prettierrc"
        if prettier_path.exists():
            try:
                content = prettier_path.read_text()
                if content.strip().startswith("{"):
                    return json.loads(content)
                else:
                    return {"enabled": True, "type": "prettier"}
            except (json.JSONDecodeError, OSError):
                pass

        prettier_yaml = project_path / ".prettierrc.yaml"
        if prettier_yaml.exists():
            return {"enabled": True, "type": "prettier"}

        return None

    def _detect_patterns(self, project_path: Path) -> list[DetectedPattern]:
        """Detect common patterns in project."""
        patterns: list[DetectedPattern] = []

        # Check for test patterns
        test_files = list(project_path.glob("**/*test*.py")) + list(project_path.glob("**/*.test.{ts,tsx,js,jsx}"))
        if test_files:
            patterns.append(
                DetectedPattern(
                    category="testing",
                    pattern="test_file_pattern",
                    examples=[str(f.relative_to(project_path)) for f in test_files[:3]],
                    confidence=0.9,
                )
            )

        # Check for API route patterns
        route_patterns = self._detect_route_patterns(project_path)
        if route_patterns:
            patterns.append(
                DetectedPattern(
                    category="api",
                    pattern="route_definition",
                    examples=route_patterns[:3],
                    confidence=0.8,
                )
            )

        # Check for model/schema patterns
        model_files = list(project_path.glob("**/models.py")) + list(project_path.glob("**/schemas.ts"))
        if model_files:
            patterns.append(
                DetectedPattern(
                    category="data_models",
                    pattern="separate_models",
                    examples=[str(f.relative_to(project_path)) for f in model_files[:3]],
                    confidence=0.85,
                )
            )

        return patterns

    def _detect_route_patterns(self, project_path: Path) -> list[str]:
        """Detect API route patterns."""
        routes: list[str] = []

        for file_path in list(project_path.glob("**/*.py"))[:10]:
            try:
                content = file_path.read_text(encoding="utf-8")
                matches = re.findall(r"@app\.route\(['\"]([^'\"]+)", content)
                routes.extend(matches)
            except (UnicodeDecodeError, OSError):
                continue

        for file_path in list(project_path.glob("**/*.{ts,js}"))[:10]:
            try:
                content = file_path.read_text(encoding="utf-8")
                matches = re.findall(r'app\.(get|post|put|delete)\([\'"]([^\'"]+)', content)
                routes.extend([m[1] for m in matches])
            except (UnicodeDecodeError, OSError):
                continue

        return routes
