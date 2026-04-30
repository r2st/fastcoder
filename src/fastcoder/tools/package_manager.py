"""Package manager tool with auto-detection."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from fastcoder.tools.shell_executor import ShellExecutor
from fastcoder.types.tools import SideEffects, ToolName, ToolResult


class PackageManagerTool:
    """Unified package manager interface with auto-detection."""

    def __init__(self, project_dir: str, shell_executor: ShellExecutor):
        """Initialize package manager."""
        self.project_dir = Path(project_dir).resolve()
        self.shell = shell_executor
        self.pm_type = self._detect_package_manager()

    def _detect_package_manager(self) -> str:
        """Auto-detect package manager from lockfiles."""
        if (self.project_dir / "pnpm-lock.yaml").exists():
            return "pnpm"
        if (self.project_dir / "yarn.lock").exists():
            return "yarn"
        if (self.project_dir / "package-lock.json").exists():
            return "npm"
        if (self.project_dir / "package.json").exists():
            return "npm"
        if (self.project_dir / "pyproject.toml").exists():
            return "pip"
        if (self.project_dir / "requirements.txt").exists():
            return "pip"
        return "npm"

    async def install(self, packages: Optional[list[str]] = None) -> ToolResult:
        """Install packages."""
        if self.pm_type in ["npm", "yarn", "pnpm"]:
            return await self._install_node(packages)
        else:
            return await self._install_python(packages)

    async def _install_node(self, packages: Optional[list[str]]) -> ToolResult:
        """Install Node.js packages."""
        if packages:
            cmd = f"{self.pm_type} install {' '.join(packages)}"
            side_effects = SideEffects(packages_added=packages)
        else:
            cmd = f"{self.pm_type} install"
            side_effects = SideEffects(packages_added=["<all dependencies>"])

        result = await self.shell.execute(cmd)
        result.side_effects = side_effects
        return result

    async def _install_python(
        self, packages: Optional[list[str]]
    ) -> ToolResult:
        """Install Python packages."""
        if packages:
            cmd = f"pip install {' '.join(packages)}"
            side_effects = SideEffects(packages_added=packages)
        else:
            cmd = "pip install -r requirements.txt"
            side_effects = SideEffects(packages_added=["<from requirements.txt>"])

        result = await self.shell.execute(cmd)
        result.side_effects = side_effects
        return result

    async def update(self, packages: Optional[list[str]] = None) -> ToolResult:
        """Update packages."""
        if self.pm_type in ["npm", "yarn", "pnpm"]:
            if packages:
                cmd = f"{self.pm_type} update {' '.join(packages)}"
            else:
                cmd = f"{self.pm_type} update"
        else:
            if packages:
                cmd = f"pip install --upgrade {' '.join(packages)}"
            else:
                cmd = "pip install --upgrade -r requirements.txt"

        result = await self.shell.execute(cmd)
        result.side_effects = SideEffects(packages_added=packages or ["<updates>"])
        return result

    async def remove(self, packages: list[str]) -> ToolResult:
        """Remove packages."""
        if self.pm_type in ["npm", "yarn", "pnpm"]:
            if self.pm_type == "npm":
                cmd = f"npm uninstall {' '.join(packages)}"
            else:
                cmd = f"{self.pm_type} remove {' '.join(packages)}"
        else:
            cmd = f"pip uninstall -y {' '.join(packages)}"

        result = await self.shell.execute(cmd)
        result.side_effects = SideEffects(packages_removed=packages)
        return result

    async def audit(self) -> ToolResult:
        """Run security audit."""
        if self.pm_type in ["npm", "yarn"]:
            cmd = f"{self.pm_type} audit"
        elif self.pm_type == "pnpm":
            cmd = "pnpm audit"
        else:
            cmd = "pip audit"

        return await self.shell.execute(cmd)

    async def get_lockfile_hash(self) -> str:
        """Get hash of lockfile for change detection."""
        lockfiles = {
            "npm": "package-lock.json",
            "yarn": "yarn.lock",
            "pnpm": "pnpm-lock.yaml",
            "pip": "requirements.txt",
        }

        lockfile = self.project_dir / lockfiles.get(self.pm_type, "")
        if lockfile.exists():
            import hashlib

            with open(lockfile, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        return ""
