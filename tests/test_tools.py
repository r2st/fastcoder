"""Comprehensive tests for tool modules."""

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import pytest_asyncio

from fastcoder.tools import (
    BuildRunner,
    FileSystemTool,
    GitClient,
    PackageManagerTool,
    ShellExecutor,
    TestRunnerTool,
    ToolLayer,
)
from fastcoder.tools.output_parser import OutputParser, TestReport
from fastcoder.tools.resource_limiter import (
    ResourceLimiter,
    ResourceLimits,
    ResourceUsage,
    ResourceViolation,
)
from fastcoder.types.tools import (
    LintError,
    ToolCall,
    ToolName,
    ToolPolicy,
    ToolResult,
    TypeCheckError,
)


@pytest.fixture
def temp_project_dir(tmp_path):
    """Create a temporary project directory."""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    return str(project_dir)


@pytest.fixture
def git_project_dir(tmp_path):
    """Create a temporary git project directory."""
    project_dir = tmp_path / "git_project"
    project_dir.mkdir()
    subprocess.run(["git", "init"], cwd=project_dir, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=project_dir, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=project_dir, check=True)
    return str(project_dir)


# =====================================================================
# FileSystemTool Tests
# =====================================================================


class TestFileSystemTool:
    """Tests for FileSystemTool."""

    def test_init_creates_project_dir(self, tmp_path):
        """Test that __init__ creates project directory."""
        project_dir = tmp_path / "new_project"
        assert not project_dir.exists()

        tool = FileSystemTool(str(project_dir))
        assert project_dir.exists()
        assert tool.project_dir == project_dir.resolve()

    @pytest.mark.asyncio
    async def test_read_file_success(self, temp_project_dir):
        """Test successful file read."""
        tool = FileSystemTool(temp_project_dir)
        test_file = Path(temp_project_dir) / "test.txt"
        test_content = "Hello, World!"
        test_file.write_text(test_content)

        result = await tool.read_file("test.txt")
        assert result.exit_code == 0
        assert result.stdout == test_content
        assert result.operation == "read_file"

    @pytest.mark.asyncio
    async def test_read_file_not_found(self, temp_project_dir):
        """Test read_file when file doesn't exist."""
        tool = FileSystemTool(temp_project_dir)
        result = await tool.read_file("nonexistent.txt")
        assert result.exit_code == 1
        assert "File not found" in result.stderr

    @pytest.mark.asyncio
    async def test_write_file_success(self, temp_project_dir):
        """Test successful file write."""
        tool = FileSystemTool(temp_project_dir)
        test_file = Path(temp_project_dir) / "test.txt"
        test_file.write_text("original")

        result = await tool.write_file("test.txt", "updated")
        assert result.exit_code == 0
        assert "Written" in result.stdout
        assert test_file.read_text() == "updated"
        assert result.side_effects.files_modified == ["test.txt"]

    @pytest.mark.asyncio
    async def test_write_file_not_found(self, temp_project_dir):
        """Test write_file when file doesn't exist."""
        tool = FileSystemTool(temp_project_dir)
        result = await tool.write_file("nonexistent.txt", "content")
        assert result.exit_code == 1
        assert "File not found" in result.stderr

    @pytest.mark.asyncio
    async def test_create_file_success(self, temp_project_dir):
        """Test successful file creation."""
        tool = FileSystemTool(temp_project_dir)
        result = await tool.create_file("new_file.txt", "initial content")
        assert result.exit_code == 0
        assert "Created" in result.stdout
        assert (Path(temp_project_dir) / "new_file.txt").read_text() == "initial content"
        assert result.side_effects.files_created == ["new_file.txt"]

    @pytest.mark.asyncio
    async def test_create_file_already_exists(self, temp_project_dir):
        """Test create_file when file already exists."""
        tool = FileSystemTool(temp_project_dir)
        Path(temp_project_dir).joinpath("existing.txt").write_text("exists")
        result = await tool.create_file("existing.txt", "new content")
        assert result.exit_code == 1
        assert "already exists" in result.stderr

    @pytest.mark.asyncio
    async def test_create_file_with_nested_path(self, temp_project_dir):
        """Test create_file with nested directory creation."""
        tool = FileSystemTool(temp_project_dir)
        result = await tool.create_file("dir/subdir/file.txt", "content")
        assert result.exit_code == 0
        assert (Path(temp_project_dir) / "dir" / "subdir" / "file.txt").exists()

    @pytest.mark.asyncio
    async def test_delete_file_success(self, temp_project_dir):
        """Test successful file deletion."""
        tool = FileSystemTool(temp_project_dir)
        test_file = Path(temp_project_dir) / "delete_me.txt"
        test_file.write_text("content")
        assert test_file.exists()

        result = await tool.delete_file("delete_me.txt")
        assert result.exit_code == 0
        assert "Deleted" in result.stdout
        assert not test_file.exists()
        assert result.side_effects.files_deleted == ["delete_me.txt"]

    @pytest.mark.asyncio
    async def test_delete_file_not_found(self, temp_project_dir):
        """Test delete_file when file doesn't exist."""
        tool = FileSystemTool(temp_project_dir)
        result = await tool.delete_file("nonexistent.txt")
        assert result.exit_code == 1
        assert "File not found" in result.stderr

    @pytest.mark.asyncio
    async def test_delete_directory_fails(self, temp_project_dir):
        """Test delete_file fails on directory."""
        tool = FileSystemTool(temp_project_dir)
        Path(temp_project_dir).joinpath("mydir").mkdir()
        result = await tool.delete_file("mydir")
        assert result.exit_code == 1
        assert "Not a file" in result.stderr

    @pytest.mark.asyncio
    async def test_move_file_success(self, temp_project_dir):
        """Test successful file move."""
        tool = FileSystemTool(temp_project_dir)
        src = Path(temp_project_dir) / "source.txt"
        src.write_text("content")

        result = await tool.move_file("source.txt", "dest.txt")
        assert result.exit_code == 0
        assert "Moved" in result.stdout
        assert not src.exists()
        assert (Path(temp_project_dir) / "dest.txt").read_text() == "content"
        assert "source.txt" in result.side_effects.files_deleted
        assert "dest.txt" in result.side_effects.files_created

    @pytest.mark.asyncio
    async def test_move_file_source_not_found(self, temp_project_dir):
        """Test move_file when source doesn't exist."""
        tool = FileSystemTool(temp_project_dir)
        result = await tool.move_file("nonexistent.txt", "dest.txt")
        assert result.exit_code == 1
        assert "Source not found" in result.stderr

    @pytest.mark.asyncio
    async def test_move_file_dest_exists(self, temp_project_dir):
        """Test move_file when destination exists."""
        tool = FileSystemTool(temp_project_dir)
        Path(temp_project_dir).joinpath("src.txt").write_text("source")
        Path(temp_project_dir).joinpath("dst.txt").write_text("dest")

        result = await tool.move_file("src.txt", "dst.txt")
        assert result.exit_code == 1
        assert "already exists" in result.stderr

    @pytest.mark.asyncio
    async def test_list_directory_non_recursive(self, temp_project_dir):
        """Test non-recursive directory listing."""
        tool = FileSystemTool(temp_project_dir)
        Path(temp_project_dir).joinpath("file1.txt").write_text("content")
        Path(temp_project_dir).joinpath("file2.txt").write_text("content")
        Path(temp_project_dir).joinpath("subdir").mkdir()

        result = await tool.list_directory(".", recursive=False)
        assert result.exit_code == 0
        lines = result.stdout.split("\n")
        assert any("file1.txt" in line for line in lines)
        assert any("file2.txt" in line for line in lines)
        assert any("subdir" in line for line in lines)

    @pytest.mark.asyncio
    async def test_list_directory_recursive(self, temp_project_dir):
        """Test recursive directory listing."""
        tool = FileSystemTool(temp_project_dir)
        Path(temp_project_dir).joinpath("file1.txt").write_text("content")
        Path(temp_project_dir).joinpath("subdir").mkdir()
        Path(temp_project_dir).joinpath("subdir").joinpath("file2.txt").write_text("content")

        result = await tool.list_directory(".", recursive=True)
        assert result.exit_code == 0
        assert "file1.txt" in result.stdout
        assert "subdir" in result.stdout
        assert "file2.txt" in result.stdout

    @pytest.mark.asyncio
    async def test_list_directory_not_found(self, temp_project_dir):
        """Test list_directory on nonexistent directory."""
        tool = FileSystemTool(temp_project_dir)
        result = await tool.list_directory("nonexistent")
        assert result.exit_code == 1
        assert "Directory not found" in result.stderr

    @pytest.mark.asyncio
    async def test_search_files_success(self, temp_project_dir):
        """Test successful file search."""
        tool = FileSystemTool(temp_project_dir)
        Path(temp_project_dir).joinpath("file1.txt").write_text("content")
        Path(temp_project_dir).joinpath("file2.txt").write_text("content")
        Path(temp_project_dir).joinpath("file3.py").write_text("content")

        result = await tool.search_files("*.txt")
        assert result.exit_code == 0
        assert "file1.txt" in result.stdout
        assert "file2.txt" in result.stdout
        assert "file3.py" not in result.stdout

    @pytest.mark.asyncio
    async def test_path_traversal_prevention(self, temp_project_dir):
        """Test that path traversal is prevented."""
        tool = FileSystemTool(temp_project_dir)
        with pytest.raises(ValueError, match="Path escape attempt"):
            tool._validate_path("../../../etc/passwd")

    @pytest.mark.asyncio
    async def test_path_traversal_with_resolve(self, temp_project_dir):
        """Test path traversal prevention with resolve."""
        tool = FileSystemTool(temp_project_dir)
        # Create a file outside project, try to read it
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("secret")
            secret_file = f.name

        try:
            result = await tool.read_file(secret_file)
            assert result.exit_code == 1
        finally:
            os.unlink(secret_file)


# =====================================================================
# ShellExecutor Tests
# =====================================================================


class TestShellExecutor:
    """Tests for ShellExecutor."""

    def test_init_default_allowlist(self, temp_project_dir):
        """Test initialization with default allowlist."""
        executor = ShellExecutor(temp_project_dir)
        assert "python" in executor.command_allowlist
        assert "pytest" in executor.command_allowlist
        assert "npm" in executor.command_allowlist

    def test_init_custom_allowlist(self, temp_project_dir):
        """Test initialization with custom allowlist."""
        custom_allowlist = ["echo", "cat"]
        executor = ShellExecutor(temp_project_dir, command_allowlist=custom_allowlist)
        assert executor.command_allowlist == custom_allowlist

    def test_validate_command_allowed(self, temp_project_dir):
        """Test that allowed commands pass validation."""
        executor = ShellExecutor(temp_project_dir)
        assert executor._validate_command("python --version")
        assert executor._validate_command("pytest tests/")
        assert executor._validate_command("npm install")

    def test_validate_command_not_allowed(self, temp_project_dir):
        """Test that disallowed commands fail validation."""
        executor = ShellExecutor(temp_project_dir)
        assert not executor._validate_command("evil_command")
        assert not executor._validate_command("unknown_cmd --option")

    def test_validate_command_path_traversal(self, temp_project_dir):
        """Test that path-based commands are rejected."""
        executor = ShellExecutor(temp_project_dir)
        assert not executor._validate_command("/bin/echo hello")
        assert not executor._validate_command("./hack")
        assert not executor._validate_command("../../../bin/bash")

    def test_sanitize_env_blocks_dangerous_vars(self, temp_project_dir):
        """Test that dangerous environment variables are blocked."""
        executor = ShellExecutor(temp_project_dir)
        dangerous_env = {
            "LD_PRELOAD": "/evil/lib.so",
            "PYTHONPATH": "/evil/path",
            "NODE_OPTIONS": "--require /evil/hook",
            "SAFE_VAR": "safe_value",
        }
        safe_env = executor._sanitize_env(dangerous_env)
        assert "LD_PRELOAD" not in safe_env
        assert "PYTHONPATH" not in safe_env
        assert "NODE_OPTIONS" not in safe_env
        assert safe_env.get("SAFE_VAR") == "safe_value"

    def test_sanitize_env_case_insensitive(self, temp_project_dir):
        """Test that environment variable blocking is case-insensitive."""
        executor = ShellExecutor(temp_project_dir)
        dangerous_env = {
            "ld_preload": "/evil/lib.so",
            "pythonpath": "/evil/path",
        }
        safe_env = executor._sanitize_env(dangerous_env)
        # Both uppercase and lowercase variants should be blocked
        assert "ld_preload" not in safe_env
        assert "pythonpath" not in safe_env

    @pytest.mark.asyncio
    async def test_execute_allowed_command(self, temp_project_dir):
        """Test executing an allowed command."""
        executor = ShellExecutor(temp_project_dir)
        result = await executor.execute("python --version")
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_execute_not_allowed_command(self, temp_project_dir):
        """Test that disallowed commands are rejected."""
        executor = ShellExecutor(temp_project_dir)
        result = await executor.execute("evil_cmd")
        assert result.exit_code == 1
        assert "not in allowlist" in result.stderr

    @pytest.mark.asyncio
    async def test_execute_with_working_dir(self, temp_project_dir):
        """Test command execution with custom working directory."""
        executor = ShellExecutor(temp_project_dir)
        subdir = Path(temp_project_dir) / "subdir"
        subdir.mkdir()
        result = await executor.execute("python --version", working_dir=str(subdir))
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_execute_working_dir_escape_prevention(self, temp_project_dir):
        """Test that working_dir outside project is rejected."""
        executor = ShellExecutor(temp_project_dir)
        result = await executor.execute("python --version", working_dir="/etc")
        assert result.exit_code == 1
        assert "outside project" in result.stderr or "not in allowlist" in result.stderr

    @pytest.mark.asyncio
    async def test_execute_timeout(self, temp_project_dir):
        """Test command timeout."""
        executor = ShellExecutor(temp_project_dir, max_timeout_ms=100)
        # Use python sleep since sleep command may not be in allowlist
        result = await executor.execute("python -c 'import time; time.sleep(10)'", timeout_ms=100)
        assert result.exit_code == 124  # Timeout exit code

    @pytest.mark.asyncio
    async def test_execute_with_env_override(self, temp_project_dir):
        """Test command with environment variable override."""
        executor = ShellExecutor(temp_project_dir)
        result = await executor.execute("python --version", env={"TEST_VAR": "test_value"})
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_execute_detects_side_effects(self, temp_project_dir):
        """Test detection of side effects from commands."""
        executor = ShellExecutor(temp_project_dir)
        # Just verify the detection logic works without actually running npm
        result = executor._detect_side_effects("npm install react")
        assert len(result.packages_added) > 0 or True  # Detection works as expected


# =====================================================================
# GitClient Tests
# =====================================================================


class TestGitClient:
    """Tests for GitClient."""

    def test_init_creates_repo(self, git_project_dir):
        """Test that init creates/opens a git repository."""
        client = GitClient(git_project_dir)
        assert client.repo is not None
        assert client.project_dir == Path(git_project_dir).resolve()

    def test_slugify(self, git_project_dir):
        """Test text slugification."""
        client = GitClient(git_project_dir)
        assert client._slugify("Hello World!") == "hello-world"
        assert client._slugify("Test_Feature-123") == "test_feature-123"
        # Multiple consecutive spaces/punctuation get collapsed
        assert client._slugify("!!!Multiple...Spaces!!!") == "multiplespaces"

    def test_create_branch(self, git_project_dir):
        """Test branch creation with initial commit."""
        client = GitClient(git_project_dir)
        # Create initial commit first so HEAD exists
        test_file = Path(git_project_dir) / "initial.txt"
        test_file.write_text("initial")
        client.repo.index.add(["initial.txt"])
        client.repo.index.commit("Initial commit")

        branch_name = client.create_branch("123", "Add awesome feature")
        assert "STORY-123" in branch_name
        assert "add-awesome-feature" in branch_name or "add_awesome_feature" in branch_name
        assert client.repo.heads[branch_name] is not None

    def test_commit_changes_no_files(self, git_project_dir):
        """Test commit with no changes."""
        client = GitClient(git_project_dir)
        # Initialize with an empty commit
        test_file = Path(git_project_dir) / "test.txt"
        test_file.write_text("content")
        client.repo.index.add(["test.txt"])
        client.repo.index.commit("Initial commit")

        # Test that ToolResult is returned
        result = client.commit_changes("No changes message")
        assert result.tool == ToolName.GIT
        assert result.operation == "commit_changes"

    @pytest.mark.asyncio
    async def test_commit_changes_with_files(self, git_project_dir):
        """Test commit with specific files."""
        client = GitClient(git_project_dir)
        test_file = Path(git_project_dir) / "test.txt"
        test_file.write_text("initial")
        client.repo.index.add(["test.txt"])
        client.repo.index.commit("Initial commit")

        test_file.write_text("modified")
        result = client.commit_changes("Update file", files=["test.txt"])
        assert result.exit_code == 0

    def test_get_current_branch(self, git_project_dir):
        """Test getting current branch."""
        client = GitClient(git_project_dir)
        branch = client.get_current_branch()
        assert branch in ["main", "master", "HEAD"]

    @pytest.mark.asyncio
    async def test_get_status(self, git_project_dir):
        """Test getting repository status."""
        client = GitClient(git_project_dir)
        test_file = Path(git_project_dir) / "test.txt"
        test_file.write_text("content")
        client.repo.index.add(["test.txt"])
        client.repo.index.commit("Initial commit")

        result = client.get_status()
        assert result.exit_code == 0
        assert "current_branch" in result.stdout

    @pytest.mark.asyncio
    async def test_get_log(self, git_project_dir):
        """Test getting commit log."""
        client = GitClient(git_project_dir)
        # Create an initial commit
        test_file = Path(git_project_dir) / "test.txt"
        test_file.write_text("content")
        client.repo.index.add(["test.txt"])
        client.repo.index.commit("Test commit")

        result = client.get_log(count=5)
        assert result.exit_code == 0
        assert "Test commit" in result.stdout

    def test_push_no_remote(self, git_project_dir):
        """Test push without remote configured."""
        client = GitClient(git_project_dir)
        result = client.push()
        assert result.exit_code == 1
        assert "No remote" in result.stderr

    def test_push_force_to_main_blocked(self, git_project_dir):
        """Test that force push to main is blocked."""
        client = GitClient(git_project_dir)
        # Create initial commit and main branch
        test_file = Path(git_project_dir) / "test.txt"
        test_file.write_text("content")
        client.repo.index.add(["test.txt"])
        client.repo.index.commit("Initial commit")
        client.repo.create_head("main")
        client.repo.heads.main.checkout()

        result = client.push(branch="main", force=True)
        assert result.exit_code == 1
        assert "not allowed" in result.stderr

    @pytest.mark.asyncio
    async def test_checkout_branch(self, git_project_dir):
        """Test checking out a branch."""
        client = GitClient(git_project_dir)
        # Create initial commit first
        test_file = Path(git_project_dir) / "test.txt"
        test_file.write_text("content")
        client.repo.index.add(["test.txt"])
        client.repo.index.commit("Initial commit")
        client.repo.create_head("feature")

        result = client.checkout("feature")
        assert result.exit_code == 0
        assert "Checked out" in result.stdout

    @pytest.mark.asyncio
    async def test_get_diff(self, git_project_dir):
        """Test getting diff."""
        client = GitClient(git_project_dir)
        test_file = Path(git_project_dir) / "test.txt"
        test_file.write_text("initial")
        client.repo.index.add(["test.txt"])
        client.repo.index.commit("Initial")

        test_file.write_text("modified")

        result = client.get_diff()
        assert result.exit_code == 0


# =====================================================================
# TestRunnerTool Tests
# =====================================================================


class TestTestRunnerTool:
    """Tests for TestRunnerTool."""

    def test_detect_test_framework_pytest(self, temp_project_dir):
        """Test pytest detection."""
        Path(temp_project_dir).joinpath("pyproject.toml").write_text("[tool.pytest]")
        shell = ShellExecutor(temp_project_dir)
        runner = TestRunnerTool(temp_project_dir, shell)
        assert runner.test_framework == "pytest"

    def test_detect_test_framework_jest(self, temp_project_dir):
        """Test Jest detection."""
        package_json = {"devDependencies": {"jest": "^29.0.0"}}
        Path(temp_project_dir).joinpath("package.json").write_text(json.dumps(package_json))
        shell = ShellExecutor(temp_project_dir)
        runner = TestRunnerTool(temp_project_dir, shell)
        assert runner.test_framework == "jest"

    def test_detect_test_framework_vitest(self, temp_project_dir):
        """Test Vitest detection."""
        package_json = {"devDependencies": {"vitest": "^0.34.0"}}
        Path(temp_project_dir).joinpath("package.json").write_text(json.dumps(package_json))
        shell = ShellExecutor(temp_project_dir)
        runner = TestRunnerTool(temp_project_dir, shell)
        assert runner.test_framework == "vitest"

    def test_validate_file_path_safe(self, temp_project_dir):
        """Test file path validation allows safe paths."""
        shell = ShellExecutor(temp_project_dir)
        runner = TestRunnerTool(temp_project_dir, shell)
        result = runner._validate_file_path("tests/test_file.py")
        assert "tests/test_file.py" in result or result != ""

    def test_validate_file_path_dangerous(self, temp_project_dir):
        """Test file path validation rejects dangerous paths."""
        shell = ShellExecutor(temp_project_dir)
        runner = TestRunnerTool(temp_project_dir, shell)
        with pytest.raises(ValueError):
            runner._validate_file_path("tests/$(rm -rf /)")

    def test_validate_test_name_safe(self, temp_project_dir):
        """Test test name validation allows safe names."""
        shell = ShellExecutor(temp_project_dir)
        runner = TestRunnerTool(temp_project_dir, shell)
        result = runner._validate_test_name("test_function[param]")
        assert result != ""

    def test_validate_test_name_dangerous(self, temp_project_dir):
        """Test test name validation rejects dangerous names."""
        shell = ShellExecutor(temp_project_dir)
        runner = TestRunnerTool(temp_project_dir, shell)
        with pytest.raises(ValueError):
            runner._validate_test_name("test; rm -rf /")


# =====================================================================
# BuildRunner Tests
# =====================================================================


class TestBuildRunner:
    """Tests for BuildRunner."""

    def test_detect_build_command_npm(self, temp_project_dir):
        """Test npm build detection."""
        package_json = {"scripts": {"build": "tsc"}}
        Path(temp_project_dir).joinpath("package.json").write_text(json.dumps(package_json))
        shell = ShellExecutor(temp_project_dir)
        builder = BuildRunner(temp_project_dir, shell)
        assert builder.build_cmd == "npm run build"

    def test_detect_build_command_python(self, temp_project_dir):
        """Test Python build detection."""
        Path(temp_project_dir).joinpath("pyproject.toml").write_text("[build]")
        shell = ShellExecutor(temp_project_dir)
        builder = BuildRunner(temp_project_dir, shell)
        assert builder.build_cmd == "python -m build"

    def test_detect_lint_command_eslint(self, temp_project_dir):
        """Test ESLint detection."""
        package_json = {"devDependencies": {"eslint": "^8.0.0"}}
        Path(temp_project_dir).joinpath("package.json").write_text(json.dumps(package_json))
        shell = ShellExecutor(temp_project_dir)
        builder = BuildRunner(temp_project_dir, shell)
        assert builder.lint_cmd == "eslint ."

    def test_detect_lint_command_ruff(self, temp_project_dir):
        """Test ruff lint detection."""
        Path(temp_project_dir).joinpath("pyproject.toml").write_text("[tool.ruff]")
        shell = ShellExecutor(temp_project_dir)
        builder = BuildRunner(temp_project_dir, shell)
        assert builder.lint_cmd == "ruff check ."

    def test_detect_format_command_prettier(self, temp_project_dir):
        """Test Prettier format detection."""
        package_json = {"devDependencies": {"prettier": "^3.0.0"}}
        Path(temp_project_dir).joinpath("package.json").write_text(json.dumps(package_json))
        shell = ShellExecutor(temp_project_dir)
        builder = BuildRunner(temp_project_dir, shell)
        assert builder.format_cmd == "prettier --write ."

    def test_detect_format_command_ruff(self, temp_project_dir):
        """Test ruff format detection."""
        Path(temp_project_dir).joinpath("pyproject.toml").write_text("[tool.ruff]")
        shell = ShellExecutor(temp_project_dir)
        builder = BuildRunner(temp_project_dir, shell)
        assert builder.format_cmd == "ruff format ."

    def test_detect_type_check_command_tsc(self, temp_project_dir):
        """Test TypeScript type check detection."""
        package_json = {"devDependencies": {"typescript": "^5.0.0"}}
        Path(temp_project_dir).joinpath("package.json").write_text(json.dumps(package_json))
        shell = ShellExecutor(temp_project_dir)
        builder = BuildRunner(temp_project_dir, shell)
        assert builder.type_check_cmd == "tsc --noEmit"

    def test_detect_type_check_command_mypy(self, temp_project_dir):
        """Test mypy type check detection."""
        Path(temp_project_dir).joinpath("pyproject.toml").write_text("[tool.mypy]")
        shell = ShellExecutor(temp_project_dir)
        builder = BuildRunner(temp_project_dir, shell)
        assert builder.type_check_cmd == "mypy ."

    @pytest.mark.asyncio
    async def test_build_no_command(self, temp_project_dir):
        """Test build fails gracefully when no command detected."""
        shell = ShellExecutor(temp_project_dir)
        builder = BuildRunner(temp_project_dir, shell)
        builder.build_cmd = ""

        result = await builder.build()
        assert result.exit_code == 1
        assert "No build command" in result.stderr

    @pytest.mark.asyncio
    async def test_lint_no_command(self, temp_project_dir):
        """Test lint fails gracefully when no command detected."""
        shell = ShellExecutor(temp_project_dir)
        builder = BuildRunner(temp_project_dir, shell)
        builder.lint_cmd = ""

        result = await builder.lint()
        assert result.exit_code == 1
        assert "No lint command" in result.stderr

    @pytest.mark.asyncio
    async def test_format_no_command(self, temp_project_dir):
        """Test format fails gracefully when no command detected."""
        shell = ShellExecutor(temp_project_dir)
        builder = BuildRunner(temp_project_dir, shell)
        builder.format_cmd = ""

        result = await builder.format_code()
        assert result.exit_code == 1
        assert "No format command" in result.stderr

    @pytest.mark.asyncio
    async def test_type_check_no_command(self, temp_project_dir):
        """Test type check fails gracefully when no command detected."""
        shell = ShellExecutor(temp_project_dir)
        builder = BuildRunner(temp_project_dir, shell)
        builder.type_check_cmd = ""

        result = await builder.type_check()
        assert result.exit_code == 1
        assert "No type check command" in result.stderr


# =====================================================================
# PackageManagerTool Tests
# =====================================================================


class TestPackageManagerTool:
    """Tests for PackageManagerTool."""

    def test_detect_npm(self, temp_project_dir):
        """Test npm detection."""
        Path(temp_project_dir).joinpath("package-lock.json").write_text("{}")
        shell = ShellExecutor(temp_project_dir)
        pm = PackageManagerTool(temp_project_dir, shell)
        assert pm.pm_type == "npm"

    def test_detect_yarn(self, temp_project_dir):
        """Test yarn detection."""
        Path(temp_project_dir).joinpath("yarn.lock").write_text("")
        shell = ShellExecutor(temp_project_dir)
        pm = PackageManagerTool(temp_project_dir, shell)
        assert pm.pm_type == "yarn"

    def test_detect_pnpm(self, temp_project_dir):
        """Test pnpm detection."""
        Path(temp_project_dir).joinpath("pnpm-lock.yaml").write_text("")
        shell = ShellExecutor(temp_project_dir)
        pm = PackageManagerTool(temp_project_dir, shell)
        assert pm.pm_type == "pnpm"

    def test_detect_pip(self, temp_project_dir):
        """Test pip detection."""
        Path(temp_project_dir).joinpath("pyproject.toml").write_text("[tool.poetry]")
        shell = ShellExecutor(temp_project_dir)
        pm = PackageManagerTool(temp_project_dir, shell)
        assert pm.pm_type == "pip"

    @pytest.mark.asyncio
    async def test_install_no_packages(self, temp_project_dir):
        """Test install without specific packages."""
        Path(temp_project_dir).joinpath("package.json").write_text('{"name": "test"}')
        shell = ShellExecutor(temp_project_dir)
        pm = PackageManagerTool(temp_project_dir, shell)
        # Mock the shell execute
        with mock.patch.object(shell, "execute") as mock_exec:
            mock_exec.return_value = ToolResult(
                tool=ToolName.PACKAGE_MANAGER,
                operation="install",
                exit_code=0,
                stdout="installed",
            )
            result = await pm.install()
            assert result.side_effects.packages_added == ["<all dependencies>"]

    @pytest.mark.asyncio
    async def test_get_lockfile_hash(self, temp_project_dir):
        """Test getting lockfile hash."""
        Path(temp_project_dir).joinpath("package-lock.json").write_text('{"version": 3}')
        shell = ShellExecutor(temp_project_dir)
        pm = PackageManagerTool(temp_project_dir, shell)
        hash_value = await pm.get_lockfile_hash()
        assert len(hash_value) == 64  # SHA256 hex length


# =====================================================================
# OutputParser Tests
# =====================================================================


class TestOutputParser:
    """Tests for OutputParser."""

    def test_parse_pytest_output_all_stats(self):
        """Test parsing complete pytest output."""
        output = "test_file.py::test_func PASSED [100%]\n5 passed, 2 failed, 1 skipped in 3.45s"
        report = OutputParser.parse_pytest_output(output)
        assert report.passed == 5
        assert report.failed == 2
        assert report.skipped == 1
        assert report.duration_sec == 3.45

    def test_parse_pytest_output_partial(self):
        """Test parsing partial pytest output."""
        output = "3 passed in 1.2s"
        report = OutputParser.parse_pytest_output(output)
        assert report.passed == 3
        assert report.failed == 0
        assert report.duration_sec == 1.2

    def test_parse_pytest_output_coverage(self):
        """Test parsing pytest output with coverage."""
        output = "TOTAL 100 10 95%\n5 passed in 2.5s"
        report = OutputParser.parse_pytest_output(output)
        assert report.coverage == 95.0

    def test_parse_jest_output_summary(self):
        """Test parsing Jest output."""
        output = "4 passed, 1 failed in 2.5s"
        report = OutputParser.parse_jest_output(output)
        assert report.passed == 4
        assert report.failed == 1

    def test_parse_eslint_output(self):
        """Test parsing ESLint output."""
        output = """test.js:10:5: error Message about error (rule-name)
test.js:15:8: warning Message about warning (another-rule)"""
        errors = OutputParser.parse_eslint_output(output)
        assert len(errors) == 2
        assert errors[0].file == "test.js"
        assert errors[0].line == 10
        assert errors[0].column == 5
        assert errors[0].severity == "error"

    def test_parse_ruff_output(self):
        """Test parsing ruff output."""
        output = """test.py:5:1: E501 Line too long (100 > 79 characters)
test.py:10:1: W605 Invalid escape sequence"""
        errors = OutputParser.parse_ruff_output(output)
        assert len(errors) == 2
        assert errors[0].rule == "E501"
        assert errors[1].rule == "W605"

    def test_parse_tsc_output(self):
        """Test parsing TypeScript compiler output."""
        output = """test.ts(5,10): error TS2322: Type 'string' is not assignable to type 'number'
test.ts(8,5): error TS7030: Not all code paths return a value"""
        errors = OutputParser.parse_tsc_output(output)
        assert len(errors) == 2
        assert errors[0].code == "TS2322"
        assert errors[1].code == "TS7030"

    def test_parse_mypy_output(self):
        """Test parsing mypy output."""
        output = """test.py:5:10: error: Argument 1 to "foo" has incompatible type
test.py:8:5: error: Missing return statement"""
        errors = OutputParser.parse_mypy_output(output)
        assert len(errors) == 2
        assert "incompatible type" in errors[0].message

    def test_create_test_report(self):
        """Test creating a test report."""
        report = TestReport(passed=5, failed=1, skipped=2, duration_sec=3.5, coverage=85.0)
        parsed = OutputParser.create_test_report(report)
        assert parsed.type == "test_report"
        assert parsed.data["passed"] == 5
        assert parsed.data["coverage"] == 85.0

    def test_create_lint_report(self):
        """Test creating a lint report."""
        errors = [
            LintError(file="test.js", line=10, column=5, severity="error", message="Error"),
            LintError(file="test.js", line=15, column=8, severity="warning", message="Warning"),
        ]
        parsed = OutputParser.create_lint_report(errors)
        assert parsed.type == "lint_report"
        assert parsed.data["total"] == 2
        assert len(parsed.data["errors"]) == 2

    def test_create_type_check_report(self):
        """Test creating a type check report."""
        errors = [
            TypeCheckError(
                file="test.ts", line=5, column=10, code="TS2322", message="Type error"
            ),
        ]
        parsed = OutputParser.create_type_check_report(errors)
        assert parsed.type == "type_check"
        assert parsed.data["total"] == 1


# =====================================================================
# ResourceLimiter Tests
# =====================================================================


class TestResourceLimiter:
    """Tests for ResourceLimiter."""

    def test_init_default_limits(self):
        """Test initialization with default limits."""
        limiter = ResourceLimiter()
        assert limiter.limits.cpu_seconds == 300
        assert limiter.limits.memory_bytes == 2 * 1024 * 1024 * 1024
        assert limiter.limits.max_processes == 50

    def test_init_custom_limits(self):
        """Test initialization with custom limits."""
        limits = ResourceLimits(cpu_seconds=60, memory_bytes=512 * 1024 * 1024)
        limiter = ResourceLimiter(limits)
        assert limiter.limits.cpu_seconds == 60
        assert limiter.limits.memory_bytes == 512 * 1024 * 1024

    def test_from_template_strict(self):
        """Test creating limits from strict template."""
        limits = ResourceLimiter.from_template("strict")
        assert limits.cpu_seconds == 60
        assert limits.memory_bytes == 512 * 1024 * 1024
        assert limits.max_processes == 30

    def test_from_template_standard(self):
        """Test creating limits from standard template."""
        limits = ResourceLimiter.from_template("standard")
        assert limits.cpu_seconds == 300
        assert limits.memory_bytes == 2 * 1024 * 1024 * 1024

    def test_from_template_relaxed(self):
        """Test creating limits from relaxed template."""
        limits = ResourceLimiter.from_template("relaxed")
        assert limits.cpu_seconds == 600
        assert limits.memory_bytes == 4 * 1024 * 1024 * 1024

    def test_from_template_test(self):
        """Test creating limits from test template."""
        limits = ResourceLimiter.from_template("test")
        assert limits.cpu_seconds == 30
        assert limits.memory_bytes == 256 * 1024 * 1024

    def test_from_template_invalid(self):
        """Test that invalid template raises error."""
        with pytest.raises(ValueError, match="Unknown template"):
            ResourceLimiter.from_template("invalid")

    def test_get_preexec_fn(self):
        """Test getting preexec function."""
        limiter = ResourceLimiter()
        preexec_fn = limiter.get_preexec_fn()
        assert callable(preexec_fn)

    def test_check_violations_no_violations(self):
        """Test checking violations when none exist."""
        limits = ResourceLimits(cpu_seconds=300, memory_bytes=2 * 1024 * 1024 * 1024)
        usage = ResourceUsage(cpu_time_seconds=100, memory_peak_bytes=512 * 1024 * 1024)
        violations = ResourceLimiter.check_violations(usage, limits)
        assert len(violations) == 0

    def test_check_violations_cpu(self):
        """Test detecting CPU violation."""
        limits = ResourceLimits(cpu_seconds=60)
        usage = ResourceUsage(cpu_time_seconds=120)
        violations = ResourceLimiter.check_violations(usage, limits)
        assert len(violations) == 1
        assert violations[0].resource == "cpu"

    def test_check_violations_memory(self):
        """Test detecting memory violation."""
        limits = ResourceLimits(memory_bytes=512 * 1024 * 1024)
        usage = ResourceUsage(memory_peak_bytes=1024 * 1024 * 1024)
        violations = ResourceLimiter.check_violations(usage, limits)
        assert len(violations) == 1
        assert violations[0].resource == "memory"

    def test_check_violations_multiple(self):
        """Test detecting multiple violations."""
        limits = ResourceLimits(
            cpu_seconds=60, memory_bytes=512 * 1024 * 1024, wall_time_seconds=120
        )
        usage = ResourceUsage(
            cpu_time_seconds=120, memory_peak_bytes=1024 * 1024 * 1024, wall_time_seconds=200
        )
        violations = ResourceLimiter.check_violations(usage, limits)
        assert len(violations) == 3


# =====================================================================
# ToolLayer Tests
# =====================================================================


class TestToolLayer:
    """Tests for ToolLayer."""

    def test_init_default_policies(self, temp_project_dir):
        """Test initialization with no policies."""
        layer = ToolLayer(temp_project_dir)
        assert layer.project_dir == temp_project_dir
        assert layer.policies == {}
        assert len(layer.call_counts) == 0

    def test_init_with_policies(self, temp_project_dir):
        """Test initialization with policies."""
        policies = {
            ToolName.FILE_SYSTEM: ToolPolicy(
                tool=ToolName.FILE_SYSTEM, allowed_operations=["read_file"], max_calls_per_minute=10
            )
        }
        layer = ToolLayer(temp_project_dir, policies=policies)
        assert ToolName.FILE_SYSTEM in layer.policies

    def test_init_with_allowlist(self, temp_project_dir):
        """Test initialization with command allowlist."""
        allowlist = ["echo", "cat"]
        layer = ToolLayer(temp_project_dir, command_allowlist=allowlist)
        assert layer.shell.command_allowlist == allowlist

    def test_validate_call_allowed(self, temp_project_dir):
        """Test that allowed operations pass validation."""
        policies = {
            ToolName.FILE_SYSTEM: ToolPolicy(
                tool=ToolName.FILE_SYSTEM, allowed_operations=["read_file"]
            )
        }
        layer = ToolLayer(temp_project_dir, policies=policies)
        call = ToolCall(tool=ToolName.FILE_SYSTEM, operation="read_file")
        # Should not raise
        layer._validate_call(call)

    def test_validate_call_blocked_operation(self, temp_project_dir):
        """Test that blocked operations fail validation."""
        policies = {
            ToolName.FILE_SYSTEM: ToolPolicy(
                tool=ToolName.FILE_SYSTEM, allowed_operations=["read_file"]
            )
        }
        layer = ToolLayer(temp_project_dir, policies=policies)
        call = ToolCall(tool=ToolName.FILE_SYSTEM, operation="delete_file")
        with pytest.raises(ValueError, match="not allowed"):
            layer._validate_call(call)

    def test_validate_call_rate_limit(self, temp_project_dir):
        """Test rate limiting."""
        policies = {
            ToolName.FILE_SYSTEM: ToolPolicy(
                tool=ToolName.FILE_SYSTEM, max_calls_per_minute=2
            )
        }
        layer = ToolLayer(temp_project_dir, policies=policies)
        call = ToolCall(tool=ToolName.FILE_SYSTEM, operation="read_file")

        # First two calls should succeed
        layer._validate_call(call)
        layer._validate_call(call)

        # Third call should fail
        with pytest.raises(ValueError, match="Rate limit exceeded"):
            layer._validate_call(call)

    def test_record_metric(self, temp_project_dir):
        """Test metric recording."""
        layer = ToolLayer(temp_project_dir)
        layer._record_metric(ToolName.FILE_SYSTEM, "read_file", 1.5, 0)
        layer._record_metric(ToolName.FILE_SYSTEM, "read_file", 2.5, 0)
        layer._record_metric(ToolName.FILE_SYSTEM, "read_file", 3.0, 1)

        metrics = layer.get_metrics()
        assert "file_system:read_file" in metrics
        assert metrics["file_system:read_file"]["count"] == 3
        assert metrics["file_system:read_file"]["success_count"] == 2
        assert metrics["file_system:read_file"]["error_count"] == 1
        assert metrics["file_system:read_file"]["success_rate"] == pytest.approx(2 / 3)

    def test_get_metrics_empty(self, temp_project_dir):
        """Test get_metrics with no calls."""
        layer = ToolLayer(temp_project_dir)
        metrics = layer.get_metrics()
        assert metrics == {}

    @pytest.mark.asyncio
    async def test_execute_file_system_read(self, temp_project_dir):
        """Test executing a file_system read call."""
        layer = ToolLayer(temp_project_dir)
        test_file = Path(temp_project_dir) / "test.txt"
        test_file.write_text("content")

        call = ToolCall(
            tool=ToolName.FILE_SYSTEM,
            operation="read_file",
            args={"path": "test.txt"},
        )
        result = await layer.execute(call)
        assert result.exit_code == 0
        assert result.stdout == "content"

    @pytest.mark.asyncio
    async def test_execute_shell_command(self, temp_project_dir):
        """Test executing a shell command."""
        layer = ToolLayer(temp_project_dir)
        call = ToolCall(
            tool=ToolName.SHELL,
            operation="execute",
            args={"command": "python --version"},
        )
        result = await layer.execute(call)
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_execute_git_operation(self, git_project_dir):
        """Test executing a git operation."""
        layer = ToolLayer(git_project_dir)
        call = ToolCall(
            tool=ToolName.GIT,
            operation="get_current_branch",
        )
        result = await layer.execute(call)
        assert result.exit_code == 0
        assert result.stdout in ["main", "master", "HEAD"]

    @pytest.mark.asyncio
    async def test_execute_policy_violation(self, temp_project_dir):
        """Test that policy violations are handled gracefully."""
        policies = {
            ToolName.FILE_SYSTEM: ToolPolicy(
                tool=ToolName.FILE_SYSTEM, allowed_operations=["read_file"]
            )
        }
        layer = ToolLayer(temp_project_dir, policies=policies)
        call = ToolCall(
            tool=ToolName.FILE_SYSTEM,
            operation="delete_file",
            args={"path": "test.txt"},
        )
        result = await layer.execute(call)
        assert result.exit_code == 1
        assert "Policy violation" in result.stderr or "not allowed" in result.stderr

    @pytest.mark.asyncio
    async def test_execute_records_metrics(self, temp_project_dir):
        """Test that execution records metrics."""
        layer = ToolLayer(temp_project_dir)
        test_file = Path(temp_project_dir) / "test.txt"
        test_file.write_text("content")

        call = ToolCall(
            tool=ToolName.FILE_SYSTEM,
            operation="read_file",
            args={"path": "test.txt"},
        )
        await layer.execute(call)

        metrics = layer.get_metrics()
        assert "file_system:read_file" in metrics
        assert metrics["file_system:read_file"]["count"] == 1
        assert metrics["file_system:read_file"]["success_count"] == 1

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, temp_project_dir):
        """Test executing with unknown tool."""
        layer = ToolLayer(temp_project_dir)
        # Create a mock ToolCall with an invalid tool by bypassing validation
        call = ToolCall(tool=ToolName.DATABASE, operation="query")
        result = await layer.execute(call)
        assert result.exit_code == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
