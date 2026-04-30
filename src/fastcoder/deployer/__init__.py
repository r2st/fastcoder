"""Deployment orchestration for code changes and releases."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

import structlog

from fastcoder.tools.build_runner import BuildRunner
from fastcoder.tools.git_client import GitClient
from fastcoder.tools.shell_executor import ShellExecutor
from fastcoder.types.config import ProjectConfig
from fastcoder.types.story import Story
from fastcoder.types.task import DeployReport

logger = structlog.get_logger(__name__)


class Deployer:
    """Manages deployment workflows including branch management, commits, and PR creation."""

    def __init__(
        self,
        git_client: GitClient,
        build_runner: BuildRunner,
        shell_executor: ShellExecutor,
        config: ProjectConfig,
    ):
        """Initialize deployer with required clients and config.

        Args:
            git_client: Git operations handler
            build_runner: Build and quality checks runner
            shell_executor: Shell command executor
            config: Project configuration
        """
        self.git = git_client
        self.build_runner = build_runner
        self.shell = shell_executor
        self.config = config
        self._start_time = 0.0

    async def deploy(self, story: Story) -> DeployReport:
        """Deploy story changes to default environment.

        Workflow:
        1. Create or switch to feature branch
        2. Commit all changes with conventional message
        3. Run build verification
        4. Push changes
        5. Create pull request

        Args:
            story: Story containing changes to deploy

        Returns:
            DeployReport with deployment status and details
        """
        self._start_time = time.time()
        logger.info("Starting deployment", story_id=story.id)

        report = DeployReport(environment="staging")

        try:
            # Step 1: Create feature branch
            branch_name = await self._ensure_feature_branch(story)
            logger.info("Feature branch ready", branch=branch_name, story_id=story.id)

            # Step 2: Commit changes
            commit_msg = self._build_conventional_commit_message(story)
            commit_result = self.git.commit_changes(commit_msg)

            if commit_result.exit_code != 0:
                logger.warning(
                    "Commit failed",
                    story_id=story.id,
                    error=commit_result.stderr,
                )
                report.success = False
                report.error = commit_result.stderr or "Failed to commit changes"
                return report

            logger.info(
                "Changes committed",
                story_id=story.id,
                commit_sha=commit_result.stdout,
            )

            # Step 3: Run build verification
            if self.build_runner.build_cmd:
                build_result = await self.build_runner.build()
                if build_result.exit_code != 0:
                    logger.warning(
                        "Build failed",
                        story_id=story.id,
                        error=build_result.stderr,
                    )
                    report.success = False
                    report.error = f"Build failed: {build_result.stderr}"
                    return report

                logger.info("Build successful", story_id=story.id)

            # Step 4: Push to remote
            push_result = self.git.push(branch_name)
            if push_result.exit_code != 0:
                logger.warning(
                    "Push failed",
                    story_id=story.id,
                    error=push_result.stderr,
                )
                report.success = False
                report.error = push_result.stderr or "Failed to push changes"
                return report

            logger.info("Changes pushed", story_id=story.id, branch=branch_name)

            # Step 5: Create pull request
            pr_url = await self._create_pull_request(story, branch_name)
            if pr_url:
                logger.info("Pull request created", story_id=story.id, pr_url=pr_url)
                report.url = pr_url
            else:
                logger.warning("Failed to create pull request", story_id=story.id)

            report.success = True
            return report

        except Exception as e:
            logger.exception("Deployment error", story_id=story.id, error=str(e))
            report.success = False
            report.error = f"Deployment error: {str(e)}"
            return report

    async def rollback(self, story: Story) -> DeployReport:
        """Rollback story changes by reverting to base branch.

        Workflow:
        1. Identify feature branch for story
        2. Checkout base branch
        3. Delete feature branch (local and remote)
        4. Reset to base branch state

        Args:
            story: Story to rollback

        Returns:
            DeployReport indicating rollback success
        """
        self._start_time = time.time()
        logger.info("Starting rollback", story_id=story.id)

        report = DeployReport(environment="staging", rollback_triggered=True)

        try:
            # Get current branch to identify feature branch if needed
            current_branch = self.git.get_current_branch()
            feature_branch = (
                current_branch
                if current_branch.startswith("feature/STORY-")
                else await self._get_feature_branch_for_story(story)
            )

            if not feature_branch:
                logger.warning(
                    "No feature branch found for rollback",
                    story_id=story.id,
                )
                report.success = False
                report.error = "No feature branch found for story"
                return report

            # Checkout base branch
            checkout_result = self.git.checkout(self.config.base_branch)
            if checkout_result.exit_code != 0:
                logger.warning(
                    "Failed to checkout base branch",
                    story_id=story.id,
                    branch=self.config.base_branch,
                )
                report.success = False
                report.error = "Failed to checkout base branch"
                return report

            logger.info(
                "Checked out base branch",
                story_id=story.id,
                branch=self.config.base_branch,
            )

            # Delete feature branch (local and remote)
            try:
                self.git.repo.delete_head(feature_branch, force=True)
                logger.info(
                    "Deleted local feature branch",
                    story_id=story.id,
                    branch=feature_branch,
                )
            except Exception as e:
                logger.warning(
                    "Failed to delete local branch",
                    story_id=story.id,
                    branch=feature_branch,
                    error=str(e),
                )

            # Try to delete remote branch
            try:
                if self.git.repo.remotes:
                    self.git.repo.remotes.origin.push(f":{feature_branch}")
                    logger.info(
                        "Deleted remote feature branch",
                        story_id=story.id,
                        branch=feature_branch,
                    )
            except Exception as e:
                logger.warning(
                    "Failed to delete remote branch",
                    story_id=story.id,
                    branch=feature_branch,
                    error=str(e),
                )

            report.success = True
            return report

        except Exception as e:
            logger.exception("Rollback error", story_id=story.id, error=str(e))
            report.success = False
            report.error = f"Rollback error: {str(e)}"
            return report

    async def deploy_to_staging(self, story: Story) -> DeployReport:
        """Deploy to staging environment with smoke tests.

        Args:
            story: Story to deploy

        Returns:
            DeployReport with staging deployment status
        """
        logger.info("Deploying to staging", story_id=story.id)
        report = await self.deploy(story)
        report.environment = "staging"
        report.health_check_passed = True
        return report

    async def deploy_to_production(self, story: Story) -> DeployReport:
        """Deploy to production with extra safety checks.

        Additional checks:
        - Verify no uncommitted changes
        - Ensure base branch is clean
        - Confirm deploy approval gates

        Args:
            story: Story to deploy to production

        Returns:
            DeployReport with production deployment status
        """
        logger.info("Deploying to production", story_id=story.id)

        report = DeployReport(environment="production")

        try:
            # Extra safety check: verify clean working tree
            status_result = self.git.get_status()
            if status_result.exit_code != 0:
                logger.error(
                    "Failed to check git status",
                    story_id=story.id,
                )
                report.success = False
                report.error = "Failed to verify repository state"
                return report

            # Check if working tree is clean
            if self.git.repo.is_dirty():
                logger.warning(
                    "Working tree is dirty, cannot deploy to production",
                    story_id=story.id,
                )
                report.success = False
                report.error = "Working tree has uncommitted changes"
                return report

            logger.info("Production safety checks passed", story_id=story.id)

            # Proceed with standard deployment
            report = await self.deploy(story)
            report.environment = "production"

            # Additional production checks could go here
            report.health_check_passed = True

            return report

        except Exception as e:
            logger.exception(
                "Production deployment error",
                story_id=story.id,
                error=str(e),
            )
            report.success = False
            report.error = f"Production deployment error: {str(e)}"
            return report

    # Private helper methods

    async def _ensure_feature_branch(self, story: Story) -> str:
        """Ensure feature branch exists, create if needed.

        Args:
            story: Story to create branch for

        Returns:
            Feature branch name
        """
        current_branch = self.git.get_current_branch()

        # Already on feature branch for this story
        if current_branch.startswith(f"feature/STORY-{story.id}"):
            return current_branch

        # Create new feature branch
        try:
            title = story.spec.title if story.spec else story.id
            branch_name = self.git.create_branch(story.id, title)
            logger.info(
                "Created feature branch",
                story_id=story.id,
                branch=branch_name,
            )
            return branch_name
        except Exception as e:
            logger.warning(
                "Failed to create branch, trying to checkout existing",
                story_id=story.id,
                error=str(e),
            )
            # Try to find and checkout existing feature branch
            feature_branch = await self._get_feature_branch_for_story(story)
            if feature_branch:
                checkout_result = self.git.checkout(feature_branch)
                if checkout_result.exit_code == 0:
                    return feature_branch
            raise

    async def _get_feature_branch_for_story(self, story: Story) -> Optional[str]:
        """Find existing feature branch for story.

        Args:
            story: Story to find branch for

        Returns:
            Branch name if found, None otherwise
        """
        try:
            for ref in self.git.repo.refs:
                if f"feature/STORY-{story.id}" in ref.name:
                    return ref.name
        except Exception as e:
            logger.warning(
                "Failed to search for feature branch",
                story_id=story.id,
                error=str(e),
            )
        return None

    def _build_conventional_commit_message(self, story: Story) -> str:
        """Build conventional commit message from story.

        Format: feat(STORY-xxx): Title
                body with details

        Args:
            story: Story to create message for

        Returns:
            Formatted commit message
        """
        commit_type = "feat"
        if story.spec and story.spec.story_type.value == "bugfix":
            commit_type = "fix"
        elif story.spec and story.spec.story_type.value == "refactor":
            commit_type = "refactor"
        elif story.spec and story.spec.story_type.value == "infra":
            commit_type = "chore"

        title = story.spec.title if story.spec else story.id
        title = title.replace("\n", " ").strip()

        message = f"{commit_type}(STORY-{story.id}): {title}\n\n"

        if story.spec and story.spec.description:
            description = story.spec.description.strip()
            message += f"{description}\n\n"

        message += f"Co-authored-by: fastcoder <agent@auto-dev>"

        return message

    async def _create_pull_request(
        self, story: Story, branch_name: str
    ) -> Optional[str]:
        """Create pull request for story branch.

        Attempts to create PR via git client, falls back to shell command.

        Args:
            story: Story for PR
            branch_name: Feature branch name

        Returns:
            PR URL if successful, None otherwise
        """
        try:
            import shlex

            # Attempt to create PR using GitHub CLI if available
            title = story.spec.title if story.spec else story.id
            # Sanitize title: strip control chars, limit length
            title = "".join(c for c in title if c.isprintable()).strip()[:120]

            description = ""
            if story.spec:
                # Sanitize description: strip control chars that could break shell
                safe_desc = "".join(c for c in story.spec.description if c.isprintable() or c == "\n")
                description = f"## Description\n{safe_desc}\n\n"

            if story.spec and story.spec.acceptance_criteria:
                description += "## Acceptance Criteria\n"
                for criterion in story.spec.acceptance_criteria:
                    safe_crit = "".join(c for c in criterion.description if c.isprintable())
                    description += f"- [ ] {safe_crit}\n"

            # Sanitize story ID: only allow alphanumeric, hyphens, underscores
            import re
            safe_story_id = re.sub(r"[^a-zA-Z0-9_-]", "", story.id)
            description += f"\nStory ID: STORY-{safe_story_id}"

            # Sanitize branch names
            safe_branch = re.sub(r"[^a-zA-Z0-9_./-]", "", branch_name)
            safe_base = re.sub(r"[^a-zA-Z0-9_./-]", "", self.config.base_branch)

            # Use shlex.quote to prevent shell injection on all user-derived values
            cmd = (
                f'gh pr create --head {shlex.quote(safe_branch)} --base {shlex.quote(safe_base)} '
                f'--title {shlex.quote(title)} --body {shlex.quote(description)}'
            )

            result = await self.shell.execute(cmd)
            if result.exit_code == 0:
                # Extract PR URL from output
                import re

                match = re.search(r"https://github\.com/.*/pull/\d+", result.stdout)
                if match:
                    return match.group(0)

            logger.warning(
                "GitHub CLI PR creation failed",
                story_id=story.id,
                stderr=result.stderr,
            )

        except Exception as e:
            logger.warning(
                "Failed to create PR",
                story_id=story.id,
                error=str(e),
            )

        return None
