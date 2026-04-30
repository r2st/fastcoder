"""SCIM 2.0 provider for user and group management."""

from __future__ import annotations

import json
import os
import re
import stat
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import structlog

from fastcoder.auth.types import SCIMGroup, SCIMUser, UserRole

logger = structlog.get_logger(__name__)


class SCIMProvider:
    """SCIM 2.0 RFC 7643/7644 compliant user and group management."""

    def __init__(self, persistence_path: Optional[str] = None):
        """Initialize the SCIM provider.

        Args:
            persistence_path: Optional file path for saving/loading data.
        """
        self.persistence_path = persistence_path
        self._users: dict[str, SCIMUser] = {}
        self._groups: dict[str, SCIMGroup] = {}

        if persistence_path:
            self.load(persistence_path)

        logger.info(
            "scim_provider_initialized",
            persistence_enabled=persistence_path is not None,
            user_count=len(self._users),
            group_count=len(self._groups),
        )

    # ========================================================================
    # User Operations
    # ========================================================================

    async def list_users(
        self,
        filter_expr: Optional[str] = None,
        start_index: int = 1,
        count: int = 100,
    ) -> dict[str, Any]:
        """List users with optional filtering (SCIM RFC 7643).

        Args:
            filter_expr: SCIM filter expression (e.g., 'userName eq "bjensen"').
            start_index: 1-based index for pagination.
            count: Maximum results to return.

        Returns:
            SCIM ListResponse with users array.
        """
        logger.info(
            "listing_users",
            filter_expr=filter_expr,
            start_index=start_index,
            count=count,
        )

        # Apply filter if provided
        users = list(self._users.values())
        if filter_expr:
            filter_func = self._parse_scim_filter(filter_expr)
            users = [u for u in users if filter_func(u)]

        # Apply pagination
        total_results = len(users)
        start_idx = max(0, start_index - 1)  # Convert to 0-based
        end_idx = start_idx + count
        paginated = users[start_idx:end_idx]

        # Build SCIM response
        response = {
            "schemas": ["urn:ietf:params:scim:api:messages:2.0:ListResponse"],
            "totalResults": total_results,
            "startIndex": start_index,
            "itemsPerPage": len(paginated),
            "Resources": [self._user_to_scim(u) for u in paginated],
        }

        logger.info("users_listed", count=len(paginated), total=total_results)
        return response

    async def get_user(self, user_id: str) -> Optional[SCIMUser]:
        """Get a user by ID.

        Args:
            user_id: User ID.

        Returns:
            SCIMUser or None if not found.
        """
        user = self._users.get(user_id)
        if user:
            logger.debug("user_retrieved", user_id=user_id)
        else:
            logger.debug("user_not_found", user_id=user_id)
        return user

    async def create_user(self, user_data: dict[str, Any]) -> SCIMUser:
        """Create a new user.

        Args:
            user_data: User data dictionary.

        Returns:
            Created SCIMUser.

        Raises:
            ValueError: If required fields are missing.
        """
        user_name = user_data.get("userName")
        if not user_name:
            raise ValueError("userName is required")

        user_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        user = SCIMUser(
            id=user_id,
            external_id=user_data.get("externalId"),
            user_name=user_name,
            display_name=user_data.get("displayName", user_name),
            email=user_data.get("emails", [{}])[0].get("value", "")
            if user_data.get("emails")
            else user_data.get("email", ""),
            active=user_data.get("active", True),
            roles=[UserRole(r) for r in user_data.get("roles", []) if r in UserRole._value2member_map_],
            groups=user_data.get("groups", []),
            created_at=now,
            updated_at=now,
        )

        self._users[user_id] = user
        logger.info("user_created", user_id=user_id, user_name=user_name)

        if self.persistence_path:
            self.save(self.persistence_path)

        return user

    async def update_user(self, user_id: str, user_data: dict[str, Any]) -> SCIMUser:
        """Replace user (PUT operation).

        Args:
            user_id: User ID.
            user_data: New user data.

        Returns:
            Updated SCIMUser.

        Raises:
            ValueError: If user not found.
        """
        user = self._users.get(user_id)
        if not user:
            logger.error("user_not_found_for_update", user_id=user_id)
            raise ValueError(f"User {user_id} not found")

        # Update fields
        user.user_name = user_data.get("userName", user.user_name)
        user.display_name = user_data.get("displayName", user.display_name)
        user.active = user_data.get("active", user.active)
        user.updated_at = datetime.now(timezone.utc)

        if "email" in user_data:
            user.email = user_data["email"]
        elif "emails" in user_data and user_data["emails"]:
            user.email = user_data["emails"][0].get("value", user.email)

        if "roles" in user_data:
            user.roles = [UserRole(r) for r in user_data["roles"] if r in UserRole._value2member_map_]

        if "groups" in user_data:
            user.groups = user_data["groups"]

        logger.info("user_updated", user_id=user_id)

        if self.persistence_path:
            self.save(self.persistence_path)

        return user

    async def patch_user(self, user_id: str, operations: list[dict[str, Any]]) -> SCIMUser:
        """Apply SCIM PATCH operations to a user.

        Args:
            user_id: User ID.
            operations: List of SCIM PATCH operations.

        Returns:
            Updated SCIMUser.

        Raises:
            ValueError: If user not found or operation is invalid.
        """
        user = self._users.get(user_id)
        if not user:
            logger.error("user_not_found_for_patch", user_id=user_id)
            raise ValueError(f"User {user_id} not found")

        for op in operations:
            op_type = op.get("op", "").lower()
            path = op.get("path", "")
            value = op.get("value")

            logger.debug("applying_patch_operation", user_id=user_id, op_type=op_type, path=path)

            if op_type == "add":
                self._apply_patch_add(user, path, value)
            elif op_type == "remove":
                self._apply_patch_remove(user, path)
            elif op_type == "replace":
                self._apply_patch_replace(user, path, value)

        user.updated_at = datetime.now(timezone.utc)
        logger.info("user_patched", user_id=user_id, op_count=len(operations))

        if self.persistence_path:
            self.save(self.persistence_path)

        return user

    async def delete_user(self, user_id: str) -> bool:
        """Delete/deactivate a user.

        Args:
            user_id: User ID.

        Returns:
            True if user was deleted, False if not found.
        """
        if user_id in self._users:
            # Deactivate instead of hard delete
            self._users[user_id].active = False
            self._users[user_id].updated_at = datetime.now(timezone.utc)
            logger.info("user_deleted", user_id=user_id)

            if self.persistence_path:
                self.save(self.persistence_path)

            return True

        logger.debug("user_not_found_for_deletion", user_id=user_id)
        return False

    # ========================================================================
    # Group Operations
    # ========================================================================

    async def list_groups(
        self,
        filter_expr: Optional[str] = None,
        start_index: int = 1,
        count: int = 100,
    ) -> dict[str, Any]:
        """List groups with optional filtering.

        Args:
            filter_expr: SCIM filter expression.
            start_index: 1-based index for pagination.
            count: Maximum results to return.

        Returns:
            SCIM ListResponse with groups array.
        """
        logger.info(
            "listing_groups",
            filter_expr=filter_expr,
            start_index=start_index,
            count=count,
        )

        groups = list(self._groups.values())
        if filter_expr:
            filter_func = self._parse_scim_filter(filter_expr)
            groups = [g for g in groups if filter_func(g)]

        total_results = len(groups)
        start_idx = max(0, start_index - 1)
        end_idx = start_idx + count
        paginated = groups[start_idx:end_idx]

        response = {
            "schemas": ["urn:ietf:params:scim:api:messages:2.0:ListResponse"],
            "totalResults": total_results,
            "startIndex": start_index,
            "itemsPerPage": len(paginated),
            "Resources": [self._group_to_scim(g) for g in paginated],
        }

        logger.info("groups_listed", count=len(paginated), total=total_results)
        return response

    async def get_group(self, group_id: str) -> Optional[SCIMGroup]:
        """Get a group by ID.

        Args:
            group_id: Group ID.

        Returns:
            SCIMGroup or None if not found.
        """
        group = self._groups.get(group_id)
        if group:
            logger.debug("group_retrieved", group_id=group_id)
        else:
            logger.debug("group_not_found", group_id=group_id)
        return group

    async def create_group(self, group_data: dict[str, Any]) -> SCIMGroup:
        """Create a new group.

        Args:
            group_data: Group data dictionary.

        Returns:
            Created SCIMGroup.

        Raises:
            ValueError: If displayName is missing.
        """
        display_name = group_data.get("displayName")
        if not display_name:
            raise ValueError("displayName is required")

        group_id = str(uuid.uuid4())

        members = []
        if "members" in group_data and isinstance(group_data["members"], list):
            members = group_data["members"]

        group = SCIMGroup(
            id=group_id,
            display_name=display_name,
            members=members,
        )

        self._groups[group_id] = group
        logger.info("group_created", group_id=group_id, display_name=display_name)

        if self.persistence_path:
            self.save(self.persistence_path)

        return group

    async def update_group(self, group_id: str, group_data: dict[str, Any]) -> SCIMGroup:
        """Replace group (PUT operation).

        Args:
            group_id: Group ID.
            group_data: New group data.

        Returns:
            Updated SCIMGroup.

        Raises:
            ValueError: If group not found.
        """
        group = self._groups.get(group_id)
        if not group:
            logger.error("group_not_found_for_update", group_id=group_id)
            raise ValueError(f"Group {group_id} not found")

        group.display_name = group_data.get("displayName", group.display_name)
        if "members" in group_data:
            group.members = group_data["members"]

        logger.info("group_updated", group_id=group_id)

        if self.persistence_path:
            self.save(self.persistence_path)

        return group

    async def delete_group(self, group_id: str) -> bool:
        """Delete a group.

        Args:
            group_id: Group ID.

        Returns:
            True if group was deleted, False if not found.
        """
        if group_id in self._groups:
            del self._groups[group_id]
            logger.info("group_deleted", group_id=group_id)

            if self.persistence_path:
                self.save(self.persistence_path)

            return True

        logger.debug("group_not_found_for_deletion", group_id=group_id)
        return False

    # ========================================================================
    # Persistence
    # ========================================================================

    def save(self, path: str) -> None:
        """Save users and groups to file.

        Args:
            path: File path for persistence.
        """
        try:
            data = {
                "users": {uid: u.model_dump() for uid, u in self._users.items()},
                "groups": {gid: g.model_dump() for gid, g in self._groups.items()},
            }

            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

            # Restrict file permissions to owner-only (contains user data)
            try:
                os.chmod(str(save_path), stat.S_IRUSR | stat.S_IWUSR)
            except OSError:
                pass  # Best-effort on platforms that don't support chmod

            logger.info("scim_data_saved", path=path)
        except Exception as e:
            logger.error("scim_save_error", path=path, error=str(e))
            raise

    def load(self, path: str) -> None:
        """Load users and groups from file.

        Args:
            path: File path to load from.
        """
        try:
            if not Path(path).exists():
                logger.info("scim_file_not_found", path=path)
                return

            with open(path, "r") as f:
                data = json.load(f)

            # Load users
            for uid, user_data in data.get("users", {}).items():
                try:
                    user = SCIMUser(**user_data)
                    self._users[uid] = user
                except Exception as e:
                    logger.warning("failed_to_load_user", user_id=uid, error=str(e))

            # Load groups
            for gid, group_data in data.get("groups", {}).items():
                try:
                    group = SCIMGroup(**group_data)
                    self._groups[gid] = group
                except Exception as e:
                    logger.warning("failed_to_load_group", group_id=gid, error=str(e))

            logger.info(
                "scim_data_loaded",
                path=path,
                user_count=len(self._users),
                group_count=len(self._groups),
            )
        except Exception as e:
            logger.error("scim_load_error", path=path, error=str(e))
            raise

    # ========================================================================
    # Filtering and Helpers
    # ========================================================================

    def _parse_scim_filter(self, filter_expr: str) -> Callable[[Any], bool]:
        """Parse SCIM filter expression and return matching function.

        Supports: eq, ne, co, sw, ew operators

        Args:
            filter_expr: SCIM filter string (e.g., 'userName eq "test"').

        Returns:
            Callable that returns True if object matches filter.
        """
        # Simple regex-based filter parsing
        # Pattern: attribute operator "value"
        pattern = r'(\w+)\s+(eq|ne|co|sw|ew)\s+"([^"]*)"'
        match = re.match(pattern, filter_expr.strip())

        if not match:
            logger.warning("invalid_filter_expression", filter_expr=filter_expr)
            raise ValueError(f"Invalid or unsupported SCIM filter expression: {filter_expr!r}")

        attr_name, operator, value = match.groups()

        def filter_func(obj: Any) -> bool:
            if isinstance(obj, (SCIMUser, SCIMGroup)):
                obj_value = getattr(obj, attr_name, None)
            else:
                obj_value = obj.get(attr_name)

            if obj_value is None:
                return False

            obj_str = str(obj_value).lower()
            val_str = value.lower()

            if operator == "eq":
                return obj_str == val_str
            elif operator == "ne":
                return obj_str != val_str
            elif operator == "co":
                return val_str in obj_str
            elif operator == "sw":
                return obj_str.startswith(val_str)
            elif operator == "ew":
                return obj_str.endswith(val_str)

            return True

        return filter_func

    # Whitelist of fields allowed via PATCH to prevent unauthorized attribute manipulation
    _PATCHABLE_FIELDS = frozenset({"active", "displayname", "email", "roles", "groups"})

    def _validate_patch_path(self, path: str) -> None:
        """Validate PATCH path against whitelist of allowed fields."""
        path_lower = path.lower()
        if not any(field in path_lower for field in self._PATCHABLE_FIELDS):
            raise ValueError(f"Cannot patch field: {path!r}. Allowed: {', '.join(sorted(self._PATCHABLE_FIELDS))}")

    def _apply_patch_add(self, user: SCIMUser, path: str, value: Any) -> None:
        """Apply PATCH add operation."""
        self._validate_patch_path(path)
        path_lower = path.lower()

        if "roles" in path_lower:
            if isinstance(value, list):
                for role in value:
                    if role not in user.roles and role in UserRole._value2member_map_:
                        user.roles.append(UserRole(role))
        elif "groups" in path_lower:
            if isinstance(value, list):
                user.groups.extend([g for g in value if g not in user.groups])

    def _apply_patch_remove(self, user: SCIMUser, path: str) -> None:
        """Apply PATCH remove operation."""
        self._validate_patch_path(path)
        path_lower = path.lower()

        if "roles" in path_lower:
            user.roles = []
        elif "groups" in path_lower:
            user.groups = []

    def _apply_patch_replace(self, user: SCIMUser, path: str, value: Any) -> None:
        """Apply PATCH replace operation."""
        self._validate_patch_path(path)
        path_lower = path.lower()

        if "active" in path_lower:
            user.active = bool(value)
        elif "displayname" in path_lower:
            user.display_name = str(value)
        elif "email" in path_lower:
            user.email = str(value)
        elif "roles" in path_lower:
            if isinstance(value, list):
                user.roles = [UserRole(r) for r in value if r in UserRole._value2member_map_]
        elif "groups" in path_lower:
            if isinstance(value, list):
                user.groups = value

    def _user_to_scim(self, user: SCIMUser) -> dict[str, Any]:
        """Convert SCIMUser to SCIM resource format."""
        return {
            "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
            "id": user.id,
            "externalId": user.external_id,
            "userName": user.user_name,
            "name": {"formatted": user.display_name},
            "displayName": user.display_name,
            "emails": [{"value": user.email, "primary": True}],
            "active": user.active,
            "roles": user.roles,
            "groups": user.groups,
            "meta": {
                "resourceType": "User",
                "created": user.created_at.isoformat(),
                "lastModified": user.updated_at.isoformat(),
            },
        }

    def _group_to_scim(self, group: SCIMGroup) -> dict[str, Any]:
        """Convert SCIMGroup to SCIM resource format."""
        return {
            "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Group"],
            "id": group.id,
            "displayName": group.display_name,
            "members": [{"value": member_id} for member_id in group.members],
            "meta": {
                "resourceType": "Group",
            },
        }
