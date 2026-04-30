"""SCIM 2.0 API endpoints (RFC 7643/7644)."""

from __future__ import annotations

from typing import Any, Optional

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status

from fastcoder.auth.scim_provider import SCIMProvider
from fastcoder.auth.types import UserRole

logger = structlog.get_logger(__name__)

# SCIM endpoints require a bearer token for authentication.
# The token is validated against the agent's API token.
_SCIM_BEARER_TOKEN: Optional[str] = None


def set_scim_bearer_token(token: str) -> None:
    """Set the bearer token required for SCIM API access."""
    global _SCIM_BEARER_TOKEN
    _SCIM_BEARER_TOKEN = token


async def _require_scim_auth(authorization: Optional[str] = Header(None)) -> None:
    """Dependency that enforces bearer token authentication on SCIM endpoints."""
    if _SCIM_BEARER_TOKEN is None:
        # If no token configured, SCIM endpoints are disabled
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SCIM endpoints not configured — no bearer token set",
        )
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token required for SCIM API access",
            headers={"WWW-Authenticate": "Bearer"},
        )
    import hmac
    provided_token = authorization[7:]
    if not hmac.compare_digest(provided_token, _SCIM_BEARER_TOKEN):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def create_scim_router(scim_provider: SCIMProvider) -> APIRouter:
    """Create SCIM 2.0 API router.

    All endpoints require bearer token authentication via the Authorization header.

    Args:
        scim_provider: SCIMProvider instance.

    Returns:
        FastAPI router with SCIM endpoints.
    """
    router = APIRouter(
        prefix="/scim/v2",
        tags=["SCIM"],
        dependencies=[Depends(_require_scim_auth)],
    )

    # ========================================================================
    # User Endpoints
    # ========================================================================

    @router.get("/Users")
    async def list_users(
        filter: Optional[str] = Query(None, description="SCIM filter expression"),
        startIndex: int = Query(1, ge=1, description="1-based start index for pagination"),
        count: int = Query(100, ge=1, le=1000, description="Number of results"),
    ) -> dict[str, Any]:
        """List users with optional filtering and pagination.

        Implements RFC 7643 Section 3.2.2

        Query Parameters:
            filter: SCIM filter expression (e.g., 'userName eq "test"')
            startIndex: 1-based index for pagination
            count: Max results per page

        Returns:
            SCIM ListResponse with users.
        """
        logger.info("list_users_request", filter=filter, start_index=startIndex, count=count)

        try:
            result = await scim_provider.list_users(
                filter_expr=filter,
                start_index=startIndex,
                count=count,
            )
            return result
        except Exception as e:
            logger.error("list_users_error", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to list users",
            )

    @router.get("/Users/{id}")
    async def get_user(id: str) -> dict[str, Any]:
        """Get a specific user by ID.

        Implements RFC 7643 Section 3.2.1

        Args:
            id: User ID.

        Returns:
            SCIM User resource.
        """
        logger.info("get_user_request", user_id=id)

        try:
            user = await scim_provider.get_user(id)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"User {id} not found",
                )

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
        except HTTPException:
            raise
        except Exception as e:
            logger.error("get_user_error", user_id=id, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve user",
            )

    @router.post("/Users")
    async def create_user(request: Request) -> dict[str, Any]:
        """Create a new user.

        Implements RFC 7643 Section 3.2.3

        Request Body:
            userName (required): Unique username.
            displayName: User's display name.
            email: User's email address.
            active: Whether user is active (default: true).
            roles: List of user roles.
            groups: List of group IDs.

        Returns:
            Created SCIM User resource.
        """
        logger.info("create_user_request")

        try:
            body = await request.json()
            user = await scim_provider.create_user(body)

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
        except ValueError as e:
            logger.warning("create_user_validation_error", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )
        except Exception as e:
            logger.error("create_user_error", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user",
            )

    @router.put("/Users/{id}")
    async def update_user(id: str, request: Request) -> dict[str, Any]:
        """Replace a user (PUT operation).

        Implements RFC 7643 Section 3.2.4

        Args:
            id: User ID.

        Request Body:
            Complete user data (replaces existing).

        Returns:
            Updated SCIM User resource.
        """
        logger.info("update_user_request", user_id=id)

        try:
            body = await request.json()
            user = await scim_provider.update_user(id, body)

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
        except ValueError as e:
            logger.warning("update_user_error", user_id=id, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e),
            )
        except Exception as e:
            logger.error("update_user_error", user_id=id, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update user",
            )

    @router.patch("/Users/{id}")
    async def patch_user(id: str, request: Request) -> dict[str, Any]:
        """Apply partial modifications to a user (PATCH operation).

        Implements RFC 7643 Section 3.2.5 and RFC 6902

        Args:
            id: User ID.

        Request Body:
            List of PATCH operations with op (add/remove/replace),
            path, and optional value.

        Returns:
            Updated SCIM User resource.
        """
        logger.info("patch_user_request", user_id=id)

        try:
            body = await request.json()
            operations = body.get("Operations", [])
            user = await scim_provider.patch_user(id, operations)

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
        except ValueError as e:
            logger.warning("patch_user_error", user_id=id, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e),
            )
        except Exception as e:
            logger.error("patch_user_error", user_id=id, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to patch user",
            )

    @router.delete("/Users/{id}")
    async def delete_user(id: str) -> dict[str, Any]:
        """Delete/deactivate a user.

        Implements RFC 7643 Section 3.2.6

        Args:
            id: User ID.

        Returns:
            Empty response with 204 status.
        """
        logger.info("delete_user_request", user_id=id)

        try:
            success = await scim_provider.delete_user(id)
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"User {id} not found",
                )
            return {}
        except HTTPException:
            raise
        except Exception as e:
            logger.error("delete_user_error", user_id=id, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete user",
            )

    # ========================================================================
    # Group Endpoints
    # ========================================================================

    @router.get("/Groups")
    async def list_groups(
        filter: Optional[str] = Query(None, description="SCIM filter expression"),
        startIndex: int = Query(1, ge=1, description="1-based start index"),
        count: int = Query(100, ge=1, le=1000, description="Number of results"),
    ) -> dict[str, Any]:
        """List groups with optional filtering and pagination.

        Query Parameters:
            filter: SCIM filter expression.
            startIndex: 1-based index for pagination.
            count: Max results per page.

        Returns:
            SCIM ListResponse with groups.
        """
        logger.info("list_groups_request", filter=filter, start_index=startIndex, count=count)

        try:
            result = await scim_provider.list_groups(
                filter_expr=filter,
                start_index=startIndex,
                count=count,
            )
            return result
        except Exception as e:
            logger.error("list_groups_error", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to list groups",
            )

    @router.get("/Groups/{id}")
    async def get_group(id: str) -> dict[str, Any]:
        """Get a specific group by ID.

        Args:
            id: Group ID.

        Returns:
            SCIM Group resource.
        """
        logger.info("get_group_request", group_id=id)

        try:
            group = await scim_provider.get_group(id)
            if not group:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Group {id} not found",
                )

            return {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Group"],
                "id": group.id,
                "displayName": group.display_name,
                "members": [{"value": member_id} for member_id in group.members],
                "meta": {
                    "resourceType": "Group",
                },
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error("get_group_error", group_id=id, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve group",
            )

    @router.post("/Groups")
    async def create_group(request: Request) -> dict[str, Any]:
        """Create a new group.

        Request Body:
            displayName (required): Group's display name.
            members: List of user IDs to add to group.

        Returns:
            Created SCIM Group resource.
        """
        logger.info("create_group_request")

        try:
            body = await request.json()
            group = await scim_provider.create_group(body)

            return {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Group"],
                "id": group.id,
                "displayName": group.display_name,
                "members": [{"value": member_id} for member_id in group.members],
                "meta": {
                    "resourceType": "Group",
                },
            }
        except ValueError as e:
            logger.warning("create_group_validation_error", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )
        except Exception as e:
            logger.error("create_group_error", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create group",
            )

    @router.put("/Groups/{id}")
    async def update_group(id: str, request: Request) -> dict[str, Any]:
        """Replace a group (PUT operation).

        Args:
            id: Group ID.

        Request Body:
            Complete group data (replaces existing).

        Returns:
            Updated SCIM Group resource.
        """
        logger.info("update_group_request", group_id=id)

        try:
            body = await request.json()
            group = await scim_provider.update_group(id, body)

            return {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Group"],
                "id": group.id,
                "displayName": group.display_name,
                "members": [{"value": member_id} for member_id in group.members],
                "meta": {
                    "resourceType": "Group",
                },
            }
        except ValueError as e:
            logger.warning("update_group_error", group_id=id, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e),
            )
        except Exception as e:
            logger.error("update_group_error", group_id=id, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update group",
            )

    @router.delete("/Groups/{id}")
    async def delete_group(id: str) -> dict[str, Any]:
        """Delete a group.

        Args:
            id: Group ID.

        Returns:
            Empty response with 204 status.
        """
        logger.info("delete_group_request", group_id=id)

        try:
            success = await scim_provider.delete_group(id)
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Group {id} not found",
                )
            return {}
        except HTTPException:
            raise
        except Exception as e:
            logger.error("delete_group_error", group_id=id, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete group",
            )

    # ========================================================================
    # Service Provider Configuration
    # ========================================================================

    @router.get("/ServiceProviderConfig")
    async def service_provider_config() -> dict[str, Any]:
        """Return SCIM service provider configuration.

        Implements RFC 7643 Section 5

        Returns:
            ServiceProviderConfig resource.
        """
        logger.info("service_provider_config_request")

        return {
            "schemas": ["urn:ietf:params:scim:schemas:core:2.0:ServiceProviderConfig"],
            "documentationUri": "https://tools.ietf.org/html/rfc7643",
            "patch": {
                "supported": True,
            },
            "bulk": {
                "supported": False,
            },
            "filter": {
                "supported": True,
                "maxResults": 1000,
            },
            "changePassword": {
                "supported": False,
            },
            "sort": {
                "supported": False,
            },
            "etag": {
                "supported": False,
            },
            "authenticationSchemes": [
                {
                    "name": "OAuth Bearer Token",
                    "description": "Authentication via Bearer token",
                    "specUri": "https://tools.ietf.org/html/rfc6750",
                    "type": "oauthbearertoken",
                    "primary": True,
                }
            ],
            "meta": {
                "location": "/scim/v2/ServiceProviderConfig",
                "resourceType": "ServiceProviderConfig",
            },
        }

    @router.get("/Schemas")
    async def schemas() -> dict[str, Any]:
        """Return supported SCIM schemas.

        Implements RFC 7643 Section 6

        Returns:
            ListResponse with schema definitions.
        """
        logger.info("schemas_request")

        return {
            "schemas": ["urn:ietf:params:scim:api:messages:2.0:ListResponse"],
            "totalResults": 2,
            "Resources": [
                {
                    "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Schema"],
                    "id": "urn:ietf:params:scim:schemas:core:2.0:User",
                    "name": "User",
                    "description": "User resource schema",
                    "attributes": [
                        {
                            "name": "userName",
                            "type": "string",
                            "required": True,
                        },
                        {
                            "name": "displayName",
                            "type": "string",
                        },
                        {
                            "name": "email",
                            "type": "string",
                        },
                        {
                            "name": "active",
                            "type": "boolean",
                        },
                    ],
                },
                {
                    "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Schema"],
                    "id": "urn:ietf:params:scim:schemas:core:2.0:Group",
                    "name": "Group",
                    "description": "Group resource schema",
                    "attributes": [
                        {
                            "name": "displayName",
                            "type": "string",
                            "required": True,
                        },
                        {
                            "name": "members",
                            "type": "complex",
                            "multiValued": True,
                        },
                    ],
                },
            ],
        }

    return router
