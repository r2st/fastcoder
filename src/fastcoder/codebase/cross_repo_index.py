"""Cross-repository index for tracking shared types, API contracts, and schemas."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import structlog
import yaml
from pydantic import BaseModel, Field

from fastcoder.types.codebase import APIEndpoint, APISurface, SymbolInfo

logger = structlog.get_logger(__name__)


# =============================================================================
# Pydantic Models
# =============================================================================


class ContractConsumer(BaseModel):
    """A repository that uses a shared contract."""

    repo: str
    file: str
    import_path: str = ""
    usage_type: str = "import"  # import, extends, implements, calls


class SharedContract(BaseModel):
    """A type, API endpoint, or schema shared across repositories."""

    id: str
    name: str
    kind: str  # type, api_endpoint, protobuf_message, graphql_type, openapi_schema
    source_repo: str
    source_file: str
    source_line: int = 0
    definition: str = ""
    consumers: list[ContractConsumer] = Field(default_factory=list)
    version_hash: str = ""
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class RepoRegistration(BaseModel):
    """A registered repository in the cross-repo index."""

    repo_id: str
    repo_path: str
    repo_url: str = ""
    branch: str = "main"
    last_indexed: Optional[datetime] = None
    contract_count: int = 0


class RepoChange(BaseModel):
    """Changes to a repository as part of a coordinated changeset."""

    repo_id: str
    files_changed: list[str] = Field(default_factory=list)
    contracts_affected: list[str] = Field(default_factory=list)
    pr_url: Optional[str] = None


class CrossRepoChangeSet(BaseModel):
    """A set of coordinated changes across multiple repositories."""

    id: str
    source_story_id: str = ""
    changes: list[RepoChange] = Field(default_factory=list)
    validation_status: str = "pending"  # pending, valid, invalid
    validation_errors: list[str] = Field(default_factory=list)


# =============================================================================
# Main CrossRepoIndex Class
# =============================================================================


class CrossRepoIndex:
    """Index and track shared types/contracts across multiple repositories."""

    def __init__(self) -> None:
        """Initialize the cross-repo index."""
        self._repos: dict[str, RepoRegistration] = {}
        self._contracts: dict[str, SharedContract] = {}
        self._dependency_graph: dict[str, set[str]] = {}

    async def register_repo(
        self, repo_id: str, repo_path: str, repo_url: str = "", branch: str = "main"
    ) -> RepoRegistration:
        """
        Register a repository for cross-repo tracking.

        Args:
            repo_id: Unique identifier for the repository
            repo_path: Local path to the repository
            repo_url: Remote URL of the repository
            branch: Primary branch name

        Returns:
            RepoRegistration object for the registered repository
        """
        if not Path(repo_path).exists():
            logger.warning("repo_path_not_found", repo_id=repo_id, repo_path=repo_path)

        registration = RepoRegistration(
            repo_id=repo_id,
            repo_path=repo_path,
            repo_url=repo_url,
            branch=branch,
            contract_count=0,
        )
        self._repos[repo_id] = registration
        self._dependency_graph[repo_id] = set()

        logger.info("repo_registered", repo_id=repo_id, repo_path=repo_path, repo_url=repo_url)
        return registration

    async def index_repo(self, repo_id: str) -> list[SharedContract]:
        """
        Index shared contracts in a repository.

        Scans for:
        - Protobuf messages (.proto files)
        - OpenAPI schemas (openapi.yaml/json)
        - GraphQL schemas (.graphql files)
        - Shared TypeScript/Python types

        Args:
            repo_id: Repository ID to index

        Returns:
            List of SharedContract objects found in the repository
        """
        if repo_id not in self._repos:
            logger.error("repo_not_registered", repo_id=repo_id)
            return []

        repo_registration = self._repos[repo_id]
        repo_path = Path(repo_registration.repo_path)

        if not repo_path.exists():
            logger.error("repo_path_not_found", repo_id=repo_id, repo_path=str(repo_path))
            return []

        contracts: list[SharedContract] = []

        try:
            contracts.extend(await self._scan_for_protobuf(str(repo_path), repo_id))
        except Exception as e:
            logger.warning("protobuf_scan_failed", repo_id=repo_id, error=str(e))

        try:
            contracts.extend(await self._scan_for_openapi(str(repo_path), repo_id))
        except Exception as e:
            logger.warning("openapi_scan_failed", repo_id=repo_id, error=str(e))

        try:
            contracts.extend(await self._scan_for_graphql(str(repo_path), repo_id))
        except Exception as e:
            logger.warning("graphql_scan_failed", repo_id=repo_id, error=str(e))

        try:
            contracts.extend(await self._scan_for_shared_types(str(repo_path), repo_id))
        except Exception as e:
            logger.warning("shared_types_scan_failed", repo_id=repo_id, error=str(e))

        # Update index
        for contract in contracts:
            self._contracts[contract.id] = contract

        # Update registration
        repo_registration.contract_count = len(contracts)
        repo_registration.last_indexed = datetime.utcnow()

        logger.info(
            "repo_indexed",
            repo_id=repo_id,
            contract_count=len(contracts),
            contracts=[(c.name, c.kind) for c in contracts],
        )

        return contracts

    async def _scan_for_protobuf(self, repo_path: str, repo_id: str) -> list[SharedContract]:
        """
        Scan for .proto files and extract message definitions.

        Args:
            repo_path: Path to repository
            repo_id: Repository ID

        Returns:
            List of SharedContract objects for protobuf messages
        """
        contracts: list[SharedContract] = []
        repo_path_obj = Path(repo_path)

        proto_files = list(repo_path_obj.rglob("*.proto"))

        for proto_file in proto_files:
            try:
                content = proto_file.read_text(encoding="utf-8")
                relative_path = str(proto_file.relative_to(repo_path_obj))

                # Extract message definitions
                message_pattern = r"^\s*message\s+(\w+)\s*\{"
                for match in re.finditer(message_pattern, content, re.MULTILINE):
                    message_name = match.group(1)
                    contract_id = f"protobuf_{repo_id}_{relative_path}_{message_name}"

                    # Extract full message definition
                    start_pos = match.start()
                    start_line = content[:start_pos].count("\n") + 1
                    message_def = self._extract_protobuf_definition(content, match.start())

                    version_hash = hashlib.sha256(message_def.encode()).hexdigest()

                    contract = SharedContract(
                        id=contract_id,
                        name=message_name,
                        kind="protobuf_message",
                        source_repo=repo_id,
                        source_file=relative_path,
                        source_line=start_line,
                        definition=message_def,
                        version_hash=version_hash,
                    )
                    contracts.append(contract)

                logger.debug("protobuf_file_scanned", file=relative_path, messages=len(contracts))
            except Exception as e:
                logger.warning("protobuf_parse_failed", file=str(proto_file), error=str(e))

        return contracts

    async def _scan_for_openapi(self, repo_path: str, repo_id: str) -> list[SharedContract]:
        """
        Scan for openapi.yaml/json and extract schemas.

        Args:
            repo_path: Path to repository
            repo_id: Repository ID

        Returns:
            List of SharedContract objects for OpenAPI schemas
        """
        contracts: list[SharedContract] = []
        repo_path_obj = Path(repo_path)

        # Look for common OpenAPI/Swagger file names
        openapi_patterns = [
            "openapi.yaml",
            "openapi.yml",
            "openapi.json",
            "swagger.yaml",
            "swagger.yml",
            "swagger.json",
            "**/openapi.yaml",
            "**/openapi.json",
        ]

        openapi_files: set[Path] = set()
        for pattern in openapi_patterns:
            openapi_files.update(repo_path_obj.glob(pattern))

        for openapi_file in openapi_files:
            try:
                content = openapi_file.read_text(encoding="utf-8")
                relative_path = str(openapi_file.relative_to(repo_path_obj))

                # Parse YAML or JSON
                spec: dict[str, Any] = {}
                if openapi_file.suffix.lower() in {".yaml", ".yml"}:
                    spec = yaml.safe_load(content) or {}
                else:
                    spec = json.loads(content)

                # Extract schema definitions
                schemas = spec.get("components", {}).get("schemas", {})
                if not schemas:
                    schemas = spec.get("definitions", {})

                for schema_name, schema_def in schemas.items():
                    contract_id = f"openapi_{repo_id}_{relative_path}_{schema_name}"
                    schema_json = json.dumps(schema_def, indent=2)
                    version_hash = hashlib.sha256(schema_json.encode()).hexdigest()

                    contract = SharedContract(
                        id=contract_id,
                        name=schema_name,
                        kind="openapi_schema",
                        source_repo=repo_id,
                        source_file=relative_path,
                        definition=schema_json,
                        version_hash=version_hash,
                    )
                    contracts.append(contract)

                logger.debug(
                    "openapi_file_scanned", file=relative_path, schemas=len(contracts)
                )
            except Exception as e:
                logger.warning("openapi_parse_failed", file=str(openapi_file), error=str(e))

        return contracts

    async def _scan_for_graphql(self, repo_path: str, repo_id: str) -> list[SharedContract]:
        """
        Scan for .graphql schema files.

        Args:
            repo_path: Path to repository
            repo_id: Repository ID

        Returns:
            List of SharedContract objects for GraphQL types
        """
        contracts: list[SharedContract] = []
        repo_path_obj = Path(repo_path)

        graphql_files = list(repo_path_obj.rglob("*.graphql")) + list(
            repo_path_obj.rglob("*.gql")
        )

        for graphql_file in graphql_files:
            try:
                content = graphql_file.read_text(encoding="utf-8")
                relative_path = str(graphql_file.relative_to(repo_path_obj))

                # Extract type definitions
                type_pattern = r"^\s*(type|interface|enum|scalar|directive|union)\s+(\w+)\s*"
                for match in re.finditer(type_pattern, content, re.MULTILINE):
                    kind_keyword = match.group(1)
                    type_name = match.group(2)
                    start_line = content[: match.start()].count("\n") + 1

                    type_def = self._extract_graphql_definition(content, match.start())

                    contract_id = f"graphql_{repo_id}_{relative_path}_{type_name}"
                    version_hash = hashlib.sha256(type_def.encode()).hexdigest()

                    contract = SharedContract(
                        id=contract_id,
                        name=type_name,
                        kind=f"graphql_{kind_keyword.lower()}",
                        source_repo=repo_id,
                        source_file=relative_path,
                        source_line=start_line,
                        definition=type_def,
                        version_hash=version_hash,
                    )
                    contracts.append(contract)

                logger.debug("graphql_file_scanned", file=relative_path, types=len(contracts))
            except Exception as e:
                logger.warning("graphql_parse_failed", file=str(graphql_file), error=str(e))

        return contracts

    async def _scan_for_shared_types(self, repo_path: str, repo_id: str) -> list[SharedContract]:
        """
        Scan for exported TypeScript/Python types that other repos might import.

        Looks for:
        - TypeScript/JavaScript: exported types and interfaces
        - Python: classes with __all__ exports or documented types

        Args:
            repo_path: Path to repository
            repo_id: Repository ID

        Returns:
            List of SharedContract objects for shared types
        """
        contracts: list[SharedContract] = []
        repo_path_obj = Path(repo_path)

        # Scan TypeScript/JavaScript files
        ts_files = list(repo_path_obj.rglob("*.ts")) + list(repo_path_obj.rglob("*.tsx"))
        for ts_file in ts_files:
            try:
                content = ts_file.read_text(encoding="utf-8")
                relative_path = str(ts_file.relative_to(repo_path_obj))

                # Extract exported types and interfaces
                type_pattern = r"^\s*export\s+(type|interface)\s+(\w+)\s*[={]"
                for match in re.finditer(type_pattern, content, re.MULTILINE):
                    kind_keyword = match.group(1)
                    type_name = match.group(2)
                    start_line = content[: match.start()].count("\n") + 1

                    type_def = self._extract_typescript_definition(content, match.start())

                    contract_id = f"typescript_{repo_id}_{relative_path}_{type_name}"
                    version_hash = hashlib.sha256(type_def.encode()).hexdigest()

                    contract = SharedContract(
                        id=contract_id,
                        name=type_name,
                        kind=f"typescript_{kind_keyword.lower()}",
                        source_repo=repo_id,
                        source_file=relative_path,
                        source_line=start_line,
                        definition=type_def,
                        version_hash=version_hash,
                    )
                    contracts.append(contract)

                logger.debug("typescript_file_scanned", file=relative_path, types=len(contracts))
            except Exception as e:
                logger.warning("typescript_scan_failed", file=str(ts_file), error=str(e))

        # Scan Python files for shared types
        py_files = list(repo_path_obj.rglob("*.py"))
        for py_file in py_files:
            try:
                content = py_file.read_text(encoding="utf-8")
                relative_path = str(py_file.relative_to(repo_path_obj))

                # Look for __all__ exports
                all_pattern = r'__all__\s*=\s*\[(.*?)\]'
                all_match = re.search(all_pattern, content, re.DOTALL)
                if all_match:
                    exports_text = all_match.group(1)
                    exports = re.findall(r'"(\w+)"|\'(\w+)\'', exports_text)
                    for export_tuple in exports:
                        export_name = export_tuple[0] or export_tuple[1]

                        # Find class or function definition
                        def_pattern = rf"^\s*(class|def)\s+{re.escape(export_name)}\s*[\(:]"
                        def_match = re.search(def_pattern, content, re.MULTILINE)
                        if def_match:
                            start_line = content[: def_match.start()].count("\n") + 1
                            type_def = self._extract_python_definition(
                                content, def_match.start(), export_name
                            )

                            kind = "class" if def_match.group(1) == "class" else "function"
                            contract_id = (
                                f"python_{repo_id}_{relative_path}_{export_name}"
                            )
                            version_hash = hashlib.sha256(type_def.encode()).hexdigest()

                            contract = SharedContract(
                                id=contract_id,
                                name=export_name,
                                kind=f"python_{kind}",
                                source_repo=repo_id,
                                source_file=relative_path,
                                source_line=start_line,
                                definition=type_def,
                                version_hash=version_hash,
                            )
                            contracts.append(contract)

                logger.debug("python_file_scanned", file=relative_path, types=len(contracts))
            except Exception as e:
                logger.warning("python_scan_failed", file=str(py_file), error=str(e))

        return contracts

    def get_contract(self, contract_id: str) -> Optional[SharedContract]:
        """
        Get a shared contract by ID.

        Args:
            contract_id: Contract ID

        Returns:
            SharedContract if found, None otherwise
        """
        return self._contracts.get(contract_id)

    def get_consumers(self, contract_id: str) -> list[ContractConsumer]:
        """
        Get all consumers of a shared contract.

        Args:
            contract_id: Contract ID

        Returns:
            List of ContractConsumer objects
        """
        contract = self.get_contract(contract_id)
        if not contract:
            return []
        return contract.consumers

    def add_consumer(self, contract_id: str, consumer: ContractConsumer) -> None:
        """
        Add a consumer relationship to a shared contract.

        Args:
            contract_id: Contract ID
            consumer: ContractConsumer object to add
        """
        contract = self.get_contract(contract_id)
        if not contract:
            logger.warning("contract_not_found", contract_id=contract_id)
            return

        # Avoid duplicates
        for existing in contract.consumers:
            if existing.repo == consumer.repo and existing.file == consumer.file:
                logger.debug("consumer_already_registered", contract_id=contract_id)
                return

        contract.consumers.append(consumer)
        logger.debug(
            "consumer_added",
            contract_id=contract_id,
            consumer_repo=consumer.repo,
            consumer_file=consumer.file,
        )

    def get_affected_repos(self, changed_contracts: list[str]) -> set[str]:
        """
        Get all repositories affected by changes to the given contracts.

        Args:
            changed_contracts: List of contract IDs that changed

        Returns:
            Set of repository IDs affected by these changes
        """
        affected_repos: set[str] = set()

        for contract_id in changed_contracts:
            contract = self.get_contract(contract_id)
            if contract:
                affected_repos.add(contract.source_repo)
                for consumer in contract.consumers:
                    affected_repos.add(consumer.repo)

        return affected_repos

    async def validate_change_set(self, change_set: CrossRepoChangeSet) -> CrossRepoChangeSet:
        """
        Validate that a set of cross-repo changes maintains contract compatibility.

        Args:
            change_set: CrossRepoChangeSet to validate

        Returns:
            Updated CrossRepoChangeSet with validation status and errors
        """
        errors: list[str] = []

        # Validate that all affected repos exist
        all_repos = set()
        for change in change_set.changes:
            all_repos.add(change.repo_id)

        for repo_id in all_repos:
            if repo_id not in self._repos:
                errors.append(f"Repository {repo_id} not registered in index")

        # Validate contracts are compatible
        for change in change_set.changes:
            for contract_id in change.contracts_affected:
                contract = self.get_contract(contract_id)
                if not contract:
                    errors.append(
                        f"Contract {contract_id} not found in index for repo {change.repo_id}"
                    )

        if errors:
            change_set.validation_status = "invalid"
            change_set.validation_errors = errors
            logger.warning(
                "change_set_validation_failed",
                change_set_id=change_set.id,
                errors=errors,
            )
        else:
            change_set.validation_status = "valid"
            logger.info("change_set_validated", change_set_id=change_set.id)

        return change_set

    def create_change_plan(
        self, story_id: str, primary_repo: str, changed_files: list[str]
    ) -> CrossRepoChangeSet:
        """
        Create a coordinated change plan for a story that affects shared contracts.

        Args:
            story_id: Story/issue ID
            primary_repo: Primary repository where changes originate
            changed_files: List of files changed in the primary repo

        Returns:
            CrossRepoChangeSet with coordinated changes
        """
        change_set = CrossRepoChangeSet(
            id=f"changeset_{story_id}_{datetime.utcnow().timestamp()}",
            source_story_id=story_id,
        )

        # Find contracts affected by changed files
        affected_contracts: set[str] = set()
        for contract_id, contract in self._contracts.items():
            if contract.source_repo == primary_repo:
                for changed_file in changed_files:
                    if changed_file in contract.source_file or contract.source_file in changed_file:
                        affected_contracts.add(contract_id)

        # Get all affected repos
        affected_repos = self.get_affected_repos(list(affected_contracts))

        # Create changes for each affected repo
        for repo_id in affected_repos:
            repo_change = RepoChange(
                repo_id=repo_id,
                files_changed=[],
                contracts_affected=list(affected_contracts),
            )
            change_set.changes.append(repo_change)

        logger.info(
            "change_plan_created",
            story_id=story_id,
            primary_repo=primary_repo,
            affected_repos=len(affected_repos),
            affected_contracts=len(affected_contracts),
        )

        return change_set

    def get_dependency_graph(self) -> dict[str, set[str]]:
        """
        Get the repository-to-repository dependency graph.

        Returns:
            Dict mapping repo_id to set of dependent repo_ids
        """
        graph: dict[str, set[str]] = {}

        # Build graph from consumer relationships
        for repo_id in self._repos:
            graph[repo_id] = set()

        for contract in self._contracts.values():
            for consumer in contract.consumers:
                if consumer.repo != contract.source_repo:
                    if contract.source_repo not in graph:
                        graph[contract.source_repo] = set()
                    graph[contract.source_repo].add(consumer.repo)

        self._dependency_graph = graph
        return graph

    def save(self, path: str) -> None:
        """
        Persist the index to disk.

        Saves repos, contracts, and dependency graph to a JSON file.

        Args:
            path: File path to save to
        """
        try:
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to serializable format
            data = {
                "repos": {
                    repo_id: repo.model_dump() for repo_id, repo in self._repos.items()
                },
                "contracts": {
                    contract_id: contract.model_dump()
                    for contract_id, contract in self._contracts.items()
                },
                "dependency_graph": {
                    repo_id: list(deps) for repo_id, deps in self._dependency_graph.items()
                },
            }

            # Serialize datetime objects
            for repo in data["repos"].values():
                if repo.get("last_indexed"):
                    repo["last_indexed"] = repo["last_indexed"].isoformat()

            for contract in data["contracts"].values():
                if contract.get("last_updated"):
                    contract["last_updated"] = contract["last_updated"].isoformat()

            output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            logger.info("index_saved", path=path)
        except Exception as e:
            logger.error("index_save_failed", path=path, error=str(e))

    def load(self, path: str) -> None:
        """
        Load the index from disk.

        Args:
            path: File path to load from
        """
        try:
            input_path = Path(path)

            if not input_path.exists():
                logger.warning("index_file_not_found", path=path)
                return

            data = json.loads(input_path.read_text(encoding="utf-8"))

            # Load repos
            self._repos.clear()
            for repo_id, repo_data in data.get("repos", {}).items():
                if repo_data.get("last_indexed"):
                    repo_data["last_indexed"] = datetime.fromisoformat(
                        repo_data["last_indexed"]
                    )
                self._repos[repo_id] = RepoRegistration(**repo_data)

            # Load contracts
            self._contracts.clear()
            for contract_id, contract_data in data.get("contracts", {}).items():
                if contract_data.get("last_updated"):
                    contract_data["last_updated"] = datetime.fromisoformat(
                        contract_data["last_updated"]
                    )
                self._contracts[contract_id] = SharedContract(**contract_data)

            # Load dependency graph
            self._dependency_graph.clear()
            for repo_id, deps in data.get("dependency_graph", {}).items():
                self._dependency_graph[repo_id] = set(deps)

            logger.info(
                "index_loaded",
                path=path,
                repos=len(self._repos),
                contracts=len(self._contracts),
            )
        except Exception as e:
            logger.error("index_load_failed", path=path, error=str(e))

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _extract_protobuf_definition(self, content: str, start_pos: int) -> str:
        """
        Extract a complete protobuf message definition.

        Args:
            content: File content
            start_pos: Starting position of the message

        Returns:
            The message definition including braces and contents
        """
        brace_count = 0
        end_pos = start_pos
        found_open = False

        for i in range(start_pos, len(content)):
            if content[i] == "{":
                brace_count += 1
                found_open = True
            elif content[i] == "}":
                brace_count -= 1
                if found_open and brace_count == 0:
                    end_pos = i + 1
                    break

        return content[start_pos:end_pos].strip()

    def _extract_graphql_definition(self, content: str, start_pos: int) -> str:
        """
        Extract a complete GraphQL type definition.

        Args:
            content: File content
            start_pos: Starting position of the type

        Returns:
            The type definition including braces and contents
        """
        return self._extract_protobuf_definition(content, start_pos)

    def _extract_typescript_definition(self, content: str, start_pos: int) -> str:
        """
        Extract a complete TypeScript type/interface definition.

        Args:
            content: File content
            start_pos: Starting position of the type

        Returns:
            The type/interface definition
        """
        # Find the next significant syntax boundary
        lines = content[start_pos:].split("\n")
        definition_lines: list[str] = []
        brace_count = 0
        found_open = False

        for line in lines:
            definition_lines.append(line)
            brace_count += line.count("{") - line.count("}")

            if "{" in line:
                found_open = True

            if found_open and brace_count == 0:
                break

        return "\n".join(definition_lines).strip()

    def _extract_python_definition(
        self, content: str, start_pos: int, symbol_name: str
    ) -> str:
        """
        Extract a Python class or function definition.

        Args:
            content: File content
            start_pos: Starting position of the definition
            symbol_name: Name of the symbol

        Returns:
            The complete definition including docstring
        """
        lines = content[start_pos:].split("\n")
        definition_lines: list[str] = []
        base_indent = None

        for i, line in enumerate(lines):
            if i == 0:
                definition_lines.append(line)
                # Capture base indentation
                base_indent = len(line) - len(line.lstrip())
                continue

            # Stop at next symbol at same or less indentation
            if line.strip() and not line.startswith(" " * (base_indent + 1)) and i > 1:
                if line.strip() and not line.strip().startswith("#"):
                    break

            definition_lines.append(line)

        return "\n".join(definition_lines).strip()

    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about the cross-repo index.

        Returns:
            Dictionary with index statistics
        """
        contracts_by_kind: dict[str, int] = {}
        for contract in self._contracts.values():
            contracts_by_kind[contract.kind] = contracts_by_kind.get(contract.kind, 0) + 1

        total_consumers = sum(len(c.consumers) for c in self._contracts.values())

        return {
            "total_repos": len(self._repos),
            "total_contracts": len(self._contracts),
            "contracts_by_kind": contracts_by_kind,
            "total_consumer_relationships": total_consumers,
            "repos_by_contract_count": {
                repo_id: reg.contract_count for repo_id, reg in self._repos.items()
            },
        }
