"""
Node repository implementation for the knowledge graph.

This module implements the repository pattern for Node objects, providing
a high-level interface for node management within the knowledge graph.
It handles node persistence, validation, and caching while enforcing
business rules and maintaining data integrity.

The node repository provides:
- CRUD operations for nodes
- Custom validation support
- Caching optimization
- Node dependency management
- Type-based node queries
"""

from typing import Any, Dict, List, Optional

from ..core.exceptions import ResourceNotFoundError, StorageError, ValidationError
from ..core.models import Node
from ..infrastructure.cache import LRUCache
from ..infrastructure.storage import StorageService
from ..utils.validation import DataIntegrityValidator
from .base import BaseRepository


class NodeRepository(BaseRepository[Node]):
    """
    Repository for managing Node objects in the knowledge graph.

    This repository handles the persistence and retrieval of Node objects,
    providing a high-level interface for node management while enforcing
    business rules and maintaining data integrity. It supports custom validation
    rules and optimizes performance through caching.

    Attributes:
        storage (StorageService): The underlying storage service for persistence
        cache (Optional[LRUCache]): Optional LRU cache for performance optimization
        node_validators (List[callable]): List of custom validation functions
    """

    def __init__(self, storage: StorageService, cache: Optional[LRUCache] = None):
        """
        Initialize the node repository.

        Args:
            storage: Storage service for data persistence
            cache: Optional LRU cache for performance optimization
        """
        super().__init__(storage, cache)
        self.node_validators = []

    def register_validator(self, validator_func):
        """
        Register a custom validation function for nodes.

        Custom validators allow for domain-specific validation rules beyond
        the basic integrity checks.

        Args:
            validator_func: Function that takes a Node and returns bool
        """
        self.node_validators.append(validator_func)

    async def create(self, item: Node) -> Node:
        """
        Create a new node in the knowledge graph.

        This method validates the node before creation and updates the cache
        after successful storage.

        Args:
            item: Node object to create

        Returns:
            Created node with updated metadata

        Raises:
            ValidationError: If node validation fails
            StorageError: If there's an error with storage operations
        """
        if not await self.validate(item):
            raise ValidationError(f"Node validation failed: {item.name}")

        try:
            # Create node in storage
            created_node = await self.storage.create_node(item)

            # Update cache
            await self._set_in_cache(created_node.name, created_node)

            return created_node

        except StorageError as e:
            raise StorageError(f"Failed to create node: {str(e)}")

    async def get(self, id: str) -> Node:
        """
        Retrieve a node by name.

        This method checks the cache before accessing storage and updates
        the cache after successful retrieval.

        Args:
            id: Node name/identifier

        Returns:
            Retrieved node

        Raises:
            ResourceNotFoundError: If node doesn't exist
            StorageError: If there's an error with storage operations
        """
        # Try cache first
        cached_node = await self._get_from_cache(id)
        if cached_node:
            return cached_node

        try:
            node = await self.storage.get_node(id)
            if not node:
                raise ResourceNotFoundError(f"Node not found: {id}")

            await self._set_in_cache(id, node)
            return node

        except StorageError as e:
            raise StorageError(f"Failed to get node: {str(e)}")

    async def update(self, item: Node) -> Node:
        """
        Update an existing node.

        This method validates the updated node and ensures the node
        exists before performing the update.

        Args:
            item: Node object to update

        Returns:
            Updated node

        Raises:
            ValidationError: If node validation fails
            ResourceNotFoundError: If node doesn't exist
            StorageError: If there's an error with storage operations
        """
        if not await self.validate(item):
            raise ValidationError(f"Node validation failed: {item.name}")

        try:
            # Check existence
            if not await self.storage.node_exists(item.name):
                raise ResourceNotFoundError(f"Node not found: {item.name}")

            # Update node in storage
            updated_node = await self.storage.update_node(item)

            # Update cache
            await self._set_in_cache(updated_node.name, updated_node)

            return updated_node

        except StorageError as e:
            raise StorageError(f"Failed to update node: {str(e)}")

    async def delete(self, id: str) -> None:
        """
        Delete a node by name.

        This method ensures the node exists before deletion and updates
        the cache accordingly.

        Args:
            id: Node name/identifier to delete

        Raises:
            ResourceNotFoundError: If node doesn't exist
            StorageError: If there's an error with storage operations
        """
        try:
            if not await self.storage.node_exists(id):
                raise ResourceNotFoundError(f"Node not found: {id}")

            await self.storage.delete_node(id)
            await self._remove_from_cache(id)

        except StorageError as e:
            raise StorageError(f"Failed to delete node: {str(e)}")

    async def list(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Node]:
        """
        List nodes with optional filtering.

        This method supports pagination and filtering to efficiently retrieve
        subsets of nodes.

        Args:
            filters: Optional dictionary of filters to apply
            limit: Maximum number of nodes to return
            offset: Number of nodes to skip

        Returns:
            List of nodes matching the criteria

        Raises:
            StorageError: If there's an error with storage operations
        """
        try:
            return await self.storage.list_nodes(filters=filters or {}, limit=limit, offset=offset)

        except StorageError as e:
            raise StorageError(f"Failed to list nodes: {str(e)}")

    async def validate(self, item: Node) -> bool:
        """
        Validate a node using built-in and custom validators.

        This method combines system integrity validation with custom
        domain-specific validation rules.

        Args:
            item: Node to validate

        Returns:
            True if validation passes, False otherwise
        """
        try:
            # Built-in validation
            validation_result = DataIntegrityValidator.validate_node_integrity(item)
            if not validation_result.is_valid:
                return False

            # Custom validators
            for validator in self.node_validators:
                if not await validator(item):
                    return False

            return True

        except ValidationError:
            return False

    async def get_by_type(self, entity_type: str) -> List[Node]:
        """
        Get all nodes of a specific type.

        This method provides efficient type-based querying of nodes.

        Args:
            entity_type: Type of nodes to retrieve

        Returns:
            List of nodes of the specified type

        Raises:
            StorageError: If there's an error with storage operations
        """
        try:
            return await self.storage.get_nodes_by_type(entity_type)
        except StorageError as e:
            raise StorageError(f"Failed to get nodes by type: {str(e)}")

    async def get_with_dependencies(self, name: str) -> Dict[str, Node]:
        """
        Get a node and all its dependencies.

        This method retrieves a node along with all nodes it depends on,
        providing a complete view of the node's context.

        Args:
            name: Name of the root node

        Returns:
            Dictionary mapping node names to Node objects

        Raises:
            ResourceNotFoundError: If root node doesn't exist
            StorageError: If there's an error with storage operations
        """
        try:
            node = await self.get(name)
            result = {name: node}

            for dep in node.dependencies:
                try:
                    dep_node = await self.get(dep)
                    result[dep] = dep_node
                except ResourceNotFoundError:
                    continue

            return result

        except StorageError as e:
            raise StorageError(f"Failed to get node with dependencies: {str(e)}")
