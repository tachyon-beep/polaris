"""
Node storage implementation.

This module provides persistent storage functionality for knowledge graph nodes.
It extends the base storage implementation with node-specific operations including:
- CRUD operations for nodes
- Node filtering and pagination
- Dependency tracking and resolution
- Type-based node queries
- Thread-safe operations

The storage implementation ensures data consistency and provides robust error handling
for all node operations.
"""

import logging
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...core.exceptions import NodeNotFoundError, StorageError
from ...core.models import Node, NodeMetadata
from .base import BaseStorage

logger = logging.getLogger(__name__)


class NodeStorage(BaseStorage[Node]):
    """
    Storage implementation for nodes.

    This class provides specialized storage operations for Node objects,
    extending the base storage functionality with node-specific features.
    It handles node creation, retrieval, updates, and deletion while
    maintaining data consistency and proper error handling.

    The storage is thread-safe and supports filtering, pagination, and
    dependency tracking for complex node relationships.
    """

    def __init__(self, storage_dir: str):
        """
        Initialize node storage.

        Args:
            storage_dir: Directory path for persistent node storage
        """
        super().__init__(storage_dir, "nodes.json")

    def _serialize_item(self, item: Node) -> Dict[str, Any]:
        """
        Convert a node to a dictionary for storage.

        Args:
            item: Node to serialize

        Returns:
            Dictionary representation of the node with datetime fields serialized
        """
        return asdict(
            item,
            dict_factory=lambda x: {k: self._serialize_datetime(v) for k, v in x},
        )

    def _deserialize_items(self, items_data: Dict[str, Any]) -> Dict[str, Node]:
        """
        Deserialize nodes from JSON data.

        Args:
            items_data: Dictionary of serialized nodes

        Returns:
            Dictionary mapping node names to deserialized Node objects
        """
        nodes = {}
        for name, data in items_data.items():
            data = self._deserialize_datetime(data)
            metadata = NodeMetadata(**data.pop("metadata"))
            nodes[name] = Node(**data, metadata=metadata)
        return nodes

    def _node_matches_filters(self, node: Node, filters: Dict[str, Any]) -> bool:
        """
        Check if a node matches all given filters.

        Args:
            node: Node to check
            filters: Dictionary of attribute-value pairs to match against

        Returns:
            True if node matches all filters, False otherwise
        """
        return all(self._get_nested_attr(node, key) == value for key, value in filters.items())

    async def create_node(self, node: Node) -> Node:
        """
        Create a new node.

        Creates a new node in storage with automatically set creation and
        modification timestamps.

        Args:
            node: Node to create

        Returns:
            Created node with updated metadata

        Raises:
            StorageError: If node already exists or creation fails
        """
        async with self._lock:
            try:
                if node.name in self._items:
                    raise StorageError(f"Node already exists: {node.name}")

                node.metadata.created_at = datetime.now()
                node.metadata.last_modified = datetime.now()

                self._items[node.name] = node
                await self._persist_items()

                return node

            except Exception as e:
                logger.error(f"Failed to create node: {str(e)}")
                raise StorageError(f"Node creation failed: {str(e)}")

    async def get_node(self, name: str) -> Node:
        """
        Get a node by name.

        Args:
            name: Name of the node to retrieve

        Returns:
            Retrieved node

        Raises:
            NodeNotFoundError: If node does not exist
            StorageError: If retrieval operation fails
        """
        async with self._lock:
            try:
                node = self._items.get(name)
                if not node:
                    raise NodeNotFoundError(f"Node not found: {name}")
                return node
            except Exception as e:
                logger.error(f"Failed to get node {name}: {str(e)}")
                raise StorageError(f"Failed to get node: {str(e)}")

    async def node_exists(self, name: str) -> bool:
        """
        Check if a node exists.

        Args:
            name: Name of the node to check

        Returns:
            True if node exists, False otherwise

        Raises:
            StorageError: If check operation fails
        """
        async with self._lock:
            try:
                return name in self._items
            except Exception as e:
                logger.error(f"Failed to check node existence {name}: {str(e)}")
                raise StorageError(f"Failed to check node existence: {str(e)}")

    async def update_node(self, node: Node) -> Node:
        """
        Update an existing node.

        Updates a node's data and automatically updates its last modified timestamp.

        Args:
            node: Node with updated data

        Returns:
            Updated node

        Raises:
            NodeNotFoundError: If node does not exist
            StorageError: If update operation fails
        """
        async with self._lock:
            try:
                if node.name not in self._items:
                    raise NodeNotFoundError(f"Node not found: {node.name}")

                node.metadata.last_modified = datetime.now()
                self._items[node.name] = node
                await self._persist_items()

                return node

            except Exception as e:
                logger.error(f"Failed to update node {node.name}: {str(e)}")
                raise StorageError(f"Failed to update node: {str(e)}")

    async def delete_node(self, name: str) -> None:
        """
        Delete a node.

        Args:
            name: Name of the node to delete

        Raises:
            NodeNotFoundError: If node does not exist
            StorageError: If deletion operation fails
        """
        async with self._lock:
            try:
                if name not in self._items:
                    raise NodeNotFoundError(f"Node not found: {name}")

                del self._items[name]
                await self._persist_items()

            except Exception as e:
                logger.error(f"Failed to delete node {name}: {str(e)}")
                raise StorageError(f"Failed to delete node: {str(e)}")

    async def list_nodes(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Node]:
        """
        List nodes with optional filtering and pagination.

        Args:
            filters: Optional dictionary of filters to apply
            limit: Maximum number of nodes to return
            offset: Number of nodes to skip

        Returns:
            List of nodes matching the filters

        Raises:
            StorageError: If listing operation fails or pagination parameters are invalid
        """
        async with self._lock:
            try:
                self._validate_pagination(offset, limit)

                nodes = list(self._items.values())

                if filters:
                    nodes = [node for node in nodes if self._node_matches_filters(node, filters)]

                return nodes[offset : offset + limit]

            except ValueError as e:
                raise StorageError(f"Invalid pagination parameters: {str(e)}")
            except Exception as e:
                logger.error(f"Failed to list nodes: {str(e)}")
                raise StorageError(f"Failed to list nodes: {str(e)}")

    async def get_nodes_by_type(self, entity_type: str) -> List[Node]:
        """
        Get all nodes of a specific type.

        Args:
            entity_type: Type of nodes to retrieve

        Returns:
            List of nodes of the specified type
        """
        return await self.list_nodes(filters={"entity_type": entity_type})

    async def get_nodes_with_dependencies(self, name: str) -> Dict[str, Node]:
        """
        Get a node and all its dependencies.

        Performs a depth-first traversal of node dependencies to retrieve
        all related nodes.

        Args:
            name: Name of the root node

        Returns:
            Dictionary mapping node names to Node objects for the root
            node and all its dependencies

        Raises:
            NodeNotFoundError: If root node does not exist
            StorageError: If dependency resolution fails
        """
        async with self._lock:
            try:
                if name not in self._items:
                    raise NodeNotFoundError(f"Node not found: {name}")

                result = {}
                to_process = [name]
                processed = set()

                while to_process:
                    current = to_process.pop()
                    if current in processed:
                        continue

                    node = self._items.get(current)
                    if node:
                        result[current] = node
                        processed.add(current)
                        # Add dependencies to processing queue
                        to_process.extend(dep for dep in node.dependencies if dep not in processed)

                return result

            except Exception as e:
                logger.error(f"Failed to get node dependencies for {name}: {str(e)}")
                raise StorageError(f"Failed to get node dependencies: {str(e)}")
