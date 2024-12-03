"""
Edge storage implementation.

This module provides persistent storage functionality for knowledge graph edges.
It extends the base storage implementation with edge-specific operations including:
- CRUD operations for edges
- Edge filtering and pagination
- Type-based edge queries
- Thread-safe operations

The storage implementation ensures data consistency and provides robust error handling
for all edge operations.
"""

import logging
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...core.enums import RelationType
from ...core.exceptions import EdgeNotFoundError, StorageError
from ...core.models import Edge, EdgeMetadata
from .base import BaseStorage

logger = logging.getLogger(__name__)


class EdgeStorage(BaseStorage[Edge]):
    """
    Storage implementation for edges.

    This class provides specialized storage operations for Edge objects,
    extending the base storage functionality with edge-specific features.
    It handles edge creation, retrieval, updates, and deletion while
    maintaining data consistency and proper error handling.

    The storage is thread-safe and supports filtering, pagination, and
    type-based edge queries.
    """

    def __init__(self, storage_dir: str):
        """
        Initialize edge storage.

        Args:
            storage_dir: Directory path for persistent edge storage
        """
        super().__init__(storage_dir, "edges.json")

    def _get_edge_key(self, from_node: str, to_node: str, relation_type: RelationType) -> str:
        """
        Generate a unique key for an edge.

        Args:
            from_node: Source node name
            to_node: Target node name
            relation_type: Type of relationship

        Returns:
            Unique string key for edge

        Raises:
            ValueError: If relation_type is not a valid RelationType
        """
        if not isinstance(relation_type, RelationType):
            raise ValueError(f"Invalid relation type: {relation_type}")
        return f"{from_node}:{relation_type.value}:{to_node}"

    def _serialize_item(self, item: Edge) -> Dict[str, Any]:
        """
        Convert an edge to a dictionary for storage.

        Args:
            item: Edge to serialize

        Returns:
            Dictionary representation of the edge with datetime fields serialized
        """
        return asdict(
            item,
            dict_factory=lambda x: {k: self._serialize_datetime(v) for k, v in x},
        )

    def _deserialize_items(self, items_data: Dict[str, Any]) -> Dict[str, Edge]:
        """
        Deserialize edges from JSON data.

        Args:
            items_data: Dictionary of serialized edges

        Returns:
            Dictionary mapping edge keys to deserialized Edge objects
        """
        edges = {}
        for key, data in items_data.items():
            data = self._deserialize_datetime(data)
            metadata = EdgeMetadata(**data.pop("metadata"))
            edges[key] = Edge(**data, metadata=metadata)
        return edges

    def _edge_matches_filters(self, edge: Edge, filters: Dict[str, Any]) -> bool:
        """
        Check if an edge matches all given filters.

        Args:
            edge: Edge to check
            filters: Dictionary of filters, may include $or operator for OR conditions

        Returns:
            True if edge matches all filters, False otherwise
        """
        if "$or" in filters:
            return any(
                all(self._get_nested_attr(edge, key) == value for key, value in or_filter.items())
                for or_filter in filters["$or"]
            )

        return all(self._get_nested_attr(edge, key) == value for key, value in filters.items())

    async def create_edge(self, edge: Edge) -> Edge:
        """
        Create a new edge.

        Creates a new edge in storage with automatically set creation and
        modification timestamps.

        Args:
            edge: Edge to create

        Returns:
            Created edge with updated metadata

        Raises:
            StorageError: If edge already exists or creation fails
        """
        async with self._lock:
            try:
                key = self._get_edge_key(edge.from_entity, edge.to_entity, edge.relation_type)

                if key in self._items:
                    raise StorageError(f"Edge already exists: {key}")

                edge.metadata.created_at = datetime.now()
                edge.metadata.last_modified = datetime.now()

                self._items[key] = edge
                await self._persist_items()

                return edge

            except Exception as e:
                logger.error(f"Failed to create edge: {str(e)}")
                raise StorageError(f"Edge creation failed: {str(e)}")

    async def get_edge(self, from_node: str, to_node: str, relation_type: RelationType) -> Edge:
        """
        Get an edge by its components.

        Args:
            from_node: Source node name
            to_node: Target node name
            relation_type: Type of relationship

        Returns:
            Retrieved edge

        Raises:
            EdgeNotFoundError: If edge does not exist
            StorageError: If retrieval operation fails
        """
        async with self._lock:
            try:
                key = self._get_edge_key(from_node, to_node, relation_type)
                edge = self._items.get(key)
                if not edge:
                    raise EdgeNotFoundError(f"Edge not found: {key}")
                return edge
            except Exception as e:
                logger.error(f"Failed to get edge: {str(e)}")
                raise StorageError(f"Failed to get edge: {str(e)}")

    async def update_edge(self, edge: Edge) -> Edge:
        """
        Update an existing edge.

        Updates an edge's data and automatically updates its last modified timestamp.

        Args:
            edge: Edge with updated data

        Returns:
            Updated edge

        Raises:
            EdgeNotFoundError: If edge does not exist
            StorageError: If update operation fails
        """
        async with self._lock:
            try:
                key = self._get_edge_key(edge.from_entity, edge.to_entity, edge.relation_type)

                if key not in self._items:
                    raise EdgeNotFoundError(f"Edge not found: {key}")

                edge.metadata.last_modified = datetime.now()
                self._items[key] = edge
                await self._persist_items()

                return edge

            except Exception as e:
                logger.error(f"Failed to update edge: {str(e)}")
                raise StorageError(f"Failed to update edge: {str(e)}")

    async def delete_edge(self, from_node: str, to_node: str, relation_type: RelationType) -> None:
        """
        Delete an edge.

        Args:
            from_node: Source node name
            to_node: Target node name
            relation_type: Type of relationship

        Raises:
            EdgeNotFoundError: If edge does not exist
            StorageError: If deletion operation fails
        """
        async with self._lock:
            try:
                key = self._get_edge_key(from_node, to_node, relation_type)

                if key not in self._items:
                    raise EdgeNotFoundError(f"Edge not found: {key}")

                del self._items[key]
                await self._persist_items()

            except Exception as e:
                logger.error(f"Failed to delete edge: {str(e)}")
                raise StorageError(f"Failed to delete edge: {str(e)}")

    async def list_edges(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Edge]:
        """
        List edges with optional filtering and pagination.

        Args:
            filters: Optional dictionary of filters to apply
            limit: Maximum number of edges to return
            offset: Number of edges to skip

        Returns:
            List of edges matching the filters

        Raises:
            StorageError: If listing operation fails or pagination parameters are invalid
        """
        async with self._lock:
            try:
                self._validate_pagination(offset, limit)

                edges = list(self._items.values())

                if filters:
                    edges = [edge for edge in edges if self._edge_matches_filters(edge, filters)]

                return edges[offset : offset + limit]

            except ValueError as e:
                raise StorageError(f"Invalid pagination parameters: {str(e)}")
            except Exception as e:
                logger.error(f"Failed to list edges: {str(e)}")
                raise StorageError(f"Failed to list edges: {str(e)}")

    async def get_edges_by_type(self, relation_type: RelationType) -> List[Edge]:
        """
        Get all edges of a specific type.

        Args:
            relation_type: Type of edges to retrieve

        Returns:
            List of edges of the specified type
        """
        return await self.list_edges(filters={"relation_type": relation_type})

    async def get_edges_for_node(self, node_name: str, as_source: bool = True) -> List[Edge]:
        """
        Get all edges where the given node is either source or target.

        Args:
            node_name: Name of the node
            as_source: If True, get edges where node is source,
                      if False, get edges where node is target

        Returns:
            List of edges involving the specified node
        """
        field = "from_entity" if as_source else "to_entity"
        return await self.list_edges(filters={field: node_name})
