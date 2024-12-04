"""
Storage service implementation for the knowledge graph server.

This module provides a high-level storage service that coordinates node and edge
storage operations. It offers:
- Pluggable storage backends (JSON, SQLite)
- Unified interface for node and edge operations
- Backup and restore functionality
- Resource cleanup and initialization
- Error handling and logging

The storage service acts as a facade over the storage implementation details,
providing a clean and consistent API for the rest of the system to interact with
persistent storage.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from ...core.enums import RelationType
from ...core.exceptions import StorageError
from ...core.models import Edge, Node
from .plugins import EdgeStoragePlugin, JsonEdgeStorage, JsonNodeStorage, NodeStoragePlugin
from .plugins.sqlite import SqliteEdgeStorage, SqliteNodeStorage

logger = logging.getLogger(__name__)

StorageType = Literal["json", "sqlite"]


class StorageService:
    """
    Server-side storage service for the knowledge graph.

    This class provides a unified interface for persistent storage operations,
    handling both nodes and edges. It supports multiple storage backends
    and provides comprehensive data management capabilities.

    The service manages:
    - Node lifecycle (creation, retrieval, updates, deletion)
    - Edge lifecycle (creation, retrieval, updates, deletion)
    - Storage backend configuration
    - Backup and restore operations
    - Resource initialization and cleanup

    Attributes:
        storage_dir (str): Directory for persistent storage
        storage_type (StorageType): Type of storage backend in use
        node_storage (NodeStoragePlugin): Plugin for node storage operations
        edge_storage (EdgeStoragePlugin): Plugin for edge storage operations
    """

    def __init__(self, storage_dir: str = "data", storage_type: StorageType = "json"):
        """
        Initialize the storage service.

        Args:
            storage_dir: Directory for persistent storage
            storage_type: Type of storage backend to use ("json" or "sqlite")

        Raises:
            ValueError: If an unsupported storage type is specified
        """
        os.makedirs(storage_dir, exist_ok=True)
        self.storage_dir = storage_dir
        self.storage_type = storage_type

        # Initialize appropriate storage plugins
        if storage_type == "json":
            self.node_storage: NodeStoragePlugin = JsonNodeStorage(storage_dir)
            self.edge_storage: EdgeStoragePlugin = JsonEdgeStorage(storage_dir)
        elif storage_type == "sqlite":
            self.node_storage = SqliteNodeStorage(storage_dir)
            self.edge_storage = SqliteEdgeStorage(storage_dir)
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")

    async def initialize(self) -> None:
        """
        Initialize the storage service and load existing data.

        This method must be called after creating a storage service instance
        to ensure all storage plugins are properly initialized and existing
        data is loaded.
        """
        await self.node_storage.initialize()
        await self.edge_storage.initialize()

    async def cleanup(self) -> None:
        """
        Clean up storage resources.

        This method should be called when shutting down the service to ensure
        proper resource cleanup and data persistence.
        """
        await self.node_storage.cleanup()
        await self.edge_storage.cleanup()

    async def backup(self, backup_dir: Optional[str] = None) -> None:
        """
        Create a backup of the current storage.

        Creates a timestamped backup of all storage data. If no backup directory
        is specified, creates one using the current timestamp.

        Args:
            backup_dir: Optional directory for backup files. If not provided,
                       a timestamped directory will be created.

        Raises:
            StorageError: If backup operation fails
        """
        if not backup_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = f"{self.storage_dir}_backup_{timestamp}"

        os.makedirs(backup_dir, exist_ok=True)

        await self.node_storage.backup(backup_dir)
        await self.edge_storage.backup(backup_dir)
        logger.info(f"Created backup in directory: {backup_dir}")

    async def restore_backup(self, backup_dir: str) -> None:
        """
        Restore from a backup directory.

        Restores both node and edge data from a backup directory.

        Args:
            backup_dir: Directory containing backup files

        Raises:
            StorageError: If backup directory doesn't exist or restore fails
        """
        if not os.path.exists(backup_dir):
            raise StorageError(f"Backup directory not found: {backup_dir}")

        await self.node_storage.restore_from_backup(backup_dir)
        await self.edge_storage.restore_from_backup(backup_dir)
        logger.info(f"Successfully restored from backup: {backup_dir}")

    # Node operations
    async def create_node(self, node: Node) -> Node:
        """
        Create a new node.

        Args:
            node: Node to create

        Returns:
            Created node with updated metadata

        Raises:
            StorageError: If node creation fails
        """
        return await self.node_storage.create_node(node)

    async def get_node(self, name: str) -> Node:
        """
        Get a node by name.

        Args:
            name: Name of the node to retrieve

        Returns:
            Retrieved node

        Raises:
            NodeNotFoundError: If node doesn't exist
            StorageError: If retrieval fails
        """
        return await self.node_storage.get_node(name)

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
        return await self.node_storage.node_exists(name)

    async def update_node(self, node: Node) -> Node:
        """
        Update an existing node.

        Args:
            node: Node with updated data

        Returns:
            Updated node

        Raises:
            NodeNotFoundError: If node doesn't exist
            StorageError: If update fails
        """
        return await self.node_storage.update_node(node)

    async def delete_node(self, name: str) -> None:
        """
        Delete a node and its edges.

        Args:
            name: Name of the node to delete

        Raises:
            NodeNotFoundError: If node doesn't exist
            StorageError: If deletion fails
        """
        await self.node_storage.delete_node(name)

    async def list_nodes(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Node]:
        """
        List nodes with optional filtering.

        Args:
            filters: Optional dictionary of filters to apply
            limit: Maximum number of nodes to return
            offset: Number of nodes to skip

        Returns:
            List of nodes matching the filters

        Raises:
            StorageError: If listing fails or pagination parameters are invalid
        """
        return await self.node_storage.list_nodes(filters, limit, offset)

    async def get_nodes_by_type(self, entity_type: str) -> List[Node]:
        """
        Get all nodes of a specific type.

        Args:
            entity_type: Type of nodes to retrieve

        Returns:
            List of nodes of the specified type

        Raises:
            StorageError: If retrieval fails
        """
        return await self.node_storage.get_nodes_by_type(entity_type)

    async def get_nodes_with_dependencies(self, name: str) -> Dict[str, Node]:
        """
        Get a node and all its dependencies.

        Args:
            name: Name of the root node

        Returns:
            Dictionary mapping node names to Node objects for the root
            node and all its dependencies

        Raises:
            NodeNotFoundError: If root node doesn't exist
            StorageError: If dependency resolution fails
        """
        return await self.node_storage.get_nodes_with_dependencies(name)

    # Edge operations
    async def create_edge(self, edge: Edge) -> Edge:
        """
        Create a new edge.

        Args:
            edge: Edge to create

        Returns:
            Created edge with updated metadata

        Raises:
            StorageError: If edge creation fails
        """
        return await self.edge_storage.create_edge(edge)

    async def get_edge(self, from_node: str, to_node: str, relation_type: RelationType) -> Edge:
        """
        Get an edge by its components.

        Args:
            from_node: Source node name
            to_node: Target node name
            relation_type: Type of relation

        Returns:
            Retrieved edge

        Raises:
            EdgeNotFoundError: If edge doesn't exist
            StorageError: If retrieval fails
        """
        return await self.edge_storage.get_edge(from_node, to_node, relation_type)

    async def update_edge(self, edge: Edge) -> Edge:
        """
        Update an existing edge.

        Args:
            edge: Edge with updated data

        Returns:
            Updated edge

        Raises:
            EdgeNotFoundError: If edge doesn't exist
            StorageError: If update fails
        """
        return await self.edge_storage.update_edge(edge)

    async def delete_edge(self, from_node: str, to_node: str, relation_type: RelationType) -> None:
        """
        Delete an edge.

        Args:
            from_node: Source node name
            to_node: Target node name
            relation_type: Type of relation

        Raises:
            EdgeNotFoundError: If edge doesn't exist
            StorageError: If deletion fails
        """
        await self.edge_storage.delete_edge(from_node, to_node, relation_type)

    async def list_edges(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Edge]:
        """
        List edges with optional filtering.

        Args:
            filters: Optional dictionary of filters to apply
            limit: Maximum number of edges to return
            offset: Number of edges to skip

        Returns:
            List of edges matching the filters

        Raises:
            StorageError: If listing fails or pagination parameters are invalid
        """
        return await self.edge_storage.list_edges(filters, limit, offset)

    async def get_edges_by_type(self, relation_type: RelationType) -> List[Edge]:
        """
        Get all edges of a specific type.

        Args:
            relation_type: Type of edges to retrieve

        Returns:
            List of edges of the specified type

        Raises:
            StorageError: If retrieval fails
        """
        return await self.edge_storage.get_edges_by_type(relation_type)

    async def get_edges_for_node(self, node_name: str, as_source: bool = True) -> List[Edge]:
        """
        Get all edges where the given node is either source or target.

        Args:
            node_name: Name of the node
            as_source: If True, get edges where node is source,
                      if False, get edges where node is target

        Returns:
            List of edges involving the specified node

        Raises:
            StorageError: If retrieval fails
        """
        return await self.edge_storage.get_edges_for_node(node_name, as_source)
