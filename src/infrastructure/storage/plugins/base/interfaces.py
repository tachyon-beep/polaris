"""
Core interfaces for storage plugins.

This module defines the base interfaces that all storage plugins must implement:
- StoragePlugin: Generic base interface with core storage operations
- NodeStoragePlugin: Interface for node-specific storage operations
- EdgeStoragePlugin: Interface for edge-specific storage operations

These interfaces ensure consistent behavior across different storage implementations
while allowing flexibility in how the actual storage is handled.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, TypeVar

from .....core.enums import RelationType
from .....core.models import Edge, Node

T = TypeVar("T")


class StoragePlugin(Generic[T], ABC):
    """
    Abstract base class for storage plugins.

    This class defines the core interface that all storage plugins must implement,
    providing basic storage operations like initialization, cleanup, backup and restore.

    Attributes:
        storage_dir (str): Base directory for storage operations
    """

    def __init__(self, storage_dir: str):
        """
        Initialize the storage plugin.

        Args:
            storage_dir: Directory for storage operations
        """
        self.storage_dir = storage_dir

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the storage plugin.

        This method should handle any setup required before the plugin can be used,
        such as creating directories or initializing databases.

        Raises:
            StorageError: If initialization fails
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """
        Clean up any resources used by the plugin.

        This method should handle proper cleanup of any resources allocated by the plugin,
        such as closing database connections or file handles.

        Raises:
            StorageError: If cleanup fails
        """
        pass

    @abstractmethod
    async def backup(self, backup_dir: str) -> None:
        """
        Create a backup of the current storage.

        Args:
            backup_dir: Directory where backup should be stored

        Raises:
            StorageError: If backup operation fails
        """
        pass

    @abstractmethod
    async def restore_from_backup(self, backup_dir: str) -> None:
        """
        Restore storage from a backup directory.

        Args:
            backup_dir: Directory containing the backup to restore from

        Raises:
            StorageError: If restore operation fails
        """
        pass


class NodeStoragePlugin(StoragePlugin[Node]):
    """
    Abstract base class for node storage plugins.

    This class extends StoragePlugin to provide node-specific storage operations.
    It defines the interface for creating, reading, updating, and deleting nodes,
    as well as querying nodes with various filters.
    """

    @abstractmethod
    async def create_node(self, node: Node) -> Node:
        """
        Create a new node in storage.

        Args:
            node: Node instance to create

        Returns:
            Created node instance with any storage-specific modifications

        Raises:
            StorageError: If node creation fails or node already exists
        """
        pass

    @abstractmethod
    async def get_node(self, name: str) -> Node:
        """
        Get a node by name.

        Args:
            name: Unique name of the node to retrieve

        Returns:
            Retrieved node instance

        Raises:
            NodeNotFoundError: If node does not exist
            StorageError: If retrieval fails
        """
        pass

    @abstractmethod
    async def node_exists(self, name: str) -> bool:
        """
        Check if a node exists.

        Args:
            name: Name of node to check

        Returns:
            True if node exists, False otherwise

        Raises:
            StorageError: If check fails
        """
        pass

    @abstractmethod
    async def update_node(self, node: Node) -> Node:
        """
        Update an existing node.

        Args:
            node: Node instance with updated data

        Returns:
            Updated node instance

        Raises:
            NodeNotFoundError: If node does not exist
            StorageError: If update fails
        """
        pass

    @abstractmethod
    async def delete_node(self, name: str) -> None:
        """
        Delete a node.

        Args:
            name: Name of node to delete

        Raises:
            NodeNotFoundError: If node does not exist
            StorageError: If deletion fails
        """
        pass

    @abstractmethod
    async def list_nodes(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Node]:
        """
        List nodes with optional filtering.

        Args:
            filters: Optional dictionary of filter criteria
            limit: Maximum number of nodes to return
            offset: Number of nodes to skip

        Returns:
            List of nodes matching criteria

        Raises:
            StorageError: If listing fails
        """
        pass

    @abstractmethod
    async def get_nodes_by_type(self, node_type: str) -> List[Node]:
        """
        Get all nodes of a specific type.

        Args:
            node_type: Type of nodes to retrieve

        Returns:
            List of nodes of specified type

        Raises:
            StorageError: If retrieval fails
        """
        pass

    @abstractmethod
    async def get_nodes_with_dependencies(self, name: str) -> Dict[str, Node]:
        """
        Get a node and all its dependencies.

        Args:
            name: Name of root node

        Returns:
            Dictionary mapping node names to Node instances for root node and all dependencies

        Raises:
            NodeNotFoundError: If root node does not exist
            StorageError: If retrieval fails
        """
        pass


class EdgeStoragePlugin(StoragePlugin[Edge]):
    """
    Abstract base class for edge storage plugins.

    This class extends StoragePlugin to provide edge-specific storage operations.
    It defines the interface for creating, reading, updating, and deleting edges,
    as well as querying edges with various filters.
    """

    @abstractmethod
    async def create_edge(self, edge: Edge) -> Edge:
        """
        Create a new edge.

        Args:
            edge: Edge instance to create

        Returns:
            Created edge instance with any storage-specific modifications

        Raises:
            StorageError: If edge creation fails or edge already exists
        """
        pass

    @abstractmethod
    async def get_edge(self, from_entity: str, to_entity: str, relation_type: RelationType) -> Edge:
        """
        Get an edge by its components.

        Args:
            from_entity: Source node name
            to_entity: Target node name
            relation_type: Type of relationship

        Returns:
            Retrieved edge instance

        Raises:
            EdgeNotFoundError: If edge does not exist
            StorageError: If retrieval fails
        """
        pass

    @abstractmethod
    async def update_edge(self, edge: Edge) -> Edge:
        """
        Update an existing edge.

        Args:
            edge: Edge instance with updated data

        Returns:
            Updated edge instance

        Raises:
            EdgeNotFoundError: If edge does not exist
            StorageError: If update fails
        """
        pass

    @abstractmethod
    async def delete_edge(
        self, from_entity: str, to_entity: str, relation_type: RelationType
    ) -> None:
        """
        Delete an edge.

        Args:
            from_entity: Source node name
            to_entity: Target node name
            relation_type: Type of relationship

        Raises:
            EdgeNotFoundError: If edge does not exist
            StorageError: If deletion fails
        """
        pass

    @abstractmethod
    async def list_edges(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Edge]:
        """
        List edges with optional filtering.

        Args:
            filters: Optional dictionary of filter criteria
            limit: Maximum number of edges to return
            offset: Number of edges to skip

        Returns:
            List of edges matching criteria

        Raises:
            StorageError: If listing fails
        """
        pass

    @abstractmethod
    async def get_edges_by_type(self, relation_type: RelationType) -> List[Edge]:
        """
        Get all edges of a specific relationship type.

        Args:
            relation_type: Type of relationship to retrieve

        Returns:
            List of edges of specified relationship type

        Raises:
            StorageError: If retrieval fails
        """
        pass

    @abstractmethod
    async def get_edges_for_node(self, node_name: str, as_source: bool = True) -> List[Edge]:
        """
        Get all edges where the given node is either source or target.

        Args:
            node_name: Name of node to get edges for
            as_source: If True, get edges where node is source, else target

        Returns:
            List of edges connected to node

        Raises:
            StorageError: If retrieval fails
        """
        pass
