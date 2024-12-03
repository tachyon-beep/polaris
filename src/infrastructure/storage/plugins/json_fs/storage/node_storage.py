"""JSON filesystem implementation for node storage."""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from ......core.exceptions import NodeNotFoundError, StorageError
from ......core.models import Node, NodeMetadata
from ...base import NodeStoragePlugin
from ...utils import validate_pagination
from ..constants import NODES_JSON
from ..utils.filtering import node_matches_filters
from ..utils.persistence import backup_file, load_json_file, save_json_file

logger = logging.getLogger(__name__)


class JsonNodeStorage(NodeStoragePlugin):
    """
    JSON filesystem implementation for node storage.

    This class implements the NodeStoragePlugin interface using JSON files
    for persistence. All nodes are stored in a single JSON file with their
    names as keys.

    Attributes:
        storage_dir (str): Directory where JSON files are stored
        nodes_file (str): Path to the nodes JSON file
        _nodes (Dict[str, Node]): In-memory cache of nodes
    """

    def __init__(self, storage_dir: str):
        """
        Initialize JSON node storage.

        Args:
            storage_dir: Directory for storing JSON files
        """
        super().__init__(storage_dir)
        self._nodes: Dict[str, Node] = {}
        self.nodes_file = os.path.join(storage_dir, NODES_JSON)

    async def initialize(self) -> None:
        """
        Initialize storage and load existing data.

        Loads any existing nodes from the JSON file into memory.
        Creates a new empty file if none exists.

        Raises:
            StorageError: If loading fails
        """
        try:
            nodes_data = await load_json_file(self.nodes_file)
            for name, data in nodes_data.items():
                metadata = NodeMetadata(**data.pop("metadata"))
                self._nodes[name] = Node(**data, metadata=metadata)
            logger.info(f"Loaded {len(self._nodes)} nodes from storage")
        except Exception as e:
            logger.error(f"Failed to load nodes: {str(e)}")
            raise StorageError(f"Failed to load nodes: {str(e)}")

    async def cleanup(self) -> None:
        """Clean up resources by clearing in-memory cache."""
        self._nodes.clear()

    async def backup(self, backup_dir: str) -> None:
        """
        Create a backup of node storage.

        Args:
            backup_dir: Directory to store backup

        Raises:
            StorageError: If backup fails
        """
        backup_file(self.nodes_file, backup_dir, NODES_JSON)

    async def restore_from_backup(self, backup_dir: str) -> None:
        """
        Restore nodes from backup.

        Args:
            backup_dir: Directory containing backup

        Raises:
            StorageError: If restore fails
        """
        try:
            backup_path = os.path.join(backup_dir, NODES_JSON)
            if os.path.exists(backup_path):
                self._nodes.clear()
                os.replace(backup_path, self.nodes_file)
                await self.initialize()
        except Exception as e:
            logger.error(f"Failed to restore nodes: {str(e)}")
            raise StorageError(f"Failed to restore nodes: {str(e)}")

    async def create_node(self, node: Node) -> Node:
        """Create a new node. See base class for details."""
        try:
            if node.name in self._nodes:
                raise StorageError(f"Node already exists: {node.name}")

            node.metadata.created_at = datetime.now()
            node.metadata.last_modified = datetime.now()

            self._nodes[node.name] = node
            await save_json_file(self.nodes_file, self._nodes)

            return node

        except Exception as e:
            logger.error(f"Failed to create node: {str(e)}")
            raise StorageError(f"Failed to create node: {str(e)}")

    async def get_node(self, name: str) -> Node:
        """Get a node by name. See base class for details."""
        try:
            node = self._nodes.get(name)
            if not node:
                raise NodeNotFoundError(f"Node not found: {name}")
            return node
        except Exception as e:
            logger.error(f"Failed to get node: {str(e)}")
            raise StorageError(f"Failed to get node: {str(e)}")

    async def node_exists(self, name: str) -> bool:
        """Check if a node exists. See base class for details."""
        return name in self._nodes

    async def update_node(self, node: Node) -> Node:
        """Update an existing node. See base class for details."""
        try:
            if node.name not in self._nodes:
                raise NodeNotFoundError(f"Node not found: {node.name}")

            node.metadata.last_modified = datetime.now()
            self._nodes[node.name] = node
            await save_json_file(self.nodes_file, self._nodes)

            return node

        except Exception as e:
            logger.error(f"Failed to update node: {str(e)}")
            raise StorageError(f"Failed to update node: {str(e)}")

    async def delete_node(self, name: str) -> None:
        """Delete a node. See base class for details."""
        try:
            if name not in self._nodes:
                raise NodeNotFoundError(f"Node not found: {name}")

            del self._nodes[name]
            await save_json_file(self.nodes_file, self._nodes)

        except Exception as e:
            logger.error(f"Failed to delete node: {str(e)}")
            raise StorageError(f"Failed to delete node: {str(e)}")

    async def list_nodes(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Node]:
        """List nodes with optional filtering. See base class for details."""
        try:
            validate_pagination(offset, limit)

            nodes = list(self._nodes.values())

            if filters:
                nodes = [node for node in nodes if node_matches_filters(node, filters)]

            return nodes[offset : offset + limit]

        except ValueError as e:
            raise StorageError(f"Invalid pagination parameters: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to list nodes: {str(e)}")
            raise StorageError(f"Failed to list nodes: {str(e)}")

    async def get_nodes_by_type(self, node_type: str) -> List[Node]:
        """Get all nodes of a specific type. See base class for details."""
        return await self.list_nodes(filters={"entity_type": node_type})

    async def get_nodes_with_dependencies(self, name: str) -> Dict[str, Node]:
        """Get a node and all its dependencies. See base class for details."""
        try:
            if name not in self._nodes:
                raise NodeNotFoundError(f"Node not found: {name}")

            result = {}
            to_process = [name]
            processed = set()

            while to_process:
                current = to_process.pop()
                if current in processed:
                    continue

                node = self._nodes.get(current)
                if node:
                    result[current] = node
                    processed.add(current)
                    to_process.extend(dep for dep in node.dependencies if dep not in processed)

            return result

        except Exception as e:
            logger.error(f"Failed to get node dependencies: {str(e)}")
            raise StorageError(f"Failed to get node dependencies: {str(e)}")
