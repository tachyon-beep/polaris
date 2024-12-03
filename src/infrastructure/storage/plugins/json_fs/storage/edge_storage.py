"""JSON filesystem implementation for edge storage."""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from ......core.enums import RelationType
from ......core.exceptions import EdgeNotFoundError, StorageError
from ......core.models import Edge, EdgeMetadata
from ...base import EdgeStoragePlugin
from ...utils import validate_pagination
from ..constants import EDGES_JSON
from ..utils.filtering import edge_matches_filters
from ..utils.keys import get_edge_key
from ..utils.persistence import backup_file, load_json_file, save_json_file

logger = logging.getLogger(__name__)


class JsonEdgeStorage(EdgeStoragePlugin):
    """
    JSON filesystem implementation for edge storage.

    This class implements the EdgeStoragePlugin interface using JSON files
    for persistence. All edges are stored in a single JSON file with composite
    keys based on their components.

    Attributes:
        storage_dir (str): Directory where JSON files are stored
        edges_file (str): Path to the edges JSON file
        _edges (Dict[str, Edge]): In-memory cache of edges
    """

    def __init__(self, storage_dir: str):
        """
        Initialize JSON edge storage.

        Args:
            storage_dir: Directory for storing JSON files
        """
        super().__init__(storage_dir)
        self._edges: Dict[str, Edge] = {}
        self.edges_file = os.path.join(storage_dir, EDGES_JSON)

    async def initialize(self) -> None:
        """
        Initialize storage and load existing data.

        Loads any existing edges from the JSON file into memory.
        Creates a new empty file if none exists.

        Raises:
            StorageError: If loading fails
        """
        try:
            edges_data = await load_json_file(self.edges_file)
            for key, data in edges_data.items():
                metadata = EdgeMetadata(**data.pop("metadata"))
                self._edges[key] = Edge(**data, metadata=metadata)
            logger.info(f"Loaded {len(self._edges)} edges from storage")
        except Exception as e:
            logger.error(f"Failed to load edges: {str(e)}")
            raise StorageError(f"Failed to load edges: {str(e)}")

    async def cleanup(self) -> None:
        """Clean up resources by clearing in-memory cache."""
        self._edges.clear()

    async def backup(self, backup_dir: str) -> None:
        """
        Create a backup of edge storage.

        Args:
            backup_dir: Directory to store backup

        Raises:
            StorageError: If backup fails
        """
        backup_file(self.edges_file, backup_dir, EDGES_JSON)

    async def restore_from_backup(self, backup_dir: str) -> None:
        """
        Restore edges from backup.

        Args:
            backup_dir: Directory containing backup

        Raises:
            StorageError: If restore fails
        """
        try:
            backup_path = os.path.join(backup_dir, EDGES_JSON)
            if os.path.exists(backup_path):
                self._edges.clear()
                os.replace(backup_path, self.edges_file)
                await self.initialize()
        except Exception as e:
            logger.error(f"Failed to restore edges: {str(e)}")
            raise StorageError(f"Failed to restore edges: {str(e)}")

    async def create_edge(self, edge: Edge) -> Edge:
        """Create a new edge. See base class for details."""
        try:
            key = get_edge_key(edge.from_entity, edge.to_entity, edge.relation_type)

            if key in self._edges:
                raise StorageError(f"Edge already exists: {key}")

            edge.metadata.created_at = datetime.now()
            edge.metadata.last_modified = datetime.now()

            self._edges[key] = edge
            await save_json_file(self.edges_file, self._edges)

            return edge

        except Exception as e:
            logger.error(f"Failed to create edge: {str(e)}")
            raise StorageError(f"Failed to create edge: {str(e)}")

    async def get_edge(self, from_entity: str, to_entity: str, relation_type: RelationType) -> Edge:
        """Get an edge by its components. See base class for details."""
        try:
            key = get_edge_key(from_entity, to_entity, relation_type)
            edge = self._edges.get(key)
            if not edge:
                raise EdgeNotFoundError(f"Edge not found: {key}")
            return edge
        except Exception as e:
            logger.error(f"Failed to get edge: {str(e)}")
            raise StorageError(f"Failed to get edge: {str(e)}")

    async def update_edge(self, edge: Edge) -> Edge:
        """Update an existing edge. See base class for details."""
        try:
            key = get_edge_key(edge.from_entity, edge.to_entity, edge.relation_type)

            if key not in self._edges:
                raise EdgeNotFoundError(f"Edge not found: {key}")

            edge.metadata.last_modified = datetime.now()
            self._edges[key] = edge
            await save_json_file(self.edges_file, self._edges)

            return edge

        except Exception as e:
            logger.error(f"Failed to update edge: {str(e)}")
            raise StorageError(f"Failed to update edge: {str(e)}")

    async def delete_edge(
        self, from_entity: str, to_entity: str, relation_type: RelationType
    ) -> None:
        """Delete an edge. See base class for details."""
        try:
            key = get_edge_key(from_entity, to_entity, relation_type)

            if key not in self._edges:
                raise EdgeNotFoundError(f"Edge not found: {key}")

            del self._edges[key]
            await save_json_file(self.edges_file, self._edges)

        except Exception as e:
            logger.error(f"Failed to delete edge: {str(e)}")
            raise StorageError(f"Failed to delete edge: {str(e)}")

    async def list_edges(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Edge]:
        """List edges with optional filtering. See base class for details."""
        try:
            validate_pagination(offset, limit)

            edges = list(self._edges.values())

            if filters:
                edges = [edge for edge in edges if edge_matches_filters(edge, filters)]

            return edges[offset : offset + limit]

        except ValueError as e:
            raise StorageError(f"Invalid pagination parameters: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to list edges: {str(e)}")
            raise StorageError(f"Failed to list edges: {str(e)}")

    async def get_edges_by_type(self, relation_type: RelationType) -> List[Edge]:
        """Get all edges of a specific type. See base class for details."""
        return await self.list_edges(filters={"relation_type": relation_type})

    async def get_edges_for_node(self, node_name: str, as_source: bool = True) -> List[Edge]:
        """Get all edges for a node. See base class for details."""
        field = "from_entity" if as_source else "to_entity"
        return await self.list_edges(filters={field: node_name})
