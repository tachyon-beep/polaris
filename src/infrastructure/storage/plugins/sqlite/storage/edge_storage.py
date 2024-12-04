"""SQLite implementation for edge storage."""

import logging
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiosqlite

from ......core.enums import RelationType
from ......core.exceptions import EdgeNotFoundError, StorageError
from ......core.models import Edge
from ...base import EdgeStoragePlugin
from ...utils import validate_pagination
from ..constants import (
    DELETE_EDGE,
    EDGE_SCHEMA,
    INSERT_EDGE,
    SELECT_EDGE,
    SELECT_ONE_EDGE,
    STORAGEDB,
    UPDATE_EDGE,
)
from ..utils import (
    add_pagination,
    backup_database,
    build_edge_filter_query,
    edge_to_row,
    initialize_table,
    restore_database,
    row_to_edge,
)

logger = logging.getLogger(__name__)


class SqliteEdgeStorage(EdgeStoragePlugin):
    """
    SQLite implementation for edge storage.

    This class implements the EdgeStoragePlugin interface using SQLite for persistence.
    It provides efficient querying through SQL and ensures data integrity through
    transactions.

    Attributes:
        storage_dir (str): Directory where SQLite database is stored
        db_path (str): Full path to SQLite database file
    """

    def __init__(self, storage_dir: str):
        """
        Initialize SQLite edge storage.

        Args:
            storage_dir: Directory for storing SQLite database
        """
        super().__init__(storage_dir)
        self.db_path = os.path.join(storage_dir, STORAGEDB)

    async def initialize(self) -> None:
        """
        Initialize storage and create tables.

        Creates the edges table if it doesn't exist.

        Raises:
            StorageError: If initialization fails
        """
        await initialize_table(self.db_path, EDGE_SCHEMA)
        logger.info("Initialized SQLite edge storage")

    async def cleanup(self) -> None:
        """Clean up resources."""
        pass  # SQLite connection is managed per operation

    async def backup(self, backup_dir: str) -> None:
        """
        Create a backup of edge storage.

        Args:
            backup_dir: Directory to store backup

        Raises:
            StorageError: If backup fails
        """
        backup_database(self.db_path, backup_dir, STORAGEDB)

    async def restore_from_backup(self, backup_dir: str) -> None:
        """
        Restore edges from backup.

        Args:
            backup_dir: Directory containing backup

        Raises:
            StorageError: If restore fails
        """
        restore_database(backup_dir, self.db_path, STORAGEDB)

    async def create_edge(self, edge: Edge) -> Edge:
        """Create a new edge. See base class for details."""
        try:
            edge.metadata.created_at = datetime.now()
            edge.metadata.last_modified = datetime.now()

            async with aiosqlite.connect(self.db_path) as db:
                try:
                    await db.execute(INSERT_EDGE, edge_to_row(edge))
                    await db.commit()
                except sqlite3.IntegrityError:
                    raise StorageError(
                        f"Edge already exists: {edge.from_entity} -> {edge.to_entity}"
                    )

            return edge

        except Exception as e:
            logger.error(f"Failed to create edge: {str(e)}")
            raise StorageError(f"Failed to create edge: {str(e)}")

    async def get_edge(self, from_entity: str, to_entity: str, relation_type: RelationType) -> Edge:
        """Get an edge by its components. See base class for details."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(
                    SELECT_EDGE,
                    (from_entity, to_entity, relation_type.value),
                ) as cursor:
                    row = await cursor.fetchone()
                    if not row:
                        raise EdgeNotFoundError(f"Edge not found: {from_entity} -> {to_entity}")
                    return row_to_edge(row)
        except Exception as e:
            logger.error(f"Failed to get edge: {str(e)}")
            raise StorageError(f"Failed to get edge: {str(e)}")

    async def update_edge(self, edge: Edge) -> Edge:
        """Update an existing edge. See base class for details."""
        try:
            edge.metadata.last_modified = datetime.now()

            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(
                    SELECT_ONE_EDGE,
                    (
                        edge.from_entity,
                        edge.to_entity,
                        edge.relation_type.value,
                    ),
                ) as cursor:
                    if not await cursor.fetchone():
                        raise EdgeNotFoundError(
                            f"Edge not found: {edge.from_entity} -> {edge.to_entity}"
                        )

                await db.execute(
                    UPDATE_EDGE,
                    edge_to_row(edge)[3:]
                    + (
                        edge.from_entity,
                        edge.to_entity,
                        edge.relation_type.value,
                    ),
                )
                await db.commit()

            return edge

        except Exception as e:
            logger.error(f"Failed to update edge: {str(e)}")
            raise StorageError(f"Failed to update edge: {str(e)}")

    async def delete_edge(
        self, from_entity: str, to_entity: str, relation_type: RelationType
    ) -> None:
        """Delete an edge. See base class for details."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(
                    SELECT_ONE_EDGE,
                    (from_entity, to_entity, relation_type.value),
                ) as cursor:
                    if not await cursor.fetchone():
                        raise EdgeNotFoundError(f"Edge not found: {from_entity} -> {to_entity}")

                await db.execute(
                    DELETE_EDGE,
                    (from_entity, to_entity, relation_type.value),
                )
                await db.commit()

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

            query, params = build_edge_filter_query(filters or {})
            query, params = add_pagination(query, limit, offset)

            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(query, params) as cursor:
                    return [row_to_edge(row) async for row in cursor]

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
        field = "from_node" if as_source else "to_node"
        return await self.list_edges(filters={field: node_name})
