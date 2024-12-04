"""SQLite implementation for node storage."""

import logging
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiosqlite

from ......core.exceptions import NodeNotFoundError, StorageError
from ......core.models import Node
from ...base import NodeStoragePlugin
from ...utils import validate_pagination
from ..constants import (
    DELETE_NODE,
    INSERT_NODE,
    NODE_SCHEMA,
    SELECT_NODE_BY_NAME,
    SELECT_ONE_BY_NAME,
    STORAGEDB,
    UPDATE_NODE,
)
from ..utils import (
    add_pagination,
    backup_database,
    build_node_filter_query,
    initialize_table,
    node_to_row,
    restore_database,
    row_to_node,
)

logger = logging.getLogger(__name__)


class SqliteNodeStorage(NodeStoragePlugin):
    """
    SQLite implementation for node storage.

    This class implements the NodeStoragePlugin interface using SQLite for persistence.
    It provides efficient querying through SQL and ensures data integrity through
    transactions.

    Attributes:
        storage_dir (str): Directory where SQLite database is stored
        db_path (str): Full path to SQLite database file
    """

    def __init__(self, storage_dir: str):
        """
        Initialize SQLite node storage.

        Args:
            storage_dir: Directory for storing SQLite database
        """
        super().__init__(storage_dir)
        self.db_path = os.path.join(storage_dir, STORAGEDB)

    async def initialize(self) -> None:
        """
        Initialize storage and create tables.

        Creates the nodes table if it doesn't exist.

        Raises:
            StorageError: If initialization fails
        """
        await initialize_table(self.db_path, NODE_SCHEMA)
        logger.info("Initialized SQLite node storage")

    async def cleanup(self) -> None:
        """Clean up resources."""
        pass  # SQLite connection is managed per operation

    async def backup(self, backup_dir: str) -> None:
        """
        Create a backup of node storage.

        Args:
            backup_dir: Directory to store backup

        Raises:
            StorageError: If backup fails
        """
        backup_database(self.db_path, backup_dir, STORAGEDB)

    async def restore_from_backup(self, backup_dir: str) -> None:
        """
        Restore nodes from backup.

        Args:
            backup_dir: Directory containing backup

        Raises:
            StorageError: If restore fails
        """
        restore_database(backup_dir, self.db_path, STORAGEDB)

    async def create_node(self, node: Node) -> Node:
        """Create a new node. See base class for details."""
        try:
            node.metadata.created_at = datetime.now()
            node.metadata.last_modified = datetime.now()

            async with aiosqlite.connect(self.db_path) as db:
                try:
                    await db.execute(INSERT_NODE, node_to_row(node))
                    await db.commit()
                except sqlite3.IntegrityError:
                    raise StorageError(f"Node already exists: {node.name}")

            return node

        except Exception as e:
            logger.error(f"Failed to create node: {str(e)}")
            raise StorageError(f"Failed to create node: {str(e)}")

    async def get_node(self, name: str) -> Node:
        """Get a node by name. See base class for details."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(SELECT_NODE_BY_NAME, (name,)) as cursor:
                    row = await cursor.fetchone()
                    if not row:
                        raise NodeNotFoundError(f"Node not found: {name}")
                    return row_to_node(row)
        except Exception as e:
            logger.error(f"Failed to get node: {str(e)}")
            raise StorageError(f"Failed to get node: {str(e)}")

    async def node_exists(self, name: str) -> bool:
        """Check if a node exists. See base class for details."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(SELECT_ONE_BY_NAME, (name,)) as cursor:
                    return await cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"Failed to check node existence: {str(e)}")
            raise StorageError(f"Failed to check node existence: {str(e)}")

    async def update_node(self, node: Node) -> Node:
        """Update an existing node. See base class for details."""
        try:
            node.metadata.last_modified = datetime.now()

            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(SELECT_ONE_BY_NAME, (node.name,)) as cursor:
                    if not await cursor.fetchone():
                        raise NodeNotFoundError(f"Node not found: {node.name}")

                await db.execute(
                    UPDATE_NODE,
                    node_to_row(node)[1:] + (node.name,),
                )
                await db.commit()

            return node

        except Exception as e:
            logger.error(f"Failed to update node: {str(e)}")
            raise StorageError(f"Failed to update node: {str(e)}")

    async def delete_node(self, name: str) -> None:
        """Delete a node. See base class for details."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(SELECT_ONE_BY_NAME, (name,)) as cursor:
                    if not await cursor.fetchone():
                        raise NodeNotFoundError(f"Node not found: {name}")

                await db.execute(DELETE_NODE, (name,))
                await db.commit()

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

            query, params = build_node_filter_query(filters or {})
            query, params = add_pagination(query, limit, offset)

            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(query, params) as cursor:
                    return [row_to_node(row) async for row in cursor]

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
            result = {}
            to_process = {name}
            processed = set()

            while to_process:
                current = to_process.pop()
                if current in processed:
                    continue

                try:
                    node = await self.get_node(current)
                    result[current] = node
                    processed.add(current)
                    to_process.update(dep for dep in node.dependencies if dep not in processed)
                except NodeNotFoundError:
                    continue

            return result

        except Exception as e:
            logger.error(f"Failed to get node dependencies: {str(e)}")
            raise StorageError(f"Failed to get node dependencies: {str(e)}")
