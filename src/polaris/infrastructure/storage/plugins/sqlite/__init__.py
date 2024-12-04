"""
SQLite storage plugin.

This package provides SQLite-based implementations of NodeStoragePlugin and EdgeStoragePlugin.
The implementation uses aiosqlite for async database operations and provides:
- Full CRUD operations for nodes and edges
- SQL-based filtering and pagination
- Proper handling of JSON serialization/deserialization
- Atomic transactions for data integrity
"""

from .constants import STORAGEDB
from .storage import SqliteEdgeStorage, SqliteNodeStorage

__all__ = [
    "SqliteNodeStorage",
    "SqliteEdgeStorage",
    "STORAGEDB",
]
