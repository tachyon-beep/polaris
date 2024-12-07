"""
JSON filesystem storage plugin.

This package provides implementations of NodeStoragePlugin and EdgeStoragePlugin
that store data in JSON files on the filesystem. Each implementation maintains
its own JSON file for persistence.

The implementation provides:
- Full CRUD operations for nodes and edges
- Filtering and pagination support
- Proper handling of datetime serialization/deserialization
- Atomic file operations to prevent data corruption
"""

from .constants import EDGES_JSON, NODES_JSON
from .storage import JsonEdgeStorage, JsonNodeStorage

__all__ = [
    "JsonNodeStorage",
    "JsonEdgeStorage",
    "NODES_JSON",
    "EDGES_JSON",
]
