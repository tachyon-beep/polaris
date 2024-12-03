"""
Storage plugins package.

This package provides various storage implementations:
- JSON filesystem storage
- SQLite storage
- Event store

Each plugin implements the base interfaces defined in the base package:
- StoragePlugin: Generic base interface
- NodeStoragePlugin: Interface for node storage
- EdgeStoragePlugin: Interface for edge storage

The plugins package also provides common utilities used across implementations:
- Serialization helpers
- Validation functions
- Common constants
"""

from .base import EdgeStoragePlugin, NodeStoragePlugin, StoragePlugin
from .event_store import Event, EventStore, EventType
from .json_fs import JsonEdgeStorage, JsonNodeStorage
from .sqlite import SqliteEdgeStorage, SqliteNodeStorage
from .utils import (
    deserialize_datetime,
    get_nested_attr,
    serialize_datetime,
    to_dict,
    validate_pagination,
)

__all__ = [
    # Base interfaces
    "StoragePlugin",
    "NodeStoragePlugin",
    "EdgeStoragePlugin",
    # JSON filesystem implementation
    "JsonNodeStorage",
    "JsonEdgeStorage",
    # SQLite implementation
    "SqliteNodeStorage",
    "SqliteEdgeStorage",
    # Event store
    "EventStore",
    "Event",
    "EventType",
    # Utilities
    "serialize_datetime",
    "deserialize_datetime",
    "get_nested_attr",
    "validate_pagination",
    "to_dict",
]
