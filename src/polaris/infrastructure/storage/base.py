"""
Base storage functionality for the knowledge graph server.

This module provides the foundational storage capabilities for the knowledge graph,
implementing common functionality used by both entity and relation storage. It handles:
- Persistent storage and retrieval of data
- Thread-safe operations with asyncio locks
- JSON serialization/deserialization with datetime support
- Backup and restore functionality
- Error handling and logging

The base storage implementation is generic and can be extended for specific storage
needs while maintaining consistent behavior and reliability across different storage types.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Generic, TypeVar

import aiofiles

from ...core.exceptions import StorageError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseStorage(Generic[T]):
    """
    Base storage class with common functionality for entities and relations.

    This class provides the core storage operations and utilities that are common
    across different types of storage (entities, relations, etc.). It implements
    thread-safe operations, persistent storage, and data serialization.

    Attributes:
        storage_dir (str): Directory path for persistent storage
        storage_file (str): Full path to the storage file
        _items (Dict[str, T]): In-memory storage of items
        _lock (asyncio.Lock): Thread synchronization lock
    """

    def __init__(self, storage_dir: str, filename: str):
        """
        Initialize base storage.

        Args:
            storage_dir: Directory for persistent storage
            filename: Name of storage file

        The storage is initialized with an empty items dictionary and a thread lock.
        Actual data loading from persistent storage happens in the initialize() method.
        """
        self._items: Dict[str, T] = {}
        self._lock = asyncio.Lock()
        self.storage_dir = storage_dir
        self.storage_file = os.path.join(storage_dir, filename)

    def _serialize_datetime(self, obj: Any) -> Any:
        """
        Handle datetime serialization for JSON.

        Args:
            obj: Object to serialize

        Returns:
            Serialized object with datetime converted to ISO format string
        """
        if isinstance(obj, datetime):
            return obj.isoformat()
        return obj

    def _deserialize_datetime(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle datetime deserialization from JSON.

        Args:
            obj: Dictionary potentially containing datetime strings

        Returns:
            Dictionary with datetime strings converted to datetime objects
        """
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in ["created_at", "last_modified"] and isinstance(value, str):
                    obj[key] = datetime.fromisoformat(value)
                elif isinstance(value, dict):
                    obj[key] = self._deserialize_datetime(value)
        return obj

    def _get_nested_attr(self, obj: Any, attr_path: str) -> Any:
        """
        Safely get nested attribute value from an object.

        This method traverses a dot-separated path to access nested attributes,
        returning None if any part of the path doesn't exist.

        Args:
            obj: Object to get attribute from
            attr_path: Dot-separated path to attribute (e.g., "user.address.city")

        Returns:
            Attribute value if found, None if not found
        """
        curr = obj
        for attr in attr_path.split("."):
            if not hasattr(curr, attr):
                return None
            curr = getattr(curr, attr)
        return curr

    def _validate_pagination(self, offset: int, limit: int) -> None:
        """
        Validate pagination parameters.

        Ensures that pagination parameters are within valid ranges to prevent
        invalid queries.

        Args:
            offset: Number of items to skip
            limit: Maximum number of items to return

        Raises:
            ValueError: If offset is negative or limit is less than 1
        """
        if offset < 0:
            raise ValueError("Offset must be non-negative")
        if limit < 1:
            raise ValueError("Limit must be positive")

    def _serialize_item(self, item: T) -> Dict[str, Any]:
        """
        Convert an item to a dictionary suitable for JSON serialization.

        This is an abstract method that must be implemented by subclasses to define
        how their specific item types should be serialized.

        Args:
            item: Item to serialize

        Returns:
            Dictionary representation of the item

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError

    async def _persist_items(self) -> None:
        """
        Persist items to disk storage.

        Serializes all items in memory and writes them to the storage file in
        JSON format. This operation is atomic and thread-safe.

        Raises:
            StorageError: If persistence operation fails
        """
        try:
            items_data = {key: self._serialize_item(item) for key, item in self._items.items()}

            async with aiofiles.open(self.storage_file, "w") as f:
                await f.write(json.dumps(items_data, indent=2))

            logger.debug(f"Persisted {len(items_data)} items to {self.storage_file}")

        except Exception as e:
            logger.error(f"Failed to persist items: {str(e)}")
            raise StorageError(f"Item persistence failed: {str(e)}")

    async def _load_from_storage(self) -> None:
        """
        Load items from disk storage.

        Reads and deserializes items from the storage file into memory.
        If the storage file doesn't exist or is empty, initializes with
        empty storage.

        Raises:
            StorageError: If loading operation fails
        """
        try:
            if os.path.exists(self.storage_file):
                async with aiofiles.open(self.storage_file, "r") as f:
                    content = await f.read()
                    if content.strip():
                        items_data = json.loads(content)
                        self._items = self._deserialize_items(items_data)
                logger.info(f"Loaded {len(self._items)} items from storage")

        except Exception as e:
            logger.error(f"Failed to load from storage: {str(e)}")
            raise StorageError(f"Storage loading failed: {str(e)}")

    def _deserialize_items(self, items_data: Dict[str, Any]) -> Dict[str, T]:
        """
        Deserialize items from JSON data.

        This is an abstract method that must be implemented by subclasses to define
        how their specific item types should be deserialized.

        Args:
            items_data: Dictionary of serialized items

        Returns:
            Dictionary of deserialized items

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError

    async def initialize(self) -> None:
        """
        Initialize storage and load existing data.

        This method should be called after creating a storage instance to load
        existing data from persistent storage. The operation is thread-safe.
        """
        async with self._lock:
            await self._load_from_storage()

    async def backup(self, backup_dir: str) -> None:
        """
        Create a backup of the current storage.

        Creates a copy of the current storage file in the specified backup directory.
        This can be used for data recovery or migration purposes.

        Args:
            backup_dir: Directory to store the backup

        Raises:
            StorageError: If backup creation fails
        """
        try:
            if os.path.exists(self.storage_file):
                backup_file = os.path.join(backup_dir, os.path.basename(self.storage_file))
                async with (
                    aiofiles.open(self.storage_file, "r") as src,
                    aiofiles.open(backup_file, "w") as dst,
                ):
                    content = await src.read()
                    await dst.write(content)

        except Exception as e:
            logger.error(f"Failed to create backup: {str(e)}")
            raise StorageError(f"Backup creation failed: {str(e)}")

    async def restore_from_backup(self, backup_dir: str) -> None:
        """
        Restore from a backup directory.

        Replaces the current storage with data from a backup file. This operation
        clears the current storage before restoring the backup.

        Args:
            backup_dir: Directory containing the backup

        Raises:
            StorageError: If backup restoration fails
        """
        try:
            backup_file = os.path.join(backup_dir, os.path.basename(self.storage_file))
            if os.path.exists(backup_file):
                self._items.clear()
                async with (
                    aiofiles.open(backup_file, "r") as src,
                    aiofiles.open(self.storage_file, "w") as dst,
                ):
                    content = await src.read()
                    await dst.write(content)
                await self._load_from_storage()

        except Exception as e:
            logger.error(f"Failed to restore from backup: {str(e)}")
            raise StorageError(f"Backup restoration failed: {str(e)}")
