"""
LRU cache implementation for knowledge graph entities and relations.

This module provides a thread-safe Least Recently Used (LRU) cache implementation
with time-based expiration. It offers:
- Generic caching for any type of object
- Thread-safe operations
- Time-based expiration (TTL)
- Custom serialization/deserialization
- Batch operations
- Automatic cleanup of expired items
- Size-based eviction using LRU policy

The cache is particularly optimized for storing knowledge graph entities and
relations, improving access times for frequently used data.
"""

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import asdict
from typing import Callable, Dict, Generic, List, Optional, TypeVar

from ..core.exceptions import CacheError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class LRUCache(Generic[T]):
    """
    Thread-safe LRU cache implementation with time-based expiration.

    This class implements a Least Recently Used (LRU) cache with the following features:
    - Thread-safe operations using RLock
    - Time-based expiration (TTL)
    - Maximum size limit with LRU eviction
    - Custom serialization/deserialization support
    - Batch operations for multiple items
    - Automatic cleanup of expired items

    The cache is generic and can store any type of object, with optional
    custom serialization/deserialization for complex objects.

    Attributes:
        max_size (int): Maximum number of items the cache can hold
        ttl (int): Time to live in seconds for cached items
        cache (OrderedDict): Ordered dictionary storing cached items and timestamps
        lock (threading.RLock): Reentrant lock for thread safety
        serializer (Callable): Function to serialize items for storage
        deserializer (Callable): Function to deserialize stored items
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl: int = 3600,  # Time to live in seconds
        serializer: Optional[Callable[[T], Dict]] = None,
        deserializer: Optional[Callable[[Dict], T]] = None,
    ):
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum number of items in cache (default: 1000)
            ttl: Time to live in seconds (default: 1 hour)
            serializer: Optional function to serialize items for storage
            deserializer: Optional function to deserialize stored items

        The serializer and deserializer are optional but recommended for complex
        objects to ensure proper storage and retrieval.
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict[str, tuple[T, float]] = OrderedDict()
        self.lock = threading.RLock()
        self.serializer = serializer or (lambda x: asdict(x))
        self.deserializer = deserializer or (lambda x: x)
        self._setup_logging()

    def _setup_logging(self) -> None:
        """
        Configure cache logging.

        Sets up file-based logging for cache operations with timestamp,
        log level, and formatted messages.
        """
        handler = logging.FileHandler("cache.log")
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    def get(self, key: str) -> Optional[T]:
        """
        Retrieve an item from the cache.

        This method checks for item existence and expiration, updating
        the item's position in the LRU order if it's accessed.

        Args:
            key: Cache key to retrieve

        Returns:
            Cached item if found and not expired, None otherwise

        Thread Safety:
            This method is thread-safe, protected by the cache's lock.
        """
        with self.lock:
            if key not in self.cache:
                return None

            item, timestamp = self.cache[key]
            if time.time() - timestamp > self.ttl:
                del self.cache[key]
                return None

            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return item

    def put(self, key: str, item: T) -> None:
        """
        Add an item to the cache.

        This method adds or updates an item in the cache, handling LRU eviction
        if the cache is full and updating timestamps for existing items.

        Args:
            key: Cache key
            item: Item to cache

        Raises:
            CacheError: If serialization fails or other cache operations fail

        Thread Safety:
            This method is thread-safe, protected by the cache's lock.
        """
        try:
            with self.lock:
                if key in self.cache:
                    # Update existing item
                    self.cache.move_to_end(key)
                    self.cache[key] = (item, time.time())
                else:
                    # Add new item
                    if len(self.cache) >= self.max_size:
                        # Remove least recently used item
                        self.cache.popitem(last=False)
                    self.cache[key] = (item, time.time())
        except Exception as e:
            logger.error(f"Failed to cache item with key {key}: {str(e)}")
            raise CacheError(f"Failed to cache item: {str(e)}")

    def delete(self, key: str) -> bool:
        """
        Remove an item from the cache.

        Args:
            key: Cache key to remove

        Returns:
            True if item was found and deleted, False otherwise

        Thread Safety:
            This method is thread-safe, protected by the cache's lock.
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False

    def clear(self) -> None:
        """
        Clear all items from the cache.

        Removes all items from the cache regardless of their expiration status.

        Thread Safety:
            This method is thread-safe, protected by the cache's lock.
        """
        with self.lock:
            self.cache.clear()

    def get_many(self, keys: List[str]) -> Dict[str, T]:
        """
        Retrieve multiple items from the cache.

        This method efficiently retrieves multiple items in a single operation,
        handling expiration and updating LRU order for all accessed items.

        Args:
            keys: List of cache keys to retrieve

        Returns:
            Dictionary mapping found keys to their cached items

        Thread Safety:
            This method is thread-safe, protected by the cache's lock.
        """
        result = {}
        with self.lock:
            for key in keys:
                item = self.get(key)
                if item is not None:
                    result[key] = item
        return result

    def put_many(self, items: Dict[str, T]) -> None:
        """
        Add multiple items to the cache.

        This method efficiently adds multiple items in a single operation,
        handling LRU eviction and timestamp updates as needed.

        Args:
            items: Dictionary mapping keys to items for caching

        Thread Safety:
            This method is thread-safe, protected by the cache's lock.
        """
        with self.lock:
            for key, item in items.items():
                self.put(key, item)

    def get_size(self) -> int:
        """
        Get current cache size.

        Returns:
            Number of items currently in the cache

        Thread Safety:
            This method is thread-safe, protected by the cache's lock.
        """
        with self.lock:
            return len(self.cache)

    def cleanup_expired(self) -> int:
        """
        Remove expired items from cache.

        This method removes all items that have exceeded their TTL,
        freeing up space in the cache.

        Returns:
            Number of expired items removed

        Thread Safety:
            This method is thread-safe, protected by the cache's lock.
        """
        removed = 0
        current_time = time.time()
        with self.lock:
            keys_to_remove = [
                key
                for key, (_, timestamp) in self.cache.items()
                if current_time - timestamp > self.ttl
            ]
            for key in keys_to_remove:
                del self.cache[key]
                removed += 1
        return removed
