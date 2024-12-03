"""
Base repository pattern implementation for the knowledge graph.

This module provides the foundation for all repository implementations in the system.
It defines the core interface and common functionality that specific repositories
must implement. The repository pattern abstracts the data layer, providing a more
object-oriented interface to data persistence.

The base repository includes:
- Basic CRUD operations interface
- Caching support with LRU implementation
- Abstract validation methods
- Common utility methods for cache management
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, TypeVar

from ..core.exceptions import EntityNotFoundError, ValidationError
from ..infrastructure.cache import LRUCache
from ..infrastructure.storage import StorageService

T = TypeVar("T")


class BaseRepository(ABC, Generic[T]):
    """
    Abstract base repository implementing common functionality for all repositories.

    This class serves as the foundation for all repository implementations,
    providing common functionality and enforcing a consistent interface across
    different entity types. It implements the repository pattern with optional
    caching support.

    Attributes:
        storage (StorageService): The underlying storage service for persistence
        cache (Optional[LRUCache]): Optional LRU cache for performance optimization
    """

    def __init__(self, storage: StorageService, cache: Optional[LRUCache] = None):
        """
        Initialize the repository with storage and optional cache.

        Args:
            storage: Storage service for data persistence
            cache: Optional LRU cache for performance optimization
        """
        self.storage = storage
        self.cache = cache
        if cache:
            self._setup_cache()

    def _setup_cache(self) -> None:
        """
        Initialize any cache settings specific to this repository.

        This method can be overridden by subclasses to configure
        cache-specific settings like TTL, max size, etc.
        """
        pass

    @abstractmethod
    async def create(self, item: T) -> T:
        """
        Create a new item in the repository.

        Args:
            item: The item to create

        Returns:
            The created item with any system-generated fields populated

        Raises:
            ValidationError: If the item fails validation
            StorageError: If there's an error during storage operation
        """
        pass

    @abstractmethod
    async def get(self, id: str) -> T:
        """
        Retrieve an item by its unique identifier.

        Args:
            id: The unique identifier of the item

        Returns:
            The requested item

        Raises:
            EntityNotFoundError: If the item doesn't exist
            StorageError: If there's an error during storage operation
        """
        pass

    @abstractmethod
    async def update(self, item: T) -> T:
        """
        Update an existing item.

        Args:
            item: The item to update with new values

        Returns:
            The updated item

        Raises:
            EntityNotFoundError: If the item doesn't exist
            ValidationError: If the updated item fails validation
            StorageError: If there's an error during storage operation
        """
        pass

    @abstractmethod
    async def delete(self, id: str) -> None:
        """
        Delete an item by its unique identifier.

        Args:
            id: The unique identifier of the item to delete

        Raises:
            EntityNotFoundError: If the item doesn't exist
            StorageError: If there's an error during storage operation
        """
        pass

    @abstractmethod
    async def list(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[T]:
        """
        List items with optional filtering, pagination and sorting.

        Args:
            filters: Optional dictionary of filter criteria
            limit: Maximum number of items to return
            offset: Number of items to skip for pagination

        Returns:
            List of items matching the criteria

        Raises:
            StorageError: If there's an error during storage operation
        """
        pass

    async def exists(self, id: str) -> bool:
        """
        Check if an item exists by its unique identifier.

        Args:
            id: The unique identifier to check

        Returns:
            True if the item exists, False otherwise
        """
        try:
            await self.get(id)
            return True
        except EntityNotFoundError:
            return False

    async def get_or_create(self, id: str, default_item: T) -> T:
        """
        Get an existing item or create it if it doesn't exist.

        This method provides an atomic way to ensure an item exists,
        creating it with default values if necessary.

        Args:
            id: The unique identifier to look up
            default_item: The default item to create if none exists

        Returns:
            The existing or newly created item

        Raises:
            ValidationError: If the default item fails validation
            StorageError: If there's an error during storage operation
        """
        try:
            return await self.get(id)
        except EntityNotFoundError:
            return await self.create(default_item)

    @abstractmethod
    async def validate(self, item: T) -> bool:
        """
        Validate an item before creation or update.

        Args:
            item: The item to validate

        Returns:
            True if validation passes, False otherwise

        Raises:
            ValidationError: If validation fails with specific reasons
        """
        pass

    def _cache_key(self, id: str) -> str:
        """
        Generate a cache key for an item.

        Args:
            id: The unique identifier of the item

        Returns:
            A string key suitable for cache storage
        """
        return f"{self.__class__.__name__}:{id}"

    async def _get_from_cache(self, id: str) -> Optional[T]:
        """
        Attempt to retrieve an item from cache.

        Args:
            id: The unique identifier of the item

        Returns:
            The cached item if found, None otherwise
        """
        if self.cache:
            cache_key = self._cache_key(id)
            return self.cache.get(cache_key)
        return None

    async def _set_in_cache(self, id: str, item: T) -> None:
        """
        Store an item in cache.

        Args:
            id: The unique identifier of the item
            item: The item to cache
        """
        if self.cache:
            cache_key = self._cache_key(id)
            self.cache.put(cache_key, item)

    async def _remove_from_cache(self, id: str) -> None:
        """
        Remove an item from cache.

        Args:
            id: The unique identifier of the item to remove
        """
        if self.cache:
            cache_key = self._cache_key(id)
            self.cache.delete(cache_key)
