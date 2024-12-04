"""
Caching functionality for path finding results.

This module provides a caching mechanism for path finding results to improve
performance by avoiding redundant calculations. It uses an LRU (Least Recently Used)
cache with TTL (Time To Live) support.

Features:
- Configurable cache size and TTL
- Automatic cache key generation
- Cache metrics tracking
- Adaptive TTL support

Example:
    >>> key = PathCache.get_cache_key("A", "B", "shortest")
    >>> if result := PathCache.get(key):
    ...     return result  # Cache hit
    >>> result = compute_path()
    >>> PathCache.put(key, result)  # Cache result
"""

from typing import Any, Dict, Optional

from ...infrastructure.cache import LRUCache
from .models import PathResult

# Constants
PATH_CACHE_SIZE = 10000  # Maximum number of cached results
PATH_CACHE_TTL = 3600  # Default TTL in seconds (1 hour)


class PathCache:
    """
    Handles caching of path finding results.

    This class provides a centralized caching mechanism for path finding results
    using an LRU (Least Recently Used) cache with TTL (Time To Live) support.
    The cache automatically evicts least recently used entries when it reaches
    its size limit, and entries expire after their TTL.

    Features:
        - LRU eviction policy
        - TTL-based expiration
        - Adaptive TTL support
        - Cache metrics tracking
        - Thread-safe operations

    Example:
        >>> # Check cache for existing result
        >>> key = PathCache.get_cache_key("A", "B", "shortest")
        >>> if result := PathCache.get(key):
        ...     return result
        >>> # Cache miss - compute and store result
        >>> result = compute_path()
        >>> PathCache.put(key, result)
    """

    _instance = None
    cache = LRUCache[PathResult](
        max_size=PATH_CACHE_SIZE,
        base_ttl=PATH_CACHE_TTL,
        adaptive_ttl=False,  # Disable adaptive TTL for consistent testing
        min_ttl=PATH_CACHE_TTL,
        max_ttl=PATH_CACHE_TTL,
    )
    _hits = 0
    _misses = 0

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_cache_key(
        cls,
        start_node: str,
        end_node: str,
        path_type: str,
        weight_func_name: Optional[str] = None,
        max_length: Optional[int] = None,
    ) -> str:
        """
        Generate cache key for path finding results.

        Creates a unique cache key based on the path finding parameters.
        The key includes the start and end nodes, path finding type,
        and optionally the weight function name and max length.

        Args:
            start_node: Starting node ID
            end_node: Target node ID
            path_type: Type of path finding algorithm used
            weight_func_name: Optional name of weight function
            max_length: Optional maximum path length

        Returns:
            Cache key string

        Example:
            >>> key = PathCache.get_cache_key("A", "B", "shortest", "impact_weight", 3)
            >>> print(f"Cache key: {key}")
        """
        key_parts = [start_node, end_node, path_type]
        if weight_func_name is not None:
            key_parts.append(weight_func_name)
        if max_length is not None:
            key_parts.append(str(max_length))
        return ":".join(key_parts)

    @classmethod
    def get(cls, key: str) -> Optional[PathResult]:
        """
        Get cached path result.

        Retrieves a previously cached path result. Returns None if the
        key doesn't exist or the entry has expired.

        Args:
            key: Cache key to look up

        Returns:
            Cached PathResult if found and not expired, None otherwise

        Example:
            >>> if result := PathCache.get(key):
            ...     print("Cache hit!")
            ...     return result
        """
        result = cls.cache.get(key)
        if result is not None:
            cls._hits += 1
        else:
            cls._misses += 1
        return result

    @classmethod
    def put(cls, key: str, result: PathResult) -> None:
        """
        Cache path result.

        Stores a path result in the cache. If the cache is full,
        the least recently used entry is evicted.

        Args:
            key: Cache key to store result under
            result: PathResult to cache

        Example:
            >>> PathCache.put(key, path_result)
        """
        cls.cache.put(key, result)

    @classmethod
    def get_metrics(cls) -> Dict[str, float]:
        """
        Get current cache performance metrics.

        Returns a dictionary containing various cache performance metrics:
        - hits: Number of cache hits
        - misses: Number of cache misses
        - size: Current cache size
        - hit_rate: Cache hit rate
        - avg_access_time_ms: Average access time in milliseconds

        Returns:
            Dictionary containing cache metrics

        Example:
            >>> metrics = PathCache.get_metrics()
            >>> print(f"Cache hit rate: {metrics['hit_rate']:.2%}")
        """
        total = cls._hits + cls._misses
        hit_rate = float(cls._hits) / total if total > 0 else 0.0
        metrics = cls.cache.get_metrics()
        metrics.update(
            {
                "hits": float(cls._hits),
                "misses": float(cls._misses),
                "hit_rate": hit_rate,
            }
        )
        return metrics

    @classmethod
    def clear(cls) -> None:
        """
        Clear all cached results.

        Removes all entries from the cache, resetting it to an empty state.
        This is useful for testing or when cache invalidation is needed.

        Example:
            >>> PathCache.clear()  # Clear all cached results
        """
        cls.cache.clear()
        cls._hits = 0
        cls._misses = 0

    @classmethod
    def reconfigure(cls, max_size: int = PATH_CACHE_SIZE, ttl: int = PATH_CACHE_TTL) -> None:
        """
        Reconfigure cache parameters.

        Updates cache configuration with new size and TTL values.
        This is useful for testing or runtime optimization.

        Args:
            max_size: Maximum number of entries to cache
            ttl: Time-to-live in seconds for cached entries

        Example:
            >>> PathCache.reconfigure(max_size=100, ttl=60)  # Small cache, short TTL
        """
        cls.cache = LRUCache[PathResult](
            max_size=max_size,
            base_ttl=ttl,
            adaptive_ttl=False,  # Disable adaptive TTL for consistent testing
            min_ttl=ttl,
            max_ttl=ttl,
        )
        cls._hits = 0
        cls._misses = 0
