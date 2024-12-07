"""
Graph caching system.

This module provides caching functionality for graph operations, particularly
focused on caching path finding results. It implements an LRU (Least Recently Used)
cache with TTL (Time To Live) support and adaptive cache duration based on usage patterns.
"""

from dataclasses import dataclass, field
from threading import RLock
from typing import Dict, Generic, List, Optional, TypeVar, Union
from time import time

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """
    Individual cache entry with metadata.

    Attributes:
        value (T): The cached value
        timestamp (float): When the entry was created/updated
        ttl (int): Time-to-live in seconds
        hits (int): Number of times this entry was accessed
    """

    value: T
    timestamp: float
    ttl: int
    hits: int = 0


class LRUCache(Generic[T]):
    """
    LRU cache implementation with TTL support.

    This cache automatically evicts the least recently used items when it reaches
    capacity and supports time-based expiration of entries. It also provides
    adaptive TTL based on access patterns.

    Attributes:
        _cache (Dict[str, CacheEntry[T]]): The cache storage
        _max_size (int): Maximum number of items to store
        _base_ttl (int): Base time-to-live in seconds
        _min_ttl (int): Minimum allowed TTL
        _max_ttl (int): Maximum allowed TTL
        _adaptive_ttl (bool): Whether to adjust TTL based on usage
        _lock (RLock): Thread synchronization lock
    """

    def __init__(
        self,
        max_size: int = 1000,
        base_ttl: int = 3600,
        adaptive_ttl: bool = True,
        min_ttl: Optional[int] = None,
        max_ttl: Optional[int] = None,
    ):
        """
        Initialize the cache.

        Args:
            max_size (int): Maximum number of items to store
            base_ttl (int): Base time-to-live in seconds
            adaptive_ttl (bool): Whether to adjust TTL based on usage
            min_ttl (Optional[int]): Minimum allowed TTL
            max_ttl (Optional[int]): Maximum allowed TTL
        """
        self._cache: Dict[str, CacheEntry[T]] = {}
        self._max_size = max_size
        self._base_ttl = base_ttl
        self._adaptive_ttl = adaptive_ttl
        self._min_ttl = min_ttl or base_ttl // 2
        self._max_ttl = max_ttl or base_ttl * 2
        self._lock = RLock()

    def get(self, key: str) -> Optional[T]:
        """
        Get a value from the cache.

        Args:
            key (str): Cache key

        Returns:
            Optional[T]: Cached value if it exists and hasn't expired
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None

            current_time = time()
            if current_time - entry.timestamp > entry.ttl:
                # Entry has expired
                del self._cache[key]
                return None

            # Update access metadata
            entry.hits += 1
            if self._adaptive_ttl:
                # Increase TTL for frequently accessed items
                entry.ttl = min(self._max_ttl, int(entry.ttl * (1 + 0.1 * (entry.hits // 10))))

            return entry.value

    def put(self, key: str, value: T) -> None:
        """
        Add or update a cache entry.

        Args:
            key (str): Cache key
            value (T): Value to cache
        """
        with self._lock:
            # Check if we need to evict entries
            while len(self._cache) >= self._max_size:
                self._evict_lru()

            # Calculate TTL based on existing entry if it exists
            ttl = self._base_ttl
            existing = self._cache.get(key)
            if existing and self._adaptive_ttl:
                # Adjust TTL based on previous usage
                adjusted_ttl = int(existing.ttl * (1 + 0.1 * existing.hits))
                ttl = min(self._max_ttl, adjusted_ttl)

            self._cache[key] = CacheEntry(value=value, timestamp=time(), ttl=ttl, hits=0)

    def _evict_lru(self) -> None:
        """Remove the least recently used cache entry."""
        if not self._cache:
            return

        # Find entry with oldest timestamp
        lru_key = min(self._cache.keys(), key=lambda k: self._cache[k].timestamp)
        del self._cache[lru_key]

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()

    def get_metrics(self) -> Dict[str, Union[int, float]]:
        """
        Get cache statistics and metrics.

        Returns:
            Dict[str, Union[int, float]]: Dictionary of cache metrics
        """
        with self._lock:
            if not self._cache:
                return {
                    "size": 0,
                    "max_size": self._max_size,
                    "total_hits": 0,
                    "avg_ttl": 0,
                    "expired_entries": 0,
                    "memory_usage": 0,
                }

            total_hits = sum(entry.hits for entry in self._cache.values())
            avg_ttl = int(sum(entry.ttl for entry in self._cache.values()) / len(self._cache))
            current_time = time()
            expired = sum(
                1 for entry in self._cache.values() if current_time - entry.timestamp > entry.ttl
            )

            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "total_hits": total_hits,
                "avg_ttl": avg_ttl,
                "expired_entries": expired,
                "memory_usage": len(self._cache) * 100,  # Rough estimate
            }


@dataclass
class PathCache:
    """
    Specialized cache for graph path results.

    This class extends the base LRU cache with path-specific functionality
    and optimized key generation for path queries.

    Attributes:
        _cache (LRUCache[List[List[str]]]): Underlying cache for paths
        _stats (Dict[str, int]): Usage statistics
    """

    _cache: LRUCache[List[List[str]]] = field(
        default_factory=lambda: LRUCache(max_size=1000, base_ttl=3600)
    )
    _stats: Dict[str, int] = field(default_factory=lambda: {"hits": 0, "misses": 0})

    def get_paths(
        self,
        from_node: str,
        to_node: str,
        max_depth: Optional[int] = None,
        algorithm: str = "default",
    ) -> Optional[List[List[str]]]:
        """
        Get cached paths for a path query.

        Args:
            from_node (str): Starting node
            to_node (str): Target node
            max_depth (Optional[int]): Maximum path length
            algorithm (str): Path finding algorithm used

        Returns:
            Optional[List[List[str]]]: Cached paths if they exist
        """
        key = self._make_key(from_node, to_node, max_depth, algorithm)
        result = self._cache.get(key)

        if result is not None:
            self._stats["hits"] += 1
        else:
            self._stats["misses"] += 1

        return result

    def cache_paths(
        self,
        from_node: str,
        to_node: str,
        paths: List[List[str]],
        max_depth: Optional[int] = None,
        algorithm: str = "default",
    ) -> None:
        """
        Cache paths for a path query.

        Args:
            from_node (str): Starting node
            to_node (str): Target node
            paths (List[List[str]]): Paths to cache
            max_depth (Optional[int]): Maximum path length
            algorithm (str): Path finding algorithm used
        """
        key = self._make_key(from_node, to_node, max_depth, algorithm)
        self._cache.put(key, paths)

    def _make_key(
        self, from_node: str, to_node: str, max_depth: Optional[int], algorithm: str
    ) -> str:
        """Generate a cache key for a path query."""
        return f"{from_node}|{to_node}|{max_depth}|{algorithm}"

    def clear(self) -> None:
        """Clear the path cache."""
        self._cache.clear()
        self._stats = {"hits": 0, "misses": 0}

    def get_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get cache statistics.

        Returns:
            Dict[str, Union[int, float]]: Cache statistics including
            hit/miss ratios and general metrics
        """
        metrics = self._cache.get_metrics()
        total_queries = self._stats["hits"] + self._stats["misses"]
        hit_ratio = self._stats["hits"] / total_queries if total_queries > 0 else 0

        return {**metrics, "hit_ratio": hit_ratio, "total_queries": total_queries, **self._stats}
