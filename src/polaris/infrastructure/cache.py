"""
Generic LRU cache implementation with TTL support.

This module provides a thread-safe LRU (Least Recently Used) cache implementation
with TTL (Time To Live) support and performance metrics tracking.

Features:
- LRU eviction policy
- TTL-based expiration
- Adaptive TTL support
- Thread-safe operations
- Performance metrics
"""

from collections import OrderedDict
from threading import Lock
from time import time
from typing import Dict, Generic, Optional, TypeVar

T = TypeVar("T")  # Type of cached values


class LRUCache(Generic[T]):
    """
    Thread-safe LRU cache with TTL support.

    This class implements a Least Recently Used (LRU) cache with support for:
    - Maximum size limit with LRU eviction
    - TTL (Time To Live) based expiration
    - Optional adaptive TTL
    - Thread-safe operations
    - Performance metrics tracking

    The cache automatically evicts least recently used entries when it reaches
    its size limit, and entries expire after their TTL.

    Attributes:
        max_size: Maximum number of entries to store
        base_ttl: Base time-to-live in seconds
        adaptive_ttl: Whether to adjust TTL based on access patterns
        min_ttl: Minimum TTL when using adaptive TTL
        max_ttl: Maximum TTL when using adaptive TTL
    """

    def __init__(
        self,
        max_size: int,
        base_ttl: float,
        adaptive_ttl: bool = True,
        min_ttl: Optional[float] = None,
        max_ttl: Optional[float] = None,
    ):
        """Initialize cache with given parameters."""
        self._cache: OrderedDict[str, T] = OrderedDict()
        self._expiry: Dict[str, float] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = Lock()
        self._max_size = max_size
        self._base_ttl = base_ttl
        self._adaptive_ttl = adaptive_ttl
        self._min_ttl = min_ttl if min_ttl is not None else base_ttl / 2
        self._max_ttl = max_ttl if max_ttl is not None else base_ttl * 2

        # Metrics
        self._hits = 0
        self._misses = 0
        self._total_access_time = 0.0
        self._access_count = 0

    def get(self, key: str) -> Optional[T]:
        """
        Get value from cache.

        Args:
            key: Cache key to look up

        Returns:
            Cached value if found and not expired, None otherwise
        """
        start_time = time()
        try:
            with self._lock:
                # Check if key exists and hasn't expired
                if key in self._cache and time() < self._expiry[key]:
                    # Update access metrics
                    self._hits += 1
                    self._access_times[key] = time()
                    # Move to end (most recently used)
                    value = self._cache.pop(key)
                    self._cache[key] = value
                    return value

                # Cache miss
                self._misses += 1
                # Clean up expired key if present
                if key in self._cache:
                    del self._cache[key]
                    del self._expiry[key]
                    if key in self._access_times:
                        del self._access_times[key]
                return None
        finally:
            # Update access time metrics
            access_duration = time() - start_time
            self._total_access_time += access_duration
            self._access_count += 1

    def put(self, key: str, value: T) -> None:
        """
        Store value in cache.

        Args:
            key: Cache key to store value under
            value: Value to cache
        """
        with self._lock:
            # If key exists, update it
            if key in self._cache:
                self._cache.pop(key)

            # Add new entry
            self._cache[key] = value
            self._expiry[key] = time() + self._calculate_ttl(key)
            self._access_times[key] = time()

            # Evict least recently used if over size limit
            while len(self._cache) > self._max_size:
                lru_key = next(iter(self._cache))
                del self._cache[lru_key]
                del self._expiry[lru_key]
                if lru_key in self._access_times:
                    del self._access_times[lru_key]

    def remove(self, key: str) -> None:
        """
        Remove an item from the cache.

        Args:
            key: Cache key to remove
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                del self._expiry[key]
                if key in self._access_times:
                    del self._access_times[key]

    def _calculate_ttl(self, key: str) -> float:
        """Calculate TTL for entry based on access patterns."""
        if not self._adaptive_ttl:
            return self._base_ttl

        # Calculate TTL based on access frequency
        if key in self._access_times:
            last_access = self._access_times[key]
            age = time() - last_access
            # Increase TTL for frequently accessed items
            if age < self._base_ttl / 2:
                return min(self._base_ttl * 2, self._max_ttl)
            # Decrease TTL for infrequently accessed items
            elif age > self._base_ttl * 2:
                return max(self._base_ttl / 2, self._min_ttl)

        return self._base_ttl

    def clear(self) -> None:
        """Clear all entries from cache."""
        with self._lock:
            self._cache.clear()
            self._expiry.clear()
            self._access_times.clear()
            # Reset metrics
            self._hits = 0
            self._misses = 0
            self._total_access_time = 0.0
            self._access_count = 0

    def get_metrics(self) -> Dict[str, float]:
        """
        Get cache performance metrics.

        Returns:
            Dictionary containing:
            - hits: Number of cache hits
            - misses: Number of cache misses
            - size: Current cache size
            - hit_rate: Cache hit rate
            - avg_access_time_ms: Average access time in milliseconds
        """
        with self._lock:
            total_accesses = self._hits + self._misses
            hit_rate = float(self._hits) / total_accesses if total_accesses > 0 else 0.0
            avg_access_time = (
                float(self._total_access_time * 1000) / self._access_count
                if self._access_count > 0
                else 0.0
            )
            return {
                "hits": float(self._hits),
                "misses": float(self._misses),
                "size": float(len(self._cache)),
                "hit_rate": hit_rate,
                "avg_access_time_ms": avg_access_time,
            }
