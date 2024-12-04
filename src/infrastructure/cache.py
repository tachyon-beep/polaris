"""
Enhanced caching system for the Polaris knowledge graph framework.

This module provides an advanced caching implementation with:
- LRU (Least Recently Used) eviction
- Adaptive TTL based on access patterns
- Performance monitoring capabilities
"""

from typing import TypeVar, Generic, Dict, Optional, Any
from dataclasses import dataclass
from time import time
import math
import json

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """
    Cache entry containing value and metadata for tracking usage patterns.

    Attributes:
        value: The cached value
        timestamp: Time when the entry was last accessed
        access_count: Number of times the entry has been accessed
    """

    value: T
    timestamp: float
    access_count: int


class CacheMetrics:
    """Tracks cache performance metrics."""

    def __init__(self):
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0
        self.total_access_time: float = 0
        self.access_count: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0

    @property
    def avg_access_time(self) -> float:
        """Calculate average access time in milliseconds."""
        return (self.total_access_time / self.access_count * 1000) if self.access_count > 0 else 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": self.hit_rate,
            "avg_access_time_ms": self.avg_access_time,
        }


class LRUCache(Generic[T]):
    """
    Enhanced LRU cache with adaptive TTL and performance monitoring.

    Features:
    - LRU eviction policy
    - Adaptive TTL based on access patterns
    - Performance metrics tracking
    - Configurable size and TTL parameters
    """

    def __init__(
        self,
        max_size: int = 1000,
        base_ttl: int = 3600,
        adaptive_ttl: bool = True,
        min_ttl: int = 300,
        max_ttl: int = 86400,
    ):
        """
        Initialize the cache.

        Args:
            max_size: Maximum number of entries
            base_ttl: Base time-to-live in seconds
            adaptive_ttl: Whether to adjust TTL based on access patterns
            min_ttl: Minimum TTL in seconds
            max_ttl: Maximum TTL in seconds
        """
        self._cache: Dict[str, CacheEntry[T]] = {}
        self._max_size = max_size
        self._base_ttl = base_ttl
        self._adaptive_ttl = adaptive_ttl
        self._min_ttl = min_ttl
        self._max_ttl = max_ttl
        self.metrics = CacheMetrics()

    def get(self, key: str) -> Optional[T]:
        """
        Retrieve a value from the cache.

        Args:
            key: Cache key

        Returns:
            Cached value if present and not expired, None otherwise
        """
        start_time = time()

        try:
            if key not in self._cache:
                self.metrics.misses += 1
                return None

            entry = self._cache[key]
            current_time = time()

            # Check if entry has expired
            ttl = self._calculate_ttl(entry.access_count)
            if current_time - entry.timestamp > ttl:
                del self._cache[key]
                self.metrics.evictions += 1
                self.metrics.misses += 1
                return None

            # Update access statistics
            entry.access_count += 1
            entry.timestamp = current_time
            self.metrics.hits += 1
            return entry.value

        finally:
            self.metrics.total_access_time += time() - start_time
            self.metrics.access_count += 1

    def put(self, key: str, value: T) -> None:
        """
        Store a value in the cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        # Evict least recently used items if cache is full
        if len(self._cache) >= self._max_size:
            self._evict_lru()

        self._cache[key] = CacheEntry(value=value, timestamp=time(), access_count=1)

    def _calculate_ttl(self, access_count: int) -> float:
        """
        Calculate TTL based on access frequency if adaptive TTL is enabled.

        Args:
            access_count: Number of times entry has been accessed

        Returns:
            TTL in seconds
        """
        if not self._adaptive_ttl:
            return self._base_ttl

        # Increase TTL for frequently accessed items
        ttl = self._base_ttl * (1 + math.log(access_count))
        return max(min(ttl, self._max_ttl), self._min_ttl)

    def _evict_lru(self) -> None:
        """Evict least recently used or expired items."""
        current_time = time()

        # First try to remove expired items
        expired = [
            k
            for k, e in self._cache.items()
            if current_time - e.timestamp > self._calculate_ttl(e.access_count)
        ]

        for key in expired:
            del self._cache[key]
            self.metrics.evictions += 1

        # If no expired items, remove least accessed
        if not expired and self._cache:
            key_to_remove = min(
                self._cache.items(), key=lambda x: (x[1].access_count, -x[1].timestamp)
            )[0]
            del self._cache[key_to_remove]
            self.metrics.evictions += 1

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self.metrics = CacheMetrics()

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current cache performance metrics.

        Returns:
            Dictionary containing cache metrics
        """
        return {"size": len(self._cache), "max_size": self._max_size, **self.metrics.to_dict()}
