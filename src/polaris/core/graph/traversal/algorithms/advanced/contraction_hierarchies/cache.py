"""
Thread-safe caching implementation for Contraction Hierarchies.

This module provides a thread-safe LRU cache with automatic cleanup and monitoring
capabilities, specifically designed for the Contraction Hierarchies algorithm.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Any, Dict, List, Set, TYPE_CHECKING
import threading
import time
import math
import logging
import weakref
from contextlib import contextmanager

if TYPE_CHECKING:
    from polaris.core.graph import Graph

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Thread-safe cache entry with TTL and usage tracking."""

    value: Any
    timestamp: float
    access_count: int = 0
    size_bytes: int = 0

    def __post_init__(self):
        """Calculate size if not provided."""
        if self.size_bytes == 0:
            try:
                # Rough size estimation
                self.size_bytes = len(str(self.value).encode())
            except Exception:
                self.size_bytes = 100  # Default size


class CacheStats:
    """Thread-safe container for cache statistics."""

    def __init__(self):
        self._lock = threading.Lock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "cleanups": 0,
            "errors": 0,
            "total_bytes": 0,
            "max_age": 0.0,
        }

    def increment(self, stat: str, amount: int = 1) -> None:
        """Thread-safe increment of a statistic."""
        with self._lock:
            if stat in self._stats:
                self._stats[stat] += amount

    def set(self, stat: str, value: Any) -> None:
        """Thread-safe set of a statistic."""
        with self._lock:
            if stat in self._stats:
                self._stats[stat] = value

    def get_all(self) -> Dict[str, Any]:
        """Get a copy of all statistics."""
        with self._lock:
            stats = dict(self._stats)
            # Calculate derived metrics
            total_requests = stats["hits"] + stats["misses"]
            stats["hit_ratio"] = stats["hits"] / total_requests if total_requests > 0 else 0.0
            return stats


class DynamicLRUCache:
    """Thread-safe LRU cache with automatic cleanup and monitoring."""

    def __init__(
        self,
        ttl_seconds: int = 3600,
        cleanup_interval: int = 300,
        max_size_bytes: Optional[int] = None,
        max_memory_percent: float = 0.1,
    ):
        """
        Initialize cache with configuration.

        Args:
            ttl_seconds: Time-to-live for cache entries
            cleanup_interval: Seconds between cleanup operations
            max_size_bytes: Maximum cache size in bytes
            max_memory_percent: Maximum percent of system memory to use
        """
        self._cache = OrderedDict()  # {key: CacheEntry}
        self._lock = threading.RLock()  # Reentrant lock for nested operations
        self._ttl = ttl_seconds
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = time.time()
        self._max_size_bytes = max_size_bytes
        self._max_memory_percent = max_memory_percent
        self._stats = CacheStats()

        # Start monitoring thread
        self._monitor_thread = threading.Thread(
            target=self._monitor_cache, daemon=True, name="cache-monitor"
        )
        self._monitor_thread.start()

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache with TTL check.

        Args:
            key: Cache key to retrieve

        Returns:
            Cached value or None if not found/expired
        """
        try:
            with self._lock:
                self._maybe_cleanup()

                entry = self._cache.get(key)
                if entry is None:
                    self._stats.increment("misses")
                    return None

                # Check TTL
                if time.time() - entry.timestamp > self._ttl:
                    self._remove_entry(key)
                    return None

                # Update access stats and move to end
                entry.access_count += 1
                self._cache.move_to_end(key)
                self._stats.increment("hits")

                return entry.value

        except Exception as e:
            self._stats.increment("errors")
            logger.error(f"Error getting cache key {key}: {e}")
            return None

    def set(self, key: str, value: Any, max_entries: int) -> None:
        """
        Set value in cache with size management.

        Args:
            key: Cache key
            value: Value to cache
            max_entries: Maximum number of entries to maintain
        """
        try:
            with self._lock:
                # Remove if key exists
                if key in self._cache:
                    self._remove_entry(key)

                # Create new entry
                entry = CacheEntry(value=value, timestamp=time.time())

                # Check memory constraints
                if self._would_exceed_memory_limit(entry.size_bytes):
                    self._enforce_memory_limit(entry.size_bytes)

                # Add new entry
                self._cache[key] = entry
                self._stats.increment("total_bytes", entry.size_bytes)

                # Maintain size limit
                while len(self._cache) > max_entries:
                    self._remove_oldest()

        except Exception as e:
            self._stats.increment("errors")
            logger.error(f"Error setting cache key {key}: {e}")

    def _remove_entry(self, key: str) -> None:
        """Remove a cache entry and update stats."""
        entry = self._cache.pop(key)
        self._stats.increment("evictions")
        self._stats.increment("total_bytes", -entry.size_bytes)

    def _remove_oldest(self) -> None:
        """Remove the oldest cache entry."""
        if self._cache:
            _, entry = self._cache.popitem(last=False)
            self._stats.increment("evictions")
            self._stats.increment("total_bytes", -entry.size_bytes)

    def _would_exceed_memory_limit(self, additional_bytes: int) -> bool:
        """Check if adding bytes would exceed memory limit."""
        if self._max_size_bytes:
            current_bytes = self._stats.get_all()["total_bytes"]
            return current_bytes + additional_bytes > self._max_size_bytes
        return False

    def _enforce_memory_limit(self, needed_bytes: int) -> None:
        """Remove entries until we have space for needed_bytes."""
        while self._cache and self._would_exceed_memory_limit(needed_bytes):
            self._remove_oldest()

    def _maybe_cleanup(self) -> None:
        """Periodically cleanup expired entries."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        try:
            expired_keys = [k for k, v in self._cache.items() if now - v.timestamp > self._ttl]

            for key in expired_keys:
                self._remove_entry(key)

            self._last_cleanup = now
            self._stats.increment("cleanups")

            # Update max age stat
            if self._cache:
                max_age = max(now - e.timestamp for e in self._cache.values())
                self._stats.set("max_age", max_age)

        except Exception as e:
            self._stats.increment("errors")
            logger.error(f"Error during cache cleanup: {e}")

    def _monitor_cache(self) -> None:
        """Background monitoring of cache metrics."""
        while True:
            try:
                stats = self._stats.get_all()
                logger.info("Cache statistics:")
                for metric, value in stats.items():
                    logger.info(f"  {metric}: {value}")
            except Exception as e:
                logger.error(f"Error monitoring cache: {e}")
            time.sleep(self._cleanup_interval)

    def get_stats(self) -> Dict[str, Any]:
        """Get current cache statistics."""
        return self._stats.get_all()


class CacheManager:
    """Manages multiple caches with shared configuration."""

    def __init__(self):
        self._caches: Dict[str, DynamicLRUCache] = {}
        self._lock = threading.Lock()

    def get_cache(
        self,
        name: str,
        ttl_seconds: int = 3600,
        cleanup_interval: int = 300,
    ) -> DynamicLRUCache:
        """
        Get or create a cache by name.

        Args:
            name: Cache identifier
            ttl_seconds: Cache entry lifetime
            cleanup_interval: Cleanup frequency

        Returns:
            Thread-safe LRU cache instance
        """
        with self._lock:
            if name not in self._caches:
                self._caches[name] = DynamicLRUCache(
                    ttl_seconds=ttl_seconds, cleanup_interval=cleanup_interval
                )
            return self._caches[name]

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all managed caches."""
        with self._lock:
            return {name: cache.get_stats() for name, cache in self._caches.items()}


def get_cache_size(graph: "Graph") -> int:
    """
    Calculate appropriate cache size based on graph properties.

    Args:
        graph: Graph instance to analyze

    Returns:
        Recommended cache size
    """
    try:
        node_count = len(list(graph.get_nodes()))
        edge_count = len(list(graph.get_edges()))

        # Base size scaled by graph complexity
        base_size = 10000
        complexity_factor = math.sqrt(node_count * edge_count)

        # Limit between reasonable bounds
        min_size = 1000
        max_size = 100000

        size = int(base_size * (complexity_factor / 1000))
        return max(min_size, min(size, max_size))

    except Exception as e:
        logger.error(f"Error calculating cache size: {e}")
        return 10000  # Default size


# Global cache manager instance
_cache_manager = CacheManager()


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    return _cache_manager
