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
logger.setLevel(logging.DEBUG)


class CacheError(Exception):
    """Base exception for cache-related errors."""

    pass


class InvalidValueError(CacheError):
    """Exception raised when a value cannot be cached."""

    pass


@dataclass
class CacheEntry:
    """Thread-safe cache entry with TTL and usage tracking."""

    value: Any
    timestamp: float
    access_count: int = 0
    size_bytes: int = 0

    @staticmethod
    def validate_value(value: Any) -> int:
        """
        Validate a value can be cached and return its size.

        Args:
            value: Value to validate

        Returns:
            Size of value in bytes

        Raises:
            InvalidValueError: If value cannot be cached
        """
        try:
            # First validate that the value supports len()
            logger.debug(f"Validating value supports len(): {value!r}")
            len(value)  # This will raise if value doesn't support len()

            # Then validate we can convert it to a string
            logger.debug(f"Validating value can be stringified")
            value_str = str(value)

            # Finally get the encoded size
            size = len(value_str.encode())
            logger.debug(f"Value size: {size} bytes")
            return size

        except Exception as e:
            logger.debug(f"Value validation failed: {e}")
            raise InvalidValueError(f"Value must support len() and str(): {e}")

    def __post_init__(self):
        """Calculate size if not provided."""
        if self.size_bytes == 0:
            logger.debug("CacheEntry post init - calculating size")
            self.size_bytes = self.validate_value(self.value)
            logger.debug(f"Size calculated: {self.size_bytes}")


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
                logger.debug(f"Incremented {stat} by {amount}, new value: {self._stats[stat]}")

    def set(self, stat: str, value: Any) -> None:
        """Thread-safe set of a statistic."""
        with self._lock:
            if stat in self._stats:
                self._stats[stat] = value
                logger.debug(f"Set {stat} to {value}")

    def get_all(self) -> Dict[str, Any]:
        """Get a copy of all statistics."""
        with self._lock:
            stats = dict(self._stats)
            # Calculate hit ratio from actual hits and misses
            total_requests = stats["hits"] + stats["misses"]
            stats["hit_ratio"] = stats["hits"] / total_requests if total_requests > 0 else 0.0
            logger.debug(f"Stats snapshot: {stats}")
            return stats

    def reset(self) -> None:
        """Reset all statistics to zero."""
        with self._lock:
            for key in self._stats:
                self._stats[key] = 0.0 if key == "hit_ratio" else 0
            logger.debug("Stats reset")


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
        self._last_cleanup = 0  # Start at 0 to force first cleanup
        self._max_size_bytes = max_size_bytes
        self._max_memory_percent = max_memory_percent
        self._stats = CacheStats()
        self._running = True

        logger.debug(
            f"Cache initialized with TTL={ttl_seconds}s, " f"cleanup_interval={cleanup_interval}s"
        )

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
        if key is None:
            logger.debug("Get: key is None")
            self._stats.increment("misses")
            return None

        try:
            with self._lock:
                logger.debug(f"Get: key={key!r}")
                self._maybe_cleanup()

                entry = self._cache.get(key)
                if entry is None:
                    logger.debug(f"Get: key {key!r} not found")
                    self._stats.increment("misses")
                    return None

                # Check TTL
                age = time.time() - entry.timestamp
                logger.debug(f"Entry age: {age}s, TTL: {self._ttl}s")
                if age > self._ttl:
                    logger.debug(f"Entry expired, age={age}s > ttl={self._ttl}s")
                    self._remove_entry(key)
                    self._stats.increment("misses")
                    return None

                # Update access stats and move to end
                entry.access_count += 1
                self._cache.move_to_end(key)
                self._stats.increment("hits")

                logger.debug(f"Get: returning value for {key!r}")
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
        if key is None:
            logger.debug("Set: key is None")
            self._stats.increment("errors")
            return

        try:
            logger.debug(f"Set: key={key!r}, value={value!r}")

            # Validate value before proceeding
            size_bytes = CacheEntry.validate_value(value)
            logger.debug(f"Value validated, size={size_bytes}")

            with self._lock:
                # Remove if key exists
                if key in self._cache:
                    logger.debug(f"Removing existing entry for {key!r}")
                    self._remove_entry(key)

                # Check memory constraints
                if self._would_exceed_memory_limit(size_bytes):
                    logger.debug("Enforcing memory limit")
                    self._enforce_memory_limit(size_bytes)

                # Create and add new entry
                entry = CacheEntry(value=value, timestamp=time.time(), size_bytes=size_bytes)
                self._cache[key] = entry
                self._stats.increment("total_bytes", entry.size_bytes)
                logger.debug(f"Added entry for {key!r}")

                # Maintain size limit
                while len(self._cache) > max_entries:
                    logger.debug("Cache full, removing oldest entry")
                    self._remove_oldest()

        except InvalidValueError:
            # Don't store invalid values
            self._stats.increment("errors")
            logger.error(f"Invalid value for key {key}")
        except Exception as e:
            self._stats.increment("errors")
            logger.error(f"Error setting cache key {key}: {e}")

    def _remove_entry(self, key: str) -> None:
        """Remove a cache entry and update stats."""
        entry = self._cache.pop(key)
        self._stats.increment("evictions")
        self._stats.increment("total_bytes", -entry.size_bytes)
        logger.debug(f"Removed entry for {key!r}")

    def _remove_oldest(self) -> None:
        """Remove the oldest cache entry."""
        if self._cache:
            key, entry = self._cache.popitem(last=False)
            self._stats.increment("evictions")
            self._stats.increment("total_bytes", -entry.size_bytes)
            logger.debug(f"Removed oldest entry: {key!r}")

    def _would_exceed_memory_limit(self, additional_bytes: int) -> bool:
        """Check if adding bytes would exceed memory limit."""
        if self._max_size_bytes:
            current_bytes = self._stats.get_all()["total_bytes"]
            would_exceed = current_bytes + additional_bytes > self._max_size_bytes
            logger.debug(
                f"Memory check: current={current_bytes}, "
                f"additional={additional_bytes}, "
                f"would_exceed={would_exceed}"
            )
            return would_exceed
        return False

    def _enforce_memory_limit(self, needed_bytes: int) -> None:
        """Remove entries until we have space for needed_bytes."""
        while self._cache and self._would_exceed_memory_limit(needed_bytes):
            self._remove_oldest()

    def _maybe_cleanup(self) -> None:
        """Periodically cleanup expired entries."""
        now = time.time()

        # Check if cleanup is needed
        needs_cleanup = (
            now - self._last_cleanup >= self._cleanup_interval
            or self._last_cleanup == 0  # Force first cleanup
        )

        logger.debug(
            f"Cleanup check: age={now - self._last_cleanup}s, "
            f"interval={self._cleanup_interval}s, "
            f"needs_cleanup={needs_cleanup}"
        )

        if needs_cleanup:
            try:
                # Find expired entries
                expired_keys = [k for k, v in self._cache.items() if now - v.timestamp > self._ttl]

                logger.debug(f"Found {len(expired_keys)} expired entries")

                if expired_keys:
                    # Remove expired entries
                    for key in expired_keys:
                        self._remove_entry(key)

                    # Update cleanup stats after successful removal
                    self._stats.increment("cleanups")
                    logger.debug("Cleanup complete")

                    # Update max age stat
                    if self._cache:
                        max_age = max(now - e.timestamp for e in self._cache.values())
                        self._stats.set("max_age", max_age)

                # Always update last cleanup time
                self._last_cleanup = now

            except Exception as e:
                self._stats.increment("errors")
                logger.error(f"Error during cache cleanup: {e}")

    def _monitor_cache(self) -> None:
        """Background monitoring and cleanup of cache metrics."""
        while self._running:
            try:
                # First do cleanup
                with self._lock:
                    self._maybe_cleanup()

                # Then log stats
                stats = self._stats.get_all()
                logger.info("Cache statistics:")
                for metric, value in stats.items():
                    logger.info(f"  {metric}: {value}")

            except Exception as e:
                logger.error(f"Error monitoring cache: {e}")

            time.sleep(self._cleanup_interval)

    def shutdown(self) -> None:
        """Shutdown the cache and stop monitoring."""
        logger.debug("Shutting down cache")
        self._running = False
        if self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=1.0)

    def get_stats(self) -> Dict[str, Any]:
        """Get current cache statistics."""
        return self._stats.get_all()

    def reset_stats(self) -> None:
        """Reset all statistics to zero."""
        logger.debug("Resetting stats")
        self._stats.reset()


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
                logger.debug(f"Creating new cache: {name}")
                self._caches[name] = DynamicLRUCache(
                    ttl_seconds=ttl_seconds, cleanup_interval=cleanup_interval
                )
            return self._caches[name]

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all managed caches."""
        with self._lock:
            return {name: cache.get_stats() for name, cache in self._caches.items()}

    def shutdown(self) -> None:
        """Shutdown all caches."""
        logger.debug("Shutting down all caches")
        with self._lock:
            for cache in self._caches.values():
                cache.shutdown()


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
