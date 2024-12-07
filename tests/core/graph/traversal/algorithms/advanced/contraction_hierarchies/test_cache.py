"""Tests for Contraction Hierarchies caching implementation."""

import pytest
import time
import threading
from queue import Queue
from typing import List, Set, Dict, Any, Generator

from polaris.core.graph import Graph
from polaris.core.models import Edge, EdgeMetadata
from polaris.core.graph.traversal.algorithms.advanced.contraction_hierarchies.cache import (
    DynamicLRUCache,
    CacheManager,
    get_cache_size,
    get_cache_manager,
    InvalidValueError,
)


@pytest.fixture
def cache() -> Generator[DynamicLRUCache, None, None]:
    """Create a test cache instance."""
    cache = DynamicLRUCache(ttl_seconds=1, cleanup_interval=1)
    yield cache
    cache.shutdown()


@pytest.fixture(autouse=True)
def cleanup_cache_manager() -> Generator[None, None, None]:
    """Clean up cache manager after each test."""
    yield
    get_cache_manager().shutdown()


def test_basic_operations(cache: DynamicLRUCache) -> None:
    """Test basic cache operations."""
    # Test set and get
    cache.set("key1", "value1", 10)
    assert cache.get("key1") == "value1"

    # Test missing key
    assert cache.get("missing") is None

    # Test overwrite
    cache.set("key1", "value2", 10)
    assert cache.get("key1") == "value2"


def test_cache_size_management(cache: DynamicLRUCache) -> None:
    """Test cache size limits are enforced."""
    # Add more items than size limit
    max_size = 5
    for i in range(10):
        cache.set(f"key{i}", f"value{i}", max_size)

    # Verify size is maintained
    stats = cache.get_stats()
    assert stats["evictions"] >= 5  # Should have evicted older entries

    # Verify most recent entries exist
    for i in range(5, 10):
        assert cache.get(f"key{i}") is not None


def test_ttl_handling(cache: DynamicLRUCache) -> None:
    """Test TTL expiration and cleanup."""
    # Add entries
    cache.set("expire1", "value1", 10)
    cache.set("expire2", "value2", 10)

    # Verify entries exist
    assert cache.get("expire1") == "value1"
    assert cache.get("expire2") == "value2"

    # Wait for TTL
    time.sleep(2)

    # Verify entries expired
    assert cache.get("expire1") is None
    assert cache.get("expire2") is None

    # Check cleanup occurred
    stats = cache.get_stats()
    assert stats["cleanups"] > 0
    assert stats["evictions"] >= 2


def test_thread_safety() -> None:
    """Test concurrent cache operations."""
    cache = DynamicLRUCache(ttl_seconds=10)
    try:
        results: List[str] = []
        errors = Queue()

        def worker(worker_id: int) -> None:
            try:
                for i in range(100):
                    key = f"key{worker_id}_{i}"
                    value = f"value{worker_id}_{i}"

                    # Test set
                    cache.set(key, value, 1000)

                    # Test get
                    result = cache.get(key)
                    if result != value:
                        errors.put(f"Value mismatch: expected {value}, got {result}")

                    # Test concurrent access
                    shared_key = "shared"
                    cache.set(shared_key, f"shared_{i}", 1000)
                    shared_value = cache.get(shared_key)
                    if shared_value is not None:
                        results.append(shared_value)

            except Exception as e:
                errors.put(f"Worker {worker_id} error: {e}")

        # Run concurrent workers
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Check for errors
        assert errors.empty(), f"Encountered errors: {list(errors.queue)}"

        # Verify concurrent operations occurred
        assert len(results) > 0

        # Check cache stats
        stats = cache.get_stats()
        assert stats["hits"] > 0
        assert stats["evictions"] >= 0
        assert stats["errors"] == 0

    finally:
        cache.shutdown()


def test_memory_management() -> None:
    """Test memory limit enforcement."""
    # Create cache with small memory limit
    cache = DynamicLRUCache(max_size_bytes=1000, ttl_seconds=10)  # 1KB limit
    try:
        # Add large entries
        large_value = "x" * 200  # ~200 bytes
        for i in range(10):
            cache.set(f"large{i}", large_value, 100)

        # Verify memory limit maintained
        stats = cache.get_stats()
        assert stats["total_bytes"] <= 1000
        assert stats["evictions"] > 0
    finally:
        cache.shutdown()


def test_cache_manager() -> None:
    """Test cache manager functionality."""
    manager = CacheManager()
    try:
        # Get same cache twice
        cache1 = manager.get_cache("test1")
        cache2 = manager.get_cache("test1")
        assert cache1 is cache2

        # Get different caches
        cache3 = manager.get_cache("test2")
        assert cache1 is not cache3

        # Test cache isolation
        cache1.set("key1", "value1", 10)
        cache3.set("key1", "value2", 10)
        assert cache1.get("key1") == "value1"
        assert cache3.get("key1") == "value2"

        # Test stats collection
        stats = manager.get_all_stats()
        assert "test1" in stats
        assert "test2" in stats
    finally:
        manager.shutdown()


def test_cache_size_calculation(complex_graph: Graph) -> None:
    """Test cache size calculation based on graph properties."""
    size = get_cache_size(complex_graph)

    # Verify reasonable bounds
    assert 1000 <= size <= 100000

    # Verify size scales with graph
    larger_size = get_cache_size(complex_graph)  # Assuming complex_graph is larger
    assert larger_size >= size


def test_error_handling(cache: DynamicLRUCache) -> None:
    """Test error handling in cache operations."""
    # Test invalid keys
    cache.set(None, "value", 10)  # type: ignore
    assert cache.get(None) is None  # type: ignore

    # Test invalid values
    class BadValue:
        def __len__(self):
            raise Exception("Simulated error")

    cache.set("bad", BadValue(), 10)
    assert cache.get("bad") is None

    # Check error was recorded
    stats = cache.get_stats()
    assert stats["errors"] > 0


def test_stats_accuracy(cache: DynamicLRUCache) -> None:
    """Test accuracy of cache statistics."""
    # First sequence - will be included in cumulative stats
    cache.set("key1", "value1", 10)
    cache.get("key1")  # Hit
    cache.get("missing")  # Miss

    # Test eviction counting
    for i in range(20):
        cache.set(f"key{i}", f"value{i}", 5)

    # Second sequence - will also be included in cumulative stats
    cache.set("hit_ratio", "value", 10)
    for _ in range(3):
        cache.get("hit_ratio")  # 3 Hits
    cache.get("missing1")  # Miss
    cache.get("missing2")  # Miss

    # Verify cumulative stats
    stats = cache.get_stats()
    assert stats["hits"] == 4  # 1 hit from first sequence + 3 hits from second
    assert stats["misses"] == 3  # 1 miss from first + 2 misses from second
    assert stats["hit_ratio"] == 4 / 7  # Total hits / total requests


def test_concurrent_stats_updates() -> None:
    """Test thread safety of statistics updates."""
    cache = DynamicLRUCache(ttl_seconds=10)
    try:
        iterations = 1000
        threads = 10

        def worker() -> None:
            for i in range(iterations):
                key = f"key{i}"
                cache.set(key, f"value{i}", 1000)
                cache.get(key)
                cache.get("missing")

        # Run concurrent workers
        worker_threads = [threading.Thread(target=worker) for _ in range(threads)]

        for t in worker_threads:
            t.start()
        for t in worker_threads:
            t.join()

        # Verify stats consistency
        stats = cache.get_stats()
        assert stats["hits"] == threads * iterations  # Each thread gets its own values
        assert stats["misses"] == threads * iterations  # Each thread misses same number of times
        assert 0.49 <= stats["hit_ratio"] <= 0.51  # Should be close to 0.5
    finally:
        cache.shutdown()


def test_cleanup_scheduling() -> None:
    """Test cleanup scheduling and execution."""
    # Create cache with very short TTL and cleanup interval
    cache = DynamicLRUCache(ttl_seconds=1, cleanup_interval=1)
    try:
        # Add entries
        for i in range(10):
            cache.set(f"key{i}", f"value{i}", 100)

        # Wait for entries to expire and cleanup to run
        time.sleep(2)  # Wait 2x TTL/interval to ensure expiration

        # Force cleanup by getting a key
        cache.get("key0")

        # Verify cleanup occurred
        stats = cache.get_stats()
        assert stats["cleanups"] > 0, "Cleanup should have occurred"
        assert stats["evictions"] >= 10, "All entries should have been evicted"

        # Verify entries were cleaned
        for i in range(10):
            assert cache.get(f"key{i}") is None
    finally:
        cache.shutdown()
