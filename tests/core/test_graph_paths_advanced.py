"""
Advanced test cases for graph path finding algorithms.

These tests focus on edge cases, performance limits, and robustness:
- Concurrent modifications
- Large graph handling
- Numeric edge cases
- Memory management
- Resource cleanup
"""

import threading
import time
from datetime import datetime
from typing import List, Optional, Set
import pytest
import random
import gc
import math
from concurrent.futures import ThreadPoolExecutor

from polaris.core.enums import RelationType
from polaris.core.exceptions import GraphOperationError, NodeNotFoundError
from polaris.core.graph import Graph
from polaris.core.graph_paths import PathFinding, PathResult, PathType
from polaris.core.graph_paths.cache import PathCache
from polaris.core.models import Edge, EdgeMetadata


def create_large_graph(size: int, edge_density: float = 0.1) -> Graph:
    """Create a large test graph with specified size and density."""
    edges = []
    metadata = EdgeMetadata(
        created_at=datetime.now(),
        last_modified=datetime.now(),
        confidence=0.9,
        source="test",
    )

    # Create a backbone path to ensure connectivity
    for i in range(size - 1):
        edges.append(
            Edge(
                from_entity=f"node_{i}",
                to_entity=f"node_{i+1}",
                relation_type=RelationType.DEPENDS_ON,
                metadata=metadata,
                impact_score=random.random(),  # Random value between 0 and 1
            )
        )

    # Add random edges based on density
    max_edges = size * (size - 1)
    target_edges = int(max_edges * edge_density)
    while len(edges) < target_edges:
        from_node = random.randint(0, size - 1)
        to_node = random.randint(0, size - 1)
        if from_node != to_node:
            edges.append(
                Edge(
                    from_entity=f"node_{from_node}",
                    to_entity=f"node_{to_node}",
                    relation_type=RelationType.DEPENDS_ON,
                    metadata=metadata,
                    impact_score=random.random(),  # Random value between 0 and 1
                )
            )

    return Graph(edges=edges)


@pytest.fixture
def large_graph():
    """Fixture providing a large test graph."""
    return create_large_graph(10000, 0.001)  # 10K nodes with 0.1% density


def test_concurrent_modifications():
    """Test path finding during concurrent graph modifications."""
    graph = create_large_graph(1000, 0.01)
    modification_count = 0
    path_finding_count = 0
    errors = []

    def modify_graph():
        nonlocal modification_count
        try:
            for _ in range(10):
                # Add and remove random edges
                from_node = f"node_{random.randint(0, 999)}"
                to_node = f"node_{random.randint(0, 999)}"
                edge = Edge(
                    from_entity=from_node,
                    to_entity=to_node,
                    relation_type=RelationType.DEPENDS_ON,
                    metadata=EdgeMetadata(
                        created_at=datetime.now(),
                        last_modified=datetime.now(),
                        confidence=0.9,
                        source="test",
                    ),
                    impact_score=random.random(),  # Random value between 0 and 1
                )
                graph.add_edge(edge)
                time.sleep(0.001)
                graph.remove_edge(from_node, to_node)
                modification_count += 1
        except Exception as e:
            errors.append(e)

    def find_paths():
        nonlocal path_finding_count
        try:
            for _ in range(10):
                start = f"node_{random.randint(0, 999)}"
                end = f"node_{random.randint(0, 999)}"
                try:
                    PathFinding.shortest_path(graph, start, end)
                    path_finding_count += 1
                except (GraphOperationError, NodeNotFoundError):
                    # These are expected occasionally due to concurrent modifications
                    pass
                time.sleep(0.001)
        except Exception as e:
            errors.append(e)

    # Run concurrent operations
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for _ in range(2):
            futures.append(executor.submit(modify_graph))
        for _ in range(2):
            futures.append(executor.submit(find_paths))

    assert not errors, f"Encountered errors: {errors}"
    assert modification_count > 0, "No modifications occurred"
    assert path_finding_count > 0, "No paths were found"


def test_large_graph_performance(large_graph):
    """Test performance with very large graphs."""
    # Test shortest path
    start_time = time.time()
    path = PathFinding.shortest_path(large_graph, "node_0", "node_9999")
    duration = time.time() - start_time
    assert duration < 30.0, "Shortest path took too long"  # Increased timeout for larger graphs
    assert isinstance(path, PathResult)

    # Test memory usage
    initial_memory = get_memory_usage()
    paths = list(PathFinding.all_paths(large_graph, "node_0", "node_100", max_length=3))
    final_memory = get_memory_usage()
    memory_increase = final_memory - initial_memory
    assert memory_increase < 100 * 1024 * 1024, "Excessive memory usage"  # 100MB limit


def test_edge_weight_overflow():
    """Test handling of edge weight overflow scenarios."""
    graph = Graph(
        edges=[
            Edge(
                from_entity="A",
                to_entity="B",
                relation_type=RelationType.DEPENDS_ON,
                metadata=EdgeMetadata(
                    created_at=datetime.now(),
                    last_modified=datetime.now(),
                    confidence=0.9,
                    source="test",
                ),
                impact_score=0.9,  # Valid impact score
            ),
            Edge(
                from_entity="B",
                to_entity="C",
                relation_type=RelationType.DEPENDS_ON,
                metadata=EdgeMetadata(
                    created_at=datetime.now(),
                    last_modified=datetime.now(),
                    confidence=0.9,
                    source="test",
                ),
                impact_score=0.8,  # Valid impact score
            ),
        ]
    )

    def overflow_weight_func(edge: Edge) -> float:
        # Simulate overflow by returning a very large number
        return float("inf") if edge.impact_score > 0.85 else 1.0

    with pytest.raises(ValueError, match="Path cost exceeded maximum value|Path cost overflow"):
        PathFinding.shortest_path(graph, "A", "C", weight_func=overflow_weight_func)


def test_cache_eviction(large_graph):
    """Test cache eviction behavior under memory pressure."""
    # Configure cache with small size
    PathCache.reconfigure(max_size=10, ttl=3600)

    # Perform many path findings to fill cache
    for i in range(20):  # More than cache size
        PathFinding.shortest_path(large_graph, f"node_{i}", f"node_{i+1}")

    metrics = PathFinding.get_cache_metrics()
    assert metrics["size"] <= 10, "Cache exceeded maximum size"


def test_memory_management_stress():
    """Test memory management under stress."""
    graphs = []
    initial_memory = get_memory_usage()

    try:
        # Create multiple large graphs
        for _ in range(5):
            graphs.append(create_large_graph(5000, 0.001))
            current_memory = get_memory_usage()
            if current_memory - initial_memory > 500 * 1024 * 1024:  # 500MB limit
                raise MemoryError("Excessive memory usage")

        # Perform path finding on each graph
        for graph in graphs:
            PathFinding.shortest_path(graph, "node_0", "node_4999")

    finally:
        # Clean up
        graphs.clear()
        gc.collect()

    final_memory = get_memory_usage()
    assert final_memory - initial_memory < 100 * 1024 * 1024, "Memory leak detected"


def test_path_finding_with_cycles():
    """Test path finding with cycles of varying lengths."""
    # Create a graph with multiple cycles
    edges = []
    metadata = EdgeMetadata(
        created_at=datetime.now(),
        last_modified=datetime.now(),
        confidence=0.9,
        source="test",
    )

    # Create cycles of different lengths
    for i in range(5):
        size = i + 2  # Cycles of length 2 to 6
        for j in range(size):
            edges.append(
                Edge(
                    from_entity=f"cycle_{i}_{j}",
                    to_entity=f"cycle_{i}_{(j+1)%size}",
                    relation_type=RelationType.DEPENDS_ON,
                    metadata=metadata,
                    impact_score=random.random(),  # Random value between 0 and 1
                )
            )

    graph = Graph(edges=edges)

    # Test path finding with different cycle lengths
    for i in range(5):
        paths = list(PathFinding.all_paths(graph, f"cycle_{i}_0", f"cycle_{i}_1", max_length=10))
        # Should find paths despite cycles
        assert len(paths) > 0
        # Verify no path contains a complete cycle
        for path in paths:
            nodes = set()
            for edge in path:
                assert edge.from_entity not in nodes, "Cycle detected in path"
                nodes.add(edge.from_entity)


def test_transaction_rollback():
    """Test transaction rollback with partial failures."""
    graph = create_large_graph(100, 0.1)
    initial_edge_count = len(list(graph.get_edges()))

    try:
        with graph.transaction():
            # Add some edges
            for i in range(10):
                edge = Edge(
                    from_entity=f"new_node_{i}",
                    to_entity=f"new_node_{i+1}",
                    relation_type=RelationType.DEPENDS_ON,
                    metadata=EdgeMetadata(
                        created_at=datetime.now(),
                        last_modified=datetime.now(),
                        confidence=0.9,
                        source="test",
                    ),
                    impact_score=random.random(),  # Random value between 0 and 1
                )
                graph.add_edge(edge)

            # Simulate failure
            raise ValueError("Simulated failure")
    except ValueError:
        pass

    # Verify rollback
    final_edge_count = len(list(graph.get_edges()))
    assert final_edge_count == initial_edge_count, "Transaction rollback failed"


def test_floating_point_precision():
    """Test handling of floating point precision issues in weights."""
    graph = Graph(
        edges=[
            Edge(
                from_entity="A",
                to_entity="B",
                relation_type=RelationType.DEPENDS_ON,
                metadata=EdgeMetadata(
                    created_at=datetime.now(),
                    last_modified=datetime.now(),
                    confidence=0.9,
                    source="test",
                ),
                impact_score=0.1,  # Small but valid impact score
            ),
            Edge(
                from_entity="B",
                to_entity="C",
                relation_type=RelationType.DEPENDS_ON,
                metadata=EdgeMetadata(
                    created_at=datetime.now(),
                    last_modified=datetime.now(),
                    confidence=0.9,
                    source="test",
                ),
                impact_score=0.9,  # Large but valid impact score
            ),
        ]
    )

    def precision_weight_func(edge: Edge) -> float:
        # Test handling of very small and very large weights
        return 1e-15 if edge.impact_score < 0.5 else 1e15

    # Should handle extreme values without precision loss
    path = PathFinding.shortest_path(graph, "A", "C", weight_func=precision_weight_func)
    assert isinstance(path, PathResult)
    assert len(path) == 2


def test_max_path_length_edge_cases(large_graph):
    """Test edge cases with maximum path length."""
    # Test with max_length=1
    with pytest.raises(GraphOperationError):
        PathFinding.shortest_path(large_graph, "node_0", "node_100", max_length=1)

    # Test with max_length exactly equal to shortest path
    path = PathFinding.shortest_path(large_graph, "node_0", "node_1")
    exact_length = len(path)
    result = PathFinding.shortest_path(large_graph, "node_0", "node_1", max_length=exact_length)
    assert len(result) == exact_length

    # Test with max_length one less than shortest path
    with pytest.raises(GraphOperationError):
        PathFinding.shortest_path(large_graph, "node_0", "node_1", max_length=exact_length - 1)


def test_resource_cleanup():
    """Test proper cleanup of resources."""
    graph = create_large_graph(1000, 0.01)
    initial_memory = get_memory_usage()

    for _ in range(10):
        # Perform memory-intensive operations
        paths = list(PathFinding.all_paths(graph, "node_0", "node_100", max_length=5))
        gc.collect()  # Force garbage collection

    # Verify memory is properly cleaned up
    final_memory = get_memory_usage()
    memory_increase = final_memory - initial_memory
    assert memory_increase < 50 * 1024 * 1024, "Memory not properly cleaned up"


def get_memory_usage() -> int:
    """Get current memory usage in bytes."""
    import psutil

    process = psutil.Process()
    return process.memory_info().rss
