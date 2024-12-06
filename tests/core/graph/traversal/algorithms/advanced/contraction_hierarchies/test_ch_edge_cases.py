"""Tests for edge cases in Contraction Hierarchies implementation."""

import pytest
import time
from typing import Dict, List, Set
from threading import Thread
from queue import Queue
import concurrent.futures
from datetime import datetime

from polaris.core.exceptions import GraphOperationError
from polaris.core.graph import Graph
from polaris.core.models import Edge, EdgeMetadata
from polaris.core.enums import RelationType
from polaris.core.graph.traversal.algorithms.advanced.contraction_hierarchies import (
    ContractionHierarchies,
)
from polaris.core.graph.traversal.algorithms.advanced.contraction_hierarchies.utils import (
    get_performance_stats,
)


def create_edge(from_node: str, to_node: str, weight: float, base_metadata: EdgeMetadata) -> Edge:
    """Create an edge with given parameters."""
    metadata = EdgeMetadata(
        created_at=base_metadata.created_at,
        last_modified=base_metadata.last_modified,
        confidence=base_metadata.confidence,
        source=base_metadata.source,
        weight=weight,
        custom_attributes=base_metadata.custom_attributes.copy(),
    )
    return Edge(
        from_entity=from_node,
        to_entity=to_node,
        relation_type=RelationType.RELATES_TO,
        metadata=metadata,
        impact_score=0.5,
    )


@pytest.fixture
def base_metadata() -> EdgeMetadata:
    """Create base metadata for testing."""
    now = datetime.now()
    return EdgeMetadata(
        created_at=now,
        last_modified=now,
        confidence=1.0,
        source="test",
        weight=1.0,
        custom_attributes={"test": True},
    )


@pytest.fixture
def complex_graph(base_metadata: EdgeMetadata) -> Graph:
    """Create a complex graph that could produce cycles."""
    g = Graph(edges=[])
    edges = [
        ("A", "B", 1.0),
        ("B", "C", 2.0),
        ("C", "D", 1.0),
        ("B", "D", 2.5),
        ("D", "E", 1.0),
    ]
    for from_node, to_node, weight in edges:
        edge = create_edge(from_node, to_node, weight, base_metadata)
        g.add_edge(edge)
    return g


@pytest.fixture
def large_graph(base_metadata: EdgeMetadata) -> Graph:
    """Create a larger graph for performance testing."""
    g = Graph(edges=[])
    # Create a grid-like graph
    size = 10
    for i in range(size):
        for j in range(size):
            node = f"{i},{j}"
            if i < size - 1:
                edge = create_edge(node, f"{i+1},{j}", 1.0, base_metadata)
                g.add_edge(edge)
            if j < size - 1:
                edge = create_edge(node, f"{i},{j+1}", 1.0, base_metadata)
                g.add_edge(edge)
    return g


@pytest.mark.timeout(5)
def test_single_node_graph(base_metadata: EdgeMetadata) -> None:
    """Test CH behavior with a graph containing only one node."""
    g = Graph(edges=[])
    # Add a self-loop edge
    edge = create_edge("A", "A", 1.0, base_metadata)
    g.add_edge(edge)

    ch = ContractionHierarchies(g)
    ch.preprocess()

    # Test path finding to self
    result = ch.find_path("A", "A")
    assert result.path == [], "Path to self should be empty"
    assert result.total_weight == pytest.approx(0.0), "Path to self should have zero weight"

    # Test error messages with helper function
    def get_expected_error(node: str, is_start: bool) -> str:
        """Get expected error message for node not found."""
        return f"{'Start' if is_start else 'End'} node {node} not found in graph"

    error_cases = [
        ("A", "B", False),  # End node not found
        ("B", "A", True),  # Start node not found
        ("B", "C", True),  # Start node not found (different target)
    ]

    for start, end, is_start in error_cases:
        expected_error = get_expected_error(end if not is_start else start, is_start)
        with pytest.raises(GraphOperationError, match=expected_error):
            ch.find_path(start, end)


@pytest.mark.timeout(5)
def test_complex_cyclic_shortcuts(complex_graph: Graph) -> None:
    """Test path finding with shortcuts that could create cycles."""
    ch = ContractionHierarchies(complex_graph)
    ch.preprocess()

    # Test cases that could create cycles
    test_cases = [
        ("A", "E", ["A", "B", "D", "E"]),  # Should find direct path
        ("B", "E", ["B", "D", "E"]),  # Should avoid C->D->B cycle
        ("C", "E", ["C", "D", "E"]),  # Should avoid D->B->D cycle
    ]

    for start, end, expected_path in test_cases:
        result = ch.find_path(start, end)
        actual_path = [edge.from_entity for edge in result.path] + [result.path[-1].to_entity]
        assert actual_path == expected_path, f"Expected {expected_path}, got {actual_path}"


@pytest.mark.timeout(5)
def test_concurrent_preprocessing(base_metadata: EdgeMetadata) -> None:
    """Test concurrent preprocessing operations."""
    errors = Queue()
    graphs = []

    # Create multiple similar but slightly different graphs
    for i in range(3):
        g = Graph(edges=[])
        edges = [
            ("A", "B", 1.0 + i * 0.1),
            ("B", "C", 2.0 + i * 0.1),
            ("C", "D", 1.0 + i * 0.1),
        ]
        for from_node, to_node, weight in edges:
            edge = create_edge(from_node, to_node, weight, base_metadata)
            g.add_edge(edge)
        graphs.append(g)

    def preprocess_graph(graph: Graph) -> None:
        try:
            ch = ContractionHierarchies(graph)
            ch.preprocess()
            # Verify path finding works
            result = ch.find_path("A", "D")
            assert result is not None
        except Exception as e:
            errors.put(e)

    # Start concurrent preprocessing
    threads = [Thread(target=preprocess_graph, args=(g,)) for g in graphs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Check for errors
    assert (
        errors.empty()
    ), f"Encountered errors during concurrent preprocessing: {list(errors.queue)}"


@pytest.mark.timeout(5)
def test_error_messages() -> None:
    """Test standardized error messages."""
    g = Graph(edges=[])
    ch = ContractionHierarchies(g)
    ch.preprocess()

    error_cases = [
        ("A", "B", "End node B not found in graph"),
        ("B", "A", "Start node B not found in graph"),
        ("B", "C", "Start node B not found in graph"),
    ]

    for start, end, expected_error in error_cases:
        with pytest.raises(GraphOperationError, match=expected_error):
            ch.find_path(start, end)


@pytest.mark.timeout(5)
def test_path_validation(complex_graph: Graph) -> None:
    """Test path validation with cycles."""
    ch = ContractionHierarchies(complex_graph)
    ch.preprocess()

    # Test that paths with cycles are rejected
    result = ch.find_path("A", "E")
    path_nodes = [edge.from_entity for edge in result.path] + [result.path[-1].to_entity]

    # Verify no node appears more than once in the path
    seen_nodes = set()
    for node in path_nodes:
        assert node not in seen_nodes, f"Node {node} appears multiple times in path"
        seen_nodes.add(node)

    # Verify path is optimal
    assert path_nodes == ["A", "B", "D", "E"], "Path should be optimal A->B->D->E"


@pytest.mark.timeout(10)
def test_performance_monitoring(large_graph: Graph) -> None:
    """Test performance monitoring capabilities."""
    ch = ContractionHierarchies(large_graph)
    ch.preprocess()

    # Perform multiple path findings to generate metrics
    start_nodes = ["0,0", "0,5", "5,0"]
    end_nodes = ["9,9", "9,5", "5,9"]

    for start, end in zip(start_nodes, end_nodes):
        ch.find_path(start, end)

    # Get performance metrics
    metrics = get_performance_stats()

    # Verify metrics structure
    assert "path_finding_time" in metrics
    assert "witness_search_time" in metrics
    assert "shortcut_necessity_check_time" in metrics

    # Verify metric contents
    for metric_name, stats in metrics.items():
        assert "avg" in stats
        assert "min" in stats
        assert "max" in stats
        assert "std" in stats
        assert "count" in stats
        assert stats["count"] > 0


@pytest.mark.timeout(10)
def test_cache_behavior(large_graph: Graph) -> None:
    """Test caching behavior and efficiency."""
    ch = ContractionHierarchies(large_graph)
    ch.preprocess()

    # First path finding should populate cache
    start_time = time.time()
    first_result = ch.find_path("0,0", "9,9")
    first_duration = time.time() - start_time

    # Second path finding should use cache
    start_time = time.time()
    second_result = ch.find_path("0,0", "9,9")
    second_duration = time.time() - start_time

    # Verify results are identical
    assert len(first_result.path) == len(second_result.path)
    assert first_result.total_weight == second_result.total_weight

    # Second query should be faster due to caching
    assert second_duration < first_duration


@pytest.mark.timeout(15)
def test_concurrent_path_finding(large_graph: Graph) -> None:
    """Test concurrent path finding operations."""
    ch = ContractionHierarchies(large_graph)
    ch.preprocess()

    def find_paths(start_nodes: Set[str], end_nodes: Set[str]) -> List[float]:
        """Find paths between multiple node pairs."""
        results = []
        for start in start_nodes:
            for end in end_nodes:
                result = ch.find_path(start, end)
                results.append(result.total_weight)
        return results

    # Create different node sets for each thread
    thread1_starts = {"0,0", "0,5"}
    thread1_ends = {"9,9", "9,5"}
    thread2_starts = {"5,0", "5,5"}
    thread2_ends = {"9,0", "9,5"}

    # Run path finding concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(find_paths, thread1_starts, thread1_ends)
        future2 = executor.submit(find_paths, thread2_starts, thread2_ends)

        # Get results
        results1 = future1.result()
        results2 = future2.result()

    # Verify results
    assert len(results1) == len(thread1_starts) * len(thread1_ends)
    assert len(results2) == len(thread2_starts) * len(thread2_ends)
    assert all(w > 0 for w in results1 + results2)


@pytest.mark.timeout(10)
def test_memory_management(large_graph: Graph) -> None:
    """Test memory management and cleanup."""
    ch = ContractionHierarchies(large_graph)
    ch.preprocess()

    # Get initial memory stats
    initial_stats = ch.storage.get_memory_usage()

    # Perform operations that should affect memory
    for _ in range(5):
        ch.find_path("0,0", "9,9")

    # Get updated memory stats
    updated_stats = ch.storage.get_memory_usage()

    # Verify memory tracking
    assert "shortcuts" in updated_stats
    assert "node_levels" in updated_stats
    assert updated_stats["shortcuts"] >= initial_stats["shortcuts"]

    # Test cleanup
    ch.storage.clear()
    cleared_stats = ch.storage.get_memory_usage()
    assert cleared_stats["shortcuts"] == 0
    assert cleared_stats["node_levels"] == 0


@pytest.mark.timeout(5)
def test_error_handling_edge_cases(complex_graph: Graph) -> None:
    """Test error handling for various edge cases."""
    ch = ContractionHierarchies(complex_graph)
    ch.preprocess()

    # Test invalid node combinations
    with pytest.raises(GraphOperationError, match="Start node.*not found"):
        ch.find_path("NonExistent", "A")

    with pytest.raises(GraphOperationError, match="End node.*not found"):
        ch.find_path("A", "NonExistent")

    # Test with None values
    with pytest.raises(Exception):  # Type error or value error
        ch.find_path(None, "A")  # type: ignore

    with pytest.raises(Exception):  # Type error or value error
        ch.find_path("A", None)  # type: ignore

    # Test with empty strings
    with pytest.raises(GraphOperationError):
        ch.find_path("", "A")

    with pytest.raises(GraphOperationError):
        ch.find_path("A", "")
