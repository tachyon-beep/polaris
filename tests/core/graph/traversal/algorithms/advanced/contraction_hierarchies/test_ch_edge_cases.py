"""Tests for edge cases in Contraction Hierarchies implementation."""

import pytest
from typing import Dict, List
from threading import Thread
from queue import Queue

from polaris.core.exceptions import GraphOperationError
from polaris.core.graph import Graph
from polaris.core.models import Edge, EdgeMetadata
from polaris.core.graph.traversal.algorithms.advanced.contraction_hierarchies import (
    ContractionHierarchies,
)


def create_edge(from_node: str, to_node: str, weight: float, base_metadata: EdgeMetadata) -> Edge:
    """Create an edge with given parameters."""
    metadata = EdgeMetadata(
        weight=weight,
        custom_attributes={},
        properties=base_metadata.properties.copy(),
    )
    return Edge(from_entity=from_node, to_entity=to_node, metadata=metadata)


@pytest.fixture
def base_metadata() -> EdgeMetadata:
    """Create base metadata for testing."""
    return EdgeMetadata(
        weight=1.0,
        custom_attributes={},
        properties={"test": True},
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
