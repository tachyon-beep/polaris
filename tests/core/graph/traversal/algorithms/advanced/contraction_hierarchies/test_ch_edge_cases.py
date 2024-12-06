"""
Tests for Contraction Hierarchies edge cases and error handling.
"""

import pytest
from typing import List, Optional, Callable, Tuple

from polaris.core.exceptions import GraphOperationError
from polaris.core.graph import Graph
from polaris.core.graph.traversal.algorithms.advanced.contraction_hierarchies import (
    ContractionHierarchies,
)
from polaris.core.models import Edge, EdgeMetadata

from .conftest import create_edge


@pytest.mark.timeout(5)
def test_path_finding_with_filter_no_valid_path(simple_graph: Graph) -> None:
    """Test path finding with a filter function that allows no valid paths."""
    ch = ContractionHierarchies(simple_graph)
    ch.preprocess()

    # Define a filter function that disallows all paths
    def filter_func(path: List[Edge]) -> bool:
        return False

    # Attempt to find path from A to C with restrictive filter
    with pytest.raises(GraphOperationError, match="No path satisfying filter exists"):
        ch.find_path("A", "C", filter_func=filter_func)


@pytest.mark.timeout(5)
def test_empty_graph() -> None:
    """Test CH behavior with an empty graph."""
    empty_graph = Graph(edges=[])
    ch = ContractionHierarchies(empty_graph)
    ch.preprocess()

    # Verify empty state
    assert len(ch.state.shortcuts) == 0, "Empty graph should have no shortcuts"
    assert len(ch.state.node_level) == 0, "Empty graph should have no node levels"
    assert (
        len(ch.state.contracted_neighbors) == 0
    ), "Empty graph should have no contracted neighbors"

    # Test path finding in empty graph
    with pytest.raises(GraphOperationError, match="Start node A not found in graph"):
        ch.find_path("A", "B")


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

    # Test path finding to non-existent node
    with pytest.raises(GraphOperationError, match="Start node B not found in graph"):
        ch.find_path("A", "B")


@pytest.mark.timeout(5)
def test_invalid_node_combinations(simple_graph: Graph) -> None:
    """Test CH behavior with various invalid node combinations."""
    ch = ContractionHierarchies(simple_graph)
    ch.preprocess()

    test_cases = [
        ("", "A", "Empty start node"),
        ("A", "", "Empty end node"),
        ("", "", "Both nodes empty"),
        (" ", "A", "Whitespace start node"),
        ("A", " ", "Whitespace end node"),
        ("A", "A ", "End node with trailing space"),
        (" B", "C", "Start node with leading space"),
    ]

    for start, end, description in test_cases:
        with pytest.raises(GraphOperationError, match="Start node .* not found in graph"):
            ch.find_path(start, end)


@pytest.mark.timeout(5)
def test_duplicate_nodes_in_path(simple_graph: Graph, base_metadata: EdgeMetadata) -> None:
    """Test CH behavior when path contains duplicate nodes."""
    # Add edges to create a cycle
    cycle_edges = [
        ("B", "D", 1.0),
        ("D", "B", 1.0),
    ]

    for from_node, to_node, weight in cycle_edges:
        edge = create_edge(from_node, to_node, weight, base_metadata)
        simple_graph.add_edge(edge)

    ch = ContractionHierarchies(simple_graph)
    ch.preprocess()

    # Test path finding that might involve cycles
    result = ch.find_path("A", "C")

    # Verify no node appears more than once in the path (except start/end nodes in special cases)
    if result.path:
        nodes = [edge.from_entity for edge in result.path] + [result.path[-1].to_entity]
        node_counts = {node: nodes.count(node) for node in set(nodes)}
        for node, count in node_counts.items():
            assert count <= 1, f"Node {node} appears {count} times in path"


@pytest.mark.timeout(5)
def test_edge_weight_edge_cases(simple_graph: Graph, base_metadata: EdgeMetadata) -> None:
    """Test CH behavior with edge case weights."""
    # Test cases for different edge weights
    edge_cases = [
        ("X", "Y", 0.0, "Zero weight"),
        ("Y", "Z", float("inf"), "Infinite weight"),
        ("Z", "W", 1e-10, "Very small weight"),
        ("W", "V", 1e10, "Very large weight"),
    ]

    # Add test edges
    for from_node, to_node, weight, _ in edge_cases:
        edge = create_edge(from_node, to_node, weight, base_metadata)
        simple_graph.add_edge(edge)

    ch = ContractionHierarchies(simple_graph)
    ch.preprocess()

    # Test path finding with edge case weights
    for from_node, to_node, weight, description in edge_cases:
        if weight != float("inf"):
            result = ch.find_path(from_node, to_node)
            assert result.path is not None, f"{description}: Path should exist"
            assert result.total_weight == pytest.approx(
                weight
            ), f"{description}: Path weight should be {weight}"
        else:
            # Infinite weight edges should be treated as no path
            with pytest.raises(GraphOperationError, match="No path exists"):
                ch.find_path(from_node, to_node)


@pytest.mark.timeout(5)
def test_filter_edge_cases(simple_graph: Graph) -> None:
    """Test CH behavior with edge case filter functions."""
    ch = ContractionHierarchies(simple_graph)
    ch.preprocess()

    test_cases: List[Tuple[Optional[Callable[[List[Edge]], bool]], str]] = [
        (None, "No filter provided"),
        (lambda p: True, "Always true filter"),
        (lambda p: len(p) < 100, "Unreachable path length filter"),
        (lambda p: all(e.metadata.weight > 0 for e in p), "Positive weight filter"),
    ]

    for filter_func, description in test_cases:
        result = ch.find_path("A", "B", filter_func=filter_func)
        assert result.path is not None, f"{description}: Path should exist"
        assert len(result.path) == 1, f"{description}: Path should be direct A->B"
        assert result.total_weight == pytest.approx(
            1.0
        ), f"{description}: Path weight should be 1.0"


@pytest.mark.timeout(5)
def test_concurrent_modifications(simple_graph: Graph, base_metadata: EdgeMetadata) -> None:
    """Test CH behavior with concurrent graph modifications during path finding."""
    ch = ContractionHierarchies(simple_graph)
    ch.preprocess()

    # Start path finding
    result1 = ch.find_path("A", "B")
    assert result1.path is not None, "Original path should exist"
    assert result1.total_weight == pytest.approx(1.0), "Original path weight should be 1.0"

    # Modify graph
    edge = create_edge("A", "D", 1.0, base_metadata)
    simple_graph.add_edge(edge)

    # Reprocess and find new path
    ch.preprocess()
    result2 = ch.find_path("A", "D")
    assert result2.path is not None, "New path should exist after modification"
    assert result2.total_weight == pytest.approx(1.0), "New path should have weight 1.0"


@pytest.mark.timeout(5)
def test_invalid_filter_functions(simple_graph: Graph) -> None:
    """Test CH behavior with invalid filter functions."""
    ch = ContractionHierarchies(simple_graph)
    ch.preprocess()

    def invalid_filter(path: List[Edge]) -> None:
        return None  # type: ignore

    def raising_filter(path: List[Edge]) -> bool:
        raise ValueError("Filter error")

    test_cases = [
        (invalid_filter, TypeError, "Filter returning None"),
        (raising_filter, ValueError, "Filter raising exception"),
    ]

    for filter_func, expected_error, description in test_cases:
        with pytest.raises(expected_error):
            ch.find_path("A", "C", filter_func=filter_func)
