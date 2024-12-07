"""
Tests for Contraction Hierarchies path finding functionality.
"""

import pytest
from typing import List, Callable

from polaris.core.exceptions import GraphOperationError
from polaris.core.graph import Graph
from polaris.core.graph.traversal.algorithms.advanced.contraction_hierarchies import (
    ContractionHierarchies,
)
from polaris.core.models import Edge

from .conftest import create_edge


@pytest.mark.timeout(5)
@pytest.mark.parametrize(
    "start, end, expected_weight, expected_path_length, description",
    [
        ("A", "C", 3.0, 2, "Simple path A->B->C"),
        ("A", "B", 1.0, 1, "Direct path A->B"),
        ("B", "C", 2.0, 1, "Direct path B->C"),
        ("A", "A", 0.0, 0, "Path to self"),
    ],
)
def test_multiple_path_findings(
    simple_graph: Graph,
    start: str,
    end: str,
    expected_weight: float,
    expected_path_length: int,
    description: str,
) -> None:
    """Test multiple path findings with varying start and end nodes."""
    ch = ContractionHierarchies(simple_graph)
    ch.preprocess()

    if start == end:
        # Path to self should have zero weight and no edges
        result = ch.find_path(start, end)
        assert result.path == [], f"{description}: Path should be empty"
        assert result.total_weight == pytest.approx(
            0.0
        ), f"{description}: Total weight should be 0.0"
    else:
        result = ch.find_path(start, end)
        # Verify path exists
        assert result.path is not None, f"{description}: Path does not exist"
        assert len(result.path) == expected_path_length, f"{description}: Incorrect path length"

        # Check path endpoints
        assert result.path[0].from_entity == start, f"{description}: Incorrect start node"
        assert result.path[-1].to_entity == end, f"{description}: Incorrect end node"

        # Verify total weight
        assert result.total_weight == pytest.approx(
            expected_weight
        ), f"{description}: Expected weight {expected_weight}, got {result.total_weight}"


@pytest.mark.timeout(5)
@pytest.mark.parametrize(
    "start, end",
    [
        ("X", "A"),  # Non-existent start node
        ("A", "Y"),  # Non-existent end node
        ("X", "Y"),  # Both nodes non-existent
    ],
)
def test_error_handling_invalid_nodes(simple_graph: Graph, start: str, end: str) -> None:
    """Test CH raises errors for invalid nodes."""
    ch = ContractionHierarchies(simple_graph)
    ch.preprocess()

    with pytest.raises(GraphOperationError, match=f"Start node {start} not found in graph"):
        ch.find_path(start, end)


@pytest.mark.timeout(5)
def test_path_finding_with_filter(simple_graph: Graph) -> None:
    """Test path finding with a filter function."""
    ch = ContractionHierarchies(simple_graph)
    ch.preprocess()

    # Define filter functions for different scenarios
    def max_weight_filter(path: List[Edge]) -> bool:
        return sum(edge.metadata.weight for edge in path) <= 3.0

    def max_edges_filter(path: List[Edge]) -> bool:
        return len(path) <= 2

    def combined_filter(path: List[Edge]) -> bool:
        return max_weight_filter(path) and max_edges_filter(path)

    test_cases = [
        (max_weight_filter, "Weight-based filter"),
        (max_edges_filter, "Edge count filter"),
        (combined_filter, "Combined filter"),
    ]

    for filter_func, description in test_cases:
        result = ch.find_path("A", "C", filter_func=filter_func)

        # Verify path exists and meets filter criteria
        assert result.path is not None, f"{description}: Path should exist"
        assert filter_func(result.path), f"{description}: Path should satisfy filter conditions"
        assert len(result.path) > 0, f"{description}: Path should not be empty"


@pytest.mark.timeout(5)
def test_path_finding_with_disconnected_components(simple_graph: Graph, base_metadata) -> None:
    """Test path finding behavior with disconnected components."""
    # Add a disconnected component
    disconnected_edges = [
        ("X", "Y", 1.0),
        ("Y", "Z", 2.0),
    ]

    for from_node, to_node, weight in disconnected_edges:
        edge = create_edge(from_node, to_node, weight, base_metadata)
        simple_graph.add_edge(edge)

    ch = ContractionHierarchies(simple_graph)
    ch.preprocess()

    # Test path finding within components
    result = ch.find_path("X", "Z")
    assert result.path is not None, "Path within disconnected component should exist"
    assert result.total_weight == pytest.approx(3.0), "Path X->Y->Z should have weight 3.0"

    # Test path finding between components
    with pytest.raises(GraphOperationError, match="No path exists"):
        ch.find_path("A", "X")


@pytest.mark.timeout(5)
def test_path_finding_with_cycles(complex_graph: Graph) -> None:
    """Test path finding in a graph with cycles."""
    ch = ContractionHierarchies(complex_graph)
    ch.preprocess()

    test_cases = [
        ("A", "E", 7.0, ["A", "B", "C", "D", "E"]),  # Forward path through cycle
        ("E", "A", 6.0, ["E", "A"]),  # Direct path back
        ("B", "B", 0.0, ["B"]),  # Self-loop
        ("C", "A", 10.0, ["C", "E", "A"]),  # Path using cycle
    ]

    for start, end, expected_weight, expected_sequence in test_cases:
        result = ch.find_path(start, end)

        if start == end:
            assert result.path == [], f"Path from {start} to self should be empty"
            assert result.total_weight == pytest.approx(
                0.0
            ), f"Path from {start} to self should have zero weight"
        else:
            assert result.path is not None, f"Path from {start} to {end} should exist"
            actual_sequence = [edge.from_entity for edge in result.path] + [
                result.path[-1].to_entity
            ]
            assert actual_sequence == expected_sequence, (
                f"Path from {start} to {end} sequence mismatch. "
                f"Expected {expected_sequence}, got {actual_sequence}"
            )
            assert result.total_weight == pytest.approx(expected_weight), (
                f"Path from {start} to {end} weight mismatch. "
                f"Expected {expected_weight}, got {result.total_weight}"
            )


@pytest.mark.timeout(5)
def test_path_finding_with_zero_weight_edges(simple_graph: Graph, base_metadata) -> None:
    """Test path finding with zero-weight edges."""
    # Add zero-weight edges
    zero_weight_edges = [
        ("B", "D", 0.0),
        ("D", "C", 0.0),
    ]

    for from_node, to_node, weight in zero_weight_edges:
        edge = create_edge(from_node, to_node, weight, base_metadata)
        simple_graph.add_edge(edge)

    ch = ContractionHierarchies(simple_graph)
    ch.preprocess()

    # Test path finding through zero-weight edges
    result = ch.find_path("A", "C")
    assert result.path is not None, "Path through zero-weight edges should exist"
    assert result.total_weight == pytest.approx(1.0), "Path A->B->D->C should have weight 1.0"

    # Verify the path sequence
    expected_sequence = ["A", "B", "D", "C"]
    actual_sequence = [edge.from_entity for edge in result.path] + [result.path[-1].to_entity]
    assert (
        actual_sequence == expected_sequence
    ), "Path should prefer route through zero-weight edges"
