"""Tests for Hub Labeling algorithm."""

import time
from datetime import datetime

import pytest

from polaris.core.enums import RelationType
from polaris.core.exceptions import GraphOperationError
from polaris.core.graph import Graph
from polaris.core.models import Edge, EdgeMetadata
from polaris.core.graph.traversal.algorithms.advanced.hub_labeling import HubLabeling


def create_test_graph() -> Graph:
    """Create a simple test graph."""
    edges = [
        Edge(
            from_entity="A",
            to_entity="B",
            relation_type=RelationType.CONNECTS_TO,
            metadata=EdgeMetadata(
                created_at=datetime.now(),
                last_modified=datetime.now(),
                confidence=1.0,
                source="test",
                weight=1.0,
            ),
            impact_score=0.5,
        ),
        Edge(
            from_entity="B",
            to_entity="C",
            relation_type=RelationType.CONNECTS_TO,
            metadata=EdgeMetadata(
                created_at=datetime.now(),
                last_modified=datetime.now(),
                confidence=1.0,
                source="test",
                weight=2.0,
            ),
            impact_score=0.5,
        ),
    ]
    return Graph(edges)


def test_basic_path_finding():
    """Test basic path finding functionality."""
    graph = create_test_graph()
    hl = HubLabeling(graph)
    hl.preprocess()

    # Test finding shortest path
    result = hl.find_path("A", "C")
    assert len(result.path) == 2
    assert result.path[0].from_entity == "A"
    assert result.path[0].to_entity == "B"
    assert result.path[1].from_entity == "B"
    assert result.path[1].to_entity == "C"
    assert result.total_weight == 3.0


def test_error_handling():
    """Test error handling for invalid inputs."""
    graph = create_test_graph()
    hl = HubLabeling(graph)

    # Test finding path before preprocessing
    with pytest.raises(GraphOperationError):
        hl.find_path("A", "C")

    hl.preprocess()

    # Test finding path with invalid nodes
    with pytest.raises(GraphOperationError):
        hl.find_path("A", "D")

    with pytest.raises(GraphOperationError):
        hl.find_path("D", "A")


def test_query_performance():
    """Test that queries are significantly faster than preprocessing."""
    # Create a chain of nodes to test performance
    edges = []
    prev = "A"
    for i in range(10):  # Small enough to run quickly, large enough to measure
        curr = f"N{i}"
        edges.append(
            Edge(
                from_entity=prev,
                to_entity=curr,
                relation_type=RelationType.CONNECTS_TO,
                metadata=EdgeMetadata(
                    created_at=datetime.now(),
                    last_modified=datetime.now(),
                    confidence=1.0,
                    source="test",
                    weight=1.0,
                ),
                impact_score=0.5,
            )
        )
        prev = curr

    graph = Graph(edges)
    hl = HubLabeling(graph)

    # Measure preprocessing time
    start_time = time.time()
    hl.preprocess()
    preprocess_time = time.time() - start_time

    # Measure query time (average of multiple queries)
    total_query_time = 0
    num_queries = 5
    for _ in range(num_queries):
        start_time = time.time()
        result = hl.find_path("A", "N9")
        total_query_time += time.time() - start_time

    avg_query_time = total_query_time / num_queries

    # Queries should be at least 10x faster than preprocessing
    assert avg_query_time * 10 < preprocess_time


def test_memory_efficiency():
    """Test that memory usage scales reasonably with graph size."""
    # Create a small complete graph (worst case for hub labeling)
    edges = []
    nodes = ["A", "B", "C", "D"]
    for i, from_node in enumerate(nodes):
        for to_node in nodes[i + 1 :]:
            edges.append(
                Edge(
                    from_entity=from_node,
                    to_entity=to_node,
                    relation_type=RelationType.CONNECTS_TO,
                    metadata=EdgeMetadata(
                        created_at=datetime.now(),
                        last_modified=datetime.now(),
                        confidence=1.0,
                        source="test",
                        weight=1.0,
                    ),
                    impact_score=0.5,
                )
            )

    graph = Graph(edges)
    hl = HubLabeling(graph)
    hl.preprocess()

    # Check label size
    total_labels = 0
    for node in nodes:
        total_labels += len(hl.state.get_forward_labels(node).labels)
        total_labels += len(hl.state.get_backward_labels(node).labels)

    # In a complete graph we expect O(n^2) labels total
    n = len(nodes)
    max_expected_labels = n * (n - 1)  # Theoretical upper bound
    assert (
        total_labels <= max_expected_labels
    ), f"Too many labels: {total_labels} (expected <= {max_expected_labels})"
