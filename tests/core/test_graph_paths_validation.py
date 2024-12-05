"""
Tests for graph path validation functionality.

This module provides comprehensive tests for the PathResult class,
focusing on validation and edge cases.
"""

from datetime import datetime
from typing import List

import pytest

from polaris.core.enums import EntityType, RelationType
from polaris.core.graph import Graph
from polaris.core.graph.traversal.path_models import PathResult, PathValidationError
from polaris.core.models import Edge, EdgeMetadata, Node, NodeMetadata


@pytest.fixture
def sample_edge_metadata() -> EdgeMetadata:
    """Create sample edge metadata for testing."""
    return EdgeMetadata(
        created_at=datetime.now(),
        last_modified=datetime.now(),
        confidence=0.9,
        source="test",
        weight=1.0,
    )


@pytest.fixture
def sample_edges(sample_edge_metadata) -> List[Edge]:
    """Create a list of valid edges for testing."""
    return [
        Edge(
            from_entity="A",
            to_entity="B",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        ),
        Edge(
            from_entity="B",
            to_entity="C",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.7,
        ),
    ]


@pytest.fixture
def sample_graph(sample_edges) -> Graph:
    """Create a sample graph with test edges."""
    return Graph(edges=sample_edges)


def test_path_length_consistency(sample_edges):
    """Test that path length matches actual number of edges."""
    # Test with correct length
    path = PathResult(path=sample_edges, total_weight=1.5)
    assert len(path) == 2
    assert len(path.path) == 2

    # Test with empty path
    empty_path = PathResult(path=[], total_weight=0.0)
    assert len(empty_path) == 0
    assert len(empty_path.path) == 0


def test_path_type_validation(sample_edge_metadata):
    """Test validation of path element types."""
    invalid_edge = {"from": "A", "to": "B"}  # Not an Edge object
    edges = [
        Edge(
            from_entity="A",
            to_entity="B",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        ),
        invalid_edge,  # type: ignore
    ]

    with pytest.raises(TypeError) as exc:
        PathResult(path=edges, total_weight=1.0)  # type: ignore
    assert "path must contain only Edge objects" in str(exc.value)


def test_edge_attribute_validation(sample_edge_metadata):
    """Test validation of required edge attributes."""
    # Test creating an edge with missing required attributes
    with pytest.raises(TypeError) as exc:
        # Attempt to create an Edge without required attributes
        Edge(  # type: ignore
            from_entity="A",
            # to_entity is missing
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        )
    assert "missing 1 required positional argument: 'to_entity'" in str(exc.value)

    # Test with empty string values
    with pytest.raises(ValueError) as exc:
        Edge(
            from_entity="",  # Empty string
            to_entity="B",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        )
    assert "source node must be a non-empty string" in str(exc.value)


def test_maximum_path_length(sample_edges):
    """Test enforcement of maximum path length constraint."""
    # Create a path exceeding maximum length
    long_path = sample_edges * 100  # Create very long path
    path = PathResult(path=long_path, total_weight=sum(edge.metadata.weight for edge in long_path))

    with pytest.raises(PathValidationError) as exc:
        path.validate(Graph(edges=[]), max_length=50)
    assert "exceeds maximum allowed length" in str(exc.value)


def test_weight_comparison_precision(sample_edges):
    """Test precision handling in weight comparisons."""
    # Create a graph with the test edges
    graph = Graph(edges=sample_edges)

    # Create path with expected weight from weight function
    weight_func = lambda _: 1.0000001
    path = PathResult(path=sample_edges[:1], total_weight=1.0000001)

    # Should not raise error with default epsilon
    path.validate(graph, weight_func=weight_func)

    # Test with custom epsilon
    path.validate(graph, weight_func=weight_func, weight_epsilon=1e-6)

    # Test with difference outside epsilon
    path.total_weight = 1.1
    with pytest.raises(PathValidationError) as exc:
        path.validate(graph, weight_func=lambda _: 1.0, weight_epsilon=1e-6)
    assert "Weight mismatch" in str(exc.value)


def test_none_value_handling(sample_edge_metadata):
    """Test handling of None values in path list."""
    edges = [
        Edge(
            from_entity="A",
            to_entity="B",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        ),
        None,  # type: ignore
    ]

    with pytest.raises(TypeError) as exc:
        PathResult(path=edges, total_weight=1.0)  # type: ignore
    assert "must contain only Edge objects" in str(exc.value)


def test_empty_path_validation():
    """Test validation of empty paths."""
    path = PathResult(path=[], total_weight=0.0)

    # Empty path should be valid
    path.validate(Graph(edges=[]))

    # Length should be 0
    assert len(path) == 0

    # Nodes should be empty list
    assert path.nodes == []


def test_path_continuity(sample_edge_metadata):
    """Test validation of path continuity."""
    # Create disconnected edges
    edges = [
        Edge(
            from_entity="A",
            to_entity="B",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        ),
        Edge(
            from_entity="C",  # Disconnected from previous edge
            to_entity="D",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.7,
        ),
    ]

    path = PathResult(path=edges, total_weight=1.5)

    with pytest.raises(PathValidationError) as exc:
        path.validate(Graph(edges=edges))
    assert "Path discontinuity" in str(exc.value)


def test_self_loop_detection(sample_edge_metadata):
    """Test detection of self-loops in path."""
    # Create edge with self-loop
    edge = Edge(
        from_entity="A",
        to_entity="A",  # Self-loop
        relation_type=RelationType.DEPENDS_ON,
        metadata=sample_edge_metadata,
        impact_score=0.8,
    )

    path = PathResult(path=[edge], total_weight=1.0)

    with pytest.raises(PathValidationError) as exc:
        path.validate(Graph(edges=[edge]))
    assert "Self-loop detected" in str(exc.value)


def test_graph_edge_existence(sample_edges, sample_graph):
    """Test validation of edge existence in graph."""
    # Create path with edge not in graph
    new_edge = Edge(
        from_entity="X",
        to_entity="Y",
        relation_type=RelationType.DEPENDS_ON,
        metadata=sample_edges[0].metadata,
        impact_score=0.8,
    )

    path = PathResult(path=[new_edge], total_weight=1.0)

    with pytest.raises(PathValidationError) as exc:
        path.validate(sample_graph)
    assert "Edge from X to Y not found in graph" in str(exc.value)


def test_weight_epsilon_validation():
    """Test validation of weight_epsilon parameter."""
    path = PathResult(path=[], total_weight=0.0)

    # Test invalid epsilon type
    with pytest.raises(TypeError) as exc:
        path.validate(Graph(edges=[]), weight_epsilon="0.1")  # type: ignore
    assert "weight_epsilon must be a numeric value" in str(exc.value)

    # Test negative epsilon
    with pytest.raises(ValueError) as exc:
        path.validate(Graph(edges=[]), weight_epsilon=-0.1)
    assert "weight_epsilon must be positive" in str(exc.value)

    # Test zero epsilon
    with pytest.raises(ValueError) as exc:
        path.validate(Graph(edges=[]), weight_epsilon=0)
    assert "weight_epsilon must be positive" in str(exc.value)


def test_max_length_validation():
    """Test validation of max_length parameter."""
    path = PathResult(path=[], total_weight=0.0)

    # Test invalid max_length type
    with pytest.raises(TypeError) as exc:
        path.validate(Graph(edges=[]), max_length=1.5)  # type: ignore
    assert "max_length must be an integer" in str(exc.value)

    # Test negative max_length
    with pytest.raises(ValueError) as exc:
        path.validate(Graph(edges=[]), max_length=-1)
    assert "max_length must be non-negative" in str(exc.value)
