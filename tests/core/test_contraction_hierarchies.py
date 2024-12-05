"""
Tests for Contraction Hierarchies implementation.
"""

import pytest
from datetime import datetime

from polaris.core.exceptions import GraphOperationError
from polaris.core.graph import Graph
from polaris.core.graph.traversal.algorithms.advanced.contraction_hierarchies import (
    ContractionHierarchies,
)
from polaris.core.graph.traversal.algorithms.advanced.contraction_hierarchies.models import (
    Shortcut,
)
from polaris.core.models import Edge, EdgeMetadata
from polaris.core.enums import RelationType


@pytest.fixture
def graph() -> Graph:
    """Create test graph."""
    # Initialize with empty edge list
    g = Graph(edges=[])

    # Create base metadata for edges
    base_metadata = EdgeMetadata(
        created_at=datetime.now(),
        last_modified=datetime.now(),
        confidence=1.0,
        source="test",
        weight=1.0,
    )

    # Add test edges
    edges = [
        Edge(
            from_entity="A",
            to_entity="B",
            relation_type=RelationType.CONNECTS_TO,
            metadata=EdgeMetadata(
                created_at=base_metadata.created_at,
                last_modified=base_metadata.last_modified,
                confidence=base_metadata.confidence,
                source=base_metadata.source,
                weight=1.0,
            ),
            impact_score=0.5,
        ),
        Edge(
            from_entity="B",
            to_entity="C",
            relation_type=RelationType.CONNECTS_TO,
            metadata=EdgeMetadata(
                created_at=base_metadata.created_at,
                last_modified=base_metadata.last_modified,
                confidence=base_metadata.confidence,
                source=base_metadata.source,
                weight=2.0,
            ),
            impact_score=0.5,
        ),
        Edge(
            from_entity="A",
            to_entity="C",
            relation_type=RelationType.CONNECTS_TO,
            metadata=EdgeMetadata(
                created_at=base_metadata.created_at,
                last_modified=base_metadata.last_modified,
                confidence=base_metadata.confidence,
                source=base_metadata.source,
                weight=5.0,
            ),
            impact_score=0.5,
        ),
    ]

    for edge in edges:
        g.add_edge(edge)

    return g


def test_preprocessing(graph: Graph) -> None:
    """Test preprocessing creates valid hierarchy."""
    ch = ContractionHierarchies(graph)
    ch.preprocess()

    # Verify state exists
    assert ch.state is not None

    # Check node levels assigned
    assert len(ch.state.node_level) == len(graph.get_nodes())

    # Verify shortcuts created
    shortcuts = ch.state.shortcuts
    assert len(shortcuts) > 0

    # Check contracted neighbors tracked
    assert len(ch.state.contracted_neighbors) > 0


def test_path_finding(graph: Graph) -> None:
    """Test path finding returns correct paths."""
    ch = ContractionHierarchies(graph)
    ch.preprocess()

    # Find path
    result = ch.find_path("A", "C")

    # Verify path exists
    assert result.path is not None
    assert len(result.path) > 0

    # Check path endpoints
    assert result.path[0].from_entity == "A"
    assert result.path[-1].to_entity == "C"

    # Verify total weight
    assert result.total_weight == 3.0  # A->B->C = 1 + 2


def test_invalid_path(graph: Graph) -> None:
    """Test error raised for non-existent path."""
    ch = ContractionHierarchies(graph)
    ch.preprocess()

    with pytest.raises(GraphOperationError):
        ch.find_path("A", "D")  # D doesn't exist


def test_preprocessing_required(graph: Graph) -> None:
    """Test error raised if paths found before preprocessing."""
    ch = ContractionHierarchies(graph)

    with pytest.raises(GraphOperationError):
        ch.find_path("A", "C")


def test_shortcut_creation(graph: Graph) -> None:
    """Test shortcuts created correctly during preprocessing."""
    ch = ContractionHierarchies(graph)
    ch.preprocess()

    # Get shortcuts
    shortcuts = ch.state.shortcuts

    # Verify shortcut properties
    for shortcut in shortcuts.values():
        # Check shortcut structure
        assert isinstance(shortcut, Shortcut)
        assert shortcut.via_node is not None
        assert shortcut.lower_edge is not None
        assert shortcut.upper_edge is not None

        # Verify weight consistency
        expected_weight = shortcut.lower_edge.metadata.weight + shortcut.upper_edge.metadata.weight
        assert abs(shortcut.edge.metadata.weight - expected_weight) < 1e-10


def test_path_with_shortcuts(graph: Graph) -> None:
    """Test path finding works with shortcuts."""
    # Create base metadata for new edges
    base_metadata = EdgeMetadata(
        created_at=datetime.now(),
        last_modified=datetime.now(),
        confidence=1.0,
        source="test",
        weight=1.0,
    )

    # Add more edges to force shortcut creation
    graph.add_edge(
        Edge(
            from_entity="C",
            to_entity="D",
            relation_type=RelationType.CONNECTS_TO,
            metadata=EdgeMetadata(
                created_at=base_metadata.created_at,
                last_modified=base_metadata.last_modified,
                confidence=base_metadata.confidence,
                source=base_metadata.source,
                weight=1.0,
            ),
            impact_score=0.5,
        )
    )
    graph.add_edge(
        Edge(
            from_entity="D",
            to_entity="E",
            relation_type=RelationType.CONNECTS_TO,
            metadata=EdgeMetadata(
                created_at=base_metadata.created_at,
                last_modified=base_metadata.last_modified,
                confidence=base_metadata.confidence,
                source=base_metadata.source,
                weight=1.0,
            ),
            impact_score=0.5,
        )
    )

    ch = ContractionHierarchies(graph)
    ch.preprocess()

    # Find path that should use shortcuts
    result = ch.find_path("A", "E")

    # Verify path exists and is correct
    assert result.path is not None
    assert len(result.path) > 0
    assert result.path[0].from_entity == "A"
    assert result.path[-1].to_entity == "E"

    # Check total weight
    expected_weight = 4.0  # A->B->C->D->E = 1 + 2 + 1 + 1
    assert abs(result.total_weight - expected_weight) < 1e-10
