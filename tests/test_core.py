"""
Core functionality tests.
"""

from datetime import datetime

import pytest

from src.core.enums import RelationType
from src.core.graph import Graph
from src.core.models import Edge, EdgeMetadata


@pytest.fixture
def sample_edge_metadata():
    """Fixture providing sample edge metadata."""
    return EdgeMetadata(
        created_at=datetime.now(),
        last_modified=datetime.now(),
        confidence=0.9,
        source="test_source",
    )


def test_core_edge_creation(sample_edge_metadata):
    """Test creating a graph with edges."""
    edge = Edge(
        from_entity="A",
        to_entity="B",
        relation_type=RelationType.DEPENDS_ON,
        metadata=sample_edge_metadata,
        impact_score=0.8,
    )

    assert edge.from_entity == "A"
    assert edge.to_entity == "B"
    assert edge.relation_type == RelationType.DEPENDS_ON
    assert edge.metadata.confidence == pytest.approx(0.9)
    assert edge.impact_score == pytest.approx(0.8)


def test_core_node_neighbors(sample_edge_metadata):
    """Test retrieving node neighbors."""
    edge1 = Edge(
        from_entity="A",
        to_entity="B",
        relation_type=RelationType.DEPENDS_ON,
        metadata=sample_edge_metadata,
        impact_score=0.8,
    )
    edge2 = Edge(
        from_entity="A",
        to_entity="C",
        relation_type=RelationType.DEPENDS_ON,
        metadata=sample_edge_metadata,
        impact_score=0.8,
    )

    graph = Graph(edges=[edge1, edge2])
    neighbors = graph.get_neighbors("A")

    assert len(neighbors) == 2
    assert "B" in neighbors
    assert "C" in neighbors


def test_core_edge_retrieval(sample_edge_metadata):
    """Test retrieving edges between nodes."""
    edge = Edge(
        from_entity="A",
        to_entity="B",
        relation_type=RelationType.DEPENDS_ON,
        metadata=sample_edge_metadata,
        impact_score=0.8,
    )

    graph = Graph(edges=[edge])
    retrieved_edge = graph.get_edge("A", "B")

    assert retrieved_edge is not None
    assert retrieved_edge.from_entity == "A"
    assert retrieved_edge.to_entity == "B"
    assert retrieved_edge.relation_type == RelationType.DEPENDS_ON


def test_core_node_degree(sample_edge_metadata):
    """Test calculating node degrees."""
    edges = [
        Edge(
            from_entity="A",
            to_entity="B",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        ),
        Edge(
            from_entity="A",
            to_entity="C",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        ),
    ]

    graph = Graph(edges=edges)
    assert graph.get_degree("A") == 2  # Two outgoing edges
    assert graph.get_degree("B") == 0  # No outgoing edges
    assert graph.get_degree("C") == 0  # No outgoing edges


def test_core_empty_graph():
    """Test operations on an empty graph."""
    graph = Graph(edges=[])
    assert len(graph.get_neighbors("A")) == 0
    assert graph.get_edge("A", "B") is None
    assert graph.get_degree("A") == 0
