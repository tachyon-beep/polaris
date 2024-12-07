"""Tests for graph component analysis."""

from polaris.core.enums import RelationType
from polaris.core.graph import Graph
from polaris.core.graph_operations.components import ComponentAnalysis
from polaris.core.models import Edge, EdgeMetadata


def test_component_analysis(sample_edge_metadata):
    """Test basic component analysis functionality."""
    edges = [
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
            impact_score=0.8,
        ),
        Edge(
            from_entity="D",
            to_entity="E",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        ),
    ]
    analyzer = ComponentAnalysis(edges)
    components = analyzer.get_components()

    assert len(components) == 2  # Two separate components
    assert analyzer.get_component_count() == 2

    # Check component membership
    assert analyzer.are_connected("A", "B")
    assert analyzer.are_connected("B", "C")
    assert analyzer.are_connected("A", "C")
    assert analyzer.are_connected("D", "E")
    assert not analyzer.are_connected("A", "D")


def test_isolated_nodes(sample_edge_metadata):
    """Test detection of isolated nodes."""
    edges = [
        Edge(
            from_entity="A",
            to_entity="B",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        ),
        Edge(
            from_entity="C",
            to_entity="C",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        ),  # Self-loop
        Edge(
            from_entity="D",
            to_entity="E",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        ),
    ]
    analyzer = ComponentAnalysis(edges)
    isolated = analyzer.get_isolated_nodes()

    assert "C" in isolated  # Node with only self-loop is isolated
    assert "A" not in isolated
    assert "B" not in isolated


def test_largest_component(sample_edge_metadata):
    """Test finding the largest component."""
    edges = [
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
            impact_score=0.8,
        ),
        Edge(
            from_entity="C",
            to_entity="D",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        ),
        Edge(
            from_entity="X",
            to_entity="Y",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        ),
    ]
    analyzer = ComponentAnalysis(edges)
    largest = analyzer.get_largest_component()

    assert len(largest) == 4  # A-B-C-D component
    assert all(node in largest for node in ["A", "B", "C", "D"])
    assert "X" not in largest
    assert "Y" not in largest
