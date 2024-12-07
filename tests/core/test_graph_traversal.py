"""Tests for graph traversal operations."""

from polaris.core.enums import RelationType
from polaris.core.graph_operations.components import ComponentAnalysis
from polaris.core.models import Edge, EdgeMetadata


def test_traversal_with_filter(sample_edge_metadata):
    """Test traversal with a filter function."""
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
    ]
    analyzer = ComponentAnalysis(edges)
    components = analyzer.get_components()

    assert len(components) == 1  # Single connected component
    assert all(len(comp) > 0 for comp in components)


def test_traversal_with_cycles(sample_edge_metadata):
    """Test traversal on a graph with cycles."""
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
            to_entity="A",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        ),  # Cycle
    ]
    analyzer = ComponentAnalysis(edges)
    components = analyzer.get_components()

    assert len(components) == 1  # Single strongly connected component
    assert len(components[0]) == 3  # All nodes in same component


def test_traversal_with_one_way_relationships(sample_edge_metadata):
    """Test traversal with one-way relationships."""
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
    ]
    analyzer = ComponentAnalysis(edges)
    components = analyzer.get_components()

    assert len(components) == 1  # Single component
    assert len(components[0]) == 3  # All nodes reachable


def test_traversal_with_isolated_nodes(sample_edge_metadata):
    """Test traversal with isolated nodes."""
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
            to_entity="D",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        ),  # Self-loop
    ]
    analyzer = ComponentAnalysis(edges)
    isolated = analyzer.get_isolated_nodes()

    assert "C" in isolated  # Self-loops count as isolated
    assert "D" in isolated  # Self-loops count as isolated
    assert "A" not in isolated
    assert "B" not in isolated
