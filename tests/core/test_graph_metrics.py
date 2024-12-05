"""Tests for graph metrics calculations."""

from polaris.core.enums import RelationType
from polaris.core.graph import Graph
from polaris.core.graph_operations.metrics import MetricsCalculator
from polaris.core.models import Edge, EdgeMetadata


def test_metrics_triangle_graph(sample_edge_metadata):
    """Test metrics for a fully connected triangle graph."""
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
        ),
    ]
    calculator = MetricsCalculator(edges)
    metrics = calculator.calculate_metrics()

    assert metrics.node_count == 3
    assert metrics.edge_count == 3
    assert metrics.density == 0.5  # 3 edges out of 6 possible directed edges
    assert metrics.connected_components == 1
    assert metrics.clustering_coefficient > 0


def test_metrics_path_graph(sample_edge_metadata):
    """Test metrics for a directed path graph."""
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
    calculator = MetricsCalculator(edges)
    metrics = calculator.calculate_metrics()

    assert metrics.node_count == 4
    assert metrics.edge_count == 3
    assert metrics.connected_components == 1
    assert metrics.average_path_length > 1.0


def test_metrics_disconnected_graph(sample_edge_metadata):
    """Test metrics for a disconnected graph."""
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
            to_entity="D",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        ),
    ]
    calculator = MetricsCalculator(edges)
    metrics = calculator.calculate_metrics()

    assert metrics.connected_components == 2
    assert metrics.diameter is None  # No finite diameter for disconnected graph


def test_metrics_empty_graph():
    """Test metrics for an empty graph."""
    calculator = MetricsCalculator([])
    metrics = calculator.calculate_metrics()

    assert metrics.node_count == 0
    assert metrics.edge_count == 0
    assert metrics.density == 0.0
    assert metrics.connected_components == 0
    assert metrics.diameter is None
