"""
Tests for graph metrics calculations.
"""

from datetime import datetime

import pytest

from src.core.enums import RelationType
from src.core.graph import Graph
from src.core.graph_metrics import MetricsCalculator
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


@pytest.fixture
def triangle_graph(sample_edge_metadata):
    """
    Fixture providing a triangle graph (fully connected 3 nodes).
    A <-> B <-> C <-> A (bidirectional edges)
    """
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
            to_entity="A",
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
            to_entity="B",
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
        Edge(
            from_entity="A",
            to_entity="C",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        ),
    ]
    return Graph(edges=edges)


@pytest.fixture
def path_graph(sample_edge_metadata):
    """
    Fixture providing a path graph.
    A -> B -> C -> D (directed path)
    """
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
    return Graph(edges=edges)


@pytest.fixture
def disconnected_graph(sample_edge_metadata):
    """
    Fixture providing a disconnected graph.
    A -> B    C -> D
    """
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
    return Graph(edges=edges)


@pytest.fixture
def large_graph(sample_edge_metadata):
    """
    Fixture providing a large graph that exceeds the exact calculation thresholds.
    Creates a grid-like structure of N x N nodes.
    """
    N = 50  # This will create a graph with 2500 nodes
    edges = []

    # Create horizontal edges
    for i in range(N):
        for j in range(N - 1):
            node1 = f"node_{i}_{j}"
            node2 = f"node_{i}_{j+1}"
            edges.append(
                Edge(
                    from_entity=node1,
                    to_entity=node2,
                    relation_type=RelationType.DEPENDS_ON,
                    metadata=sample_edge_metadata,
                    impact_score=0.8,
                )
            )

    # Create vertical edges
    for i in range(N - 1):
        for j in range(N):
            node1 = f"node_{i}_{j}"
            node2 = f"node_{i+1}_{j}"
            edges.append(
                Edge(
                    from_entity=node1,
                    to_entity=node2,
                    relation_type=RelationType.DEPENDS_ON,
                    metadata=sample_edge_metadata,
                    impact_score=0.8,
                )
            )

    return Graph(edges=edges)


def test_metrics_triangle_graph(triangle_graph):
    """Test metrics for a fully connected triangle graph."""
    metrics = MetricsCalculator.calculate_metrics(triangle_graph)

    assert metrics.node_count == 3
    assert metrics.edge_count == 6  # Bidirectional edges between all nodes
    assert metrics.average_degree == pytest.approx(
        4.0, abs=1e-10
    )  # Each node has 2 in + 2 out edges
    assert metrics.density == pytest.approx(1.0, abs=1e-10)  # All possible edges exist
    assert metrics.clustering_coefficient == pytest.approx(1.0, abs=1e-10)  # Perfect clustering
    assert metrics.connected_components == 1
    assert metrics.diameter == 1  # Direct connections between all nodes
    assert metrics.average_path_length == pytest.approx(1.0, abs=1e-10)


def test_metrics_path_graph(path_graph):
    """Test metrics for a directed path graph."""
    metrics = MetricsCalculator.calculate_metrics(path_graph)

    assert metrics.node_count == 4
    assert metrics.edge_count == 3
    assert metrics.average_degree == pytest.approx(1.5, abs=1e-10)
    assert metrics.density == pytest.approx(
        0.25, abs=1e-10
    )  # 3 edges out of 12 possible directed edges
    assert metrics.clustering_coefficient == pytest.approx(0.0, abs=1e-10)  # No triangles
    assert metrics.connected_components == 1
    assert metrics.diameter is None  # No complete paths between all nodes in directed graph
    # For path A->B->C->D:
    # Paths: A->B (1), A->B->C (2), A->B->C->D (3), B->C (1), B->C->D (2), C->D (1)
    # Total path length = 1 + 2 + 3 + 1 + 2 + 1 = 10
    # Number of paths = 6
    # Average path length = 10/6 ≈ 1.67
    assert metrics.average_path_length == pytest.approx(1.67, abs=0.01)


def test_metrics_disconnected_graph(disconnected_graph):
    """Test metrics for a disconnected graph."""
    metrics = MetricsCalculator.calculate_metrics(disconnected_graph)

    assert metrics.node_count == 4
    assert metrics.edge_count == 2
    assert metrics.average_degree == pytest.approx(1.0, abs=1e-10)
    assert metrics.density == pytest.approx(
        1 / 6, abs=1e-10
    )  # 2 edges out of 12 possible directed edges
    assert metrics.clustering_coefficient == pytest.approx(0.0, abs=1e-10)
    assert metrics.connected_components == 2
    assert metrics.diameter is None  # No path between components
    assert metrics.average_path_length == pytest.approx(1.0, abs=1e-10)  # Only count existing paths


def test_metrics_empty_graph():
    """Test metrics for an empty graph."""
    graph = Graph(edges=[])
    metrics = MetricsCalculator.calculate_metrics(graph)

    assert metrics.node_count == 0
    assert metrics.edge_count == 0
    assert metrics.average_degree == pytest.approx(0.0, abs=1e-10)
    assert metrics.density == pytest.approx(0.0, abs=1e-10)
    assert metrics.clustering_coefficient == pytest.approx(0.0, abs=1e-10)
    assert metrics.connected_components == 0
    assert metrics.diameter is None
    assert metrics.average_path_length == pytest.approx(0.0, abs=1e-10)


def test_metrics_single_node(sample_edge_metadata):
    """Test metrics for a graph with a single node."""
    # For a single isolated node, create an edge that points to itself
    edges = [
        Edge(
            from_entity="A",
            to_entity="A",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        )
    ]
    graph = Graph(edges=edges)
    metrics = MetricsCalculator.calculate_metrics(graph)

    assert metrics.node_count == 1
    assert metrics.edge_count == 1  # Self-loop
    assert metrics.average_degree == pytest.approx(
        2.0, abs=1e-10
    )  # One in and one out from self-loop
    assert metrics.density == pytest.approx(
        1.0, abs=1e-10
    )  # All possible edges exist (just the self-loop)
    assert metrics.clustering_coefficient == pytest.approx(0.0, abs=1e-10)
    assert metrics.connected_components == 1
    assert metrics.diameter == 0
    assert metrics.average_path_length == pytest.approx(0.0, abs=1e-10)


def test_metrics_star_graph(sample_edge_metadata):
    """Test metrics for a directed star graph (edges from center to leaves only)."""
    edges = [
        Edge(
            from_entity="A",
            to_entity=f"Node{i}",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        )
        for i in range(5)
    ]
    graph = Graph(edges=edges)
    metrics = MetricsCalculator.calculate_metrics(graph)

    assert metrics.node_count == 6
    assert metrics.edge_count == 5
    assert metrics.average_degree == pytest.approx(5 / 3, abs=1e-10)  # (5*2)/6 nodes
    assert metrics.clustering_coefficient == pytest.approx(0.0, abs=1e-10)  # No triangles
    assert metrics.connected_components == 1
    assert metrics.diameter is None  # No paths between leaf nodes in directed graph
    assert metrics.average_path_length == pytest.approx(1.0, abs=1e-10)  # Only direct paths exist


def test_metrics_complex_components(sample_edge_metadata):
    """Test metrics for a graph with multiple components of varying density."""
    # Component 1: Dense triangle
    # Component 2: Sparse path
    # Component 3: Isolated node (represented by a self-loop)
    edges = [
        # Dense triangle
        Edge(
            from_entity="A1",
            to_entity="A2",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        ),
        Edge(
            from_entity="A2",
            to_entity="A3",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        ),
        Edge(
            from_entity="A3",
            to_entity="A1",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        ),
        # Sparse path
        Edge(
            from_entity="B1",
            to_entity="B2",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        ),
        Edge(
            from_entity="B2",
            to_entity="B3",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        ),
        # Isolated node (represented as self-loop)
        Edge(
            from_entity="C1",
            to_entity="C1",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        ),
    ]
    graph = Graph(edges=edges)
    metrics = MetricsCalculator.calculate_metrics(graph)

    assert metrics.node_count == 7  # Including isolated node C1
    assert metrics.edge_count == 6  # Including self-loop for C1
    assert metrics.connected_components == 3  # Triangle, path, and isolated node
    assert metrics.diameter is None  # No complete paths between all components
    assert metrics.clustering_coefficient > 0  # Due to the triangle component


def test_path_metrics_exact_empty():
    """Test exact path metrics calculation with empty graph."""
    graph = Graph(edges=[])
    result = MetricsCalculator._calculate_path_metrics_exact(graph, set())
    assert result.diameter is None
    assert result.avg_path_length == pytest.approx(0.0, abs=1e-10)
    assert result.sample_size == 0
    assert result.confidence == pytest.approx(1.0, abs=1e-10)


def test_path_metrics_exact_single_node(sample_edge_metadata):
    """Test exact path metrics calculation with single node graph."""
    edges = [
        Edge(
            from_entity="A",
            to_entity="A",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        )
    ]
    graph = Graph(edges=edges)
    result = MetricsCalculator._calculate_path_metrics_exact(graph, {"A"})
    assert result.diameter == 0
    assert result.avg_path_length == pytest.approx(0.0, abs=1e-10)
    assert result.sample_size == 1
    assert result.confidence == pytest.approx(1.0, abs=1e-10)


def test_path_metrics_exact_disconnected(disconnected_graph):
    """Test exact path metrics calculation with disconnected graph."""
    result = MetricsCalculator._calculate_path_metrics_exact(
        disconnected_graph, disconnected_graph.get_nodes()
    )
    assert result.diameter is None  # No path between components
    assert result.avg_path_length == pytest.approx(1.0, abs=1e-10)  # Only count existing paths
    assert result.confidence == pytest.approx(1.0, abs=1e-10)


def test_large_graph_metrics(large_graph):
    """Test metrics calculation with a large graph that triggers approximate calculations."""
    metrics = MetricsCalculator.calculate_metrics(large_graph)

    # Basic metrics should be exact
    assert metrics.node_count == 2500  # 50x50 grid
    assert metrics.edge_count == 4900  # 2450 horizontal + 2450 vertical edges

    # These metrics should use approximation
    assert 0.0 <= metrics.clustering_coefficient <= 1.0
    assert metrics.diameter is not None  # Should be approximately 98 (grid diameter)
    assert metrics.average_path_length > 0

    # Test approximate clustering directly
    nodes = large_graph.get_nodes()
    clustering, confidence = MetricsCalculator._calculate_clustering_approximate(
        large_graph, nodes, MetricsCalculator.MIN_SAMPLE_SIZE
    )
    assert 0.0 <= clustering <= 1.0
    assert 0.0 <= confidence <= 1.0

    # Test approximate path metrics directly
    path_metrics = MetricsCalculator._calculate_path_metrics_approximate(
        large_graph, nodes, MetricsCalculator.MIN_SAMPLE_SIZE
    )
    assert path_metrics.diameter is not None
    assert path_metrics.avg_path_length > 0
    assert path_metrics.confidence > 0
    assert path_metrics.sample_size >= MetricsCalculator.MIN_SAMPLE_SIZE


def test_path_metrics_approximate_empty():
    """Test approximate path metrics calculation with empty graph."""
    graph = Graph(edges=[])
    result = MetricsCalculator._calculate_path_metrics_approximate(graph, set(), 100)
    assert result.diameter is None
    assert result.avg_path_length == pytest.approx(0.0, abs=1e-10)
    assert result.sample_size == 0
    assert result.confidence == pytest.approx(1.0, abs=1e-10)


def test_path_metrics_approximate_single_node(sample_edge_metadata):
    """Test approximate path metrics calculation with single node graph."""
    edges = [
        Edge(
            from_entity="A",
            to_entity="A",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        )
    ]
    graph = Graph(edges=edges)
    result = MetricsCalculator._calculate_path_metrics_approximate(graph, {"A"}, 100)
    assert result.diameter == 0
    assert result.avg_path_length == pytest.approx(0.0, abs=1e-10)
    assert result.sample_size == 1
    assert result.confidence == pytest.approx(1.0, abs=1e-10)


def test_clustering_approximate_empty():
    """Test approximate clustering calculation with empty graph."""
    graph = Graph(edges=[])
    clustering, confidence = MetricsCalculator._calculate_clustering_approximate(graph, set(), 100)
    assert clustering == pytest.approx(0.0, abs=1e-10)
    assert confidence == pytest.approx(1.0, abs=1e-10)


def test_clustering_approximate_single_node(sample_edge_metadata):
    """Test approximate clustering calculation with single node graph."""
    edges = [
        Edge(
            from_entity="A",
            to_entity="A",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        )
    ]
    graph = Graph(edges=edges)
    clustering, confidence = MetricsCalculator._calculate_clustering_approximate(graph, {"A"}, 100)
    assert clustering == pytest.approx(0.0, abs=1e-10)
    assert confidence == pytest.approx(1.0, abs=1e-10)
