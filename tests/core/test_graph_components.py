"""
Tests for graph component analysis functionality.
"""

from datetime import datetime

import pytest

from polaris.core.enums import RelationType
from polaris.core.graph import Graph
from polaris.core.graph_components import ComponentAnalysis
from polaris.core.models import Edge, EdgeMetadata


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
def disconnected_graph(sample_edge_metadata):
    """
    Fixture providing a graph with multiple disconnected components:
    Component 1: A -> B -> C
    Component 2: D -> E
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
            from_entity="D",
            to_entity="E",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        ),
    ]
    return Graph(edges=edges)


@pytest.fixture
def clustered_graph(sample_edge_metadata):
    """
    Fixture providing a graph with different clustering patterns:
    A -> B -> C -> A (fully connected triangle)
    D -> E -> F (path, no clustering)
    """
    edges = [
        # Triangle component - bidirectional edges
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
        # Path component
        Edge(
            from_entity="D",
            to_entity="E",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        ),
        Edge(
            from_entity="E",
            to_entity="F",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        ),
    ]
    return Graph(edges=edges)


@pytest.fixture
def cyclic_graph(sample_edge_metadata):
    """
    Fixture providing a graph with strongly connected components:
    Component 1: A -> B -> C -> A (cycle)
    Component 2: D -> E -> F -> D (cycle)
    Component 3: G -> H (no cycle)
    """
    edges = [
        # First cycle
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
        # Second cycle
        Edge(
            from_entity="D",
            to_entity="E",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        ),
        Edge(
            from_entity="E",
            to_entity="F",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        ),
        Edge(
            from_entity="F",
            to_entity="D",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        ),
        # Non-cyclic path
        Edge(
            from_entity="G",
            to_entity="H",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        ),
    ]
    return Graph(edges=edges)


def test_find_components_disconnected(disconnected_graph):
    """Test finding components in a disconnected graph."""
    components = ComponentAnalysis.find_components(disconnected_graph)

    # Should find two components
    assert len(components) == 2

    # Verify each component
    component_sets = set(frozenset(comp) for comp in components)
    assert frozenset({"A", "B", "C"}) in component_sets  # First component
    assert frozenset({"D", "E"}) in component_sets  # Second component


def test_find_components_empty_graph():
    """Test finding components in an empty graph."""
    graph = Graph(edges=[])
    components = ComponentAnalysis.find_components(graph)
    assert len(components) == 0


def test_find_components_single_node(sample_edge_metadata):
    """Test finding components with a single-node graph."""
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
    components = ComponentAnalysis.find_components(graph)

    assert len(components) == 1
    assert list(components[0]) == ["A"]


def test_clustering_coefficient_triangle(clustered_graph):
    """Test clustering coefficient for a node in a triangle (fully connected)."""
    coef = ComponentAnalysis.calculate_clustering_coefficient(clustered_graph, "A")
    assert coef == pytest.approx(1.0)  # All neighbors are connected to each other


def test_clustering_coefficient_path(clustered_graph):
    """Test clustering coefficient for a node in a path (no clustering)."""
    coef = ComponentAnalysis.calculate_clustering_coefficient(clustered_graph, "E")
    assert coef == pytest.approx(0.0)  # Neighbors are not connected to each other


def test_clustering_coefficient_single_neighbor(sample_edge_metadata):
    """Test clustering coefficient for a node with only one neighbor."""
    edges = [
        Edge(
            from_entity="A",
            to_entity="B",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        )
    ]
    graph = Graph(edges=edges)
    coef = ComponentAnalysis.calculate_clustering_coefficient(graph, "A")
    assert coef == pytest.approx(0.0)  # Coefficient is 0 for nodes with < 2 neighbors


def test_clustering_coefficient_isolated_node(sample_edge_metadata):
    """Test clustering coefficient for an isolated node."""
    edges = [
        Edge(
            from_entity="B",
            to_entity="C",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        )
    ]
    graph = Graph(edges=edges)
    coef = ComponentAnalysis.calculate_clustering_coefficient(graph, "A")
    assert coef == pytest.approx(0.0)  # Coefficient is 0 for isolated nodes


def test_clustering_coefficient_star_pattern(sample_edge_metadata):
    """Test clustering coefficient for center node in a star pattern."""
    # Create a star pattern where A is connected to multiple nodes,
    # but those nodes aren't connected to each other
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
        Edge(
            from_entity="A",
            to_entity="D",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        ),
    ]
    graph = Graph(edges=edges)
    coef = ComponentAnalysis.calculate_clustering_coefficient(graph, "A")
    assert coef == pytest.approx(0.0)  # No connections between neighbors


def test_clustering_coefficient_partial_clustering(sample_edge_metadata):
    """Test clustering coefficient for partially clustered node."""
    # Create a pattern where some neighbors are connected but not all
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
            from_entity="A",
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
        Edge(
            from_entity="A",
            to_entity="D",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        ),
        Edge(
            from_entity="D",
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
    ]
    graph = Graph(edges=edges)
    coef = ComponentAnalysis.calculate_clustering_coefficient(graph, "A")
    # One out of three possible connections exists between neighbors
    assert coef == pytest.approx(1 / 3)


def test_collect_all_nodes(sample_edge_metadata):
    """Test _collect_all_nodes method."""
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
    graph = Graph(edges=edges)
    nodes = ComponentAnalysis._collect_all_nodes(graph)
    assert nodes == {"A", "B", "C"}


def test_build_undirected_adjacency(sample_edge_metadata):
    """Test _build_undirected_adjacency method."""
    # Test with directed edges
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
    graph = Graph(edges=edges)
    undirected = ComponentAnalysis._build_undirected_adjacency(graph)

    # Check bidirectional connections
    assert "B" in undirected["A"]
    assert "A" in undirected["B"]
    assert "C" in undirected["B"]
    assert "B" in undirected["C"]


def test_find_component_dfs(sample_edge_metadata):
    """Test _find_component_dfs method."""
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
    graph = Graph(edges=edges)
    undirected = ComponentAnalysis._build_undirected_adjacency(graph)
    visited = set()

    # Test finding first component
    component = ComponentAnalysis._find_component_dfs("A", undirected, visited)
    assert component == {"A", "B", "C"}

    # Test finding second component
    component = ComponentAnalysis._find_component_dfs("D", undirected, visited)
    assert component == {"D", "E"}


def test_complex_component_structure(sample_edge_metadata):
    """Test finding components in a complex graph structure."""
    # Create a graph with multiple interconnected components:
    # Component 1: A <-> B <-> C (bidirectional)
    # Component 2: D -> E -> F -> D (cycle)
    # Component 3: G (isolated)
    edges = [
        # Component 1
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
        # Component 2
        Edge(
            from_entity="D",
            to_entity="E",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        ),
        Edge(
            from_entity="E",
            to_entity="F",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        ),
        Edge(
            from_entity="F",
            to_entity="D",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        ),
    ]
    graph = Graph(edges=edges)
    components = ComponentAnalysis.find_components(graph)

    assert len(components) == 2  # Two connected components (G is not in edges)
    component_sets = set(frozenset(comp) for comp in components)
    assert frozenset({"A", "B", "C"}) in component_sets
    assert frozenset({"D", "E", "F"}) in component_sets


def test_directed_vs_undirected_components(sample_edge_metadata):
    """Test component finding with directed vs undirected edges."""
    # Create a graph where nodes are connected only in one direction:
    # A -> B -> C (directed path)
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
    graph = Graph(edges=edges)
    components = ComponentAnalysis.find_components(graph)

    # Should be treated as one component despite directed edges
    assert len(components) == 1
    assert components[0] == {"A", "B", "C"}


def test_mixed_edge_clustering(sample_edge_metadata):
    """Test clustering coefficient with mixed directed/undirected edges."""
    # Create a graph where some neighbor relationships are one-way
    # and others are bidirectional
    edges = [
        # Bidirectional edge A <-> B
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
        # One-way edges A -> C and C -> B
        Edge(
            from_entity="A",
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
    ]
    graph = Graph(edges=edges)
    coef = ComponentAnalysis.calculate_clustering_coefficient(graph, "A")

    # Total Possible Directed Edges Between Neighbors:
    #
    #   From B to C
    #   From C to B
    #   So, total possible edges = 2
    #
    # Actual Edges:
    #
    #   C → B exists
    #   B → C does not exist
    #   So, actual edges = 1
    #
    # Clustering Coefficient:
    #
    #   Coefficient = actual edges / possible edges = 1 / 2 = 0.5

    assert coef == pytest.approx(0.5)


def test_find_strongly_connected_components_empty():
    """Test finding strongly connected components in an empty graph."""
    graph = Graph(edges=[])
    components = ComponentAnalysis.find_strongly_connected_components(graph)
    assert len(components) == 0


def test_find_strongly_connected_components_single_node(sample_edge_metadata):
    """Test finding strongly connected components with a single node."""
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
    components = ComponentAnalysis.find_strongly_connected_components(graph)
    assert len(components) == 1
    assert list(components[0]) == ["A"]


def test_find_strongly_connected_components_cycle(cyclic_graph):
    """Test finding strongly connected components in a graph with cycles."""
    components = ComponentAnalysis.find_strongly_connected_components(cyclic_graph)

    # Should find four components:
    # 1. A->B->C->A cycle
    # 2. D->E->F->D cycle
    # 3. G (single node, as it only has outgoing edge)
    # 4. H (single node, as it only has incoming edge)
    assert len(components) == 4

    # Convert components to frozensets for easier comparison
    component_sets = set(frozenset(comp) for comp in components)

    # Verify each component
    assert frozenset({"A", "B", "C"}) in component_sets  # First cycle
    assert frozenset({"D", "E", "F"}) in component_sets  # Second cycle
    assert frozenset({"G"}) in component_sets  # Single node with outgoing edge
    assert frozenset({"H"}) in component_sets  # Single node with incoming edge


def test_find_strongly_connected_components_no_cycles(sample_edge_metadata):
    """Test finding strongly connected components in a graph without cycles."""
    # Create a linear graph: A -> B -> C
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
    graph = Graph(edges=edges)
    components = ComponentAnalysis.find_strongly_connected_components(graph)

    # Each node should be its own component since there are no cycles
    assert len(components) == 3
    component_sets = set(frozenset(comp) for comp in components)
    assert frozenset({"A"}) in component_sets
    assert frozenset({"B"}) in component_sets
    assert frozenset({"C"}) in component_sets


def test_find_strongly_connected_components_nested_cycles(sample_edge_metadata):
    """Test finding strongly connected components with nested cycles."""
    # Create a graph with nested cycles:
    # A -> B -> C -> A (outer cycle)
    #      ^    |
    #      |    v
    #      D <- E     (inner cycle)
    edges = [
        # Outer cycle
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
        # Connection to inner cycle
        Edge(
            from_entity="C",
            to_entity="E",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        ),
        # Inner cycle
        Edge(
            from_entity="E",
            to_entity="D",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        ),
        Edge(
            from_entity="D",
            to_entity="B",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        ),
    ]
    graph = Graph(edges=edges)
    components = ComponentAnalysis.find_strongly_connected_components(graph)

    # Should find all nodes in one strongly connected component
    # since they're all mutually reachable
    assert len(components) == 1
    assert components[0] == {"A", "B", "C", "D", "E"}
