"""
Tests for core graph functionality.
"""

from datetime import datetime
from typing import Iterator, List, cast

import pytest

from polaris.core.enums import RelationType
from polaris.core.exceptions import EdgeNotFoundError, NodeNotFoundError
from polaris.core.graph import Graph
from polaris.core.models import Edge, EdgeMetadata
from polaris.core.graph_paths import PathFinding, PathType, PathResult
from polaris.core.graph_paths.cache import PathCache


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
def sample_edges(sample_edge_metadata):
    """Fixture providing a set of test edges."""
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
            impact_score=0.8,
        ),
        Edge(
            from_entity="A",
            to_entity="C",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        ),
        # Add reverse edges to test bidirectional relationships
        Edge(
            from_entity="B",
            to_entity="A",
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


def test_graph_creation(sample_edges):
    """Test creating a graph from edges."""
    graph = Graph(edges=sample_edges)

    # Test that all nodes are present with correct edges
    neighbors_a = graph.get_neighbors("A")
    neighbors_b = graph.get_neighbors("B")
    neighbors_c = graph.get_neighbors("C")

    assert len(neighbors_a) == 2  # A connects to B and C
    assert len(neighbors_b) == 2  # B connects to A and C
    assert len(neighbors_c) == 1  # C connects to B

    assert "B" in neighbors_a
    assert "C" in neighbors_a
    assert "A" in neighbors_b
    assert "C" in neighbors_b
    assert "B" in neighbors_c


def test_get_edge(sample_edges):
    """Test retrieving specific edges from the graph."""
    graph = Graph(edges=sample_edges)

    # Test existing edge
    edge_ab = graph.get_edge("A", "B")
    assert edge_ab is not None
    assert edge_ab.from_entity == "A"
    assert edge_ab.to_entity == "B"
    assert edge_ab.relation_type == RelationType.DEPENDS_ON

    # Test reverse edge exists
    edge_ba = graph.get_edge("B", "A")
    assert edge_ba is not None
    assert edge_ba.from_entity == "B"
    assert edge_ba.to_entity == "A"

    # Test non-existent edge
    edge_ac = graph.get_edge("A", "D")
    assert edge_ac is None


def test_get_degree(sample_edges):
    """Test calculating node degrees."""
    graph = Graph(edges=sample_edges)

    assert graph.get_degree("A") == 2  # Two outgoing edges (to B and C)
    assert graph.get_degree("B") == 2  # Two outgoing edges (to A and C)
    assert graph.get_degree("C") == 1  # One outgoing edge (to B)


def test_empty_graph():
    """Test creating an empty graph."""
    graph = Graph(edges=[])

    assert len(graph.get_neighbors("A")) == 0
    assert graph.get_edge("A", "B") is None
    assert graph.get_degree("A") == 0


def test_graph_with_self_loop(sample_edge_metadata):
    """Test graph with a self-loop edge."""
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
    neighbors = graph.get_neighbors("A")

    assert len(neighbors) == 1
    assert "A" in neighbors
    assert graph.get_degree("A") == 1


def test_graph_with_bidirectional_edges(sample_edge_metadata):
    """Test graph with explicit bidirectional edges."""
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
    ]

    graph = Graph(edges=edges)

    neighbors_a = graph.get_neighbors("A")
    neighbors_b = graph.get_neighbors("B")

    assert len(neighbors_a) == 1
    assert len(neighbors_b) == 1
    assert "B" in neighbors_a
    assert "A" in neighbors_b
    assert graph.get_degree("A") == 1
    assert graph.get_degree("B") == 1


def test_get_neighbors_nonexistent_node(sample_edges):
    """Test getting neighbors for a node that doesn't exist."""
    graph = Graph(edges=sample_edges)

    neighbors = graph.get_neighbors("NonExistent")
    assert len(neighbors) == 0


def test_multiple_edges_same_nodes(sample_edge_metadata):
    """Test handling multiple edges between the same nodes with different relations."""
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
            to_entity="B",
            relation_type=RelationType.CALLS,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        ),
    ]

    graph = Graph(edges=edges)
    neighbors = graph.get_neighbors("A")

    assert len(neighbors) == 1  # Should only count unique neighbor nodes
    assert "B" in neighbors
    assert graph.get_degree("A") == 1  # Degree counts unique neighbors


def test_add_edge(sample_edge_metadata):
    """Test adding a single edge to the graph."""
    graph = Graph(edges=[])
    edge = Edge(
        from_entity="A",
        to_entity="B",
        relation_type=RelationType.DEPENDS_ON,
        metadata=sample_edge_metadata,
        impact_score=0.8,
    )

    graph.add_edge(edge)

    assert graph.has_edge("A", "B")
    assert graph.get_edge_count() == 1
    assert graph.get_degree("A") == 1
    assert graph.get_degree("B") == 0
    assert "B" in graph.get_neighbors("A")


def test_remove_edge(sample_edges):
    """Test removing an edge from the graph."""
    graph = Graph(edges=sample_edges)
    initial_edge_count = graph.get_edge_count()

    graph.remove_edge("A", "B")

    assert not graph.has_edge("A", "B")
    assert graph.get_edge_count() == initial_edge_count - 1
    assert "B" not in graph.get_neighbors("A")


def test_remove_edge_cleanup(sample_edge_metadata):
    """Test edge cleanup after removal."""
    graph = Graph(edges=[])
    edge = Edge(
        from_entity="A",
        to_entity="B",
        relation_type=RelationType.DEPENDS_ON,
        metadata=sample_edge_metadata,
        impact_score=0.8,
    )
    graph.add_edge(edge)

    # Remove the only edge from A
    graph.remove_edge("A", "B")

    # Verify A's adjacency entry is removed since it has no more edges
    assert "A" not in graph.adjacency
    assert graph.get_edge_count() == 0


def test_remove_edge_with_remaining_edges(sample_edge_metadata):
    """Test edge removal when node still has other edges."""
    graph = Graph(edges=[])
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
    graph.add_edge(edge1)
    graph.add_edge(edge2)

    # Remove one edge but A should still have another
    graph.remove_edge("A", "B")

    # Verify A's adjacency entry remains since it has another edge
    assert "A" in graph.adjacency
    assert len(graph.adjacency["A"]) == 1
    assert graph.has_edge("A", "C")


def test_remove_nonexistent_edge():
    """Test removing an edge that doesn't exist."""
    graph = Graph(edges=[])

    with pytest.raises(EdgeNotFoundError):
        graph.remove_edge("A", "B")


def test_transaction_success(sample_edge_metadata):
    """Test successful transaction with multiple operations."""
    graph = Graph(edges=[])
    edge1 = Edge(
        from_entity="A",
        to_entity="B",
        relation_type=RelationType.DEPENDS_ON,
        metadata=sample_edge_metadata,
        impact_score=0.8,
    )
    edge2 = Edge(
        from_entity="B",
        to_entity="C",
        relation_type=RelationType.DEPENDS_ON,
        metadata=sample_edge_metadata,
        impact_score=0.8,
    )

    with graph.transaction():
        graph.add_edge(edge1)
        graph.add_edge(edge2)

    assert graph.has_edge("A", "B")
    assert graph.has_edge("B", "C")
    assert graph.get_edge_count() == 2


def test_transaction_rollback(sample_edge_metadata):
    """Test transaction rollback on error."""
    graph = Graph(edges=[])
    edge = Edge(
        from_entity="A",
        to_entity="B",
        relation_type=RelationType.DEPENDS_ON,
        metadata=sample_edge_metadata,
        impact_score=0.8,
    )

    try:
        with graph.transaction():
            graph.add_edge(edge)
            # This should raise an EdgeNotFoundError and trigger rollback
            graph.remove_edge("X", "Y")
    except EdgeNotFoundError:
        pass

    # Verify the graph is in its original state
    assert not graph.has_edge("A", "B")
    assert graph.get_edge_count() == 0


def test_transaction_custom_exception(sample_edge_metadata):
    """Test transaction rollback with a custom exception."""
    graph = Graph(edges=[])
    edge = Edge(
        from_entity="A",
        to_entity="B",
        relation_type=RelationType.DEPENDS_ON,
        metadata=sample_edge_metadata,
        impact_score=0.8,
    )

    class CustomError(Exception):
        pass

    try:
        with graph.transaction():
            graph.add_edge(edge)
            raise CustomError("Test error")
    except CustomError:
        pass

    # Verify the graph is in its original state
    assert not graph.has_edge("A", "B")
    assert graph.get_edge_count() == 0


def test_transaction_nested(sample_edge_metadata):
    """Test nested transactions."""
    graph = Graph(edges=[])
    edge1 = Edge(
        from_entity="A",
        to_entity="B",
        relation_type=RelationType.DEPENDS_ON,
        metadata=sample_edge_metadata,
        impact_score=0.8,
    )
    edge2 = Edge(
        from_entity="B",
        to_entity="C",
        relation_type=RelationType.DEPENDS_ON,
        metadata=sample_edge_metadata,
        impact_score=0.8,
    )

    with graph.transaction():
        graph.add_edge(edge1)
        with graph.transaction():
            graph.add_edge(edge2)

    assert graph.has_edge("A", "B")
    assert graph.has_edge("B", "C")
    assert graph.get_edge_count() == 2


def test_add_edges_batch(sample_edge_metadata):
    """Test adding multiple edges in a batch."""
    graph = Graph(edges=[])
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

    graph.add_edges_batch(edges)

    assert graph.has_edge("A", "B")
    assert graph.has_edge("B", "C")
    assert graph.get_edge_count() == 2
    assert graph.get_degree("A") == 1
    assert graph.get_degree("B") == 1


def test_add_edges_batch_rollback(sample_edge_metadata):
    """Test batch add operation rollback on error."""
    graph = Graph(edges=[])
    edges = [
        Edge(
            from_entity="A",
            to_entity="B",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        ),
        None,  # This will cause an error
    ]

    try:
        graph.add_edges_batch(edges)
    except AttributeError:
        pass

    # Verify no edges were added due to rollback
    assert not graph.has_edge("A", "B")
    assert graph.get_edge_count() == 0


def test_remove_edges_batch(sample_edges):
    """Test removing multiple edges in a batch."""
    graph = Graph(edges=sample_edges)
    initial_edge_count = graph.get_edge_count()
    edges_to_remove = [("A", "B"), ("B", "C")]

    graph.remove_edges_batch(edges_to_remove)

    assert not graph.has_edge("A", "B")
    assert not graph.has_edge("B", "C")
    assert graph.get_edge_count() == initial_edge_count - 2


def test_remove_edges_batch_rollback(sample_edges):
    """Test batch remove operation rollback on error."""
    graph = Graph(edges=sample_edges)
    initial_edge_count = graph.get_edge_count()
    edges_to_remove = [("A", "B"), ("X", "Y")]  # Second edge doesn't exist

    try:
        graph.remove_edges_batch(edges_to_remove)
    except EdgeNotFoundError:
        pass

    # Verify no edges were removed due to rollback
    assert graph.has_edge("A", "B")
    assert graph.get_edge_count() == initial_edge_count


def test_find_paths(sample_edges):
    """Test finding paths between nodes."""
    graph = Graph(edges=sample_edges)

    # Test direct path
    result = graph.find_paths("A", "B")
    assert isinstance(result, PathResult)
    assert result.nodes == ["A", "B"]

    # Test indirect path
    results = graph.find_paths("A", "C", path_type=PathType.ALL)
    paths = list(cast(Iterator[PathResult], results))
    assert len(paths) == 2

    # Verify both possible paths exist
    path_nodes = [path.nodes for path in paths]
    assert ["A", "C"] in path_nodes  # Direct path
    assert ["A", "B", "C"] in path_nodes  # Indirect path


def test_find_paths_with_max_depth(sample_edges):
    """Test finding paths with maximum depth limit."""
    graph = Graph(edges=sample_edges)

    # Test with max_depth=1 (only direct paths)
    results = graph.find_paths("A", "C", path_type=PathType.ALL, max_depth=1)
    paths = list(cast(Iterator[PathResult], results))
    assert len(paths) == 1
    assert paths[0].nodes == ["A", "C"]


def test_path_caching(sample_edges):
    """Test that paths are properly cached and retrieved."""
    # Clear cache before testing
    PathCache.clear()

    graph = Graph(edges=sample_edges)

    # First call should compute paths
    result1 = graph.find_paths("A", "C")
    assert isinstance(result1, PathResult)

    # Second call should retrieve from cache
    result2 = graph.find_paths("A", "C")
    assert isinstance(result2, PathResult)

    assert result1.nodes == result2.nodes
    assert result1.total_weight == result2.total_weight

    # Verify cache hit
    metrics = PathFinding.get_cache_metrics()
    assert metrics["hits"] > 0


def test_path_cache_invalidation(sample_edges, sample_edge_metadata):
    """Test that path cache is invalidated when graph structure changes."""
    # Clear cache before testing
    PathCache.clear()

    graph = Graph(edges=sample_edges)

    # Cache some paths
    _ = graph.find_paths("A", "C")

    # Add a new edge that creates a new path
    new_edge = Edge(
        from_entity="A",
        to_entity="D",
        relation_type=RelationType.DEPENDS_ON,
        metadata=sample_edge_metadata,
        impact_score=0.8,
    )
    graph.add_edge(new_edge)

    # Cache should be cleared
    metrics = PathFinding.get_cache_metrics()
    assert metrics["size"] == 0


def test_get_edge_safe_nonexistent_nodes(sample_edges):
    """Test get_edge_safe with non-existent nodes."""
    graph = Graph(edges=sample_edges)

    # Test with non-existent source node
    with pytest.raises(NodeNotFoundError) as exc_info:
        graph.get_edge_safe("NonExistent", "B")
    assert "Source node 'NonExistent' not found" in str(exc_info.value)

    # Test with non-existent target node
    with pytest.raises(NodeNotFoundError) as exc_info:
        graph.get_edge_safe("A", "NonExistent")
    assert "Target node 'NonExistent' not found" in str(exc_info.value)

    # Test with existing nodes but no edge
    with pytest.raises(EdgeNotFoundError) as exc_info:
        graph.get_edge_safe("A", "A")
    assert "No edge exists from 'A' to 'A'" in str(exc_info.value)


def test_from_edges_classmethod(sample_edges):
    """Test creating a graph using the from_edges classmethod."""
    # Create graph using from_edges
    graph1 = Graph.from_edges(sample_edges)

    # Create graph using constructor
    graph2 = Graph(edges=sample_edges)

    # Both graphs should have the same structure
    assert graph1.get_edge_count() == graph2.get_edge_count()
    assert graph1.get_nodes() == graph2.get_nodes()

    # Check specific edges exist in both graphs
    for edge in sample_edges:
        assert graph1.has_edge(edge.from_entity, edge.to_entity)
        assert graph2.has_edge(edge.from_entity, edge.to_entity)
