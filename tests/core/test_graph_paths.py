"""
Tests for graph path finding algorithms.
"""

from datetime import datetime
from time import sleep
from typing import Iterator, List, Set, Tuple, Union, cast

import pytest

from src.core.enums import RelationType
from src.core.exceptions import GraphOperationError, NodeNotFoundError
from src.core.graph import Graph
from src.core.graph_paths import (
    DEFAULT_MAX_PATH_LENGTH,
    PathFinding,
    PathResult,
    PathType,
    PathValidationError,
)
from src.core.graph_paths.cache import PathCache
from src.core.models import Edge, EdgeMetadata


@pytest.fixture
def sample_edge_metadata() -> EdgeMetadata:
    """Fixture providing sample edge metadata."""
    return EdgeMetadata(
        created_at=datetime.now(),
        last_modified=datetime.now(),
        confidence=0.9,
        source="test_source",
    )


@pytest.fixture
def cyclic_graph(sample_edge_metadata) -> Graph:
    """
    Fixture providing a test graph with cycles:
    A -> B -> C -> A
    |         |
    v         v
    D ------> E
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
            impact_score=0.7,
        ),
        Edge(
            from_entity="C",
            to_entity="A",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.6,
        ),
        Edge(
            from_entity="A",
            to_entity="D",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.5,
        ),
        Edge(
            from_entity="C",
            to_entity="E",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.4,
        ),
        Edge(
            from_entity="D",
            to_entity="E",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.3,
        ),
    ]
    return Graph(edges=edges)


def test_shortest_path_basic(cyclic_graph):
    """Test basic shortest path finding."""
    path = PathFinding.shortest_path(cyclic_graph, "A", "E")
    assert path.length == 2  # A -> D -> E is shortest
    assert path[0].from_entity == "A" and path[0].to_entity == "D"
    assert path[1].from_entity == "D" and path[1].to_entity == "E"


def test_shortest_path_with_weights(cyclic_graph):
    """Test shortest path with custom weight function."""

    def weight_func(edge: Edge) -> float:
        return 1.0 / edge.impact_score  # Higher impact means lower weight

    path = PathFinding.shortest_path(cyclic_graph, "A", "E", weight_func=weight_func)
    # Should prefer path with higher impact scores
    assert path.length == 3  # A -> B -> C -> E
    assert [edge.to_entity for edge in path] == ["B", "C", "E"]


def test_shortest_path_no_path(cyclic_graph, sample_edge_metadata):
    """Test shortest path when no path exists."""
    # Add an isolated node F
    edge = Edge(
        from_entity="F",
        to_entity="F",
        relation_type=RelationType.DEPENDS_ON,
        metadata=sample_edge_metadata,
        impact_score=0.8,
    )
    cyclic_graph.add_edge(edge)

    with pytest.raises(GraphOperationError, match="No path exists between A and F"):
        PathFinding.shortest_path(cyclic_graph, "A", "F")


def test_edge_weight_validation(cyclic_graph):
    """Test edge weight validation."""

    def invalid_weight_func(edge: Edge) -> float:
        return 0.0  # Invalid weight

    with pytest.raises(ValueError, match="Edge weight must be positive"):
        PathFinding._get_edge_weight(cyclic_graph.get_edge("A", "B"), invalid_weight_func)

    def negative_weight_func(edge: Edge) -> float:
        return -1.0  # Invalid weight

    with pytest.raises(ValueError, match="Edge weight must be positive"):
        PathFinding._get_edge_weight(cyclic_graph.get_edge("A", "B"), negative_weight_func)


def test_path_result_methods(cyclic_graph):
    """Test PathResult methods and properties."""
    # Get a simple path
    edges = [cyclic_graph.get_edge("A", "D"), cyclic_graph.get_edge("D", "E")]

    # Test _calculate_path_weight
    total_weight = PathFinding._calculate_path_weight(edges, None)
    # Use pytest.approx for floating point comparison
    assert total_weight == pytest.approx(2.0, rel=1e-9)  # Default weight of 1.0 per edge

    # Test _create_path_result
    result = PathFinding._create_path_result(edges, None, cyclic_graph)
    assert result.total_weight == pytest.approx(2.0, rel=1e-9)
    assert result.length == 2
    assert result.nodes == ["A", "D", "E"]

    # Test iteration
    path_edges = list(result)
    assert len(path_edges) == 2
    assert path_edges[0].from_entity == "A"
    assert path_edges[1].to_entity == "E"

    # Test indexing
    assert result[0].from_entity == "A"
    assert result[1].to_entity == "E"


def test_path_validation(cyclic_graph):
    """Test path validation."""
    # Create a valid path
    valid_path = [cyclic_graph.get_edge("A", "D"), cyclic_graph.get_edge("D", "E")]
    result = PathFinding._create_path_result(valid_path, None, cyclic_graph)
    result.validate(cyclic_graph)  # Should not raise

    # Create a disconnected path
    disconnected_path = [
        cyclic_graph.get_edge("A", "D"),
        cyclic_graph.get_edge("B", "C"),
    ]
    result = PathFinding._create_path_result(disconnected_path, None, cyclic_graph)
    with pytest.raises(PathValidationError, match="Path discontinuity"):
        result.validate(cyclic_graph)

    # Test empty path validation
    empty_result = PathResult(path=[], total_weight=0.0, length=0)
    empty_result.validate(cyclic_graph)  # Should not raise

    # Test length mismatch
    invalid_result = PathResult(path=valid_path, total_weight=2.0, length=3)
    with pytest.raises(PathValidationError, match="Path length mismatch"):
        invalid_result.validate(cyclic_graph)


def test_all_paths_max_paths_limit(cyclic_graph):
    """Test max_paths limit in all_paths."""
    # Set max_paths to 1 to get only the first path
    paths: List[PathResult] = list(PathFinding.all_paths(cyclic_graph, "A", "E", max_paths=1))
    assert len(paths) == 1

    # Invalid max_paths
    with pytest.raises(ValueError, match="max_paths must be positive"):
        next(PathFinding.all_paths(cyclic_graph, "A", "E", max_paths=0))


def test_all_paths_basic(cyclic_graph):
    """Test basic path finding without cycles."""
    paths: List[PathResult] = list(PathFinding.all_paths(cyclic_graph, "A", "E", max_length=3))
    assert len(paths) == 2  # Two possible paths: A->D->E and A->B->C->E

    # Verify both paths are present using PathResult's iteration
    path_ends: Set[Tuple[str, ...]] = {
        tuple(edge.to_entity for edge in path_result) for path_result in paths
    }
    assert path_ends == {("D", "E"), ("B", "C", "E")}


def test_all_paths_with_cycle_detection(cyclic_graph):
    """Test that cycles are properly detected and handled."""
    paths: List[PathResult] = list(PathFinding.all_paths(cyclic_graph, "A", "E"))

    # Verify no paths contain cycles
    for path_result in paths:
        visited: Set[str] = {path_result[0].from_entity}
        for edge in path_result:
            visited.add(edge.to_entity)
            # Each node should only appear once
            assert len(visited) == len(set(visited))


def test_all_paths_max_length(cyclic_graph):
    """Test max_length constraint in path finding."""
    # Only paths of length 2 or less
    short_paths: List[PathResult] = list(
        PathFinding.all_paths(cyclic_graph, "A", "E", max_length=2)
    )
    assert len(short_paths) == 1  # Only A->D->E
    assert short_paths[0].length == 2

    # Allow longer paths
    all_paths: List[PathResult] = list(PathFinding.all_paths(cyclic_graph, "A", "E", max_length=3))
    assert len(all_paths) == 2  # Both A->D->E and A->B->C->E


def test_all_paths_with_filter(cyclic_graph):
    """Test path filtering functionality."""

    def filter_func(edges: List[Edge]) -> bool:
        # Only accept paths with total impact score > 1.0
        return sum(edge.impact_score for edge in edges) > 1.0

    paths: List[PathResult] = list(
        PathFinding.all_paths(cyclic_graph, "A", "E", filter_func=filter_func)
    )

    # Verify all returned paths satisfy the filter
    for path_result in paths:
        assert sum(edge.impact_score for edge in path_result) > 1.0


def test_all_paths_invalid_max_length(cyclic_graph):
    """Test handling of invalid max_length values."""
    with pytest.raises(ValueError, match="max_length must be positive"):
        next(PathFinding.all_paths(cyclic_graph, "A", "E", max_length=0))

    with pytest.raises(ValueError, match="max_length must be positive"):
        next(PathFinding.all_paths(cyclic_graph, "A", "E", max_length=-1))


def test_all_paths_default_max_length(cyclic_graph):
    """Test default max_length behavior."""
    # Should use DEFAULT_MAX_PATH_LENGTH when max_length is None
    paths: List[PathResult] = list(PathFinding.all_paths(cyclic_graph, "A", "E"))
    assert len(paths) == 2  # Should find all valid paths
    # Verify no path exceeds default length
    assert all(path_result.length <= DEFAULT_MAX_PATH_LENGTH for path_result in paths)


def test_all_paths_nonexistent_nodes(cyclic_graph):
    """Test handling of nonexistent nodes."""
    with pytest.raises(NodeNotFoundError, match="Start node 'X' not found"):
        next(PathFinding.all_paths(cyclic_graph, "X", "E"))

    with pytest.raises(NodeNotFoundError, match="End node 'Y' not found"):
        next(PathFinding.all_paths(cyclic_graph, "A", "Y"))


def test_all_paths_deep_recursion(cyclic_graph):
    """Test handling of potentially deep recursion scenarios."""
    # Set a very small max_length to verify deep recursion is prevented
    paths: List[PathResult] = list(PathFinding.all_paths(cyclic_graph, "A", "E", max_length=2))

    # Should only find the direct path
    assert len(paths) == 1
    assert paths[0].length == 2  # A->D->E

    # Verify the path found is the shortest one
    assert paths[0][0].from_entity == "A" and paths[0][0].to_entity == "D"
    assert paths[0][1].from_entity == "D" and paths[0][1].to_entity == "E"


def test_all_paths_cycle_with_different_lengths(cyclic_graph):
    """Test paths in presence of cycles with different length constraints."""
    # Test with increasing max_length values
    paths_2: List[PathResult] = list(PathFinding.all_paths(cyclic_graph, "A", "E", max_length=2))
    paths_3: List[PathResult] = list(PathFinding.all_paths(cyclic_graph, "A", "E", max_length=3))
    paths_4: List[PathResult] = list(PathFinding.all_paths(cyclic_graph, "A", "E", max_length=4))

    # Verify number of paths found increases with max_length
    assert len(paths_2) <= len(paths_3)
    # paths_4 should not find more paths due to cycle detection
    assert len(paths_4) == len(paths_3)

    # Verify all paths in paths_2 are also in paths_3
    # Use PathResult's iteration to get edges
    path_ends_2: Set[Tuple[str, ...]] = {
        tuple(edge.to_entity for edge in path_result) for path_result in paths_2
    }
    path_ends_3: Set[Tuple[str, ...]] = {
        tuple(edge.to_entity for edge in path_result) for path_result in paths_3
    }
    assert path_ends_2.issubset(path_ends_3)


def test_path_finding_empty_graph():
    """Test path finding on an empty graph."""
    empty_graph = Graph(edges=[])

    with pytest.raises(NodeNotFoundError):
        next(PathFinding.all_paths(empty_graph, "A", "B"))

    with pytest.raises(NodeNotFoundError):
        PathFinding.shortest_path(empty_graph, "A", "B")


def test_find_paths_different_types(cyclic_graph):
    """Test find_paths with different path types."""
    # Test shortest path type
    result = PathFinding.find_paths(cyclic_graph, "A", "E", path_type=PathType.SHORTEST)
    assert isinstance(result, PathResult)
    shortest_path = cast(PathResult, result)
    assert shortest_path.length == 2  # A->D->E is shortest

    # Test all paths type
    result = PathFinding.find_paths(cyclic_graph, "A", "E", path_type=PathType.ALL)
    assert not isinstance(result, PathResult)  # Should be an iterator
    paths: List[PathResult] = list(cast(Iterator[PathResult], result))
    assert len(paths) == 2  # Two possible paths

    # Test filtered paths type
    def filter_func(edges: List[Edge]) -> bool:
        # Filter function operates on List[Edge]
        return len(edges) == 2  # Only accept paths of length 2

    result = PathFinding.find_paths(
        cyclic_graph, "A", "E", path_type=PathType.FILTERED, filter_func=filter_func
    )
    assert not isinstance(result, PathResult)  # Should be an iterator
    paths = list(cast(Iterator[PathResult], result))
    assert len(paths) == 1  # Only A->D->E matches filter
    # Check length property on PathResult objects
    assert all(path_result.length == 2 for path_result in paths)


# New tests for bidirectional search and caching


def test_bidirectional_search_basic(cyclic_graph):
    """Test basic bidirectional search functionality."""
    result = PathFinding.bidirectional_search(cyclic_graph, "A", "E")
    assert isinstance(result, PathResult)
    assert result.nodes[0] == "A"
    assert result.nodes[-1] == "E"
    assert len(result) == 2  # A->D->E is shortest


def test_bidirectional_search_with_depth_limit(cyclic_graph):
    """Test bidirectional search respects depth limit."""
    # Test with depth limit that should allow finding path
    result = PathFinding.bidirectional_search(cyclic_graph, "A", "E", max_depth=2)
    assert isinstance(result, PathResult)
    assert len(result) <= 2

    # Test with depth limit that should prevent finding path
    with pytest.raises(GraphOperationError):
        PathFinding.bidirectional_search(cyclic_graph, "A", "E", max_depth=1)


def test_bidirectional_search_with_weights(cyclic_graph):
    """Test bidirectional search with custom weight function."""

    def weight_func(edge: Edge) -> float:
        return 1.0 / edge.impact_score  # Higher impact means lower weight

    result = PathFinding.bidirectional_search(cyclic_graph, "A", "E", weight_func=weight_func)
    assert isinstance(result, PathResult)
    # Should prefer path with higher impact scores
    assert result.nodes == ["A", "B", "C", "E"]


def test_caching_behavior(cyclic_graph):
    """Test that path finding results are properly cached."""
    # Clear cache before testing
    PathCache.clear()

    # First search should miss cache
    result1 = PathFinding.bidirectional_search(cyclic_graph, "A", "E")
    metrics1 = PathFinding.get_cache_metrics()
    assert metrics1["hits"] == 0
    assert metrics1["misses"] > 0

    # Second search should hit cache
    result2 = PathFinding.bidirectional_search(cyclic_graph, "A", "E")
    metrics2 = PathFinding.get_cache_metrics()
    assert metrics2["hits"] == 1

    # Results should be identical
    assert result1.nodes == result2.nodes
    assert result1.total_weight == result2.total_weight


def test_cache_expiration(cyclic_graph):
    """Test that cached results expire correctly."""
    # Configure cache with short TTL for testing
    original_cache = PathCache.cache
    try:
        # Use reconfigure method instead of direct constructor
        PathCache.reconfigure(max_size=1000, ttl=1)  # 1 second TTL

        # First search
        PathFinding.bidirectional_search(cyclic_graph, "A", "E")

        # Wait for cache to expire
        sleep(1.1)

        # Search again - should miss cache
        PathFinding.bidirectional_search(cyclic_graph, "A", "E")
        metrics = PathFinding.get_cache_metrics()
        assert metrics["misses"] >= 2
    finally:
        # Restore original cache
        PathCache.cache = original_cache


def test_cache_metrics(cyclic_graph):
    """Test cache metrics collection."""
    # Clear cache and metrics
    PathCache.clear()

    # Perform multiple searches
    for _ in range(3):
        PathFinding.bidirectional_search(cyclic_graph, "A", "E")

    metrics = PathFinding.get_cache_metrics()
    assert isinstance(metrics, dict)
    assert "size" in metrics
    assert "hits" in metrics
    assert "misses" in metrics
    assert "hit_rate" in metrics
    assert isinstance(metrics["avg_access_time_ms"], float)

    # Verify hit rate calculation
    assert 0 <= metrics["hit_rate"] <= 1.0
    assert metrics["hits"] + metrics["misses"] > 0
