"""
Tests for graph traversal algorithms.
"""

from datetime import datetime

import pytest

from src.core.enums import RelationType
from src.core.exceptions import NodeNotFoundError
from src.core.graph import Graph
from src.core.graph_traversal import GraphTraversal, TraversalStrategy
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
def sample_graph(sample_edge_metadata):
    """
    Fixture providing a test graph with the following structure:
    A -> B -> D
    |    |
    v    v
    C <- E
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
            from_entity="A",
            to_entity="C",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.7,
        ),
        Edge(
            from_entity="B",
            to_entity="D",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.6,
        ),
        Edge(
            from_entity="B",
            to_entity="E",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.5,
        ),
        Edge(
            from_entity="E",
            to_entity="C",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.4,
        ),
    ]
    return Graph(edges=edges)


def test_bfs_traversal(sample_graph):
    """Test breadth-first search traversal."""
    traversal = GraphTraversal()
    visited = []

    # Collect nodes and their depths from BFS
    for node, depth in traversal.bfs(sample_graph, "A"):
        visited.append((node, depth))

    # Verify the traversal order and depths
    assert visited[0] == ("A", 0)  # Start node at depth 0

    # Level 1 nodes (depth 1)
    level1_nodes = {node for node, depth in visited if depth == 1}
    assert level1_nodes == {"B", "C"}

    # Level 2 nodes (depth 2)
    level2_nodes = {node for node, depth in visited if depth == 2}
    assert level2_nodes == {"D", "E"}


def test_dfs_traversal(sample_graph):
    """Test depth-first search traversal."""
    traversal = GraphTraversal()
    visited = []

    # Collect nodes and their depths from DFS
    for node, depth in traversal.dfs(sample_graph, "A"):
        visited.append((node, depth))

    # Verify basic properties of the traversal
    assert visited[0] == ("A", 0)  # Start node at depth 0
    assert len(visited) == 5  # All nodes should be visited exactly once
    assert set(node for node, _ in visited) == {"A", "B", "C", "D", "E"}

    # In a directed graph, nodes can be reached through different paths
    # C can be reached directly from A (depth 1) or through B->E->C (depth 3)
    # The actual depth depends on which path is traversed first
    for node, depth in visited:
        if node == "B":
            assert depth == 1  # B is always at depth 1
        elif node in {"D", "E"}:
            assert depth == 2  # D and E are always at depth 2


def test_bfs_with_max_depth(sample_graph):
    """Test BFS traversal with maximum depth limit."""
    traversal = GraphTraversal()
    visited = []

    # Only traverse to depth 1
    for node, depth in traversal.bfs(sample_graph, "A", max_depth=1):
        visited.append((node, depth))

    assert len(visited) == 3  # Should only visit A, B, and C
    assert visited[0] == ("A", 0)
    assert set(node for node, depth in visited if depth == 1) == {"B", "C"}
    assert all(depth <= 1 for _, depth in visited)


def test_dfs_with_max_depth(sample_graph):
    """Test DFS traversal with maximum depth limit."""
    traversal = GraphTraversal()
    visited = []

    # Only traverse to depth 1
    for node, depth in traversal.dfs(sample_graph, "A", max_depth=1):
        visited.append((node, depth))

    assert len(visited) == 3  # Should only visit A, B, and C
    assert visited[0] == ("A", 0)
    assert set(node for node, depth in visited if depth == 1) == {"B", "C"}
    assert all(depth <= 1 for _, depth in visited)


def test_traversal_with_filter(sample_edge_metadata):
    """Test traversal with a filter function."""
    traversal = GraphTraversal()

    # Create a graph where D can reach E and F
    edges = [
        Edge(
            from_entity="D",
            to_entity="E",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        ),
        Edge(
            from_entity="D",
            to_entity="F",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.8,
        ),
    ]
    graph = Graph(edges=edges)

    visited = []

    # Filter to only include nodes that come alphabetically after 'C'
    def filter_func(node: str) -> bool:
        return node > "C"

    for node, depth in traversal.bfs(graph, "D", filter_func=filter_func):
        visited.append((node, depth))

    # Should include D, E, and F since they all pass the filter
    assert set(node for node, _ in visited) == {"D", "E", "F"}


def test_traversal_with_filter_no_matches(sample_graph):
    """Test traversal with a filter that matches no nodes."""
    traversal = GraphTraversal()
    visited = []

    def filter_func(node: str) -> bool:
        return node > "Z"  # No nodes will match this

    for node, depth in traversal.bfs(sample_graph, "A", filter_func=filter_func):
        visited.append((node, depth))

    assert len(visited) == 0  # No nodes should be visited


def test_traversal_from_nonexistent_node(sample_graph):
    """Test traversal starting from a node that doesn't exist."""
    traversal = GraphTraversal()

    # BFS from nonexistent node should raise NodeNotFoundError
    with pytest.raises(NodeNotFoundError) as exc_info:
        list(traversal.bfs(sample_graph, "NonExistent"))
    assert "Start node 'NonExistent' not found in the graph" in str(exc_info.value)

    # DFS from nonexistent node should raise NodeNotFoundError
    with pytest.raises(NodeNotFoundError) as exc_info:
        list(traversal.dfs(sample_graph, "NonExistent"))
    assert "Start node 'NonExistent' not found in the graph" in str(exc_info.value)


def test_traversal_empty_graph():
    """Test traversal on an empty graph."""
    graph = Graph(edges=[])
    traversal = GraphTraversal()

    # BFS on empty graph should raise NodeNotFoundError
    with pytest.raises(NodeNotFoundError) as exc_info:
        list(traversal.bfs(graph, "A"))
    assert "Start node 'A' not found in the graph" in str(exc_info.value)

    # DFS on empty graph should raise NodeNotFoundError
    with pytest.raises(NodeNotFoundError) as exc_info:
        list(traversal.dfs(graph, "A"))
    assert "Start node 'A' not found in the graph" in str(exc_info.value)


def test_traversal_with_cycles(sample_edge_metadata):
    """Test traversal on a graph with cycles."""
    # Create a graph with a cycle: A -> B -> C -> A
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
    ]
    graph = Graph(edges=edges)
    traversal = GraphTraversal()

    # Test BFS with cycle
    visited_bfs = list(traversal.bfs(graph, "A"))
    assert len(visited_bfs) == 3  # Should visit each node exactly once
    assert set(node for node, _ in visited_bfs) == {"A", "B", "C"}

    # Test DFS with cycle
    visited_dfs = list(traversal.dfs(graph, "A"))
    assert len(visited_dfs) == 3  # Should visit each node exactly once
    assert set(node for node, _ in visited_dfs) == {"A", "B", "C"}


def test_traversal_with_max_depth_and_filter(sample_graph):
    """Test traversal with both max_depth and filter_func."""
    traversal = GraphTraversal()

    def filter_func(node: str) -> bool:
        return node in {"A", "B", "D"}

    # Test BFS with both constraints
    visited_bfs = list(traversal.bfs(sample_graph, "A", max_depth=2, filter_func=filter_func))
    assert len(visited_bfs) == 3
    assert visited_bfs[0] == ("A", 0)
    assert ("B", 1) in visited_bfs
    assert ("D", 2) in visited_bfs

    # Test DFS with both constraints
    visited_dfs = list(traversal.dfs(sample_graph, "A", max_depth=2, filter_func=filter_func))
    assert len(visited_dfs) == 3
    assert visited_dfs[0] == ("A", 0)
    assert ("B", 1) in visited_dfs
    assert ("D", 2) in visited_dfs


def test_filter_excluding_start_node(sample_graph):
    """
    Test traversal with a filter that excludes the start node.
    When the start node is filtered out, the traversal should yield no nodes.
    """
    traversal = GraphTraversal()

    def filter_func(node: str) -> bool:
        return node != "A"

    # BFS should yield no nodes when start node is filtered out
    visited_bfs = list(traversal.bfs(sample_graph, "A", filter_func=filter_func))
    assert len(visited_bfs) == 0

    # DFS should yield no nodes when start node is filtered out
    visited_dfs = list(traversal.dfs(sample_graph, "A", filter_func=filter_func))
    assert len(visited_dfs) == 0


def test_multiple_paths_graph(sample_edge_metadata):
    """Test traversal on a graph with multiple paths to the same node."""
    # Create a diamond-shaped graph: A -> B -> D
    #                               A -> C -> D
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
            impact_score=0.7,
        ),
        Edge(
            from_entity="B",
            to_entity="D",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.6,
        ),
        Edge(
            from_entity="C",
            to_entity="D",
            relation_type=RelationType.DEPENDS_ON,
            metadata=sample_edge_metadata,
            impact_score=0.5,
        ),
    ]
    graph = Graph(edges=edges)
    traversal = GraphTraversal()

    # Test BFS - D should be visited only once and at depth 2
    visited_bfs = list(traversal.bfs(graph, "A"))
    assert len(visited_bfs) == 4  # A, B, C, D
    d_nodes = [(node, depth) for node, depth in visited_bfs if node == "D"]
    assert len(d_nodes) == 1
    assert d_nodes[0][1] == 2  # D should be at depth 2

    # Test DFS - verify exact traversal order
    visited_dfs = list(traversal.dfs(graph, "A"))
    assert len(visited_dfs) == 4
    # DFS should follow one path completely before the other
    assert visited_dfs[0] == ("A", 0)
    # The exact order depends on implementation, but should be consistent
    possible_orders = [
        [("A", 0), ("B", 1), ("D", 2), ("C", 1)],
        [("A", 0), ("C", 1), ("D", 2), ("B", 1)],
    ]
    assert visited_dfs in possible_orders


def test_zero_max_depth(sample_graph):
    """Test traversal with max_depth=0."""
    traversal = GraphTraversal()

    # BFS with zero depth
    visited_bfs = list(traversal.bfs(sample_graph, "A", max_depth=0))
    assert len(visited_bfs) == 1
    assert visited_bfs[0] == ("A", 0)

    # DFS with zero depth
    visited_dfs = list(traversal.dfs(sample_graph, "A", max_depth=0))
    assert len(visited_dfs) == 1
    assert visited_dfs[0] == ("A", 0)


def test_traversal_with_one_way_relationships(sample_edge_metadata):
    """Test traversal with one-way relationships."""
    # Create a graph where relationships are strictly one-way:
    # A -> B -> C, but no back edges
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
    ]
    graph = Graph(edges=edges)
    traversal = GraphTraversal()

    # Test traversal from A (should reach all nodes)
    visited_from_a = list(traversal.bfs(graph, "A"))
    assert len(visited_from_a) == 3
    assert set(node for node, _ in visited_from_a) == {"A", "B", "C"}

    # Test traversal from B (should only reach B and C)
    visited_from_b = list(traversal.bfs(graph, "B"))
    assert len(visited_from_b) == 2
    assert set(node for node, _ in visited_from_b) == {"B", "C"}

    # Test traversal from C (should only contain C)
    visited_from_c = list(traversal.bfs(graph, "C"))
    assert len(visited_from_c) == 1
    assert visited_from_c[0] == ("C", 0)


def test_traversal_with_isolated_nodes(sample_edge_metadata):
    """Test traversal with isolated nodes (no outgoing edges)."""
    # Create a graph with some isolated nodes:
    # A -> B -> C, D (isolated), E (isolated)
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
    ]
    graph = Graph(edges=edges)
    traversal = GraphTraversal()

    # Test traversal from isolated node D should raise NodeNotFoundError
    with pytest.raises(NodeNotFoundError) as exc_info:
        list(traversal.bfs(graph, "D"))
    assert "Start node 'D' not found in the graph" in str(exc_info.value)

    # Test traversal from node with no outgoing edges (C)
    visited_from_c = list(traversal.bfs(graph, "C"))
    assert len(visited_from_c) == 1
    assert visited_from_c[0] == ("C", 0)

    # Test DFS from isolated node should raise NodeNotFoundError
    with pytest.raises(NodeNotFoundError) as exc_info:
        list(traversal.dfs(graph, "D"))
    assert "Start node 'D' not found in the graph" in str(exc_info.value)


def test_recursive_dfs_traversal(sample_graph):
    """Test recursive DFS traversal."""
    traversal = GraphTraversal()
    visited = []

    # Use recursive DFS strategy
    for node, depth in traversal.dfs(sample_graph, "A", strategy=TraversalStrategy.RECURSIVE_DFS):
        visited.append((node, depth))

    # Verify basic properties
    assert visited[0] == ("A", 0)  # Start node at depth 0
    assert len(visited) == 5  # All nodes should be visited exactly once
    assert set(node for node, _ in visited) == {"A", "B", "C", "D", "E"}

    # Test with max_depth
    visited_limited = list(
        traversal.dfs(sample_graph, "A", max_depth=1, strategy=TraversalStrategy.RECURSIVE_DFS)
    )
    assert len(visited_limited) == 3  # Should only visit A, B, and C
    assert all(depth <= 1 for _, depth in visited_limited)


def test_recursive_dfs_with_filter(sample_graph):
    """Test recursive DFS with filter function."""
    traversal = GraphTraversal()

    def filter_func(node: str) -> bool:
        return node in {"A", "B", "D"}

    visited = list(
        traversal.dfs(
            sample_graph,
            "A",
            filter_func=filter_func,
            strategy=TraversalStrategy.RECURSIVE_DFS,
        )
    )

    assert len(visited) == 3  # Only A, B, and D should be visited
    assert set(node for node, _ in visited) == {"A", "B", "D"}


def test_traverse_with_different_strategies(sample_graph):
    """Test traverse method with different strategies."""
    traversal = GraphTraversal()

    # Test BFS strategy
    bfs_visited = list(traversal.traverse(sample_graph, "A", strategy=TraversalStrategy.BFS))
    assert len(bfs_visited) == 5
    assert bfs_visited[0] == ("A", 0)
    level1_nodes = {node for node, depth in bfs_visited if depth == 1}
    assert level1_nodes == {"B", "C"}

    # Test iterative DFS strategy
    dfs_visited = list(
        traversal.traverse(sample_graph, "A", strategy=TraversalStrategy.ITERATIVE_DFS)
    )
    assert len(dfs_visited) == 5
    assert dfs_visited[0] == ("A", 0)

    # Test recursive DFS strategy
    recursive_dfs_visited = list(
        traversal.traverse(sample_graph, "A", strategy=TraversalStrategy.RECURSIVE_DFS)
    )
    assert len(recursive_dfs_visited) == 5
    assert recursive_dfs_visited[0] == ("A", 0)

    # Test with max_depth and filter
    def filter_func(node: str) -> bool:
        return node in {"A", "B", "D"}

    filtered_visited = list(
        traversal.traverse(
            sample_graph,
            "A",
            strategy=TraversalStrategy.BFS,
            max_depth=2,
            filter_func=filter_func,
        )
    )
    assert len(filtered_visited) == 3
    assert all(node in {"A", "B", "D"} for node, _ in filtered_visited)
    assert all(depth <= 2 for _, depth in filtered_visited)
