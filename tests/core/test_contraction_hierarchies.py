"""Tests for Contraction Hierarchies algorithm."""

import pytest
from datetime import datetime

from polaris.core.exceptions import GraphOperationError
from polaris.core.graph import Graph
from polaris.core.graph.traversal.algorithms.advanced.contraction_hierarchies import (
    ContractionHierarchies,
    Shortcut,
)
from polaris.core.models import Edge, EdgeMetadata
from polaris.core.enums import RelationType


def create_test_graph() -> Graph:
    """Create a simple test graph for Contraction Hierarchies testing."""
    # Create base metadata for edges
    base_metadata = EdgeMetadata(
        created_at=datetime.now(),
        last_modified=datetime.now(),
        confidence=1.0,
        source="test",
        weight=1.0,
    )

    # Create edges forming a diamond shape: A -> B -> D
    #                                       \-> C ->/
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
                weight=2.0,
            ),
            impact_score=0.5,
        ),
        Edge(
            from_entity="B",
            to_entity="D",
            relation_type=RelationType.CONNECTS_TO,
            metadata=EdgeMetadata(
                created_at=base_metadata.created_at,
                last_modified=base_metadata.last_modified,
                confidence=base_metadata.confidence,
                source=base_metadata.source,
                weight=3.0,
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
                weight=2.0,
            ),
            impact_score=0.5,
        ),
        Edge(
            from_entity="C",
            to_entity="D",
            relation_type=RelationType.CONNECTS_TO,
            metadata=EdgeMetadata(
                created_at=base_metadata.created_at,
                last_modified=base_metadata.last_modified,
                confidence=base_metadata.confidence,
                source=base_metadata.source,
                weight=3.0,
            ),
            impact_score=0.5,
        ),
    ]

    return Graph(edges)


def test_initialization():
    """Test Contraction Hierarchies initialization."""
    graph = create_test_graph()
    ch = ContractionHierarchies(graph)

    assert ch.graph == graph
    assert not ch._preprocessed
    assert len(ch.node_level) == 0
    assert len(ch.shortcuts) == 0
    assert len(ch.contracted_neighbors) == 0


def test_preprocessing():
    """Test preprocessing and shortcut creation."""
    graph = create_test_graph()
    ch = ContractionHierarchies(graph)

    ch.preprocess()

    assert ch._preprocessed
    assert len(ch.node_level) == len(graph.get_nodes())

    # Verify node levels are unique and continuous
    levels = set(ch.node_level.values())
    assert len(levels) == len(ch.node_level)
    assert min(levels) == 0
    assert max(levels) == len(levels) - 1


def test_find_path_without_preprocessing():
    """Test that finding path without preprocessing raises error."""
    graph = create_test_graph()
    ch = ContractionHierarchies(graph)

    with pytest.raises(GraphOperationError):
        ch.find_path("A", "D")


def test_path_finding():
    """Test basic path finding."""
    graph = create_test_graph()
    ch = ContractionHierarchies(graph)
    ch.preprocess()

    result = ch.find_path("A", "D")

    assert result.path, "Path should not be empty"
    assert result.total_weight == 5.0, f"Expected total_weight to be 5.0, got {result.total_weight}"

    # Verify path length
    assert len(result.path) == 2, f"Expected path length 2, got {len(result.path)}"

    # Verify path starts at A and ends at D
    assert (
        result.path[0].from_entity == "A"
    ), f"Expected first edge from 'A', got {result.path[0].from_entity}"
    assert (
        result.path[1].to_entity == "D"
    ), f"Expected second edge to 'D', got {result.path[1].to_entity}"

    # Verify path goes through either B or C (both are valid)
    middle_node = result.path[0].to_entity
    assert middle_node in ["B", "C"], f"Path should go through B or C, got {middle_node}"
    assert (
        result.path[1].from_entity == middle_node
    ), f"Expected second edge from {middle_node}, got {result.path[1].from_entity}"

    # Verify path consistency
    current = "A"
    total_weight = 0
    for edge in result.path:
        assert (
            edge.from_entity == current
        ), f"Expected from_entity {current}, got {edge.from_entity}"
        current = edge.to_entity
        total_weight += edge.metadata.weight
    assert current == "D", f"Path does not end at 'D', ends at '{current}'"
    assert total_weight == result.total_weight, "Total weight mismatch in PathResult"


def test_shortcut_unpacking():
    """Test that shortcuts are properly unpacked in the final path."""
    graph = create_test_graph()
    ch = ContractionHierarchies(graph)
    ch.preprocess()

    result = ch.find_path("A", "D")

    # Verify no shortcuts in final path
    for edge in result.path:
        assert edge.relation_type == RelationType.CONNECTS_TO
        assert "Shortcut" not in str(edge.context or "")


def test_nonexistent_path():
    """Test handling of nonexistent paths."""
    # Create graph without path to D
    edges = [
        Edge(
            from_entity="A",
            to_entity="B",
            relation_type=RelationType.CONNECTS_TO,
            metadata=EdgeMetadata(
                created_at=datetime.now(),
                last_modified=datetime.now(),
                confidence=1.0,
                source="test",
                weight=4.0,
            ),
            impact_score=0.5,
        ),
        Edge(
            from_entity="C",
            to_entity="D",
            relation_type=RelationType.CONNECTS_TO,
            metadata=EdgeMetadata(
                created_at=datetime.now(),
                last_modified=datetime.now(),
                confidence=1.0,
                source="test",
                weight=2.0,
            ),
            impact_score=0.5,
        ),
    ]
    graph = Graph(edges)

    ch = ContractionHierarchies(graph)
    ch.preprocess()

    with pytest.raises(GraphOperationError):
        ch.find_path("A", "D")


def test_path_filter():
    """Test path finding with filter function."""
    graph = create_test_graph()
    ch = ContractionHierarchies(graph)
    ch.preprocess()

    # Filter that rejects paths containing node B
    def filter_func(path):
        return not any(edge.to_entity == "B" for edge in path)

    result = ch.find_path("A", "D", filter_func=filter_func)
    assert not any(edge.to_entity == "B" for edge in result.path)
    # Should use the A->C->D path
    assert len(result.path) == 2
    assert result.path[0].from_entity == "A"
    assert result.path[0].to_entity == "C"
    assert result.path[1].from_entity == "C"
    assert result.path[1].to_entity == "D"


def test_weight_function():
    """Test path finding with custom weight function."""
    graph = create_test_graph()
    ch = ContractionHierarchies(graph)
    ch.preprocess()

    # Custom weight function that doubles all weights
    def weight_func(edge):
        return edge.metadata.weight * 2

    result = ch.find_path("A", "D", weight_func=weight_func)

    # Verify total weight reflects doubled weights
    # The path should be the same but with doubled weights
    normal_result = ch.find_path("A", "D")
    expected_weight = sum(edge.metadata.weight for edge in normal_result.path) * 2
    assert result.total_weight == expected_weight


def test_memory_management():
    """Test memory management during preprocessing and path finding."""
    graph = create_test_graph()
    ch = ContractionHierarchies(graph, max_memory_mb=1000)

    # Should complete without memory errors
    ch.preprocess()
    result = ch.find_path("A", "D")
    assert result.path


def test_node_importance():
    """Test node importance calculation."""
    graph = create_test_graph()
    ch = ContractionHierarchies(graph)

    # Calculate importance for each node
    importances = {node: ch._calculate_node_importance(node) for node in graph.get_nodes()}

    # Verify all nodes have non-negative importance
    assert all(importance >= 0 for importance in importances.values())

    # Verify nodes with more edges have higher importance
    # B and C have same number of edges (2 each)
    assert importances["B"] == importances["C"]
    # A has more edges than D
    assert importances["A"] > importances["D"]


def test_witness_search():
    """Test witness path search during shortcut creation."""
    graph = create_test_graph()
    ch = ContractionHierarchies(graph)

    # Test shortcut necessity for a path where witness exists
    # A->C->D is a witness for A->B->D
    assert not ch._is_shortcut_necessary("A", "D", "B", 10.0)

    # Test shortcut necessity where no witness exists
    # No path from A to D avoiding both B and C
    assert ch._is_shortcut_necessary("A", "D", "B", 4.0)


def test_shortcut_creation():
    """Test shortcut creation during node contraction."""
    graph = create_test_graph()
    ch = ContractionHierarchies(graph)

    # Contract node B
    shortcuts = ch._contract_node("B")

    # Verify shortcut properties
    assert len(shortcuts) > 0
    for u, v in shortcuts:
        shortcut = ch.shortcuts[(u, v)]
        assert shortcut.via_node == "B"
        assert shortcut.edge.metadata.source == "contraction_hierarchies"
        assert "Shortcut" in str(shortcut.edge.context)
        # Verify shortcut weight is sum of component edges
        assert (
            shortcut.edge.metadata.weight
            == shortcut.lower_edge.metadata.weight + shortcut.upper_edge.metadata.weight
        )
