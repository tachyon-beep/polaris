"""Tests for A* with Landmarks (ALT) algorithm."""

import pytest
from datetime import datetime

from polaris.core.exceptions import GraphOperationError
from polaris.core.graph import Graph
from polaris.core.graph_paths.algorithms.advanced.alt import ALTPathFinder
from polaris.core.models import Edge, EdgeMetadata
from polaris.core.enums import RelationType
from polaris.core.graph_paths.models import PathValidationError


def create_test_graph(include_direct_path: bool = True) -> Graph:
    """Create a simple test graph for ALT algorithm testing."""
    # Create base metadata for edges
    base_metadata = EdgeMetadata(
        created_at=datetime.now(),
        last_modified=datetime.now(),
        confidence=1.0,
        source="test",
        weight=1.0,
    )

    # Create edges with weights
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
                weight=4.0,
            ),
            impact_score=0.5,
        ),
        Edge(
            from_entity="B",
            to_entity="C",
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
            from_entity="C",
            to_entity="D",
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
            from_entity="D",
            to_entity="E",
            relation_type=RelationType.CONNECTS_TO,
            metadata=EdgeMetadata(
                created_at=base_metadata.created_at,
                last_modified=base_metadata.last_modified,
                confidence=base_metadata.confidence,
                source=base_metadata.source,
                weight=1.0,
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
                weight=8.0,
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
                weight=6.0,
            ),
            impact_score=0.5,
        ),
        Edge(
            from_entity="C",
            to_entity="E",
            relation_type=RelationType.CONNECTS_TO,
            metadata=EdgeMetadata(
                created_at=base_metadata.created_at,
                last_modified=base_metadata.last_modified,
                confidence=base_metadata.confidence,
                source=base_metadata.source,
                weight=4.0,
            ),
            impact_score=0.5,
        ),
    ]

    # Add direct path from A to E for path filter test if requested
    if include_direct_path:
        edges.append(
            Edge(
                from_entity="A",
                to_entity="E",
                relation_type=RelationType.CONNECTS_TO,
                metadata=EdgeMetadata(
                    created_at=base_metadata.created_at,
                    last_modified=base_metadata.last_modified,
                    confidence=base_metadata.confidence,
                    source=base_metadata.source,
                    weight=10.0,
                ),
                impact_score=0.5,
            )
        )

    return Graph(edges)


def test_alt_initialization():
    """Test ALT pathfinder initialization."""
    graph = create_test_graph()
    finder = ALTPathFinder(graph, num_landmarks=2)

    assert finder.graph == graph
    assert finder.num_landmarks == 2
    assert not finder._preprocessed
    assert len(finder.landmarks) == 0


def test_preprocessing():
    """Test landmark selection and preprocessing."""
    graph = create_test_graph()
    finder = ALTPathFinder(graph, num_landmarks=2)

    finder.preprocess()

    assert finder._preprocessed
    assert len(finder.landmarks) == 2

    # Verify landmark distances are computed
    for landmark_dist in finder.landmarks.values():
        assert len(landmark_dist.forward) > 0
        assert len(landmark_dist.backward) > 0


def test_find_path_without_preprocessing():
    """Test that finding path without preprocessing raises error."""
    graph = create_test_graph()
    finder = ALTPathFinder(graph)

    with pytest.raises(GraphOperationError):
        finder.find_path("A", "E")


def test_unidirectional_path_finding():
    """Test unidirectional ALT path finding."""
    graph = create_test_graph()
    finder = ALTPathFinder(graph, num_landmarks=2)
    finder.preprocess()

    result = finder.find_path("A", "E", bidirectional=False)

    assert result.path
    assert result.total_weight > 0
    assert len(result.path) > 0

    # Verify path exists in graph
    current = "A"
    for edge in result.path:
        assert edge.from_entity == current
        current = edge.to_entity
    assert current == "E"


def test_bidirectional_path_finding():
    """Test bidirectional ALT path finding."""
    graph = create_test_graph()
    finder = ALTPathFinder(graph, num_landmarks=2)
    finder.preprocess()

    result = finder.find_path("A", "E", bidirectional=True)

    assert result.path
    assert result.total_weight > 0
    assert len(result.path) > 0

    # Verify path exists in graph
    current = "A"
    for edge in result.path:
        assert edge.from_entity == current
        current = edge.to_entity
    assert current == "E"


def test_max_length_constraint():
    """Test path finding with max length constraint."""
    # Create graph without direct path for max length test
    graph = create_test_graph(include_direct_path=False)
    finder = ALTPathFinder(graph, num_landmarks=2)
    finder.preprocess()

    # Should fail with very short max length
    with pytest.raises(PathValidationError):
        finder.find_path("A", "E", max_length=1)

    # Should succeed with sufficient max length
    result = finder.find_path("A", "E", max_length=5)
    assert len(result.path) <= 5


def test_nonexistent_path():
    """Test handling of nonexistent paths."""
    # Create base metadata
    base_metadata = EdgeMetadata(
        created_at=datetime.now(),
        last_modified=datetime.now(),
        confidence=1.0,
        source="test",
        weight=1.0,
    )

    # Create graph without paths to E
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
                weight=4.0,
            ),
            impact_score=0.5,
        ),
        Edge(
            from_entity="B",
            to_entity="C",
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
            from_entity="C",
            to_entity="D",
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
    ]
    graph = Graph(edges)

    finder = ALTPathFinder(graph, num_landmarks=2)
    finder.preprocess()

    with pytest.raises(GraphOperationError):
        finder.find_path("A", "E")


def test_path_filter():
    """Test path finding with filter function."""
    graph = create_test_graph()
    finder = ALTPathFinder(graph, num_landmarks=2)
    finder.preprocess()

    # Filter that rejects paths containing node C
    def filter_func(path):
        return not any(edge.to_entity == "C" for edge in path)

    result = finder.find_path("A", "E", filter_func=filter_func)
    assert not any(edge.to_entity == "C" for edge in result.path)
    # Should use the direct A->E path
    assert len(result.path) == 1
    assert result.path[0].from_entity == "A"
    assert result.path[0].to_entity == "E"


def test_weight_function():
    """Test path finding with custom weight function."""
    graph = create_test_graph()
    finder = ALTPathFinder(graph, num_landmarks=2)
    finder.preprocess()

    # Custom weight function that doubles all weights
    def weight_func(edge):
        return edge.metadata.weight * 2

    result = finder.find_path("A", "E", weight_func=weight_func)

    # Verify total weight reflects doubled weights
    # The path should be the same but with doubled weights
    normal_result = finder.find_path("A", "E")
    expected_weight = sum(edge.metadata.weight for edge in normal_result.path) * 2
    assert result.total_weight == expected_weight


def test_landmark_selection():
    """Test landmark selection with small graph."""
    graph = create_test_graph()

    # Test with num_landmarks greater than nodes
    finder = ALTPathFinder(graph, num_landmarks=10)
    finder.preprocess()

    assert len(finder.landmarks) == len(graph.get_nodes())


def test_heuristic_computation():
    """Test heuristic computation."""
    graph = create_test_graph()
    finder = ALTPathFinder(graph, num_landmarks=2)

    # Before preprocessing, heuristic should return 0
    h = finder._compute_heuristic("A", "E", None)
    assert h == 0

    finder.preprocess()

    # After preprocessing, heuristic should return non-negative value
    h = finder._compute_heuristic("A", "E", None)
    assert h >= 0


def test_memory_management():
    """Test memory management during preprocessing and path finding."""
    graph = create_test_graph()
    finder = ALTPathFinder(graph, num_landmarks=2, max_memory_mb=1000)

    # Should complete without memory errors
    finder.preprocess()
    result = finder.find_path("A", "E")
    assert result.path
