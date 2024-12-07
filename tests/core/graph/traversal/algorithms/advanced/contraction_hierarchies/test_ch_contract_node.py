"""
Unit tests for the contract_node method of ContractionPreprocessor.
Tests focus on validating the behavior of single node contractions.
"""

from datetime import datetime
from math import isclose  # Moved standard import before third-party

import pytest

from polaris.core.graph.traversal.algorithms.advanced.contraction_hierarchies.models import (
    ContractionState,
)
from polaris.core.graph.traversal.algorithms.advanced.contraction_hierarchies.preprocessor import (
    ContractionPreprocessor,
)
from polaris.core.graph.traversal.algorithms.advanced.contraction_hierarchies.storage import (
    ContractionStorage,
)
from polaris.core.graph.traversal.algorithms.advanced.contraction_hierarchies.utils import (
    Graph,
    Edge,
    EdgeMetadata,
    RelationType,
)


class MockGraph(Graph):
    """MockGraph simulates a graph structure for testing."""

    def __init__(self):
        super().__init__([])

    def add_edge_simple(
        self,
        from_node: str,
        to_node: str,
        weight: float = 1.0,
        confidence: float = 1.0,
        source: str = "test",
    ) -> None:
        """Create and add an edge with simplified parameters."""
        now = datetime.now()
        edge = Edge(
            from_entity=from_node,
            to_entity=to_node,
            relation_type=RelationType.CONNECTS_TO,
            impact_score=1.0,
            metadata=EdgeMetadata(
                created_at=now,
                last_modified=now,
                confidence=confidence,
                source=source,
                weight=weight,
            ),
        )
        self.add_edge(edge)


@pytest.fixture(name="test_graph")
def fixture_graph():
    """Provides a fresh MockGraph instance."""
    return MockGraph()


@pytest.fixture(name="test_preprocessor")
def fixture_preprocessor(test_graph):
    """Provides a ContractionPreprocessor with clean state."""
    state = ContractionState()
    storage = ContractionStorage(state)
    return ContractionPreprocessor(test_graph, storage)


def test_single_node_no_edges(test_preprocessor):
    """Verify contracting an isolated node."""
    result = test_preprocessor.contract_node("A")
    state = test_preprocessor.storage.get_state()

    assert not result, "Isolated node should create no shortcuts"
    assert "A" in state.contracted_neighbors
    assert not state.contracted_neighbors["A"], "Isolated node should have no neighbors"


def test_single_edge(test_preprocessor, test_graph):
    """Verify contracting a node with a single edge."""
    test_graph.add_edge_simple("A", "B", weight=1.0)

    result = test_preprocessor.contract_node("A")
    state = test_preprocessor.storage.get_state()

    assert not result, "Single edge should create no shortcuts"
    assert state.contracted_neighbors["A"] == {"B"}, "Should record B as neighbor"


def test_basic_shortcut_creation(test_preprocessor, test_graph):
    """Verify basic shortcut creation for a three-node path."""
    test_graph.add_edge_simple("X", "Y", weight=2.0)
    test_graph.add_edge_simple("Y", "Z", weight=3.0)

    shortcuts = test_preprocessor.contract_node("Y")
    state = test_preprocessor.storage.get_state()

    assert len(shortcuts) == 1, "Should create one shortcut"
    assert shortcuts[0] == ("X", "Z"), "Shortcut should connect X to Z"
    assert isclose(
        state.shortcuts[("X", "Z")].edge.metadata.weight, 5.0, rel_tol=1e-9
    ), "Weights should sum"


def test_zero_weight_edges(test_preprocessor, test_graph):
    """Verify handling of zero-weight edges."""
    test_graph.add_edge_simple("A", "B", weight=0.0)
    test_graph.add_edge_simple("B", "C", weight=0.0)

    shortcuts = test_preprocessor.contract_node("B")
    state = test_preprocessor.storage.get_state()

    assert len(shortcuts) == 1, "Should create shortcut even with zero weights"
    assert isclose(
        state.shortcuts[("A", "C")].edge.metadata.weight, 0.0, rel_tol=1e-9
    ), "Zero weights should sum to zero"


def test_large_weight_edges(test_preprocessor, test_graph):
    """Verify handling of very large weights."""
    large_weight = 1e15
    test_graph.add_edge_simple("A", "B", weight=large_weight)
    test_graph.add_edge_simple("B", "C", weight=large_weight)

    shortcuts = test_preprocessor.contract_node("B")
    state = test_preprocessor.storage.get_state()

    assert len(shortcuts) == 1, "Should handle large weights"
    assert isclose(
        state.shortcuts[("A", "C")].edge.metadata.weight, 2e15, rel_tol=1e-9
    ), "Should sum large weights correctly"


def test_floating_point_precision(test_preprocessor, test_graph):
    """Verify handling of floating point weights."""
    test_graph.add_edge_simple("A", "B", weight=0.1)
    test_graph.add_edge_simple("B", "C", weight=0.2)

    shortcuts = test_preprocessor.contract_node("B")
    state = test_preprocessor.storage.get_state()

    assert len(shortcuts) == 1
    assert isclose(
        state.shortcuts[("A", "C")].edge.metadata.weight, 0.3, rel_tol=1e-9
    ), "Should handle floating point precision"


def test_invalid_node_contraction(test_preprocessor):
    """Verify error handling for invalid node inputs."""
    with pytest.raises(ValueError, match="Node 'nonexistent' not found"):
        test_preprocessor.contract_node("nonexistent")


def test_already_contracted_node(test_preprocessor, test_graph):
    """Verify contracting an already contracted node."""
    test_graph.add_edge_simple("A", "B")
    test_preprocessor.contract_node("A")

    with pytest.raises(ValueError, match="Node 'A' already contracted"):
        test_preprocessor.contract_node("A")


def test_self_loop_only(test_preprocessor, test_graph):
    """Verify handling of a node with only a self-loop."""
    test_graph.add_edge_simple("A", "A", weight=1.0)

    result = test_preprocessor.contract_node("A")
    state = test_preprocessor.storage.get_state()

    assert not result, "Self-loop should not create shortcuts"
    assert state.contracted_neighbors["A"] == {"A"}, "Self-loop should be recorded in neighbors"


def test_multiple_parallel_paths(test_preprocessor, test_graph):
    """Verify handling of multiple parallel paths through node."""
    # Direct path
    test_graph.add_edge_simple("X", "Z", weight=10.0)

    # Path through Y with lower total weight
    test_graph.add_edge_simple("X", "Y", weight=2.0)
    test_graph.add_edge_simple("Y", "Z", weight=3.0)

    # Another path through Y with higher weight
    test_graph.add_edge_simple("X", "Y", weight=4.0)
    test_graph.add_edge_simple("Y", "Z", weight=4.0)

    shortcuts = test_preprocessor.contract_node("Y")
    state = test_preprocessor.storage.get_state()

    assert len(shortcuts) == 1, "Should create one shortcut using shortest path"
    assert isclose(
        state.shortcuts[("X", "Z")].edge.metadata.weight, 5.0, rel_tol=1e-9
    ), "Should use shortest path weight"


def test_confidence_handling(test_preprocessor, test_graph):
    """Verify confidence values in shortcuts."""
    test_graph.add_edge_simple("A", "B", weight=1.0, confidence=0.8)
    test_graph.add_edge_simple("B", "C", weight=2.0, confidence=0.5)

    _ = test_preprocessor.contract_node("B")
    state = test_preprocessor.storage.get_state()

    shortcut = state.shortcuts[("A", "C")]
    assert isclose(
        shortcut.edge.metadata.confidence, 0.4, rel_tol=1e-9
    ), "Confidence should multiply"


def test_metadata_preservation(test_preprocessor, test_graph):
    """Verify metadata handling in shortcuts."""
    source = "test_source"
    test_graph.add_edge_simple("A", "B", weight=1.0, source=source)
    test_graph.add_edge_simple("B", "C", weight=2.0, source=source)

    _ = test_preprocessor.contract_node("B")
    state = test_preprocessor.storage.get_state()

    shortcut = state.shortcuts[("A", "C")]
    assert shortcut.edge.metadata.source == source, "Source should be preserved"
    assert isinstance(shortcut.edge.metadata.created_at, datetime), "Should have valid timestamp"


def test_equal_weight_alternative_paths(test_preprocessor, test_graph):
    """Verify handling of equal-weight alternative paths."""
    # Two equal-weight paths from A to C
    test_graph.add_edge_simple("A", "B", weight=2.0)
    test_graph.add_edge_simple("B", "C", weight=3.0)
    test_graph.add_edge_simple("A", "C", weight=5.0)

    shortcuts = test_preprocessor.contract_node("B")
    state = test_preprocessor.storage.get_state()

    assert len(shortcuts) <= 1, "Should not create redundant shortcuts for equal weights"
    if shortcuts:
        assert isclose(state.shortcuts[shortcuts[0]].edge.metadata.weight, 5.0, rel_tol=1e-9)


def test_state_after_sequential_contractions(test_preprocessor, test_graph):
    """Verify state consistency after contracting multiple nodes sequentially."""
    test_graph.add_edge_simple("A", "B", weight=1.0)
    test_graph.add_edge_simple("B", "C", weight=2.0)
    test_graph.add_edge_simple("C", "D", weight=3.0)

    test_preprocessor.contract_node("B")
    test_preprocessor.contract_node("C")
    state = test_preprocessor.storage.get_state()

    assert "B" in state.contracted_neighbors
    assert "C" in state.contracted_neighbors
    assert ("A", "D") in state.shortcuts, "Final shortcut should exist"
    assert isclose(
        state.shortcuts[("A", "D")].edge.metadata.weight, 6.0, rel_tol=1e-9
    ), "Weights should sum correctly"


def test_contraction_with_cycle(test_preprocessor, test_graph):
    """Verify correct behavior when contracting nodes in a cyclic graph."""
    test_graph.add_edge_simple("A", "B", weight=1.0)
    test_graph.add_edge_simple("B", "C", weight=2.0)
    test_graph.add_edge_simple("C", "A", weight=3.0)

    shortcuts = test_preprocessor.contract_node("B")
    state = test_preprocessor.storage.get_state()

    assert len(shortcuts) == 1, "Should create one shortcut to maintain cycle"
    assert ("A", "C") in state.shortcuts, "Shortcut should connect A to C"
    assert isclose(
        state.shortcuts[("A", "C")].edge.metadata.weight, 3.0, rel_tol=1e-9
    ), "Should use shortest path through cycle"


def test_disconnected_components(test_preprocessor, test_graph):
    """Verify contraction in a graph with disconnected components."""
    test_graph.add_edge_simple("A", "B", weight=1.0)
    test_graph.add_edge_simple("X", "Y", weight=2.0)

    shortcuts_a = test_preprocessor.contract_node("A")
    shortcuts_x = test_preprocessor.contract_node("X")

    assert not shortcuts_a, "No shortcuts for disconnected node A"
    assert not shortcuts_x, "No shortcuts for disconnected node X"
