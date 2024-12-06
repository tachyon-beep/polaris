"""
Tests for Contraction Hierarchies preprocessing functionality.
"""

import pytest
from datetime import datetime

from polaris.core.exceptions import GraphOperationError
from polaris.core.graph import Graph
from polaris.core.graph.traversal.algorithms.advanced.contraction_hierarchies import (
    ContractionHierarchies,
)
from polaris.core.models import Edge, EdgeMetadata
from polaris.core.enums import RelationType

from .conftest import create_edge


@pytest.mark.timeout(5)
def test_preprocessing(simple_graph: Graph) -> None:
    """Test preprocessing creates valid hierarchy."""
    ch = ContractionHierarchies(simple_graph)
    ch.preprocess()

    # Verify state exists
    assert ch.state is not None

    # Check node levels assigned
    assert len(ch.state.node_level) == len(simple_graph.get_nodes())
    for node in simple_graph.get_nodes():
        assert node in ch.state.node_level, f"Node {node} missing from node_level"
        assert ch.state.node_level[node] >= 0, f"Invalid level for node {node}"

    # Verify shortcuts created
    shortcuts = ch.state.shortcuts
    assert len(shortcuts) > 0

    # Check contracted neighbors tracked
    assert len(ch.state.contracted_neighbors) > 0
    for node in simple_graph.get_nodes():
        assert (
            node in ch.state.contracted_neighbors
        ), f"Node {node} missing from contracted_neighbors"


@pytest.mark.timeout(5)
def test_preprocessing_required(simple_graph: Graph) -> None:
    """Test error raised if paths found before preprocessing."""
    ch = ContractionHierarchies(simple_graph)

    with pytest.raises(
        GraphOperationError, match="Graph must be preprocessed before finding paths"
    ):
        ch.find_path("A", "C")


@pytest.mark.timeout(10)
def test_dynamic_graph_changes(simple_graph: Graph, base_metadata: EdgeMetadata) -> None:
    """Test CH handling of dynamic graph changes."""
    ch = ContractionHierarchies(simple_graph)
    ch.preprocess()

    # Record initial state
    initial_shortcuts = len(ch.state.shortcuts)

    # Add a new edge that affects existing shortcuts
    new_edge = create_edge("C", "D", 1.0, base_metadata)
    simple_graph.add_edge(new_edge)
    ch.preprocess()  # Re-preprocess to incorporate changes

    # Verify that the new shortcut is created
    shortcuts = ch.state.shortcuts

    assert ("A", "D") in shortcuts, "Expected shortcut A->D to be created after adding C->D"
    result = ch.find_path("A", "D")
    assert result.total_weight == pytest.approx(4.0), "Path A->B->C->D should have total weight 4.0"

    # Verify the number of shortcuts has increased
    assert (
        len(shortcuts) > initial_shortcuts
    ), "Number of shortcuts should increase after adding edge"


@pytest.mark.timeout(5)
def test_preprocessing_empty_graph() -> None:
    """Test preprocessing on an empty graph."""
    empty_graph = Graph(edges=[])
    ch = ContractionHierarchies(empty_graph)
    ch.preprocess()

    # Verify empty state
    assert len(ch.state.shortcuts) == 0, "Empty graph should have no shortcuts"
    assert len(ch.state.node_level) == 0, "Empty graph should have no node levels"
    assert (
        len(ch.state.contracted_neighbors) == 0
    ), "Empty graph should have no contracted neighbors"


@pytest.mark.timeout(5)
def test_preprocessing_single_node(base_metadata: EdgeMetadata) -> None:
    """Test preprocessing on a graph with a single node."""
    g = Graph(edges=[])
    # Add a self-loop edge
    edge = create_edge("A", "A", 1.0, base_metadata)
    g.add_edge(edge)

    ch = ContractionHierarchies(g)
    ch.preprocess()

    # Verify state for single node
    assert len(ch.state.shortcuts) == 0, "Single node graph should have no shortcuts"
    assert len(ch.state.node_level) == 1, "Single node graph should have one node level"
    assert (
        len(ch.state.contracted_neighbors) == 1
    ), "Single node graph should have one contracted neighbor entry"
    assert "A" in ch.state.node_level, "Node A should have a level assigned"
    assert ch.state.node_level["A"] == 0, "Node A should be at level 0"


@pytest.mark.timeout(5)
def test_preprocessing_disconnected_components(
    simple_graph: Graph, base_metadata: EdgeMetadata
) -> None:
    """Test preprocessing with disconnected components."""
    # Add a disconnected component
    disconnected_edges = [
        ("X", "Y", 1.0),
        ("Y", "Z", 2.0),
    ]

    for from_node, to_node, weight in disconnected_edges:
        edge = create_edge(from_node, to_node, weight, base_metadata)
        simple_graph.add_edge(edge)

    ch = ContractionHierarchies(simple_graph)
    ch.preprocess()

    # Verify preprocessing handles disconnected components
    assert len(ch.state.node_level) == 6, "Should have levels for all nodes in both components"
    assert (
        len(ch.state.contracted_neighbors) > 0
    ), "Should have contracted neighbors in both components"

    # Verify shortcuts are created within components but not between them
    shortcuts = ch.state.shortcuts
    for u, v in shortcuts.keys():
        # Check that shortcuts don't connect across components
        assert not (
            (u in ["A", "B", "C"] and v in ["X", "Y", "Z"])
            or (u in ["X", "Y", "Z"] and v in ["A", "B", "C"])
        ), "Shortcuts should not connect disconnected components"


@pytest.mark.timeout(5)
def test_preprocessing_with_negative_weights(
    simple_graph: Graph, base_metadata: EdgeMetadata
) -> None:
    """Test preprocessing behavior with negative edge weights."""
    # Add an edge with negative weight
    edge = create_edge("A", "D", -1.0, base_metadata)
    simple_graph.add_edge(edge)

    ch = ContractionHierarchies(simple_graph)

    # Should raise error for negative weights
    with pytest.raises(GraphOperationError, match="Negative edge weights are not supported"):
        ch.preprocess()


@pytest.mark.timeout(5)
def test_preprocessing_idempotency(simple_graph: Graph) -> None:
    """Test that preprocessing multiple times produces consistent results."""
    ch = ContractionHierarchies(simple_graph)

    # First preprocessing
    ch.preprocess()
    first_shortcuts = set(ch.state.shortcuts.keys())
    first_levels = dict(ch.state.node_level)

    # Second preprocessing
    ch.preprocess()
    second_shortcuts = set(ch.state.shortcuts.keys())
    second_levels = dict(ch.state.node_level)

    # Results should be consistent
    assert (
        first_shortcuts == second_shortcuts
    ), "Shortcuts should be consistent across preprocessings"
    assert first_levels == second_levels, "Node levels should be consistent across preprocessings"


@pytest.mark.timeout(5)
def test_preprocessing_with_self_loops(simple_graph: Graph, base_metadata: EdgeMetadata) -> None:
    """Test preprocessing behavior with self-loop edges."""
    # Add self-loops to all nodes
    for node in simple_graph.get_nodes():
        edge = create_edge(node, node, 1.0, base_metadata)
        simple_graph.add_edge(edge)

    ch = ContractionHierarchies(simple_graph)
    ch.preprocess()

    # Verify preprocessing handles self-loops
    shortcuts = ch.state.shortcuts
    for u, v in shortcuts.keys():
        assert u != v, "Shortcuts should not include self-loops"

    # Verify node levels are still assigned correctly
    for node in simple_graph.get_nodes():
        assert node in ch.state.node_level, f"Node {node} should have a level assigned"
        assert ch.state.node_level[node] >= 0, f"Node {node} should have non-negative level"


@pytest.mark.timeout(5)
def test_preprocessing_with_parallel_edges(
    simple_graph: Graph, base_metadata: EdgeMetadata
) -> None:
    """Test preprocessing behavior with parallel edges between nodes."""
    # Add parallel edges with different weights
    parallel_edges = [
        ("A", "B", 2.0),  # Parallel to existing A->B edge
        ("B", "C", 1.0),  # Parallel to existing B->C edge
    ]

    for from_node, to_node, weight in parallel_edges:
        edge = create_edge(from_node, to_node, weight, base_metadata)
        simple_graph.add_edge(edge)

    ch = ContractionHierarchies(simple_graph)
    ch.preprocess()

    # Verify preprocessing uses shortest edges for shortcuts
    result = ch.find_path("A", "C")
    assert result.total_weight == pytest.approx(2.0), "Path should use shortest parallel edges"
