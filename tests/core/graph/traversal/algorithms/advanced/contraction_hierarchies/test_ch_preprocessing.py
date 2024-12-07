"""
Tests for Contraction Hierarchies preprocessing functionality.
"""

import random
import datetime
from dataclasses import replace

import pytest

from polaris.core.exceptions import GraphOperationError
from polaris.core.graph import Graph
from polaris.core.graph.traversal.algorithms.advanced.contraction_hierarchies import (
    ContractionHierarchies,
)
from polaris.core.models import EdgeMetadata
from .conftest import create_edge


# Global `base_metadata` to be reused across tests
base_metadata = EdgeMetadata(
    created_at=datetime.datetime.now(),
    last_modified=datetime.datetime.now(),
    confidence=1.0,
    source="test",
    bidirectional=False,
    temporal=False,
    weight=1.0,
    custom_attributes={},
)


@pytest.mark.timeout(5)
def test_preprocessing_required_for_pathfinding(simple_graph: Graph) -> None:
    """Test that an error is raised if pathfinding is attempted before preprocessing."""
    ch = ContractionHierarchies(simple_graph)

    with pytest.raises(
        GraphOperationError, match="Graph must be preprocessed before accessing state"
    ):
        ch.find_path("A", "C")


@pytest.mark.timeout(5)
def test_preprocessing_empty_graph() -> None:
    """Test preprocessing on an empty graph."""
    empty_graph = Graph(edges=[])
    ch = ContractionHierarchies(empty_graph)
    ch.preprocess()

    # Verify that all states are empty
    assert len(ch.state.shortcuts) == 0, "Empty graph should have no shortcuts"
    assert len(ch.state.node_level) == 0, "Empty graph should have no node levels"
    assert (
        len(ch.state.contracted_neighbors) == 0
    ), "Empty graph should have no contracted neighbors"


@pytest.mark.timeout(5)
def test_preprocessing_single_node() -> None:
    """Test preprocessing on a graph with a single node."""
    g = Graph(edges=[])
    edge = create_edge("A", "A", 1.0, base_metadata)  # Add a self-loop
    g.add_edge(edge)

    ch = ContractionHierarchies(g)
    ch.preprocess()

    # Verify the single node's state
    assert len(ch.state.shortcuts) == 0, "Single node graph should have no shortcuts"
    assert ch.storage.get_node_level("A") == 0, "Single node graph should have one node level"
    assert (
        len(ch.state.contracted_neighbors) == 1
    ), "Single node graph should have one contracted neighbor entry"

    assert "A" in ch.state.node_level, "Node A should have a level assigned"
    assert ch.state.node_level["A"] == 0, "Node A should be at level 0"


@pytest.mark.timeout(5)
def test_preprocessing_with_negative_weights(simple_graph: Graph) -> None:
    """Test that preprocessing raises an error for graphs with negative edge weights."""
    edge = create_edge("A", "D", -1.0, base_metadata)  # Negative weight edge
    simple_graph.add_edge(edge)

    ch = ContractionHierarchies(simple_graph)

    with pytest.raises(GraphOperationError, match="Negative edge weights are not supported"):
        ch.preprocess()


@pytest.mark.timeout(5)
def test_preprocessing_with_self_loops(simple_graph: Graph) -> None:
    """Test preprocessing handles graphs with self-loops."""
    # Add self-loops to all nodes
    for node in simple_graph.get_nodes():
        edge = create_edge(node, node, 1.0, base_metadata)
        simple_graph.add_edge(edge)

    ch = ContractionHierarchies(simple_graph)
    ch.preprocess()

    # Verify no self-loops in shortcuts
    shortcuts = ch.state.shortcuts
    for u, v in shortcuts.keys():
        assert u != v, "Shortcuts should not include self-loops"


@pytest.mark.timeout(5)
def test_preprocessing_with_parallel_edges(simple_graph: Graph) -> None:
    """Test preprocessing correctly handles parallel edges."""
    # Add parallel edges with different weights
    parallel_edges = [
        ("A", "B", 2.0),
        ("B", "C", 1.0),
    ]

    for from_node, to_node, weight in parallel_edges:
        edge = create_edge(from_node, to_node, weight, base_metadata)
        simple_graph.add_edge(edge)

    ch = ContractionHierarchies(simple_graph)
    ch.preprocess()

    # Verify that shortcuts use the shortest parallel edge
    result = ch.find_path("A", "C")
    assert result.total_weight == pytest.approx(2.0), "Path should use the shortest parallel edges"


@pytest.mark.timeout(5)
def test_preprocessing_custom_metadata() -> None:
    """Test preprocessing on edges with custom metadata."""
    custom_metadata = replace(base_metadata, custom_attributes={"custom_field": "test_value"})
    edge = create_edge("A", "B", 1.0, custom_metadata)
    graph = Graph(edges=[edge])

    ch = ContractionHierarchies(graph)
    ch.preprocess()

    shortcut = ch.state.shortcuts.get(("A", "B"))
    assert shortcut is not None, "Expected shortcut does not exist"
    assert (
        shortcut.edge.metadata.custom_attributes["custom_field"] == "test_value"
    ), "Custom metadata should persist in shortcuts"


@pytest.mark.timeout(10)
def test_preprocessing_dynamic_edge_update(simple_graph: Graph) -> None:
    """Test that preprocessing dynamically updates after edge changes."""
    ch = ContractionHierarchies(simple_graph)
    ch.preprocess()

    # Add a new edge that affects existing shortcuts
    new_edge = create_edge("C", "D", 1.0, base_metadata)
    simple_graph.add_edge(new_edge)

    # Re-preprocess for the new edge
    ch.preprocessor.preprocess(new_edge=("C", "D"))

    # Verify that new shortcuts are created
    shortcuts = ch.state.shortcuts
    assert ("A", "D") in shortcuts, "Expected shortcut A->D to be created after adding C->D"

    # Verify state changes dynamically
    assert "C" in ch.state.contracted_neighbors
    assert "D" in ch.state.contracted_neighbors

    # Verify that the path uses the correct shortcuts
    result = ch.find_path("A", "D")
    assert result.total_weight == pytest.approx(
        4.0
    ), "Path A->B->C->D should have a total weight of 4.0"
    assert result.path == ["A", "B", "C", "D"], "Path should follow the correct sequence"


@pytest.mark.timeout(5)
def test_preprocessing_disconnected_components(simple_graph: Graph) -> None:
    """Test preprocessing handles disconnected graph components."""
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

    # Verify all nodes have levels
    assert len(ch.state.node_level) == len(
        simple_graph.get_nodes()
    ), "All nodes should have levels assigned, including disconnected components"

    # Verify that shortcuts exist only within components
    shortcuts = ch.state.shortcuts
    component_1_shortcuts = [(u, v) for u, v in shortcuts if u in ["A", "B", "C"]]
    component_2_shortcuts = [(u, v) for u, v in shortcuts if u in ["X", "Y", "Z"]]

    assert len(component_1_shortcuts) > 0, "Shortcuts should exist within Component 1"
    assert len(component_2_shortcuts) > 0, "Shortcuts should exist within Component 2"

    for u, v in shortcuts.keys():
        assert not (
            (u in ["A", "B", "C"] and v in ["X", "Y", "Z"])
            or (u in ["X", "Y", "Z"] and v in ["A", "B", "C"])
        ), "Shortcuts should not connect disconnected components"


@pytest.mark.timeout(5)
def test_preprocessing_large_dense_graph() -> None:
    """Test preprocessing on a large dense graph."""

    def create_dense_graph(num_nodes: int, density: float) -> Graph:
        edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if random.random() <= density:
                    edges.append(
                        create_edge(f"N{i}", f"N{j}", random.uniform(1.0, 10.0), base_metadata)
                    )
        return Graph(edges=edges)

    dense_graph = create_dense_graph(num_nodes=1000, density=0.8)
    ch = ContractionHierarchies(dense_graph)
    ch.preprocess()

    assert len(ch.state.shortcuts) > 0, "Dense graph should have many shortcuts"


@pytest.mark.timeout(5)
def test_preprocessing_cyclic_graph() -> None:
    """Test preprocessing on a cyclic graph."""
    cyclic_edges = [
        ("A", "B", 1.0),
        ("B", "C", 1.0),
        ("C", "A", 1.0),
    ]
    graph = Graph(edges=[create_edge(*edge, base_metadata) for edge in cyclic_edges])
    ch = ContractionHierarchies(graph)
    ch.preprocess()

    assert len(ch.state.shortcuts) > 0, "Cyclic graph should generate shortcuts"
    assert all(
        u != v for u, v in ch.state.shortcuts.keys()
    ), "No self-loops should exist in shortcuts"
