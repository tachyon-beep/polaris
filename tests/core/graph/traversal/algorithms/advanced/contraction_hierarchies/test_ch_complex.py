"""
Tests for Contraction Hierarchies complex scenarios.
"""

import pytest
import random
from typing import List, Dict, Set, Tuple
from datetime import datetime

from polaris.core.graph import Graph
from polaris.core.graph.traversal.algorithms.advanced.contraction_hierarchies import (
    ContractionHierarchies,
)
from polaris.core.models import Edge, EdgeMetadata
from polaris.core.enums import RelationType

from .conftest import create_edge


def generate_grid_graph(size: int, base_metadata: EdgeMetadata) -> Graph:
    """Generate a grid graph of given size."""
    g = Graph(edges=[])

    # Create grid edges
    for i in range(size):
        for j in range(size):
            node = f"{i},{j}"
            # Connect to right neighbor
            if j < size - 1:
                edge = create_edge(node, f"{i},{j+1}", 1.0, base_metadata)
                g.add_edge(edge)
            # Connect to bottom neighbor
            if i < size - 1:
                edge = create_edge(node, f"{i+1},{j}", 1.0, base_metadata)
                g.add_edge(edge)

    return g


def generate_random_graph(
    num_nodes: int,
    edge_density: float,
    base_metadata: EdgeMetadata,
    min_weight: float = 1.0,
    max_weight: float = 10.0,
) -> Graph:
    """Generate a random graph with given parameters."""
    if not 0 <= edge_density <= 1:
        raise ValueError("Edge density must be between 0 and 1")
    if min_weight > max_weight:
        raise ValueError("min_weight must be less than or equal to max_weight")

    g = Graph(edges=[])
    nodes = [str(i) for i in range(num_nodes)]

    # Create a ring to ensure connectivity
    for i in range(num_nodes):
        from_node = nodes[i]
        to_node = nodes[(i + 1) % num_nodes]
        weight = random.uniform(min_weight, max_weight)
        edge = create_edge(from_node, to_node, weight, base_metadata)
        g.add_edge(edge)
        # Add reverse edge for better connectivity
        edge = create_edge(to_node, from_node, weight, base_metadata)
        g.add_edge(edge)

    # Add additional random edges based on density
    max_additional_edges = int((num_nodes * (num_nodes - 1) - 2 * num_nodes) * edge_density)
    edges_added = 0
    attempts = 0
    max_attempts = max_additional_edges * 10  # Prevent infinite loops

    while edges_added < max_additional_edges and attempts < max_attempts:
        from_node = random.choice(nodes)
        to_node = random.choice([n for n in nodes if n != from_node])
        if not g.get_edge(from_node, to_node):
            weight = random.uniform(min_weight, max_weight)
            edge = create_edge(from_node, to_node, weight, base_metadata)
            g.add_edge(edge)
            edges_added += 1
        attempts += 1

    return g


@pytest.mark.timeout(5)
def test_complex_cyclic_graph(complex_graph: Graph) -> None:
    """Test CH on a complex cyclic graph."""
    ch = ContractionHierarchies(complex_graph)
    ch.preprocess()

    # Test various paths in the cyclic graph
    test_cases = [
        ("A", "E", 6.0, ["A", "B", "D", "E"]),  # Path through shortcut
        ("E", "B", 6.0, ["E", "C", "D", "B"]),  # Path involving a cycle (E->C=3, C->D=1, D->B=2)
        ("D", "A", 8.0, ["D", "E", "A"]),  # Path using backward edge
        ("C", "C", 0.0, ["C"]),  # Self-loop case
    ]

    for start, end, expected_weight, expected_sequence in test_cases:
        result = ch.find_path(start, end)

        if start == end:
            assert result.path == [], f"Path from {start} to {end} should be empty"
            assert result.total_weight == pytest.approx(
                0.0
            ), f"Path from {start} to {end} should have zero weight"
        else:
            assert result.path is not None, f"Path from {start} to {end} should exist"
            actual_sequence = [edge.from_entity for edge in result.path] + [
                result.path[-1].to_entity
            ]
            assert actual_sequence == expected_sequence, (
                f"Path from {start} to {end} sequence mismatch. "
                f"Expected {expected_sequence}, got {actual_sequence}"
            )
            assert result.total_weight == pytest.approx(expected_weight), (
                f"Path from {start} to {end} weight mismatch. "
                f"Expected {expected_weight}, got {result.total_weight}"
            )


@pytest.mark.timeout(10)
def test_grid_graph(base_metadata: EdgeMetadata) -> None:
    """Test CH on a grid graph."""
    grid_size = 5
    grid_graph = generate_grid_graph(grid_size, base_metadata)

    ch = ContractionHierarchies(grid_graph)
    ch.preprocess()

    # Test diagonal path
    start = "0,0"
    end = f"{grid_size-1},{grid_size-1}"
    result = ch.find_path(start, end)

    assert result.path is not None, "Path across grid should exist"
    assert len(result.path) == 2 * (grid_size - 1), "Path should use optimal number of edges"
    assert result.total_weight == pytest.approx(
        2 * (grid_size - 1)
    ), "Path weight should match grid distance"

    # Verify path stays within grid bounds
    for edge in result.path:
        for node in [edge.from_entity, edge.to_entity]:
            i, j = map(int, node.split(","))
            assert 0 <= i < grid_size and 0 <= j < grid_size, "Path should stay within grid bounds"


@pytest.mark.timeout(15)
@pytest.mark.parametrize(
    "num_nodes, edge_density",
    [
        (10, 0.3),  # Sparse graph
        (10, 0.7),  # Dense graph
    ],
)
def test_random_graphs(base_metadata: EdgeMetadata, num_nodes: int, edge_density: float) -> None:
    """Test CH on random graphs with different characteristics."""
    random_graph = generate_random_graph(num_nodes, edge_density, base_metadata)

    ch = ContractionHierarchies(random_graph)
    ch.preprocess()

    # Test multiple random paths
    nodes = list(random_graph.get_nodes())
    for _ in range(5):
        start = random.choice(nodes)
        end = random.choice([n for n in nodes if n != start])

        result = ch.find_path(start, end)
        assert result.path is not None, f"Path from {start} to {end} should exist"

        # Verify path correctness
        assert result.path[0].from_entity == start, "Path should start at start node"
        assert result.path[-1].to_entity == end, "Path should end at end node"

        # Verify path connectivity
        for i in range(len(result.path) - 1):
            assert (
                result.path[i].to_entity == result.path[i + 1].from_entity
            ), "Path should be connected"


@pytest.mark.timeout(10)
def test_stress_preprocessing(base_metadata: EdgeMetadata) -> None:
    """Test CH preprocessing under stress conditions."""
    # Create a dense graph with many potential shortcuts
    num_nodes = 15
    edge_density = 0.8
    stress_graph = generate_random_graph(num_nodes, edge_density, base_metadata)

    ch = ContractionHierarchies(stress_graph)
    ch.preprocess()

    # Verify preprocessing results
    assert len(ch.state.node_level) == num_nodes, "All nodes should have levels assigned"
    assert len(ch.state.contracted_neighbors) > 0, "Contracted neighbors should be tracked"

    # Verify shortcut properties
    for (u, v), shortcut in ch.state.shortcuts.items():
        assert shortcut.via_node is not None, f"Shortcut {u}->{v} missing via_node"
        assert shortcut.lower_edge is not None, f"Shortcut {u}->{v} missing lower_edge"
        assert shortcut.upper_edge is not None, f"Shortcut {u}->{v} missing upper_edge"


@pytest.mark.timeout(10)
def test_performance_scaling(base_metadata: EdgeMetadata) -> None:
    """Test CH performance scaling with graph size."""
    sizes = [5, 10, 15]  # Different graph sizes
    density = 0.4  # Moderate density

    for size in sizes:
        graph = generate_random_graph(size, density, base_metadata)

        ch = ContractionHierarchies(graph)
        ch.preprocess()

        # Test path finding between random pairs of nodes
        nodes = list(graph.get_nodes())
        for _ in range(3):
            start = random.choice(nodes)
            end = random.choice([n for n in nodes if n != start])

            result = ch.find_path(start, end)
            assert result.path is not None, f"Path finding failed for graph size {size}"


@pytest.mark.timeout(10)
def test_incremental_updates(complex_graph: Graph, base_metadata: EdgeMetadata) -> None:
    """Test CH behavior with incremental graph updates."""
    ch = ContractionHierarchies(complex_graph)
    ch.preprocess()

    # Record initial state
    initial_shortcuts = len(ch.state.shortcuts)

    # Make incremental changes
    updates = [
        ("A", "F", 1.0),  # Add new node
        ("F", "C", 2.0),  # Connect to existing node
        ("F", "E", 1.5),  # Create alternative path
    ]

    for from_node, to_node, weight in updates:
        # Add new edge
        edge = create_edge(from_node, to_node, weight, base_metadata)
        complex_graph.add_edge(edge)

        # Reprocess and verify
        ch.preprocess()

        # Test path finding through new edges
        result = ch.find_path(from_node, to_node)
        assert result.path is not None, f"Path {from_node}->{to_node} should exist"
        assert result.total_weight == pytest.approx(weight), (
            f"Path {from_node}->{to_node} weight mismatch. "
            f"Expected {weight}, got {result.total_weight}"
        )
