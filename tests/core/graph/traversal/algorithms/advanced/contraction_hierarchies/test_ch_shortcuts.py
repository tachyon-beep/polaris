"""
Tests for Contraction Hierarchies shortcut creation and validation.
"""

import pytest
from typing import Dict, Tuple, List

from polaris.core.graph import Graph
from polaris.core.graph.traversal.algorithms.advanced.contraction_hierarchies import (
    ContractionHierarchies,
)
from polaris.core.models import Edge, EdgeMetadata
from polaris.core.enums import RelationType

from .conftest import create_edge


@pytest.mark.timeout(5)
@pytest.mark.parametrize(
    "modify_weights, expected_shortcut_exists, expected_shortcut_weight, expected_via_node, description",
    [
        (
            {"A->C": 3.0},  # A->C exists with weight equal to A->B->C
            True,
            3.0,
            "B",
            "Shortcut A->C should exist with weight 3.0",
        ),
        (
            {"A->C": 10.0},  # A->C is too costly, shortcut should preserve A->B->C
            True,
            3.0,
            "B",
            "Shortcut A->C should exist with weight 3.0 despite original A->C weight being 10.0",
        ),
        (
            {"A->C": 1.5},  # A->C is cheaper than A->B->C, shortcut A->C should not be necessary
            False,
            None,
            None,
            "Shortcut A->C should not exist as direct path A->C is cheaper",
        ),
    ],
)
def test_shortcut_creation_and_necessity(
    simple_graph: Graph,
    base_metadata: EdgeMetadata,
    modify_weights: Dict[str, float],
    expected_shortcut_exists: bool,
    expected_shortcut_weight: float,
    expected_via_node: str,
    description: str,
) -> None:
    """Test shortcut creation and necessity based on edge weight modifications."""
    # Modify the graph edges as per the test case
    for edge_str, new_weight in modify_weights.items():
        from_node, to_node = edge_str.split("->")
        edge = simple_graph.get_edge(from_node, to_node)
        if edge:
            edge.metadata.weight = new_weight
        else:
            # If the edge doesn't exist, add it
            new_edge = create_edge(from_node, to_node, new_weight, base_metadata)
            simple_graph.add_edge(new_edge)

    ch = ContractionHierarchies(simple_graph)
    ch.preprocess()

    shortcuts = ch.state.shortcuts

    if expected_shortcut_exists:
        # Shortcut A->C should exist with specified weight
        assert ("A", "C") in shortcuts, f"{description}: Expected shortcut A->C to exist"
        shortcut = shortcuts[("A", "C")]
        assert shortcut.edge.metadata.weight == pytest.approx(expected_shortcut_weight), (
            f"{description}: Shortcut weight mismatch. "
            f"Expected {expected_shortcut_weight}, got {shortcut.edge.metadata.weight}"
        )

        # Validate shortcut properties
        assert (
            shortcut.via_node == expected_via_node
        ), f"{description}: Shortcut A->C should be via node {expected_via_node}"
        assert (
            shortcut.lower_edge.from_entity == "A" and shortcut.lower_edge.to_entity == "B"
        ), f"{description}: Shortcut lower edge should be A->B"
        assert (
            shortcut.upper_edge.from_entity == "B" and shortcut.upper_edge.to_entity == "C"
        ), f"{description}: Shortcut upper edge should be B->C"
    else:
        # Shortcut A->C should not exist
        assert ("A", "C") not in shortcuts, f"{description}: Unexpected shortcut A->C exists"


@pytest.mark.timeout(5)
def test_overlapping_shortcuts(complex_graph: Graph) -> None:
    """Test CH with overlapping shortcuts."""
    ch = ContractionHierarchies(complex_graph)
    ch.preprocess()

    # Find path from A to E
    result = ch.find_path("A", "E")

    # Expected path: A->B->C->D->E
    assert result.path is not None, "Path A->B->C->D->E should exist"
    assert len(result.path) == 4, "Path A->B->C->D->E should have 4 edges"
    assert result.total_weight == pytest.approx(7.0), (
        "Path A->B->C->D->E should have total weight 7.0, " f"got {result.total_weight}"
    )

    # Verify the actual path sequence
    expected_sequence = ["A", "B", "C", "D", "E"]
    actual_sequence = [edge.from_entity for edge in result.path] + [result.path[-1].to_entity]
    assert (
        actual_sequence == expected_sequence
    ), "Path sequence mismatch for A->E with overlapping shortcuts"


@pytest.mark.timeout(5)
@pytest.mark.parametrize(
    "additional_edges, expected_shortcuts, description",
    [
        (
            [("C", "D", 1.0), ("D", "E", 1.0)],
            [("A", "D"), ("B", "E"), ("C", "E")],
            "Linear chain shortcuts",
        ),
        (
            [("C", "D", 1.0), ("D", "B", 1.0)],
            [("A", "D"), ("C", "B")],
            "Cycle shortcuts",
        ),
        (
            [("A", "D", 1.0), ("D", "C", 1.0)],
            [("A", "C"), ("B", "D")],
            "Alternative path shortcuts",
        ),
    ],
)
def test_shortcut_patterns(
    simple_graph: Graph,
    base_metadata: EdgeMetadata,
    additional_edges: List[Tuple[str, str, float]],
    expected_shortcuts: List[Tuple[str, str]],
    description: str,
) -> None:
    """Test different patterns of shortcut creation."""
    # Add additional edges
    for from_node, to_node, weight in additional_edges:
        edge = create_edge(from_node, to_node, weight, base_metadata)
        simple_graph.add_edge(edge)

    ch = ContractionHierarchies(simple_graph)
    ch.preprocess()

    shortcuts = ch.state.shortcuts

    # Verify expected shortcuts exist
    for from_node, to_node in expected_shortcuts:
        assert (
            from_node,
            to_node,
        ) in shortcuts, f"{description}: Expected shortcut {from_node}->{to_node} missing"

    # Verify shortcut properties
    for from_node, to_node in shortcuts:
        shortcut = shortcuts[(from_node, to_node)]
        assert shortcut.via_node is not None, f"{description}: Shortcut missing via_node"
        assert shortcut.lower_edge is not None, f"{description}: Shortcut missing lower_edge"
        assert shortcut.upper_edge is not None, f"{description}: Shortcut missing upper_edge"
        assert shortcut.edge.metadata.weight >= 0, f"{description}: Shortcut has negative weight"


@pytest.mark.timeout(5)
def test_shortcut_weight_consistency(complex_graph: Graph) -> None:
    """Test consistency of shortcut weights with original paths."""
    ch = ContractionHierarchies(complex_graph)
    ch.preprocess()

    shortcuts = ch.state.shortcuts

    for (from_node, to_node), shortcut in shortcuts.items():
        # Calculate the weight of the path through the via node
        path_weight = shortcut.lower_edge.metadata.weight + shortcut.upper_edge.metadata.weight

        # Verify shortcut weight matches path weight
        assert shortcut.edge.metadata.weight == pytest.approx(path_weight), (
            f"Shortcut {from_node}->{to_node} weight {shortcut.edge.metadata.weight} "
            f"doesn't match path weight {path_weight}"
        )

        # Verify shortcut weight is less than or equal to any direct edge
        direct_edge = complex_graph.get_edge(from_node, to_node)
        if direct_edge:
            assert shortcut.edge.metadata.weight <= direct_edge.metadata.weight, (
                f"Shortcut {from_node}->{to_node} weight {shortcut.edge.metadata.weight} "
                f"is greater than direct edge weight {direct_edge.metadata.weight}"
            )


@pytest.mark.timeout(5)
def test_shortcut_transitivity(complex_graph: Graph) -> None:
    """Test transitivity of shortcuts (if shortcuts can be shortcuts of shortcuts)."""
    ch = ContractionHierarchies(complex_graph)
    ch.preprocess()

    shortcuts = ch.state.shortcuts

    # Check if any shortcut's endpoints are also connected by another shortcut
    for (from_node, to_node), shortcut in shortcuts.items():
        via_node = shortcut.via_node

        # Look for shortcuts that could be combined
        if (from_node, via_node) in shortcuts and (via_node, to_node) in shortcuts:
            # Calculate weights
            direct_weight = shortcut.edge.metadata.weight
            combined_weight = (
                shortcuts[(from_node, via_node)].edge.metadata.weight
                + shortcuts[(via_node, to_node)].edge.metadata.weight
            )

            # The direct shortcut should be at least as good as the combined shortcuts
            assert direct_weight <= pytest.approx(combined_weight), (
                f"Shortcut {from_node}->{to_node} weight {direct_weight} "
                f"is greater than combined shortcuts weight {combined_weight}"
            )
