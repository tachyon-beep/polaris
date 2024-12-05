"""
Utility functions for Contraction Hierarchies algorithm.

This module provides helper functions for node importance calculation,
shortcut necessity checking, and path validation.
"""

from typing import Dict, List, Optional, Set, TYPE_CHECKING

from polaris.core.graph.traversal.utils import WeightFunc, get_edge_weight
from polaris.core.models import Edge
from .models import SHORTCUT_TYPE

if TYPE_CHECKING:
    from polaris.core.graph import Graph


def calculate_node_importance(
    node: str,
    graph: "Graph",
    contracted_neighbors: Dict[str, Set[str]],
    node_level: Dict[str, int],
    shortcut_count: int,
) -> float:
    """
    Calculate importance of a node for contraction ordering.

    Uses multiple factors:
    - Edge difference (shortcuts needed - original edges)
    - Number of contracted neighbors
    - Current level in hierarchy
    - Node degree (in + out)

    Args:
        node: Node to calculate importance for
        graph: Graph instance
        contracted_neighbors: Map of node to contracted neighbors
        node_level: Map of node to contraction level
        shortcut_count: Number of shortcuts needed

    Returns:
        Importance score (higher means more important)
    """
    # Get original edges
    original_edges = len(list(graph.get_neighbors(node)))

    # Get contracted neighbor count
    contracted_count = len(contracted_neighbors.get(node, set()))

    # Get current level
    level = node_level.get(node, 0)

    # Calculate importance based on multiple factors
    importance = (
        5 * shortcut_count  # Weight the cost of adding shortcuts
        + 2 * contracted_count  # Consider neighborhood complexity
        + level  # Preserve hierarchy levels
        + original_edges  # Base importance on connectivity
    )

    return importance


def is_shortcut_necessary(
    source: str,
    target: str,
    via: str,
    shortcut_weight: float,
    graph: "Graph",
    weight_func: Optional[WeightFunc] = None,
) -> bool:
    """
    Determine if shortcut is necessary using witness search.

    Args:
        source: Source node
        target: Target node
        via: Node being contracted
        shortcut_weight: Weight of potential shortcut
        graph: Graph instance
        weight_func: Optional custom weight function

    Returns:
        True if shortcut is necessary, False if witness path exists
    """
    from heapq import heappop, heappush

    # Initialize distances and priority queue
    distances = {source: 0.0}
    pq = [(0.0, source)]
    visited = set()

    while pq:
        dist, node = heappop(pq)

        if node == target:
            # Found a witness path that's better than the shortcut
            return dist > shortcut_weight - 1e-10  # Allow for floating point error

        if node in visited or node == via:  # Don't go through contracted node
            continue

        visited.add(node)

        # Sort neighbors for deterministic behavior
        neighbors = sorted(graph.get_neighbors(node))
        for neighbor in neighbors:
            if neighbor == via:  # Don't go through contracted node
                continue

            edge = graph.get_edge(node, neighbor)
            if not edge:
                continue

            edge_weight = get_edge_weight(edge, weight_func)
            new_dist = dist + edge_weight
            if new_dist >= shortcut_weight + 1e-10:  # Early termination with epsilon
                continue

            if neighbor not in distances or new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heappush(pq, (new_dist, neighbor))

    # No witness path found within weight limit
    return True


def unpack_shortcut(shortcut_path: List[Edge], graph: "Graph") -> List[Edge]:
    """
    Recursively unpack shortcuts into original edges.

    Args:
        shortcut_path: Path potentially containing shortcuts
        graph: Graph instance

    Returns:
        Path with shortcuts expanded to original edges
    """
    unpacked_path = []
    for edge in shortcut_path:
        if edge.relation_type == SHORTCUT_TYPE:
            # Get shortcut details from context
            via_node = edge.context.split()[-1]  # Extract node from "Shortcut via X"

            # Get component edges
            lower_edge = graph.get_edge(edge.from_entity, via_node)
            upper_edge = graph.get_edge(via_node, edge.to_entity)

            if lower_edge and upper_edge:
                # Recursively unpack component edges
                unpacked_path.extend(unpack_shortcut([lower_edge, upper_edge], graph))
        else:
            unpacked_path.append(edge)

    return unpacked_path


def validate_shortcuts(shortcuts: Dict[str, Edge], graph: "Graph") -> bool:
    """
    Validate shortcut consistency.

    Args:
        shortcuts: Map of shortcut edges
        graph: Graph instance

    Returns:
        True if shortcuts are valid, False otherwise
    """
    for edge in shortcuts.values():
        if edge.relation_type != SHORTCUT_TYPE:
            return False

        # Extract via node from context
        if not edge.context or not edge.context.startswith("Shortcut via "):
            return False

        via_node = edge.context.split()[-1]

        # Check component edges exist
        lower_edge = graph.get_edge(edge.from_entity, via_node)
        upper_edge = graph.get_edge(via_node, edge.to_entity)

        if not lower_edge or not upper_edge:
            return False

        # Verify weight consistency
        shortcut_weight = edge.metadata.weight
        actual_weight = lower_edge.metadata.weight + upper_edge.metadata.weight

        if abs(shortcut_weight - actual_weight) > 1e-10:  # Allow for floating point error
            return False

    return True
