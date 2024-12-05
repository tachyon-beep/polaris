"""
Utility functions for Hub Labeling algorithm.

This module provides helper functions for label pruning, distance calculations,
and path validation.
"""

import time
from typing import Dict, List, Optional, Set, Protocol, Tuple, TYPE_CHECKING
from collections import deque

from polaris.core.graph.traversal.utils import get_edge_weight
from polaris.core.graph.traversal.cache import PathCache
from polaris.core.graph.traversal.path_models import PathResult, PerformanceMetrics
from polaris.core.models import Edge
from .models import HubLabel, HubLabelSet, HubLabelState

if TYPE_CHECKING:
    from polaris.core.graph import Graph


class GraphLike(Protocol):
    """Protocol defining required graph operations."""

    def get_nodes(self) -> Set[str]: ...

    def get_neighbors(self, node: str, reverse: bool = False) -> Set[str]: ...

    def get_edge(self, from_node: str, to_node: str) -> Optional[Edge]: ...


def validate_path_through_hub(source: str, target: str, hub: str, graph: GraphLike) -> bool:
    """
    Verify path exists through hub.

    Args:
        source: Source node
        target: Target node
        hub: Hub node to validate path through
        graph: Graph instance

    Returns:
        True if valid path exists through hub, False otherwise
    """
    # Check forward path exists
    current = source
    visited = {current}
    while current != hub:
        found_next = False
        for neighbor in graph.get_neighbors(current):
            if neighbor not in visited and (neighbor == hub or graph.get_edge(neighbor, hub)):
                current = neighbor
                visited.add(current)
                found_next = True
                break
        if not found_next:
            return False

    # Check backward path exists
    current = target
    visited = {current}
    while current != hub:
        found_next = False
        for neighbor in graph.get_neighbors(current, reverse=True):
            if neighbor not in visited and (neighbor == hub or graph.get_edge(hub, neighbor)):
                current = neighbor
                visited.add(current)
                found_next = True
                break
        if not found_next:
            return False

    return True


def build_forward_path(source: str, hub: str, state: HubLabelState, graph: GraphLike) -> List[Edge]:
    """Build path from source to hub."""
    path = []
    current = source
    visited = {current}

    while current != hub:
        next_label = state.get_forward_labels(current).get_label(hub)
        if next_label is None:
            return []

        if next_label.first_hop is not None:
            next_node = next_label.first_hop.to_entity
            if next_node in visited:  # Cycle detection
                return []
            path.append(next_label.first_hop)
            current = next_node
            visited.add(current)
        else:
            # Try direct edge
            direct_edge = graph.get_edge(current, hub)
            if direct_edge is None:
                return []
            path.append(direct_edge)
            break

    return path


def build_backward_path(
    hub: str, target: str, state: HubLabelState, graph: GraphLike
) -> List[Edge]:
    """Build path from hub to target."""
    path = []
    current = target
    visited = {current}

    while current != hub:
        next_label = state.get_backward_labels(current).get_label(hub)
        if next_label is None:
            return []

        if next_label.first_hop is not None:
            next_node = next_label.first_hop.from_entity
            if next_node in visited:  # Cycle detection
                return []
            path.append(next_label.first_hop)
            current = next_node
            visited.add(current)
        else:
            # Try direct edge
            direct_edge = graph.get_edge(hub, current)
            if direct_edge is None:
                return []
            path.append(direct_edge)
            break

    return list(reversed(path))


def reconstruct_path(
    source: str,
    target: str,
    state: HubLabelState,
    graph: GraphLike,
) -> List[Edge]:
    """
    Enhanced path reconstruction with better validation.

    Args:
        source: Source node
        target: Target node
        state: Algorithm state
        graph: Graph instance

    Returns:
        List of edges forming shortest path
    """
    # First find best hub and distances
    min_dist = float("inf")
    best_hub = None
    best_forward_label = None
    best_backward_label = None

    forward_labels = state.get_forward_labels(source)
    backward_labels = state.get_backward_labels(target)

    # Try direct path first
    direct_edge = graph.get_edge(source, target)
    if direct_edge:
        return [direct_edge]

    # Find best hub with validated paths
    for forward_label in forward_labels.labels:
        backward_label = backward_labels.get_label(forward_label.hub)
        if backward_label:
            total_dist = forward_label.distance + backward_label.distance
            if total_dist < min_dist:
                # Validate path exists through this hub
                if validate_path_through_hub(source, target, forward_label.hub, graph):
                    min_dist = total_dist
                    best_hub = forward_label.hub
                    best_forward_label = forward_label
                    best_backward_label = backward_label

    if not best_hub:
        return []

    # Build path through best hub
    forward_path = build_forward_path(source, best_hub, state, graph)
    backward_path = build_backward_path(best_hub, target, state, graph)

    if not forward_path or not backward_path:
        return []

    path = forward_path + backward_path

    # Validate total path weight matches computed distance
    total_weight = sum(get_edge_weight(edge) for edge in path)
    if abs(total_weight - min_dist) > 1e-10:  # Account for floating point error
        return []

    # Cache valid path
    cache_key = PathCache.get_cache_key(source, target, "hub_label_path")
    result = PathResult(path=path, total_weight=total_weight)
    PathCache.put(cache_key, result)

    return path


def _compute_reachability_metrics(node: str, graph: GraphLike) -> Tuple[int, int, float]:
    """
    Compute reachability metrics for a node using BFS.

    Args:
        node: Node to analyze
        graph: Graph instance

    Returns:
        Tuple of (forward_reach_count, backward_reach_count, avg_path_length)
    """

    def bfs_reach(start: str, reverse: bool = False) -> Tuple[int, float]:
        visited = set([start])
        queue = deque([(start, 0)])  # (node, distance)
        total_dist = 0
        reach_count = 0

        while queue:
            current, dist = queue.popleft()
            reach_count += 1
            total_dist += dist

            for neighbor in graph.get_neighbors(current, reverse=reverse):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        avg_dist = total_dist / reach_count if reach_count > 0 else float("inf")
        return reach_count, avg_dist

    # Compute forward and backward reachability
    forward_count, forward_avg_dist = bfs_reach(node, reverse=False)
    backward_count, backward_avg_dist = bfs_reach(node, reverse=True)

    # Average path length considers both directions
    avg_path_length = (forward_avg_dist + backward_avg_dist) / 2

    return forward_count, backward_count, avg_path_length


def calculate_node_importance(
    node: str,
    graph: GraphLike,
    hub_order: Dict[str, int],  # Kept for API compatibility but not used
) -> float:
    """
    Calculate importance of a node for hub selection.

    Uses a combination of metrics:
    1. Reachability (how many nodes can be reached from/to this node)
    2. Path centrality (how many shortest paths go through this node)
    3. Average path length (shorter paths through node = more important)

    Args:
        node: Node to calculate importance for
        graph: Graph instance
        hub_order: Map of node to ordering value (kept for API compatibility)

    Returns:
        Importance score (higher means more important)
    """
    # Get basic connectivity metrics
    in_neighbors = graph.get_neighbors(node, reverse=True)
    out_neighbors = graph.get_neighbors(node)
    total_degree = len(in_neighbors) + len(out_neighbors)

    # Early exit for isolated nodes
    if total_degree == 0:
        return 0.0

    # Compute reachability metrics
    forward_reach, backward_reach, avg_path_length = _compute_reachability_metrics(node, graph)

    # Calculate path coverage score
    # Higher score for nodes that can reach many other nodes in both directions
    coverage_score = (forward_reach * backward_reach) / (avg_path_length + 1)

    # Calculate centrality score based on potential paths through node
    centrality_score = len(in_neighbors) * len(out_neighbors)

    # Detect if node is part of a chain
    is_chain_node = total_degree <= 2

    if is_chain_node:
        # For chain nodes, prioritize central positions
        # Nodes with balanced forward/backward reach are more central
        balance_ratio = min(forward_reach, backward_reach) / max(forward_reach, backward_reach)
        chain_score = balance_ratio * coverage_score
        return chain_score
    else:
        # For non-chain nodes, combine metrics with appropriate weights
        # Coverage is most important, followed by centrality
        importance = (
            0.6 * coverage_score  # Path coverage
            + 0.3 * centrality_score / total_degree  # Normalized centrality
            + 0.1 * (forward_reach + backward_reach)  # Total reachability
        )

        # Penalize leaf nodes (degree 1) as they make poor hubs
        if total_degree == 1:
            importance *= 0.1

        return importance


def should_prune_label(
    source: str,
    hub: str,
    distance: float,
    state: HubLabelState,
) -> bool:
    """
    Check if a label can be pruned.

    A label can be pruned if:
    1. A shorter path exists through a more important hub
    2. The label is redundant for path coverage

    Args:
        source: Source node
        hub: Hub node
        distance: Distance to hub
        state: Current algorithm state

    Returns:
        True if label can be pruned, False otherwise
    """
    # Never prune self labels
    if source == hub:
        return False

    # Get existing labels
    forward_labels = state.get_forward_labels(source)
    backward_labels = state.get_backward_labels(hub)

    # Check paths through more important hubs
    min_dist = float("inf")
    for label in forward_labels.labels:
        # Only consider more important hubs
        if state.get_hub_order(label.hub) < state.get_hub_order(hub):
            # Check if we can reach the target through this hub
            hub_to_target = backward_labels.get_distance(label.hub)
            if hub_to_target is not None:
                min_dist = min(min_dist, label.distance + hub_to_target)

    # Prune if we have a significantly better path
    # Use a smaller threshold (0.95) to be more aggressive about pruning
    if min_dist < float("inf"):
        return min_dist < distance * 0.95

    return False


def compute_distance(
    source: str,
    target: str,
    state: HubLabelState,
) -> Optional[float]:
    """
    Compute shortest path distance using hub labels.

    Args:
        source: Source node
        target: Target node
        state: Algorithm state

    Returns:
        Shortest path distance if path exists, None otherwise
    """
    # Check cache first
    cache_key = PathCache.get_cache_key(source, target, "hub_label_distance")
    if cached_result := PathCache.get(cache_key):
        return cached_result.total_weight

    forward_labels = state.get_forward_labels(source)
    backward_labels = state.get_backward_labels(target)

    # Find minimum distance through a common hub
    min_dist = float("inf")
    for forward_label in forward_labels.labels:
        backward_dist = backward_labels.get_distance(forward_label.hub)
        if backward_dist is not None:
            min_dist = min(min_dist, forward_label.distance + backward_dist)

    if min_dist < float("inf"):
        # Cache the computed distance
        result = PathResult(path=[], total_weight=min_dist)
        PathCache.put(cache_key, result)
        return min_dist
    return None
