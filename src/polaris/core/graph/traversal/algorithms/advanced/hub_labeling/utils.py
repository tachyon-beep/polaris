"""
Utility functions for Hub Labeling algorithm.

This module provides helper functions for label pruning, distance calculations,
and path validation.
"""

import time
from typing import Dict, List, Optional, Set, Protocol, Tuple, TYPE_CHECKING

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


def calculate_node_importance(
    node: str,
    graph: GraphLike,
    hub_order: Dict[str, int],
) -> float:
    """
    Calculate importance of a node for hub selection.

    Uses betweenness centrality approximation - nodes that connect
    different parts of the graph are more important.

    Args:
        node: Node to calculate importance for
        graph: Graph instance
        hub_order: Map of node to ordering value

    Returns:
        Importance score (higher means more important)
    """
    # Get incoming and outgoing neighbors
    in_neighbors = graph.get_neighbors(node, reverse=True)
    out_neighbors = graph.get_neighbors(node)

    # Calculate how many unique paths go through this node
    paths_through = len(in_neighbors) * len(out_neighbors)

    # Add degree centrality as a secondary factor
    degree = len(in_neighbors) + len(out_neighbors)

    # Get current ordering level (default to 0 if not set)
    level = hub_order.get(node, 0)

    # Calculate distance from center of graph
    total_dist = 0
    min_weight = float("inf")
    max_weight = 0.0
    for neighbor in in_neighbors | out_neighbors:
        edge = graph.get_edge(node, neighbor) or graph.get_edge(neighbor, node)
        if edge:
            weight = get_edge_weight(edge)
            total_dist += weight
            min_weight = min(min_weight, weight)
            max_weight = max(max_weight, weight)
    avg_dist = total_dist / degree if degree > 0 else float("inf")

    # For chain graphs, make middle nodes more important
    if degree <= 2:  # Node is part of a chain
        # Boost importance based on number of paths through node
        return paths_through + 0.1 * degree + level
    else:
        # For other graphs, use betweenness centrality approximation
        return (1.0 / (avg_dist + 1)) * paths_through + 0.1 * degree + level


def should_prune_label(
    source: str,
    hub: str,
    distance: float,
    state: HubLabelState,
) -> bool:
    """
    Check if a label can be pruned.

    A label can be pruned if the distance can be computed using
    existing labels through a more important (lower ordered) hub.

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

    # Check if we can reach through more important hubs
    min_dist = float("inf")
    forward_labels = state.get_forward_labels(source)
    backward_labels = state.get_backward_labels(hub)

    # Try all possible paths through more important hubs
    for label in forward_labels.labels:
        # Only consider more important hubs
        if state.get_hub_order(label.hub) < state.get_hub_order(hub):
            # Check if we can reach the target through this hub
            hub_to_target = backward_labels.get_distance(label.hub)
            if hub_to_target is not None:
                min_dist = min(min_dist, label.distance + hub_to_target)

    # For chain graphs, be less aggressive about pruning
    # to maintain connectivity for long paths
    if min_dist < float("inf"):
        return min_dist < distance  # Only prune if strictly better path exists
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


def find_best_hub(
    source: str,
    target: str,
    state: HubLabelState,
    graph: GraphLike,
) -> Tuple[Optional[str], Optional[HubLabel], Optional[HubLabel]]:
    """
    Find the best hub for path reconstruction.

    Args:
        source: Source node
        target: Target node
        state: Algorithm state
        graph: Graph instance

    Returns:
        Tuple of (best hub, forward label, backward label) or (None, None, None) if no valid hub found
    """
    forward_labels = state.get_forward_labels(source)
    backward_labels = state.get_backward_labels(target)
    best_dist = float("inf")
    best_hub = None
    best_forward_label = None
    best_backward_label = None

    for forward_label in forward_labels.labels:
        backward_label = backward_labels.get_label(forward_label.hub)
        if backward_label is not None:
            dist = forward_label.distance + backward_label.distance
            if dist < best_dist:
                # Check if we can reconstruct the path through this hub
                can_reach_hub = (
                    forward_label.first_hop is not None
                    or forward_label.hub == source
                    or graph.get_edge(source, forward_label.hub) is not None
                )
                can_reach_target = (
                    backward_label.first_hop is not None
                    or backward_label.hub == target
                    or graph.get_edge(backward_label.hub, target) is not None
                )
                if can_reach_hub and can_reach_target:
                    best_dist = dist
                    best_hub = forward_label.hub
                    best_forward_label = forward_label
                    best_backward_label = backward_label

    return best_hub, best_forward_label, best_backward_label


def reconstruct_path(
    source: str,
    target: str,
    state: HubLabelState,
    graph: GraphLike,
) -> List[Edge]:
    """
    Reconstruct shortest path using hub labels.

    Args:
        source: Source node
        target: Target node
        state: Algorithm state
        graph: Graph instance

    Returns:
        List of edges forming shortest path
    """
    # Check cache first
    cache_key = PathCache.get_cache_key(source, target, "hub_label_path")
    if cached_result := PathCache.get(cache_key):
        return cached_result.path

    metrics = PerformanceMetrics(operation="hub_label_path_reconstruction", start_time=time.time())

    # Get shortest path distance through hubs
    hub_dist = compute_distance(source, target, state)
    if hub_dist is None:
        return []

    # Check if direct edge is shortest path
    direct_edge = graph.get_edge(source, target)
    if direct_edge:
        direct_dist = get_edge_weight(direct_edge)
        if abs(direct_dist - hub_dist) < 1e-10:  # Account for floating point error
            return [direct_edge]

    # Find best hub with valid path
    best_hub, forward_label, backward_label = find_best_hub(source, target, state, graph)
    if best_hub is None:
        return []

    # Build path through best hub
    path = []

    # Forward path to hub
    if source != best_hub:
        if forward_label and forward_label.first_hop:
            # Follow first_hop edges to hub
            current = source
            visited = {current}  # Prevent cycles
            while current != best_hub:
                next_label = state.get_forward_labels(current).get_label(best_hub)
                if next_label is None or next_label.first_hop is None:
                    return []
                next_node = next_label.first_hop.to_entity
                if next_node in visited:  # Cycle detection
                    return []
                path.append(next_label.first_hop)
                current = next_node
                visited.add(current)
        else:
            # Direct edge to hub
            direct_edge = graph.get_edge(source, best_hub)
            if direct_edge is None:
                return []
            path.append(direct_edge)

    # Backward path from target to hub
    if target != best_hub:
        backward_path = []
        if backward_label and backward_label.first_hop:
            # Follow first_hop edges to hub
            current = target
            visited = {current}  # Prevent cycles
            while current != best_hub:
                next_label = state.get_backward_labels(current).get_label(best_hub)
                if next_label is None or next_label.first_hop is None:
                    return []
                next_node = next_label.first_hop.from_entity
                if next_node in visited:  # Cycle detection
                    return []
                backward_path.append(next_label.first_hop)
                current = next_node
                visited.add(current)
        else:
            # Direct edge from hub
            direct_edge = graph.get_edge(best_hub, target)
            if direct_edge is None:
                return []
            backward_path.append(direct_edge)

        # Combine paths
        path.extend(reversed(backward_path))

    # Validate total path weight matches computed distance
    total_weight = sum(get_edge_weight(edge) for edge in path)
    if abs(total_weight - hub_dist) > 1e-10:  # Account for floating point error
        return []  # Path weight doesn't match expected distance

    # Create and validate path result
    metrics.path_length = len(path)
    metrics.end_time = time.time()

    if path:  # Only cache valid paths
        result = PathResult(path=path, total_weight=total_weight)
        PathCache.put(cache_key, result)

    return path
