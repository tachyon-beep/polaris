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

    # Try all possible hubs
    for forward_label in forward_labels.labels:
        backward_label = backward_labels.get_label(forward_label.hub)
        if backward_label is not None:
            total_dist = forward_label.distance + backward_label.distance
            if total_dist < best_dist:
                # Check if paths exist in both directions
                forward_path_exists = False
                backward_path_exists = False

                # Check forward path
                if forward_label.first_hop is not None:
                    forward_path_exists = True
                elif forward_label.hub == source:
                    forward_path_exists = True
                elif graph.get_edge(source, forward_label.hub) is not None:
                    forward_path_exists = True

                # Check backward path
                if backward_label.first_hop is not None:
                    backward_path_exists = True
                elif backward_label.hub == target:
                    backward_path_exists = True
                elif graph.get_edge(backward_label.hub, target) is not None:
                    backward_path_exists = True

                if forward_path_exists and backward_path_exists:
                    best_dist = total_dist
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
    visited = set()  # Track visited nodes for cycle detection

    # Forward path to hub
    if source != best_hub:
        current = source
        visited.add(current)

        while current != best_hub:
            next_label = state.get_forward_labels(current).get_label(best_hub)
            if next_label is None:
                return []

            # Try using first_hop if available
            if next_label.first_hop is not None:
                next_node = next_label.first_hop.to_entity
                if next_node in visited:  # Cycle detection
                    return []
                path.append(next_label.first_hop)
                current = next_node
                visited.add(current)
            else:
                # Try direct edge
                direct_edge = graph.get_edge(current, best_hub)
                if direct_edge is None:
                    return []
                path.append(direct_edge)
                break

    # Backward path from hub to target
    if target != best_hub:
        backward_path = []
        current = target
        visited.add(current)

        while current != best_hub:
            next_label = state.get_backward_labels(current).get_label(best_hub)
            if next_label is None:
                return []

            # Try using first_hop if available
            if next_label.first_hop is not None:
                next_node = next_label.first_hop.from_entity
                if next_node in visited:  # Cycle detection
                    return []
                backward_path.append(next_label.first_hop)
                current = next_node
                visited.add(current)
            else:
                # Try direct edge
                direct_edge = graph.get_edge(best_hub, current)
                if direct_edge is None:
                    return []
                backward_path.append(direct_edge)
                break

        # Combine paths
        path.extend(reversed(backward_path))

    # Validate path
    if not path:
        return []

    # Check path connectivity
    for i in range(len(path) - 1):
        if path[i].to_entity != path[i + 1].from_entity:
            return []  # Path is not connected

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
