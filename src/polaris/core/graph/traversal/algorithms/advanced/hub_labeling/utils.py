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


def validate_path_continuity(path: List[Edge]) -> bool:
    """
    Validate that a path is continuous with no gaps.

    Args:
        path: List of edges forming a path

    Returns:
        True if path is continuous, False otherwise
    """
    if not path:
        return True

    for i in range(len(path) - 1):
        if path[i].to_entity != path[i + 1].from_entity:
            return False
    return True


def validate_path_through_hub(source: str, target: str, hub: str, graph: GraphLike) -> bool:
    """
    Verify path exists through hub with cycle detection.

    Args:
        source: Source node
        target: Target node
        hub: Hub node to validate path through
        graph: Graph instance

    Returns:
        True if valid path exists through hub, False otherwise
    """

    def bfs_to_hub(start: str, reverse: bool = False) -> bool:
        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            current, path = queue.popleft()
            if current == hub:
                return True

            neighbors = graph.get_neighbors(current, reverse=reverse)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
                    if len(path) > len(graph.get_nodes()):  # Cycle detection
                        return False
        return False

    # Check both forward and backward paths exist
    return bfs_to_hub(source, reverse=False) and bfs_to_hub(target, reverse=True)


def build_forward_path(source: str, hub: str, state: HubLabelState, graph: GraphLike) -> List[Edge]:
    """Build path from source to hub with cycle detection."""
    path = []
    current = source
    visited = {current}
    path_length = 0
    max_path_length = len(graph.get_nodes())  # Maximum possible path length

    while current != hub and path_length < max_path_length:
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
            path_length += 1
        else:
            direct_edge = graph.get_edge(current, hub)
            if direct_edge is None:
                return []
            path.append(direct_edge)
            break

    return [] if path_length >= max_path_length else path


def build_backward_path(
    hub: str, target: str, state: HubLabelState, graph: GraphLike
) -> List[Edge]:
    """Build path from hub to target with cycle detection."""
    path = []
    current = target
    visited = {current}
    path_length = 0
    max_path_length = len(graph.get_nodes())  # Maximum possible path length

    while current != hub and path_length < max_path_length:
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
            path_length += 1
        else:
            direct_edge = graph.get_edge(hub, current)
            if direct_edge is None:
                return []
            path.append(direct_edge)
            break

    return [] if path_length >= max_path_length else list(reversed(path))


def validate_path_distance(path: List[Edge], expected_distance: float) -> bool:
    """
    Validate that path distance matches expected distance.

    Args:
        path: List of edges forming a path
        expected_distance: Expected total distance

    Returns:
        True if path distance matches expected, False otherwise
    """
    if not path:
        return expected_distance == 0

    actual_distance = sum(get_edge_weight(edge) for edge in path)
    return abs(actual_distance - expected_distance) < 1e-10


def reconstruct_path(
    source: str,
    target: str,
    state: HubLabelState,
    graph: GraphLike,
) -> List[Edge]:
    """
    Enhanced path reconstruction with comprehensive validation.

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

    # Try direct path first
    direct_edge = graph.get_edge(source, target)
    if direct_edge:
        direct_dist = get_edge_weight(direct_edge)
        # Verify no shorter path exists through hubs
        min_hub_dist = compute_distance(source, target, state)
        if min_hub_dist is None or direct_dist <= min_hub_dist:
            return [direct_edge]

    # Find best hub and distances
    min_dist = float("inf")
    best_hub = None
    forward_labels = state.get_forward_labels(source)
    backward_labels = state.get_backward_labels(target)

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

    if not best_hub:
        return []

    # Build path through best hub
    forward_path = build_forward_path(source, best_hub, state, graph)
    backward_path = build_backward_path(best_hub, target, state, graph)

    if not forward_path or not backward_path:
        return []

    # Combine paths
    path = forward_path + backward_path

    # Validate path
    if not validate_path_continuity(path):
        return []

    if not validate_path_distance(path, min_dist):
        return []

    # Cache valid path
    result = PathResult(
        path=path,
        total_weight=min_dist,
    )
    PathCache.put(cache_key, result)

    return path


def calculate_node_importance(
    node: str,
    graph: GraphLike,
    hub_order: Dict[str, int],
) -> float:
    """
    Calculate importance of a node for hub selection.

    Uses a combination of metrics:
    1. Reachability (how many nodes can be reached from/to this node)
    2. Path centrality (how many shortest paths go through this node)
    3. Average path length (shorter paths through node = more important)
    4. Topology position (nodes in critical positions are more important)

    Args:
        node: Node to calculate importance for
        graph: Graph instance
        hub_order: Current hub ordering (for considering existing hubs)

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
    coverage_score = (forward_reach * backward_reach) / (avg_path_length + 1)

    # Calculate centrality score
    centrality_score = len(in_neighbors) * len(out_neighbors)

    # Detect topology position
    is_bridge = _is_bridge_node(node, graph)
    is_chain = total_degree <= 2
    is_hub = total_degree > 4

    # Base importance score
    importance = (
        0.4 * coverage_score  # Path coverage
        + 0.3 * centrality_score / total_degree  # Normalized centrality
        + 0.2 * (forward_reach + backward_reach) / len(graph.get_nodes())  # Normalized reachability
        + 0.1 * (1.0 / (avg_path_length + 1))  # Path length efficiency
    )

    # Apply topology-based adjustments
    if is_bridge:
        importance *= 1.5  # Boost bridge nodes
    if is_chain:
        importance *= 0.7  # Reduce chain nodes
    if is_hub:
        importance *= 1.3  # Boost high-degree nodes

    # Consider existing hubs
    if hub_order:
        nearby_hubs = sum(1 for n in in_neighbors.union(out_neighbors) if n in hub_order)
        if nearby_hubs > 0:
            importance *= 0.8  # Reduce importance if near existing hubs

    return importance


def _is_bridge_node(node: str, graph: GraphLike) -> bool:
    """Check if node is a bridge (removing it would disconnect the graph)."""
    neighbors = graph.get_neighbors(node, reverse=True).union(graph.get_neighbors(node))
    if len(neighbors) < 2:
        return False

    # Check if all neighbors can reach each other without going through node
    for n1 in neighbors:
        for n2 in neighbors:
            if n1 != n2:
                # Try to find path not using node
                queue = deque([n1])
                visited = {n1, node}
                found_path = False

                while queue and not found_path:
                    current = queue.popleft()
                    if current == n2:
                        found_path = True
                        break

                    for next_node in graph.get_neighbors(current).union(
                        graph.get_neighbors(current, reverse=True)
                    ):
                        if next_node not in visited:
                            visited.add(next_node)
                            queue.append(next_node)

                if not found_path:
                    return True  # No path exists between neighbors without using node

    return False


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
        visited = {start}
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

    forward_count, forward_avg_dist = bfs_reach(node, reverse=False)
    backward_count, backward_avg_dist = bfs_reach(node, reverse=True)
    avg_path_length = (forward_avg_dist + backward_avg_dist) / 2

    return forward_count, backward_count, avg_path_length


def should_prune_label(source: str, hub: str, distance: float, state: HubLabelState) -> bool:
    """
    Check if a label can be pruned.

    A label can be pruned if:
    1. A shorter path exists through a more important hub
    2. The label is redundant for path coverage
    3. The distance exceeds reasonable bounds

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
        if state.get_hub_order(label.hub) < state.get_hub_order(hub):
            hub_to_target = backward_labels.get_distance(label.hub)
            if hub_to_target is not None:
                min_dist = min(min_dist, label.distance + hub_to_target)

    # Prune if significantly better path exists
    if min_dist < float("inf"):
        return min_dist < distance * 0.95

    # Prune if distance exceeds reasonable bounds
    avg_label_dist = 0.0
    num_labels = 0
    for label in forward_labels.labels:
        avg_label_dist += label.distance
        num_labels += 1
    if num_labels > 0:
        avg_label_dist /= num_labels
        if distance > avg_label_dist * 3:  # Distance is unusually high
            return True

    return False


def compute_distance(source: str, target: str, state: HubLabelState) -> Optional[float]:
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
            total_dist = forward_label.distance + backward_dist
            if total_dist < min_dist:
                min_dist = total_dist

    if min_dist < float("inf"):
        # Cache the computed distance
        result = PathResult(path=[], total_weight=min_dist)
        PathCache.put(cache_key, result)
        return min_dist
    return None
