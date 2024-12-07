"""
Utility functions for Hub Labeling algorithm.

This module provides helper functions for label pruning, distance calculations,
and path validation.
"""

from typing import Dict, List, Optional, Set, Protocol, Tuple, TYPE_CHECKING
from collections import deque

from polaris.core.graph.traversal.utils import get_edge_weight
from polaris.core.graph.traversal.cache import PathCache
from polaris.core.graph.traversal.path_models import PathResult
from polaris.core.models import Edge
from .models import HubLabelState

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

    # Try direct path first
    direct_edge = graph.get_edge(source, target)
    if direct_edge:
        direct_dist = get_edge_weight(direct_edge)
        min_hub_dist = compute_distance(source, target, state)
        if min_hub_dist is None or direct_dist <= min_hub_dist:
            return [direct_edge]

    # Try path through intermediate nodes
    path = []
    current = source
    visited = {current}

    while current != target:
        # Get forward labels from current node
        forward_labels = state.get_forward_labels(current)

        # Try direct path to target
        target_label = forward_labels.get_label(target)
        if target_label is not None and target_label.first_hop is not None:
            path.append(target_label.first_hop)
            current = target_label.first_hop.to_entity
            visited.add(current)
            continue

        # Try path through intermediate nodes
        min_dist = float("inf")
        best_hop = None
        best_next = None

        for label in forward_labels.labels:
            if label.hub in visited:
                continue

            if label.first_hop:
                next_node = label.first_hop.to_entity
                if next_node in visited:
                    continue

                # Check if we can reach target from this node
                next_labels = state.get_forward_labels(next_node)
                next_target_label = next_labels.get_label(target)
                if next_target_label is not None:
                    total_dist = label.distance + next_target_label.distance
                    if total_dist < min_dist:
                        min_dist = total_dist
                        best_hop = label.first_hop
                        best_next = next_node

        if best_hop and best_next:  # Ensure both best_hop and best_next are not None
            path.append(best_hop)
            current = best_next
            visited.add(current)
        else:
            return []

    # Validate path
    if not validate_path_continuity(path):
        return []

    # Cache valid path
    result = PathResult(
        path=path,
        total_weight=sum(get_edge_weight(edge) for edge in path),
    )
    PathCache.put(cache_key, result)

    return path


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

    # Try direct path first
    forward_labels = state.get_forward_labels(source)
    target_label = forward_labels.get_label(target)
    if target_label is not None:
        return target_label.distance

    # Try paths through intermediate nodes
    min_dist = float("inf")
    visited = {source}
    queue = [(source, 0.0)]  # (node, distance)

    while queue:
        current, dist = queue.pop(0)

        # Try direct path to target from current node
        current_labels = state.get_forward_labels(current)
        target_label = current_labels.get_label(target)
        if target_label is not None:
            total_dist = dist + target_label.distance
            min_dist = min(min_dist, total_dist)
            continue

        # Try paths through neighbors
        for label in current_labels.labels:
            if label.hub in visited:
                continue

            if label.first_hop:
                next_node = label.first_hop.to_entity
                if next_node not in visited:
                    visited.add(next_node)
                    queue.append((next_node, dist + get_edge_weight(label.first_hop)))

    if min_dist < float("inf"):
        # Cache the computed distance
        result = PathResult(path=[], total_weight=min_dist)
        PathCache.put(cache_key, result)
        return min_dist

    return None
