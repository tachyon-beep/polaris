"""
Utility functions for Contraction Hierarchies algorithm.

This module provides helper functions for node importance calculation,
shortcut necessity checking, and path validation.
"""

from typing import Dict, List, Optional, Set, Tuple
import logging
import statistics
import threading
import time
from hashlib import sha256
from collections import defaultdict
import ast
from heapq import heappop, heappush
from polaris.core.graph.traversal.utils import WeightFunc
from polaris.core.graph import Graph
from polaris.core.models import Edge, EdgeMetadata
from polaris.core.enums import RelationType
from polaris.core.exceptions import GraphOperationError
from .models import SHORTCUT_TYPE
from .cache import get_cache_manager, get_cache_size

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

EPSILON = 1e-10  # Consistent epsilon value for floating point comparisons


# Thread-safe performance monitoring
class PerformanceMonitor:
    """Monitor and log performance metrics."""

    def __init__(self):
        self.metrics = defaultdict(list)
        self._lock = threading.Lock()

    def record_metric(self, name: str, value: float):
        """Record a performance metric."""
        with self._lock:
            self.metrics[name].append(value)

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistical summary of recorded metrics."""
        stats = {}
        for name, values in self.metrics.items():
            if values:
                stats[name] = {
                    "avg": statistics.mean(values),
                    "min": min(values),
                    "max": max(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0,
                    "count": len(values),
                }
        return stats


# Global performance monitor
_performance = PerformanceMonitor()

# Enhanced thread safety with RLock
_graph_lock = threading.RLock()


def _compute_graph_hash(graph: "Graph", source: str, target: str, via: str) -> str:
    """
    Compute a stable hash for graph state and query parameters.

    Args:
        graph: Graph instance
        source: Source node
        target: Target node
        via: Node being contracted

    Returns:
        String hash representing current state
    """
    start_time = time.time()
    try:
        # Get relevant subgraph edges (only those that could affect the witness path)
        relevant_edges = []
        visited = {source}
        to_visit = [source]

        while to_visit:
            node = to_visit.pop(0)
            for neighbor in graph.get_neighbors(node):
                if neighbor == via:  # Skip the node being contracted
                    continue
                edge = graph.get_edge(node, neighbor)
                if edge and neighbor not in visited:
                    relevant_edges.append((edge.from_entity, edge.to_entity, edge.metadata.weight))
                    visited.add(neighbor)
                    if neighbor != target:  # Don't explore beyond target
                        to_visit.append(neighbor)

        # Sort for stable hash
        relevant_edges.sort()

        # Combine with query parameters
        state = (relevant_edges, source, target, via)
        return sha256(str(state).encode()).hexdigest()

    finally:
        duration = time.time() - start_time
        _performance.record_metric("hash_computation_time", duration)


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
    - Local clustering coefficient

    Args:
        node: Node to calculate importance for
        graph: Graph instance
        contracted_neighbors: Map of node to contracted neighbors
        node_level: Map of node to contraction level
        shortcut_count: Number of shortcuts needed

    Returns:
        Importance score (higher means more important)
    """

    start_time = time.time()
    try:
        with _graph_lock:
            # Get neighbors
            neighbors = list(graph.get_neighbors(node))
            incoming = list(graph.get_neighbors(node, reverse=True))
            original_edges = len(neighbors) + len(incoming)

            # Calculate clustering
            if original_edges > 1:
                possible_connections = original_edges * (original_edges - 1) / 2
                actual_connections = sum(
                    1
                    for n1 in neighbors
                    for n2 in neighbors[neighbors.index(n1) + 1 :]
                    if graph.has_edge(n1, n2) or graph.has_edge(n2, n1)
                )
                clustering = actual_connections / possible_connections
            else:
                clustering = 0

            contracted_count = len(contracted_neighbors.get(node, set()))
            level = node_level.get(node, 0)

            # Modified importance calculation
            importance = (
                3 * shortcut_count  # Reduced weight on shortcuts
                + contracted_count  # Reduced weight on neighborhood
                + level
                + original_edges * 2  # Increased weight on edges
                + clustering
                - len(neighbors) * len(incoming)  # Penalize high path potential
            )

            logger.debug(
                "Node %s importance: shortcuts=%d, contracted=%d, level=%d,"
                "edges=%d, clustering=%.3f, importance=%f",
                node,
                shortcut_count,
                contracted_count,
                level,
                original_edges,
                clustering,
                importance,
            )

            return importance

    finally:
        duration = time.time() - start_time
        _performance.record_metric("importance_calculation_time", duration)


def _witness_search(
    graph_hash: str,
    source: str,
    target: str,
    via: str,
    shortcut_weight: float,
    graph: "Graph",
) -> Tuple[bool, List[str]]:
    cache = get_cache_manager().get_cache("witness_search")
    cache_key = f"{graph_hash}:{shortcut_weight}"

    cached = cache.get(cache_key)
    if cached:
        try:
            return ast.literal_eval(cached)
        except (ValueError, SyntaxError):
            pass

    distances = {source: 0.0}
    pq = [(0.0, source)]
    visited = set()
    paths = {source: [source]}  # Track paths for each node

    while pq:
        dist, node = heappop(pq)
        if node == target:
            result = (dist > shortcut_weight + EPSILON, paths[node])
            cache.set(cache_key, str(result), get_cache_size(graph))
            return result

        if node in visited or node == via:
            continue

        visited.add(node)

        for neighbor in sorted(graph.get_neighbors(node)):
            if neighbor == via:
                continue

            edge = graph.get_edge(node, neighbor)
            if not edge:
                continue

            new_dist = dist + edge.metadata.weight
            if new_dist > shortcut_weight + EPSILON:
                continue

            if neighbor not in distances or new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                paths[neighbor] = paths[node] + [neighbor]
                heappush(pq, (new_dist, neighbor))

    result = (True, paths.get(target, []))
    cache.set(cache_key, str(result), get_cache_size(graph))
    return result


def is_shortcut_necessary(
    source: str,
    target: str,
    via: str,
    shortcut_weight: float,
    graph: "Graph",
    weight_func: Optional[WeightFunc] = None,
) -> Tuple[bool, List[str]]:
    """
    Determine if shortcut is necessary using witness search.
    Uses custom caching to avoid redundant calculations.

    Args:
        source: Source node
        target: Target node
        via: Node being contracted
        shortcut_weight: Weight of potential shortcut
        graph: Graph instance
        weight_func: Optional custom weight function

    Returns:
        True if shortcut is necessary, False if witness path exists

    Raises:
        GraphOperationError: If there's an error during shortcut necessity check
    """
    start_time = time.time()
    try:
        # Compute hash of relevant graph state
        graph_hash = _compute_graph_hash(graph, source, target, via)

        # Use witness search with caching
        with _graph_lock:
            is_necessary = _witness_search(graph_hash, source, target, via, shortcut_weight, graph)

            # Log cache statistics periodically
            cache = get_cache_manager().get_cache("witness_search")
            cache_stats = cache.get_stats()
            if cache_stats["hits"] % 1000 == 0:
                logger.info("Witness search cache stats: %s", cache_stats)

            return is_necessary

    except Exception as e:
        logger.error("Error in shortcut necessity check: %s", e)
        raise GraphOperationError(f"Failed to check shortcut necessity: {str(e)}") from e
    finally:
        duration = time.time() - start_time
        _performance.record_metric("shortcut_necessity_check_time", duration)


def unpack_shortcut(shortcut_path: List[Edge], graph: "Graph") -> List[Edge]:
    """
    Recursively unpack shortcuts into original edges.

    Args:
        shortcut_path: Path potentially containing shortcuts
        graph: Graph instance

    Returns:
        Path with shortcuts expanded to original edges

    Raises:
        GraphOperationError: If shortcut unpacking fails
    """
    start_time = time.time()
    try:
        unpacked_path = []
        for edge in shortcut_path:
            if (
                edge.metadata.custom_attributes.get("is_shortcut")
                and edge.relation_type == SHORTCUT_TYPE
            ):
                # Get via node
                via_node = edge.attributes.get("via_node")
                if via_node is None and edge.context:
                    try:
                        via_node = edge.context.split()[-1]
                    except (AttributeError, IndexError):
                        via_node = None

                if via_node is None:
                    raise GraphOperationError(
                        f"Edge {edge.from_entity}->{edge.to_entity} "
                        f"marked as shortcut but missing via node"
                    )

                # Get component edges with thread safety
                with _graph_lock:
                    lower_edge = graph.get_edge(edge.from_entity, via_node)
                    upper_edge = graph.get_edge(via_node, edge.to_entity)

                if lower_edge and upper_edge:
                    # Recursively unpack component edges
                    unpacked_path.extend(unpack_shortcut([lower_edge, upper_edge], graph))
                else:
                    raise GraphOperationError(
                        f"Failed to unpack shortcut: missing component edges for "
                        f"{edge.from_entity}->{via_node}->{edge.to_entity}"
                    )
            else:
                unpacked_path.append(edge)

        return unpacked_path

    finally:
        duration = time.time() - start_time
        _performance.record_metric("shortcut_unpacking_time", duration)


def validate_shortcuts(shortcuts: Dict[str, Edge], graph: "Graph") -> bool:
    """
    Validate shortcut consistency.

    Args:
        shortcuts: Map of shortcut edges
        graph: Graph instance

    Returns:
        True if shortcuts are valid, False otherwise
    """
    start_time = time.time()
    try:
        with _graph_lock:
            for edge in shortcuts.values():
                if edge.relation_type != SHORTCUT_TYPE:
                    logger.error(
                        "Invalid shortcut %s->%s: wrong relation type %s",
                        edge.from_entity,
                        edge.to_entity,
                        edge.relation_type,
                    )
                    return False

                # Get via node
                via_node = edge.attributes.get("via_node")
                if via_node is None and edge.context:
                    try:
                        via_node = edge.context.split()[-1]
                    except (AttributeError, IndexError):
                        logger.error(
                            "Invalid shortcut %s->%s: missing or invalid via node",
                            edge.from_entity,
                            edge.to_entity,
                        )
                        return False

                # Check component edges exist
                lower_edge = graph.get_edge(edge.from_entity, via_node)
                upper_edge = graph.get_edge(via_node, edge.to_entity)

                if not lower_edge or not upper_edge:
                    logger.error(
                        "Invalid shortcut %s->%s: missing component edges via %s",
                        edge.from_entity,
                        edge.to_entity,
                        via_node,
                    )
                    return False

                # Verify weight consistency
                shortcut_weight = edge.metadata.weight
                actual_weight = lower_edge.metadata.weight + upper_edge.metadata.weight

                if abs(shortcut_weight - actual_weight) > EPSILON:
                    logger.error(
                        "Invalid shortcut %s->%s: weight mismatch %s != %s",
                        edge.from_entity,
                        edge.to_entity,
                        shortcut_weight,
                        actual_weight,
                    )
                    return False

        return True

    finally:
        duration = time.time() - start_time
        _performance.record_metric("shortcut_validation_time", duration)


def get_performance_stats() -> Dict[str, Dict[str, float]]:
    """Get performance statistics for utility functions."""
    return _performance.get_statistics()
