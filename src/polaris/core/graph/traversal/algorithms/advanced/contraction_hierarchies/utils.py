"""
Utility functions for Contraction Hierarchies algorithm.

This module provides helper functions for node importance calculation,
shortcut necessity checking, and path validation.
"""

from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING, Callable, Any
import logging
import statistics
import threading
import time
from functools import lru_cache, wraps
from hashlib import sha256
from collections import defaultdict

from polaris.core.graph.traversal.utils import WeightFunc, get_edge_weight
from polaris.core.models import Edge
from polaris.core.exceptions import GraphOperationError
from .models import SHORTCUT_TYPE

if TYPE_CHECKING:
    from polaris.core.graph import Graph

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


def get_cache_size(graph: "Graph") -> int:
    """Calculate appropriate cache size based on graph size."""
    node_count = len(list(graph.get_nodes()))
    # Base size of 10000, scaled by number of nodes
    return max(10000, min(100000, node_count * 100))


class DynamicLRUCache:
    """Thread-safe LRU cache with dynamic sizing."""

    def __init__(self):
        self._cache = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key in self._cache:
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def set(self, key: str, value: Any, max_size: int):
        """Set value in cache with size limit."""
        with self._lock:
            self._cache[key] = value
            if len(self._cache) > max_size:
                # Remove oldest entry
                self._cache.pop(next(iter(self._cache)))

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self._lock:
            return {"hits": self._hits, "misses": self._misses, "size": len(self._cache)}


# Global cache instance
_witness_cache = DynamicLRUCache()


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
            # Get original edges and neighbors
            neighbors = list(graph.get_neighbors(node))
            original_edges = len(neighbors)

            # Calculate local clustering coefficient
            if original_edges > 1:
                possible_connections = original_edges * (original_edges - 1) / 2
                actual_connections = 0
                for i, n1 in enumerate(neighbors):
                    for n2 in neighbors[i + 1 :]:
                        if graph.has_edge(n1, n2) or graph.has_edge(n2, n1):
                            actual_connections += 1
                clustering = actual_connections / possible_connections
            else:
                clustering = 0

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
                + 3 * clustering  # Consider local structure
            )

            logger.debug(
                f"Node {node} importance calculation:\n"
                f"  Shortcut count: {shortcut_count}\n"
                f"  Contracted neighbors: {contracted_count}\n"
                f"  Level: {level}\n"
                f"  Original edges: {original_edges}\n"
                f"  Clustering coefficient: {clustering:.3f}\n"
                f"  Final importance: {importance}"
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
) -> bool:
    """
    Implementation of witness path search.

    Args:
        graph_hash: Hash of relevant graph state
        source: Source node
        target: Target node
        via: Node being contracted
        shortcut_weight: Weight of potential shortcut
        graph: Graph instance

    Returns:
        True if shortcut is necessary, False if witness path exists
    """
    from heapq import heappop, heappush

    start_time = time.time()
    try:
        # Check cache first
        cache_key = f"{graph_hash}:{shortcut_weight}"
        cached_result = _witness_cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        # Initialize distances and priority queue
        distances = {source: 0.0}
        pq = [(0.0, source)]
        visited = set()

        while pq:
            dist, node = heappop(pq)

            if node == target:
                # Found a witness path that's better than or equal to the shortcut
                result = dist > shortcut_weight + EPSILON
                _witness_cache.set(cache_key, result, get_cache_size(graph))
                return result

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

                edge_weight = edge.metadata.weight  # Use raw weight for caching
                new_dist = dist + edge_weight

                if new_dist > shortcut_weight + EPSILON:
                    continue

                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    heappush(pq, (new_dist, neighbor))

        # No witness path found within weight limit
        result = True
        _witness_cache.set(cache_key, result, get_cache_size(graph))
        return result

    finally:
        duration = time.time() - start_time
        _performance.record_metric("witness_search_time", duration)


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
            cache_stats = _witness_cache.get_stats()
            if cache_stats["hits"] % 1000 == 0:
                logger.info(f"Witness search cache stats: {cache_stats}")

            return is_necessary

    except Exception as e:
        logger.error(f"Error in shortcut necessity check: {str(e)}")
        raise GraphOperationError(f"Failed to check shortcut necessity: {str(e)}")

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
                        f"Invalid shortcut {edge.from_entity}->{edge.to_entity}: "
                        f"wrong relation type {edge.relation_type}"
                    )
                    return False

                # Get via node
                via_node = edge.attributes.get("via_node")
                if via_node is None and edge.context:
                    try:
                        via_node = edge.context.split()[-1]
                    except (AttributeError, IndexError):
                        logger.error(
                            f"Invalid shortcut {edge.from_entity}->{edge.to_entity}: "
                            f"missing or invalid via node"
                        )
                        return False

                # Check component edges exist
                lower_edge = graph.get_edge(edge.from_entity, via_node)
                upper_edge = graph.get_edge(via_node, edge.to_entity)

                if not lower_edge or not upper_edge:
                    logger.error(
                        f"Invalid shortcut {edge.from_entity}->{edge.to_entity}: "
                        f"missing component edges via {via_node}"
                    )
                    return False

                # Verify weight consistency
                shortcut_weight = edge.metadata.weight
                actual_weight = lower_edge.metadata.weight + upper_edge.metadata.weight

                if abs(shortcut_weight - actual_weight) > EPSILON:
                    logger.error(
                        f"Invalid shortcut {edge.from_entity}->{edge.to_entity}: "
                        f"weight mismatch {shortcut_weight} != {actual_weight}"
                    )
                    return False

        return True

    finally:
        duration = time.time() - start_time
        _performance.record_metric("shortcut_validation_time", duration)


def get_performance_stats() -> Dict[str, Dict[str, float]]:
    """Get performance statistics for utility functions."""
    return _performance.get_statistics()
