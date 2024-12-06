"""
Utility functions for Contraction Hierarchies algorithm.

This module provides helper functions for node importance calculation,
shortcut necessity checking, and path validation.
"""

from typing import Dict, List, Optional, Set, TYPE_CHECKING
import logging
from functools import lru_cache
from threading import Lock
from hashlib import sha256

from polaris.core.graph.traversal.utils import WeightFunc, get_edge_weight
from polaris.core.models import Edge
from polaris.core.exceptions import GraphOperationError
from .models import SHORTCUT_TYPE

if TYPE_CHECKING:
    from polaris.core.graph import Graph

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

EPSILON = 1e-10  # Consistent epsilon value for floating point comparisons

# Global locks for thread safety
_graph_lock = Lock()
_cache_lock = Lock()


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
    with _graph_lock:
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

        logger.debug(
            f"Node {node} importance calculation:\n"
            f"  Shortcut count: {shortcut_count}\n"
            f"  Contracted neighbors: {contracted_count}\n"
            f"  Level: {level}\n"
            f"  Original edges: {original_edges}\n"
            f"  Final importance: {importance}"
        )

        return importance


@lru_cache(maxsize=10000)
def _cached_witness_search(
    graph_hash: str,
    source: str,
    target: str,
    via: str,
    shortcut_weight: float,
    graph: "Graph",
) -> bool:
    """
    Cached implementation of witness path search.

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

    # Initialize distances and priority queue
    distances = {source: 0.0}
    pq = [(0.0, source)]
    visited = set()

    while pq:
        dist, node = heappop(pq)

        if node == target:
            # Found a witness path that's better than or equal to the shortcut
            logger.debug(
                f"Found path to target with distance {dist} "
                f"(shortcut weight: {shortcut_weight})"
            )
            # Use consistent epsilon comparison
            return dist > shortcut_weight + EPSILON

        if node in visited or node == via:  # Don't go through contracted node
            continue

        visited.add(node)
        logger.debug(f"Visiting node {node} with distance {dist}")

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

            # Use consistent epsilon comparison
            if new_dist > shortcut_weight + EPSILON:
                continue

            if neighbor not in distances or new_dist < distances[neighbor]:
                logger.debug(f"    Updating distance to {neighbor}: {new_dist}")
                distances[neighbor] = new_dist
                heappush(pq, (new_dist, neighbor))

    # No witness path found within weight limit
    logger.debug("No witness path found, shortcut is necessary")
    return True


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
    Uses LRU cache to avoid redundant calculations.

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
    logger.debug(
        f"Checking if shortcut {source}->{target} (via {via}) "
        f"with weight {shortcut_weight} is necessary"
    )

    try:
        # Compute hash of relevant graph state
        graph_hash = _compute_graph_hash(graph, source, target, via)

        # Use cached witness search
        with _cache_lock:
            is_necessary = _cached_witness_search(
                graph_hash, source, target, via, shortcut_weight, graph
            )

            # Log cache statistics periodically
            cache_info = _cached_witness_search.cache_info()
            if cache_info.hits % 1000 == 0:  # Log every 1000 hits
                logger.info(f"Witness search cache stats: {cache_info}")

            return is_necessary

    except Exception as e:
        logger.error(f"Error in shortcut necessity check: {e}")
        # Default to creating shortcut on error
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
    logger.debug("Unpacking shortcuts in path:")
    for edge in shortcut_path:
        logger.debug(f"  {edge.from_entity}->{edge.to_entity} (weight: {edge.metadata.weight})")

    unpacked_path = []
    for edge in shortcut_path:
        # Check if this is actually a shortcut - regular edges should be passed through
        if (
            edge.metadata.custom_attributes.get("is_shortcut")
            and edge.relation_type == SHORTCUT_TYPE
        ):
            # Get via node from attributes first, then context as fallback
            via_node = edge.attributes.get("via_node")
            if via_node is None and edge.context:
                try:
                    via_node = edge.context.split()[-1]
                except (AttributeError, IndexError):
                    via_node = None

            if via_node is None:
                logger.warning(
                    f"Edge {edge.from_entity}->{edge.to_entity} "
                    f"marked as shortcut but missing via node"
                )
                unpacked_path.append(edge)
                continue

            logger.debug(f"Unpacking shortcut {edge.from_entity}->{edge.to_entity} via {via_node}")

            # Get component edges with thread safety
            with _graph_lock:
                lower_edge = graph.get_edge(edge.from_entity, via_node)
                upper_edge = graph.get_edge(via_node, edge.to_entity)

            if lower_edge and upper_edge:
                # Recursively unpack component edges
                logger.debug("  Found component edges, recursively unpacking")
                unpacked_path.extend(unpack_shortcut([lower_edge, upper_edge], graph))
            else:
                # If we can't unpack a marked shortcut, that's an error
                raise GraphOperationError(
                    f"Failed to unpack shortcut: missing component edges for {edge.from_entity}->{via_node}->{edge.to_entity}"
                )
        else:
            # Not a shortcut, add directly to path
            logger.debug(f"Regular edge {edge.from_entity}->{edge.to_entity}, adding to path")
            unpacked_path.append(edge)

    logger.debug("Final unpacked path:")
    for edge in unpacked_path:
        logger.debug(f"  {edge.from_entity}->{edge.to_entity} (weight: {edge.metadata.weight})")

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
    logger.debug("Validating shortcuts...")

    with _graph_lock:
        for edge in shortcuts.values():
            if edge.relation_type != SHORTCUT_TYPE:
                logger.debug(
                    f"Invalid shortcut {edge.from_entity}->{edge.to_entity}: "
                    f"wrong relation type {edge.relation_type}"
                )
                return False

            # Get via node from attributes first, then context as fallback
            via_node = edge.attributes.get("via_node")
            if via_node is None and edge.context:
                try:
                    via_node = edge.context.split()[-1]
                except (AttributeError, IndexError):
                    logger.debug(
                        f"Invalid shortcut {edge.from_entity}->{edge.to_entity}: "
                        f"missing or invalid via node"
                    )
                    return False

            # Check component edges exist
            lower_edge = graph.get_edge(edge.from_entity, via_node)
            upper_edge = graph.get_edge(via_node, edge.to_entity)

            if not lower_edge or not upper_edge:
                logger.debug(
                    f"Invalid shortcut {edge.from_entity}->{edge.to_entity}: "
                    f"missing component edges via {via_node}"
                )
                return False

            # Verify weight consistency
            shortcut_weight = edge.metadata.weight
            actual_weight = lower_edge.metadata.weight + upper_edge.metadata.weight

            if abs(shortcut_weight - actual_weight) > EPSILON:
                logger.debug(
                    f"Invalid shortcut {edge.from_entity}->{edge.to_entity}: "
                    f"weight mismatch {shortcut_weight} != {actual_weight}"
                )
                return False

    logger.debug("All shortcuts valid")
    return True
