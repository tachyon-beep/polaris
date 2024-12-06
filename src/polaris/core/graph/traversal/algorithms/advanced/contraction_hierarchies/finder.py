"""
Path finding implementation for Contraction Hierarchies.

This module handles path finding using the preprocessed contraction hierarchy,
including bidirectional search and path reconstruction.
"""

from typing import Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
import logging
from copy import deepcopy
from heapq import heappop, heappush
from threading import Lock

from polaris.core.exceptions import GraphOperationError
from polaris.core.graph.traversal.path_models import PathResult
from polaris.core.graph.traversal.utils import (
    WeightFunc,
    create_path_result,
    get_edge_weight,
    validate_path,
)
from polaris.core.models import Edge
from .models import ContractionState, Shortcut
from .storage import ContractionStorage
from .utils import unpack_shortcut

if TYPE_CHECKING:
    from polaris.core.graph import Graph

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ContractionPathFinder:
    """
    Handles path finding using Contraction Hierarchies.

    Features:
    - Bidirectional search
    - Path reconstruction
    - Shortcut unpacking
    - Cycle prevention
    - Thread safety
    """

    def __init__(
        self,
        graph: "Graph",
        state: ContractionState,
        storage: ContractionStorage,
    ):
        """
        Initialize path finder.

        Args:
            graph: Graph instance
            state: Preprocessed algorithm state
            storage: Storage manager
        """
        self.graph = graph
        self.state = state
        self.storage = storage
        self._graph_lock = Lock()  # Thread safety for graph operations
        self._forward_visited = set()  # Global visited set for forward search
        self._backward_visited = set()  # Global visited set for backward search
        self._error_messages = {
            "start_not_found": lambda node: f"Start node {node} not found in graph",
            "end_not_found": lambda node: f"End node {node} not found in graph",
            "no_path": lambda start, end: f"No path exists between {start} and {end}",
            "cycle_detected": lambda node: f"Cycle detected through node {node}",
        }

    def _reset_visited_sets(self):
        """Reset visited sets before each path finding operation."""
        self._forward_visited.clear()
        self._backward_visited.clear()

    def find_path(
        self,
        start_node: str,
        end_node: str,
        max_length: Optional[int] = None,
        filter_func: Optional[Callable[[List[Edge]], bool]] = None,
        weight_func: Optional[WeightFunc] = None,
        **kwargs,
    ) -> PathResult:
        """
        Find shortest path using contraction hierarchies.

        Args:
            start_node: Starting node ID
            end_node: Target node ID
            max_length: Maximum path length
            filter_func: Optional path filter
            weight_func: Optional weight function
            **kwargs: Additional options:
                validate: Whether to validate result (default: True)

        Returns:
            PathResult containing shortest path

        Raises:
            GraphOperationError: If no path exists
        """
        # Reset visited sets at start of path finding
        self._reset_visited_sets()

        if start_node == end_node:
            return PathResult(path=[], total_weight=0.0)

        validate = kwargs.get("validate", True)
        debug = kwargs.get("debug", True)  # Enable debug by default

        # Try all possible paths if filter is provided
        if filter_func:
            # First try direct path
            try:
                path = self._find_path(start_node, end_node, weight_func, debug)
                if filter_func(path):
                    result = create_path_result(path, weight_func)
                    if validate:
                        validate_path(path, self.graph, weight_func, max_length)
                    return result
            except GraphOperationError:
                pass

            # Try alternative paths through different nodes
            # Sort nodes for deterministic behavior
            nodes = sorted(self.graph.get_nodes())
            for node in nodes:
                if node == start_node or node == end_node:
                    continue
                try:
                    # Find path through this node
                    path1 = self._find_path(start_node, node, weight_func, debug)
                    path2 = self._find_path(node, end_node, weight_func, debug)
                    path = path1 + path2
                    if filter_func(path):
                        result = create_path_result(path, weight_func)
                        if validate:
                            validate_path(path, self.graph, weight_func, max_length)
                        return result
                except GraphOperationError:
                    continue

            raise GraphOperationError(self._error_messages["no_path"](start_node, end_node))

        # No filter, find shortest path
        path = self._find_path(start_node, end_node, weight_func, debug)
        result = create_path_result(path, weight_func)
        if validate:
            validate_path(
                path, self.graph, weight_func, max_length, allow_cycles=False
            )  # Disallow cycles
        return result

    def _find_path(
        self,
        start_node: str,
        end_node: str,
        weight_func: Optional[WeightFunc],
        debug: bool = False,
    ) -> List[Edge]:
        """Find shortest path without filter."""
        if debug:
            logger.debug(f"Starting bidirectional search from {start_node} to {end_node}")
            logger.debug(f"Node levels: {self.state.node_level}")
            logger.debug(f"Available shortcuts: {self.state.shortcuts}")

        if not self.graph.has_node(start_node):
            raise GraphOperationError(self._error_messages["start_not_found"](start_node))
        if not self.graph.has_node(end_node):
            raise GraphOperationError(self._error_messages["end_not_found"](end_node))

        # Initialize searches
        forward_distances: Dict[str, float] = {start_node: 0.0}
        backward_distances: Dict[str, float] = {end_node: 0.0}

        forward_pq = [(0.0, start_node)]
        backward_pq = [(0.0, end_node)]

        forward_predecessors: Dict[str, Tuple[str, Edge]] = {}
        backward_predecessors: Dict[str, Tuple[str, Edge]] = {}

        # Track best path
        best_dist = float("inf")
        meeting_node = None

        # Track visited nodes for current search
        forward_visited = set()
        backward_visited = set()

        iteration = 0

        try:
            # Bidirectional search
            while forward_pq and backward_pq:  # Changed condition to require both queues
                if iteration > 100000:  # Increased limit
                    logger.error("Maximum iterations reached during path finding")
                    raise GraphOperationError("Path finding exceeded maximum iterations")
                iteration += 1

                if debug and iteration % 1000 == 0:
                    logger.debug(f"Iteration {iteration}")
                    logger.debug(f"Forward queue: {forward_pq}")
                    logger.debug(f"Backward queue: {backward_pq}")
                    logger.debug(f"Best distance so far: {best_dist}")

                # Process both directions in each iteration
                if forward_pq[0][0] < backward_pq[0][0]:
                    best_dist, meeting_node = self._process_forward_search(
                        forward_pq,
                        forward_visited,
                        forward_distances,
                        backward_distances,
                        forward_predecessors,
                        best_dist,
                        meeting_node,
                        weight_func,
                        debug,
                    )
                else:
                    best_dist, meeting_node = self._process_backward_search(
                        backward_pq,
                        backward_visited,
                        backward_distances,
                        forward_distances,
                        backward_predecessors,
                        best_dist,
                        meeting_node,
                        weight_func,
                        debug,
                    )

                # Early termination if both queues' minimum distances sum to more than best_dist
                if forward_pq and backward_pq:
                    min_forward = forward_pq[0][0]
                    min_backward = backward_pq[0][0]
                    if min_forward + min_backward >= best_dist:
                        if debug:
                            logger.debug(
                                f"Early termination: min_forward={min_forward}, "
                                f"min_backward={min_backward}, best_dist={best_dist}"
                            )
                        break

        except MemoryError:
            if debug:
                logger.error("Memory limit exceeded during path finding")
            raise

        if meeting_node is None:
            raise GraphOperationError(self._error_messages["no_path"](start_node, end_node))

        if debug:
            logger.debug(f"Path found with meeting point at {meeting_node}")
            logger.debug(f"Forward distances: {forward_distances}")
            logger.debug(f"Backward distances: {backward_distances}")

        # Reconstruct path
        path = self._reconstruct_path(
            start_node, end_node, meeting_node, forward_predecessors, backward_predecessors
        )

        if debug:
            logger.debug("Final path:")
            for edge in path:
                logger.debug(
                    f"  {edge.from_entity}->{edge.to_entity} (weight: {edge.metadata.weight})"
                )

        return path

    def _expand_search(
        self,
        node: str,
        dist: float,
        distances: Dict[str, float],
        predecessors: Dict[str, Tuple[str, Edge]],
        pq: List[Tuple[float, str]],
        is_forward: bool,
        weight_func: Optional[WeightFunc],
        debug: bool = False,
    ) -> None:
        """Expand search in one direction."""
        if debug:
            direction = "forward" if is_forward else "backward"
            logger.debug(f"Expanding {direction} search from {node}")

        # Use global visited sets for each direction
        visited_set = self._forward_visited if is_forward else self._backward_visited

        # Get edges to process
        edges_to_process = []

        # Get regular edges
        neighbors = self.graph.get_neighbors(node, reverse=not is_forward)
        for neighbor in neighbors:
            if is_forward:
                edge = self.graph.get_edge(node, neighbor)
                if edge:
                    edges_to_process.append((neighbor, edge))
            else:
                edge = self.graph.get_edge(neighbor, node)
                if edge:
                    edges_to_process.append((neighbor, edge))

        # Get shortcuts
        if is_forward:
            for (u, v), shortcut in self.state.shortcuts.items():
                if u == node:
                    edges_to_process.append((v, shortcut.edge))
        else:
            for (u, v), shortcut in self.state.shortcuts.items():
                if v == node:
                    edges_to_process.append((u, shortcut.edge))

        if debug:
            logger.debug(f"Found {len(edges_to_process)} edges to process")

        # Process all edges
        for neighbor, edge in edges_to_process:
            # Enhanced cycle detection using contracted neighbors
            is_shortcut = edge.metadata.custom_attributes.get("is_shortcut", False)
            current_path_set = set(predecessors.keys())

            def _detect_cycle(node: str, path_set: Set[str], visited: Set[str]) -> bool:
                """
                Detect cycles in path while considering contracted neighbors.

                Args:
                    node: Current node to check
                    path_set: Set of nodes in current path
                    visited: Set of already visited nodes in this cycle check

                Returns:
                    bool: True if cycle detected, False otherwise
                """
                if node in path_set:
                    return True

                if node in visited:
                    return False

                visited.add(node)

                # Check contracted neighbors
                contracted = self.state.get_contracted_neighbors(node)
                for neighbor in contracted:
                    if neighbor in path_set or _detect_cycle(neighbor, path_set, visited):
                        return True

                return False

            if _detect_cycle(neighbor, current_path_set, set()):
                if debug:
                    logger.debug(f"Skipping {neighbor} - would create cycle")
                continue

            # Skip if moving to a lower level node (unless it's a shortcut)
            if (
                not is_shortcut
                and neighbor in self.state.node_level
                and node in self.state.node_level
                and self.state.node_level[neighbor] < self.state.node_level[node]
            ):
                if debug:
                    logger.debug(
                        f"Skipping {neighbor} - lower level "
                        f"({self.state.node_level[neighbor]} < {self.state.node_level[node]})"
                    )
                continue

            # Create forward edge if needed
            if not is_forward:
                edge = self._create_forward_edge(edge, edge.to_entity, edge.from_entity)

            edge_weight = get_edge_weight(edge, weight_func)
            new_dist = dist + edge_weight

            if debug:
                logger.debug(f"Edge {edge.from_entity}->{edge.to_entity} weight: {edge_weight}")
                logger.debug(f"New distance to {neighbor}: {new_dist}")

            # Update distance if better path found
            if (
                neighbor not in distances
                or new_dist < distances[neighbor]
                or (
                    new_dist == distances[neighbor]
                    and edge.to_entity < predecessors[neighbor][1].to_entity
                )
            ):
                if debug:
                    logger.debug(f"Updating distance to {neighbor}: {new_dist}")
                distances[neighbor] = new_dist
                predecessors[neighbor] = (node, edge)
                heappush(pq, (new_dist, neighbor))

            # Mark as visited for regular edges
            if not is_shortcut:
                visited_set.add(neighbor)

    def _process_forward_search(
        self,
        forward_pq: List[Tuple[float, str]],
        forward_visited: Set[str],
        forward_distances: Dict[str, float],
        backward_distances: Dict[str, float],
        forward_predecessors: Dict[str, Tuple[str, Edge]],
        best_dist: float,
        meeting_node: Optional[str],
        weight_func: Optional[WeightFunc],
        debug: bool = False,
    ) -> Tuple[float, Optional[str]]:
        """Process one step of forward search."""
        if not forward_pq:
            return best_dist, meeting_node

        dist, node = heappop(forward_pq)
        if node in forward_visited:
            return best_dist, meeting_node

        if debug:
            logger.debug(f"Forward search at node {node} with distance {dist}")

        forward_visited.add(node)

        # Check for meeting point
        if node in backward_distances:
            total_dist = dist + backward_distances[node]
            if total_dist < best_dist:
                best_dist = total_dist
                meeting_node = node
                if debug:
                    logger.debug(f"Found meeting point at {node} with distance {total_dist}")

        # Only expand if we haven't found better path
        if dist <= best_dist:
            self._expand_search(
                node,
                dist,
                forward_distances,
                forward_predecessors,
                forward_pq,
                True,
                weight_func,
                debug,
            )

        return best_dist, meeting_node

    def _process_backward_search(
        self,
        backward_pq: List[Tuple[float, str]],
        backward_visited: Set[str],
        backward_distances: Dict[str, float],
        forward_distances: Dict[str, float],
        backward_predecessors: Dict[str, Tuple[str, Edge]],
        best_dist: float,
        meeting_node: Optional[str],
        weight_func: Optional[WeightFunc],
        debug: bool = False,
    ) -> Tuple[float, Optional[str]]:
        """Process one step of backward search."""
        if not backward_pq:
            return best_dist, meeting_node

        dist, node = heappop(backward_pq)
        if node in backward_visited:
            return best_dist, meeting_node

        if debug:
            logger.debug(f"Backward search at node {node} with distance {dist}")

        backward_visited.add(node)

        # Check for meeting point
        if node in forward_distances:
            total_dist = dist + forward_distances[node]
            if total_dist < best_dist:
                best_dist = total_dist
                meeting_node = node
                if debug:
                    logger.debug(f"Found meeting point at {node} with distance {total_dist}")

        # Only expand if we haven't found better path
        if dist <= best_dist:
            self._expand_search(
                node,
                dist,
                backward_distances,
                backward_predecessors,
                backward_pq,
                False,
                weight_func,
                debug,
            )

        return best_dist, meeting_node

    def _create_forward_edge(self, edge: Edge, from_node: str, to_node: str) -> Edge:
        """Create a forward edge from edge information with proper metadata copying."""
        return Edge(
            from_entity=from_node,
            to_entity=to_node,
            relation_type=edge.relation_type,
            metadata=deepcopy(edge.metadata),
            impact_score=edge.impact_score,
            attributes=edge.attributes.copy(),
            context=edge.context,
            validation_status=edge.validation_status,
            custom_metrics=edge.custom_metrics.copy(),
        )

    def _validate_edge_connection(self, edge1: Edge, edge2: Edge) -> bool:
        """Validate connection between two edges."""
        return edge1.to_entity == edge2.from_entity

    def _validate_path_continuity(self, path: List[Edge]) -> None:
        """Validate entire path continuity."""
        if not path:
            return

        for i in range(len(path) - 1):
            if not self._validate_edge_connection(path[i], path[i + 1]):
                raise GraphOperationError(
                    f"Path discontinuity between edges {i} and {i+1}: "
                    f"{path[i].to_entity} != {path[i+1].from_entity}"
                )

    def _reconstruct_path(
        self,
        start_node: str,
        end_node: str,
        meeting_node: str,
        forward_predecessors: Dict[str, Tuple[str, Edge]],
        backward_predecessors: Dict[str, Tuple[str, Edge]],
    ) -> List[Edge]:
        """Reconstruct path from predecessor maps."""
        path = []

        # Build forward path from start to meeting point
        current = meeting_node
        while current != start_node:
            if current not in forward_predecessors:
                raise GraphOperationError(
                    f"Failed to reconstruct path: missing predecessor for {current} in forward search"
                )
            prev, edge = forward_predecessors[current]
            path.insert(0, edge)  # Insert at beginning to maintain order
            current = prev

        # Build backward path from meeting point to end
        current = meeting_node
        while current != end_node:
            if current not in backward_predecessors:
                raise GraphOperationError(
                    f"Failed to reconstruct path: missing predecessor for {current} in backward search"
                )
            next_node, edge = backward_predecessors[current]

            # Create forward edge from backward edge
            forward_edge = self._create_forward_edge(
                edge=edge, from_node=current, to_node=next_node
            )
            path.append(forward_edge)
            current = next_node

        # Validate path continuity before shortcut unpacking
        self._validate_path_continuity(path)

        # Use the imported unpack_shortcut function
        return unpack_shortcut(path, self.graph)
