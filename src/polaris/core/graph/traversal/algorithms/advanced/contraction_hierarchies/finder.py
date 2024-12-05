"""
Path finding implementation for Contraction Hierarchies.

This module handles path finding using the preprocessed contraction hierarchy,
including bidirectional search and path reconstruction.
"""

from heapq import heappop, heappush
from typing import Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from polaris.core.exceptions import GraphOperationError
from polaris.core.graph.traversal.path_models import PathResult
from polaris.core.graph.traversal.utils import (
    WeightFunc,
    create_path_result,
    get_edge_weight,
    validate_path,
)
from polaris.core.models import Edge
from .models import ContractionState
from .storage import ContractionStorage
from .utils import unpack_shortcut

if TYPE_CHECKING:
    from polaris.core.graph import Graph


class ContractionPathFinder:
    """
    Handles path finding using Contraction Hierarchies.

    Features:
    - Bidirectional search
    - Path reconstruction
    - Shortcut unpacking
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
        validate = kwargs.get("validate", True)

        # Try all possible paths if filter is provided
        if filter_func:
            # First try direct path
            try:
                path = self._find_path(start_node, end_node, weight_func)
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
                    path1 = self._find_path(start_node, node, weight_func)
                    path2 = self._find_path(node, end_node, weight_func)
                    path = path1 + path2
                    if filter_func(path):
                        result = create_path_result(path, weight_func)
                        if validate:
                            validate_path(path, self.graph, weight_func, max_length)
                        return result
                except GraphOperationError:
                    continue

            raise GraphOperationError(
                f"No path satisfying filter exists between {start_node} and {end_node}"
            )

        # No filter, find shortest path
        path = self._find_path(start_node, end_node, weight_func)
        result = create_path_result(path, weight_func)
        if validate:
            validate_path(path, self.graph, weight_func, max_length)
        return result

    def _find_path(
        self,
        start_node: str,
        end_node: str,
        weight_func: Optional[WeightFunc],
    ) -> List[Edge]:
        """
        Find shortest path without filter.

        Args:
            start_node: Starting node
            end_node: Target node
            weight_func: Optional weight function

        Returns:
            List of edges forming shortest path

        Raises:
            GraphOperationError: If no path exists
        """
        # Initialize forward and backward searches
        forward_distances: Dict[str, float] = {start_node: 0.0}
        backward_distances: Dict[str, float] = {end_node: 0.0}

        forward_pq = [(0.0, start_node)]
        backward_pq = [(0.0, end_node)]

        forward_predecessors: Dict[str, Tuple[str, Edge]] = {}
        backward_predecessors: Dict[str, Tuple[str, Edge]] = {}

        # Track best path
        best_dist = float("inf")
        meeting_node = None

        # Track visited nodes
        forward_visited = set()
        backward_visited = set()

        # Bidirectional search
        while forward_pq or backward_pq:
            self.storage.check_memory()

            # Forward search
            if forward_pq:
                dist, node = heappop(forward_pq)
                if node in forward_visited:
                    continue

                forward_visited.add(node)

                if node in backward_distances:
                    total_dist = dist + backward_distances[node]
                    if total_dist < best_dist:
                        best_dist = total_dist
                        meeting_node = node

                if dist <= best_dist:
                    self._expand_search(
                        node,
                        dist,
                        forward_distances,
                        forward_predecessors,
                        forward_pq,
                        True,
                        weight_func,
                    )

            # Backward search
            if backward_pq:
                dist, node = heappop(backward_pq)
                if node in backward_visited:
                    continue

                backward_visited.add(node)

                if node in forward_distances:
                    total_dist = dist + forward_distances[node]
                    if total_dist < best_dist:
                        best_dist = total_dist
                        meeting_node = node

                if dist <= best_dist:
                    self._expand_search(
                        node,
                        dist,
                        backward_distances,
                        backward_predecessors,
                        backward_pq,
                        False,
                        weight_func,
                    )

            if (not forward_pq or forward_pq[0][0] >= best_dist) and (
                not backward_pq or backward_pq[0][0] >= best_dist
            ):
                break

        if meeting_node is None:
            raise GraphOperationError(f"No path exists between {start_node} and {end_node}")

        # Reconstruct path
        path = self._reconstruct_path(
            start_node, end_node, meeting_node, forward_predecessors, backward_predecessors
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
    ) -> None:
        """
        Expand search in one direction.

        Args:
            node: Current node
            dist: Distance to current node
            distances: Distance map
            predecessors: Predecessor map
            pq: Priority queue
            is_forward: Whether this is forward search
            weight_func: Optional weight function
        """
        # Get neighbors including shortcuts
        neighbors = set()
        # Get regular neighbors
        neighbors.update(self.graph.get_neighbors(node, reverse=not is_forward))
        # Get shortcut neighbors
        if is_forward:
            neighbors.update(v for (u, v), _ in self.state.shortcuts.items() if u == node)
        else:
            neighbors.update(u for (u, v), _ in self.state.shortcuts.items() if v == node)

        # Sort neighbors for deterministic behavior
        neighbors = sorted(neighbors)

        # Process all edges (regular and shortcuts)
        for neighbor in neighbors:
            # Only expand to nodes of higher or equal level in the hierarchy
            if (
                neighbor in self.state.node_level
                and self.state.node_level[neighbor] < self.state.node_level[node]
            ):
                continue

            # Try regular edge first
            edge = None
            if is_forward:
                edge = self.graph.get_edge(node, neighbor)
            else:
                edge = self.graph.get_edge(neighbor, node)
                if edge:
                    # For backward search, create forward edge
                    edge = Edge(
                        from_entity=edge.to_entity,
                        to_entity=edge.from_entity,
                        relation_type=edge.relation_type,
                        metadata=edge.metadata,
                        impact_score=edge.impact_score,
                        context=edge.context,
                    )

            # If no regular edge, try shortcut
            if not edge:
                if is_forward:
                    shortcut = self.state.get_shortcut(node, neighbor)
                    if shortcut:
                        edge = shortcut.edge
                else:
                    shortcut = self.state.get_shortcut(neighbor, node)
                    if shortcut:
                        # Create forward edge from backward shortcut
                        edge = Edge(
                            from_entity=shortcut.edge.to_entity,
                            to_entity=shortcut.edge.from_entity,
                            relation_type=shortcut.edge.relation_type,
                            metadata=shortcut.edge.metadata,
                            impact_score=shortcut.edge.impact_score,
                            context=shortcut.edge.context,
                        )

            if edge:
                edge_weight = get_edge_weight(edge, weight_func)
                new_dist = dist + edge_weight

                # For equal distances, prefer lexicographically smaller paths
                if (
                    neighbor not in distances
                    or new_dist < distances[neighbor]
                    or (
                        new_dist == distances[neighbor]
                        and edge.to_entity < predecessors[neighbor][1].to_entity
                    )
                ):
                    distances[neighbor] = new_dist
                    predecessors[neighbor] = (node, edge)
                    heappush(pq, (new_dist, neighbor))

    def _reconstruct_path(
        self,
        start_node: str,
        end_node: str,
        meeting_node: str,
        forward_predecessors: Dict[str, Tuple[str, Edge]],
        backward_predecessors: Dict[str, Tuple[str, Edge]],
    ) -> List[Edge]:
        """
        Reconstruct path from predecessor maps.

        Args:
            start_node: Starting node
            end_node: Target node
            meeting_node: Node where searches met
            forward_predecessors: Forward search predecessors
            backward_predecessors: Backward search predecessors

        Returns:
            Complete path from start to end
        """
        # Reconstruct forward path
        forward_path = []
        current = meeting_node
        while current in forward_predecessors:
            prev, edge = forward_predecessors[current]
            forward_path.append(edge)
            current = prev

        # Reconstruct backward path
        backward_path = []
        current = meeting_node
        while current in backward_predecessors:
            next_node, edge = backward_predecessors[current]
            backward_path.append(edge)
            current = next_node

        # Combine paths
        forward_path.reverse()
        path = forward_path + backward_path

        # Unpack shortcuts
        return unpack_shortcut(path, self.graph)
