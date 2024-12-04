"""
Transit Node Routing implementation for fast long-distance queries.

This module provides an implementation of Transit Node Routing, a preprocessing-based
technique that is particularly effective for road networks. Key features:
- Access node computation
- Transit node selection
- Distance table computation
- Locality filter for short-distance queries
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Callable
from heapq import heappush, heappop

from polaris.core.exceptions import GraphOperationError
from polaris.core.models import Edge
from polaris.core.graph import Graph
from polaris.core.graph_paths.base import PathFinder
from polaris.core.graph_paths.models import PathResult
from polaris.core.graph_paths.utils import (
    WeightFunc,
    MemoryManager,
    get_edge_weight,
    create_path_result,
    validate_path,
)


@dataclass(frozen=True)
class AccessNode:
    """
    Access node information for a vertex.

    Attributes:
        node: Transit node ID
        distance: Distance to/from transit node
        first_hop: First edge in path to transit node
    """

    node: str
    distance: float
    first_hop: Optional[Edge]


class TransitNodeRouting(PathFinder):
    """
    Transit Node Routing implementation for fast path queries.

    Features:
    - Efficient transit node selection
    - Access node computation
    - Distance table for fast queries
    - Locality filter for short paths
    """

    def __init__(
        self, graph: Graph, num_transit_nodes: int = 1000, max_memory_mb: Optional[float] = None
    ):
        """
        Initialize with graph and parameters.

        Args:
            graph: Input graph
            num_transit_nodes: Number of transit nodes to select
            max_memory_mb: Optional memory limit in MB
        """
        super().__init__(graph)
        self.memory_manager = MemoryManager(max_memory_mb)
        self.num_transit_nodes = num_transit_nodes
        self.transit_nodes: Set[str] = set()
        self.forward_access: Dict[str, List[AccessNode]] = {}
        self.backward_access: Dict[str, List[AccessNode]] = {}
        self.distances: Dict[Tuple[str, str], float] = {}
        self._preprocessed = False

    def preprocess(self) -> None:
        """
        Preprocess graph to identify transit nodes and compute distances.

        This performs three main steps:
        1. Select transit nodes based on importance
        2. Compute access nodes for each vertex
        3. Compute distances between transit nodes
        """
        print("Starting preprocessing...")

        # Select transit nodes
        print("Selecting transit nodes...")
        self.transit_nodes = self._select_transit_nodes()
        print(f"Selected {len(self.transit_nodes)} transit nodes")

        # Compute access nodes
        print("Computing access nodes...")
        total_nodes = len(self.graph.get_nodes())
        for i, node in enumerate(self.graph.get_nodes()):
            self.memory_manager.check_memory()

            # Show progress
            if (i + 1) % 100 == 0 or i + 1 == total_nodes:
                progress = ((i + 1) / total_nodes) * 100
                print(f"Computing access nodes: {progress:.1f}% complete")

            # Compute forward and backward access nodes
            self._compute_access_nodes(node)

        # Compute distances between transit nodes
        print("Computing transit node distances...")
        self._compute_transit_distances()

        self._preprocessed = True
        print("Preprocessing complete")
        print(f"Average access nodes per vertex: {self._get_average_access_nodes():.1f}")

    def _select_transit_nodes(self) -> Set[str]:
        """
        Select transit nodes using importance criteria.

        Returns nodes with highest importance based on:
        - Degree centrality
        - Betweenness estimate
        - Geographic coverage (if available)
        """
        # Calculate node importance based on degree
        importance = {}
        for node in self.graph.get_nodes():
            out_degree = len(list(self.graph.get_neighbors(node)))
            in_degree = len(list(self.graph.get_neighbors(node, reverse=True)))
            importance[node] = out_degree + in_degree

        # Select top nodes by importance
        nodes = sorted(self.graph.get_nodes(), key=lambda n: importance[n], reverse=True)
        return set(nodes[: self.num_transit_nodes])

    def _compute_access_nodes(self, node: str) -> None:
        """
        Compute forward and backward access nodes for a vertex.

        Args:
            node: Vertex to compute access nodes for
        """
        # Compute forward access nodes
        self.forward_access[node] = []
        distances = {node: 0.0}
        first_hops: Dict[str, Edge] = {}
        pq = [(0.0, node)]

        while pq:
            dist, current = heappop(pq)

            # Stop if we're too far
            if dist > self._compute_distance_threshold(node):
                break

            # Add if transit node
            if current in self.transit_nodes:
                self.forward_access[node].append(
                    AccessNode(node=current, distance=dist, first_hop=first_hops.get(current))
                )
                continue

            # Explore neighbors
            for neighbor in self.graph.get_neighbors(current):
                edge = self.graph.get_edge(current, neighbor)
                if not edge:
                    continue

                new_dist = dist + get_edge_weight(edge)
                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    if current == node:
                        first_hops[neighbor] = edge
                    elif current in first_hops:
                        first_hops[neighbor] = first_hops[current]
                    heappush(pq, (new_dist, neighbor))

        # Compute backward access nodes
        self.backward_access[node] = []
        distances = {node: 0.0}
        first_hops = {}
        pq = [(0.0, node)]

        while pq:
            dist, current = heappop(pq)

            # Stop if we're too far
            if dist > self._compute_distance_threshold(node):
                break

            # Add if transit node
            if current in self.transit_nodes:
                self.backward_access[node].append(
                    AccessNode(node=current, distance=dist, first_hop=first_hops.get(current))
                )
                continue

            # Explore neighbors
            for neighbor in self.graph.get_neighbors(current, reverse=True):
                edge = self.graph.get_edge(neighbor, current)
                if not edge:
                    continue

                new_dist = dist + get_edge_weight(edge)
                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    if current == node:
                        first_hops[neighbor] = edge
                    elif current in first_hops:
                        first_hops[neighbor] = first_hops[current]
                    heappush(pq, (new_dist, neighbor))

    def _compute_transit_distances(self) -> None:
        """Compute distances between all pairs of transit nodes."""
        total = len(self.transit_nodes)
        processed = 0

        for source in self.transit_nodes:
            self.memory_manager.check_memory()

            # Compute distances from source
            distances = {source: 0.0}
            pq = [(0.0, source)]

            while pq:
                dist, current = heappop(pq)

                for neighbor in self.graph.get_neighbors(current):
                    edge = self.graph.get_edge(current, neighbor)
                    if not edge:
                        continue

                    new_dist = dist + get_edge_weight(edge)
                    if neighbor not in distances or new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        heappush(pq, (new_dist, neighbor))

            # Store distances to other transit nodes
            for target in self.transit_nodes:
                if target in distances:
                    self.distances[(source, target)] = distances[target]

            # Show progress
            processed += 1
            if processed % 10 == 0 or processed == total:
                progress = (processed / total) * 100
                print(f"Computing transit distances: {progress:.1f}% complete")

    def _compute_distance_threshold(self, node: str) -> float:
        """
        Compute locality radius for access node computation.

        This determines how far to search for access nodes.
        Could be improved with better locality criteria.
        """
        # Simple fixed threshold - could be improved
        return 100.0

    def _get_average_access_nodes(self) -> float:
        """Calculate average number of access nodes per vertex."""
        total_access = sum(len(nodes) for nodes in self.forward_access.values()) + sum(
            len(nodes) for nodes in self.backward_access.values()
        )
        return total_access / (2 * len(self.graph.get_nodes()))

    def find_path(
        self,
        start_node: str,
        end_node: str,
        max_length: Optional[int] = None,
        max_paths: Optional[int] = None,
        filter_func: Optional[Callable[[List[Edge]], bool]] = None,
        weight_func: Optional[WeightFunc] = None,
        **kwargs,
    ) -> PathResult:
        """
        Find shortest path using transit node routing.

        Args:
            start_node: Starting node ID
            end_node: Target node ID
            max_length: Maximum path length
            max_paths: Not used (always returns single path)
            filter_func: Optional path filter
            weight_func: Optional weight function
            **kwargs: Additional options:
                validate: Whether to validate result (default: True)

        Returns:
            PathResult containing shortest path

        Raises:
            GraphOperationError: If no path exists or graph not preprocessed
        """
        if not self._preprocessed:
            raise GraphOperationError("Graph must be preprocessed before finding paths")

        validate = kwargs.get("validate", True)

        # Check if local query
        if self._is_local_query(start_node, end_node):
            return self._compute_local_path(
                start_node, end_node, max_length, filter_func, weight_func
            )

        # Get access nodes
        if start_node not in self.forward_access:
            raise GraphOperationError(f"No access nodes for start node {start_node}")
        if end_node not in self.backward_access:
            raise GraphOperationError(f"No access nodes for end node {end_node}")

        # Find best transit node pair
        best_dist = float("inf")
        best_path = None

        for s_access in self.forward_access[start_node]:
            for t_access in self.backward_access[end_node]:
                if (s_access.node, t_access.node) not in self.distances:
                    continue

                transit_dist = self.distances[(s_access.node, t_access.node)]
                total_dist = s_access.distance + transit_dist + t_access.distance

                if total_dist < best_dist:
                    # Reconstruct path
                    path = []

                    # Path to first transit node
                    current = start_node
                    while current != s_access.node:
                        edge = next(
                            a.first_hop
                            for a in self.forward_access[current]
                            if a.node == s_access.node
                        )
                        if not edge:
                            break
                        path.append(edge)
                        current = edge.to_entity

                    # Path between transit nodes
                    transit_path = self._compute_local_path(
                        s_access.node, t_access.node, None, None, weight_func
                    )
                    path.extend(transit_path.path)

                    # Path from last transit node
                    current = end_node
                    backward_path = []
                    while current != t_access.node:
                        edge = next(
                            a.first_hop
                            for a in self.backward_access[current]
                            if a.node == t_access.node
                        )
                        if not edge:
                            break
                        backward_path.append(edge)
                        current = edge.from_entity
                    path.extend(reversed(backward_path))

                    best_dist = total_dist
                    best_path = path

        if best_path is None:
            raise GraphOperationError(f"No path exists between {start_node} and {end_node}")

        # Apply filter if provided
        if filter_func and not filter_func(best_path):
            raise GraphOperationError(
                f"No path satisfying filter exists between {start_node} and {end_node}"
            )

        # Create and validate result
        result = create_path_result(best_path, weight_func)
        if validate:
            validate_path(best_path, self.graph, weight_func, max_length)

        return result

    def _is_local_query(self, source: str, target: str) -> bool:
        """
        Determine if query should be handled locally.

        Returns True if nodes share access nodes, indicating
        they are close together.
        """
        if source not in self.forward_access or target not in self.backward_access:
            return False

        source_access = {a.node for a in self.forward_access[source]}
        target_access = {a.node for a in self.backward_access[target]}
        return bool(source_access & target_access)

    def _compute_local_path(
        self,
        start_node: str,
        end_node: str,
        max_length: Optional[int],
        filter_func: Optional[Callable[[List[Edge]], bool]],
        weight_func: Optional[WeightFunc],
    ) -> PathResult:
        """
        Compute path for local query using Dijkstra's algorithm.

        Args:
            start_node: Starting node
            end_node: Target node
            max_length: Maximum path length
            filter_func: Optional path filter
            weight_func: Optional weight function

        Returns:
            PathResult for local path

        Raises:
            GraphOperationError: If no path exists
        """
        distances = {start_node: 0.0}
        predecessors: Dict[str, Tuple[str, Edge]] = {}
        pq = [(0.0, start_node)]

        while pq:
            dist, current = heappop(pq)

            if current == end_node:
                break

            for neighbor in self.graph.get_neighbors(current):
                edge = self.graph.get_edge(current, neighbor)
                if not edge:
                    continue

                new_dist = dist + get_edge_weight(edge, weight_func)
                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    predecessors[neighbor] = (current, edge)
                    heappush(pq, (new_dist, neighbor))

        if end_node not in predecessors:
            raise GraphOperationError(f"No local path exists between {start_node} and {end_node}")

        # Reconstruct path
        path = []
        current = end_node
        while current in predecessors:
            prev, edge = predecessors[current]
            path.append(edge)
            current = prev
        path.reverse()

        # Apply filter if provided
        if filter_func and not filter_func(path):
            raise GraphOperationError(
                f"No path satisfying filter exists between {start_node} and {end_node}"
            )

        # Create and validate result
        result = create_path_result(path, weight_func)
        if max_length is not None:
            validate_path(path, self.graph, weight_func, max_length)

        return result
