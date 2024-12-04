"""
Path finding algorithms for the knowledge graph.

This module provides implementations of various path finding algorithms for traversing
and analyzing paths between nodes in the knowledge graph. It includes:
- Dijkstra's algorithm for finding shortest paths
- Depth-first search for finding all possible paths
- Support for custom weight functions and path filtering
- Path validation and optimization

The path finding capabilities are essential for:
- Analyzing relationships between nodes
- Finding optimal routes through the graph
- Understanding node connections and dependencies
- Discovering indirect relationships
"""

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from heapq import heappop, heappush
from typing import Callable, Dict, Iterator, List, Optional, Set, Tuple, TypeVar, Union

from .exceptions import GraphOperationError, NodeNotFoundError
from .graph import Graph
from .models import Edge

# Type aliases for improved readability and type safety
NodeID = str
Weight = float
Distance = float
PathFilter = Callable[[List[Edge]], bool]
WeightFunc = Callable[[Edge], Weight]

# Generic type for flexible return types
T = TypeVar("T")

# Constants for path finding
DEFAULT_MAX_PATH_LENGTH = 50  # Default limit to prevent unbounded recursion
DEFAULT_MAX_PATHS = 1000  # Default limit for number of paths to return


class PathValidationError(Exception):
    """Raised when a path fails validation checks."""

    pass


class PathType(Enum):
    """Types of paths that can be found in the graph."""

    SHORTEST = "shortest"  # Shortest path by edge count or weight
    ALL = "all"  # All possible paths
    FILTERED = "filtered"  # Paths meeting specific criteria


@dataclass
class PathResult:
    """
    Container for path finding results.

    Attributes:
        path (List[Edge]): Sequence of edges forming the path
        total_weight (float): Total weight of the path
        length (int): Number of edges in the path
    """

    path: List[Edge]
    total_weight: float
    length: int

    def __len__(self) -> int:
        """Return the number of edges in the path."""
        return self.length

    def __getitem__(self, index: int) -> Edge:
        """Get an edge from the path by index."""
        return self.path[index]

    def __iter__(self):
        """Return an iterator over the path edges."""
        return iter(self.path)

    @property
    def nodes(self) -> List[str]:
        """Get the sequence of node IDs in the path."""
        if not self.path:
            return []
        result = [self.path[0].from_entity]
        result.extend(edge.to_entity for edge in self.path)
        return result

    def validate(self) -> None:
        """
        Validate the path's consistency.

        Raises:
            PathValidationError: If the path is invalid
        """
        if not self.path:
            return

        # Check node connectivity
        for i in range(len(self.path) - 1):
            if self.path[i].to_entity != self.path[i + 1].from_entity:
                raise PathValidationError(f"Path discontinuity between edges {i} and {i+1}")

        # Verify length
        if len(self.path) != self.length:
            raise PathValidationError(
                f"Path length mismatch: {len(self.path)} edges but length is {self.length}"
            )


class PathFinding:
    """
    Path finding algorithms for knowledge graph traversal.

    This class provides static methods implementing various path finding algorithms
    for discovering and analyzing paths between nodes in the knowledge graph.
    The algorithms support customization through weight functions and filters.
    """

    @staticmethod
    def _validate_nodes(graph: Graph, start_node: NodeID, end_node: NodeID) -> None:
        """
        Validate that both nodes exist in the graph.

        Args:
            graph (Graph): The graph instance
            start_node (NodeID): Starting node ID
            end_node (NodeID): Target node ID

        Raises:
            NodeNotFoundError: If either node doesn't exist
        """
        if not graph.has_node(start_node):
            raise NodeNotFoundError(f"Start node '{start_node}' not found in the graph")
        if not graph.has_node(end_node):
            raise NodeNotFoundError(f"End node '{end_node}' not found in the graph")

    @staticmethod
    def _get_edge_weight(edge: Edge, weight_func: Optional[WeightFunc]) -> Weight:
        """
        Calculate the weight of an edge using the provided weight function.

        Args:
            edge (Edge): The edge to calculate weight for
            weight_func (Optional[WeightFunc]): Optional weight function

        Returns:
            Weight: The calculated edge weight, always positive

        Raises:
            ValueError: If the weight function returns zero or negative value
        """
        weight = weight_func(edge) if weight_func is not None else 1.0
        if weight <= 0:
            raise ValueError(
                f"Edge weight must be positive. Got {weight} for "
                "edge {edge.from_entity}->{edge.to_entity}"
            )
        return weight

    @staticmethod
    def _calculate_path_weight(path: List[Edge], weight_func: Optional[WeightFunc]) -> float:
        """
        Calculate the total weight of a path.

        Args:
            path (List[Edge]): The path to calculate weight for
            weight_func (Optional[WeightFunc]): Optional weight function

        Returns:
            float: Total path weight
        """
        return sum(PathFinding._get_edge_weight(edge, weight_func) for edge in path)

    @staticmethod
    def _create_path_result(path: List[Edge], weight_func: Optional[WeightFunc]) -> PathResult:
        """
        Create a PathResult object from a path.

        Args:
            path (List[Edge]): The path to create result for
            weight_func (Optional[WeightFunc]): Optional weight function

        Returns:
            PathResult: Path result object
        """
        total_weight = PathFinding._calculate_path_weight(path, weight_func)
        return PathResult(path=path, total_weight=total_weight, length=len(path))

    @staticmethod
    def shortest_path(
        graph: Graph,
        start_node: NodeID,
        end_node: NodeID,
        weight_func: Optional[WeightFunc] = None,
    ) -> PathResult:
        """
        Find shortest path between two nodes using Dijkstra's algorithm.

        Implements Dijkstra's algorithm to find the shortest path between two
        nodes, optionally using a custom weight function for edge weights.

        Args:
            graph (Graph): The graph instance
            start_node (NodeID): Starting node ID
            end_node (NodeID): Target node ID
            weight_func (Optional[WeightFunc]): Optional function to calculate edge weights.
                                              Must return positive values.

        Returns:
            PathResult: Object containing the shortest path and its properties

        Raises:
            NodeNotFoundError: If start_node or end_node doesn't exist
            GraphOperationError: If no path exists between the nodes
            ValueError: If weight_func returns zero or negative values

        Example:
            >>> def weight_by_type(edge: Edge) -> float:
            ...     return 2.0 if edge.relation_type == RelationType.DEPENDS_ON else 1.0
            >>> result = PathFinding.shortest_path(
            ...     graph, "A", "C", weight_func=weight_by_type
            ... )
            >>> print(f"Path length: {result.length}")
        """
        PathFinding._validate_nodes(graph, start_node, end_node)

        # Initialize data structures
        distances: Dict[NodeID, float] = defaultdict(lambda: float("infinity"))
        distances[start_node] = 0
        previous: Dict[NodeID, Optional[NodeID]] = defaultdict(lambda: None)
        visited: Set[NodeID] = set()
        pq: List[Tuple[float, NodeID]] = [(0, start_node)]

        while pq:
            current_distance, current_node = heappop(pq)

            if current_node == end_node:
                break

            if current_node in visited:
                continue

            visited.add(current_node)

            for neighbor in graph.get_neighbors(current_node):
                if neighbor in visited:
                    continue

                edge = graph.get_edge(current_node, neighbor)
                if edge is None:
                    continue

                edge_weight = PathFinding._get_edge_weight(edge, weight_func)
                new_distance = current_distance + edge_weight

                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous[neighbor] = current_node
                    heappush(pq, (new_distance, neighbor))

        if distances[end_node] == float("infinity"):
            raise GraphOperationError(f"No path exists between {start_node} and {end_node}")

        # Reconstruct path
        path: List[Edge] = []
        current = end_node
        while current != start_node:
            prev = previous[current]
            if prev is None:
                break
            edge = graph.get_edge(prev, current)
            if edge is None:
                raise GraphOperationError("Path reconstruction failed")
            path.append(edge)
            current = prev

        path.reverse()
        result = PathFinding._create_path_result(path, weight_func)
        result.validate()
        return result

    @staticmethod
    def all_paths(
        graph: Graph,
        start_node: NodeID,
        end_node: NodeID,
        max_length: Optional[int] = None,
        max_paths: Optional[int] = None,
        filter_func: Optional[PathFilter] = None,
        weight_func: Optional[WeightFunc] = None,
    ) -> Iterator[PathResult]:
        """
        Find all paths between two nodes using iterative DFS with cycle prevention.

        Uses depth-first search to discover all possible paths between two nodes,
        with optional constraints on path length and composition.

        Args:
            graph (Graph): The graph instance
            start_node (NodeID): Starting node ID
            end_node (NodeID): Target node ID
            max_length (Optional[int]): Maximum path length (default: 50)
            max_paths (Optional[int]): Maximum number of paths to return (default: 1000)
            filter_func (Optional[PathFilter]): Optional function to filter paths
            weight_func (Optional[WeightFunc]): Optional function for edge weights

        Yields:
            PathResult: Objects containing each valid path and its properties

        Raises:
            NodeNotFoundError: If start_node or end_node doesn't exist
            ValueError: If max_length or max_paths is not positive
        """
        PathFinding._validate_nodes(graph, start_node, end_node)

        # Use default limits if none provided
        if max_length is None:
            max_length = DEFAULT_MAX_PATH_LENGTH
        if max_paths is None:
            max_paths = DEFAULT_MAX_PATHS

        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")
        if max_paths <= 0:
            raise ValueError(f"max_paths must be positive, got {max_paths}")

        paths_found = 0
        stack = [(start_node, [], {start_node})]  # (node, path_edges, visited)

        while stack and paths_found < max_paths:
            current, path_edges, visited = stack.pop()

            # Check if current node is the end node
            if current == end_node:
                if filter_func is None or filter_func(path_edges):
                    result = PathFinding._create_path_result(path_edges, weight_func)
                    result.validate()
                    yield result
                    paths_found += 1
                continue

            # Skip if path is too long
            if len(path_edges) >= max_length:
                continue

            # Process neighbors in reverse sorted order for consistent DFS
            neighbors = sorted(graph.get_neighbors(current), reverse=True)
            for neighbor in neighbors:
                if neighbor not in visited:
                    edge = graph.get_edge(current, neighbor)
                    if edge is None:
                        continue

                    # Add edge to path and neighbor to visited set
                    new_path_edges = path_edges + [edge]
                    new_visited = visited | {neighbor}
                    stack.append((neighbor, new_path_edges, new_visited))

    @staticmethod
    def find_paths(
        graph: Graph,
        start_node: NodeID,
        end_node: NodeID,
        path_type: PathType = PathType.SHORTEST,
        max_length: Optional[int] = None,
        max_paths: Optional[int] = None,
        filter_func: Optional[PathFilter] = None,
        weight_func: Optional[WeightFunc] = None,
    ) -> Union[PathResult, Iterator[PathResult]]:
        """
        Generic path finding method supporting multiple strategies.

        This method provides a unified interface for finding paths,
        allowing the caller to specify the desired path finding strategy.

        Args:
            graph (Graph): The graph instance
            start_node (NodeID): Starting node ID
            end_node (NodeID): Target node ID
            path_type (PathType): Type of paths to find
            max_length (Optional[int]): Maximum path length
            max_paths (Optional[int]): Maximum number of paths
            filter_func (Optional[PathFilter]): Optional path filter
            weight_func (Optional[WeightFunc]): Optional weight function

        Returns:
            Union[PathResult, Iterator[PathResult]]: Path result(s)

        Example:
            >>> result = PathFinding.find_paths(
            ...     graph, "A", "C",
            ...     path_type=PathType.SHORTEST,
            ...     weight_func=lambda e: e.metadata.weight
            ... )
        """
        if path_type == PathType.SHORTEST:
            return PathFinding.shortest_path(graph, start_node, end_node, weight_func)
        else:
            return PathFinding.all_paths(
                graph,
                start_node,
                end_node,
                max_length,
                max_paths,
                filter_func,
                weight_func,
            )
