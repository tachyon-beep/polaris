"""
Path finding algorithms for the knowledge graph.

This module provides implementations of various path finding algorithms for traversing
and analyzing paths between nodes in the knowledge graph. It includes:
- Bidirectional search for efficient path finding
- Dijkstra's algorithm for finding shortest paths
- Depth-first search for finding all possible paths
- Support for custom weight functions and path filtering
- Path validation and optimization
- Performance monitoring and caching

The path finding capabilities are essential for:
- Analyzing relationships between nodes
- Finding optimal routes through the graph
- Understanding node connections and dependencies
- Discovering indirect relationships
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from heapq import heappop, heappush
from time import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, TypeVar, Union

from ..infrastructure.cache import LRUCache
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

# Cache configuration
PATH_CACHE_SIZE = 10000
PATH_CACHE_TTL = 3600  # 1 hour


@dataclass
class PerformanceMetrics:
    """Container for path finding performance metrics."""

    operation: str
    start_time: float
    end_time: float = 0.0
    path_length: Optional[int] = None
    cache_hit: bool = False

    @property
    def duration(self) -> float:
        """Calculate operation duration in milliseconds."""
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> Dict[str, Union[str, float, int, bool, None]]:
        """Convert metrics to dictionary format."""
        return {
            "operation": self.operation,
            "duration_ms": self.duration,
            "path_length": self.path_length,
            "cache_hit": self.cache_hit,
        }


class PathValidationError(Exception):
    """Raised when a path fails validation checks."""

    pass


class PathType(Enum):
    """Types of paths that can be found in the graph."""

    SHORTEST = "shortest"  # Shortest path by edge count or weight
    BIDIRECTIONAL = "bidirectional"  # Bidirectional search for efficient path finding
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


@dataclass
class SearchState:
    """State information for bidirectional search."""

    queue: deque
    visited: Set[str]
    parent: Dict[str, Optional[str]]
    edges: Dict[Tuple[str, str], Edge]


class PathFinding:
    """
    Path finding algorithms for knowledge graph traversal.

    This class provides static methods implementing various path finding algorithms
    for discovering and analyzing paths between nodes in the knowledge graph.
    The algorithms support customization through weight functions and filters.
    """

    # Initialize cache for path results
    _path_cache = LRUCache[PathResult](
        max_size=PATH_CACHE_SIZE, base_ttl=PATH_CACHE_TTL, adaptive_ttl=True
    )

    @staticmethod
    def _get_cache_key(
        start_node: NodeID,
        end_node: NodeID,
        path_type: PathType,
        weight_func: Optional[WeightFunc] = None,
    ) -> str:
        """Generate cache key for path finding results."""
        key_parts = [start_node, end_node, path_type.value]
        if weight_func is not None:
            key_parts.append(weight_func.__name__)
        return ":".join(key_parts)

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
                f"edge {edge.from_entity}->{edge.to_entity}"
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
    def _expand_search(
        graph: Graph,
        current: SearchState,
        opposite: SearchState,
        weight_func: Optional[WeightFunc] = None,
    ) -> Optional[str]:
        """
        Expands search in one direction and checks for intersection.

        Args:
            graph: The graph instance
            current: Current search direction state
            opposite: Opposite search direction state
            weight_func: Optional weight function

        Returns:
            Intersection node ID if found, None otherwise
        """
        if not current.queue:
            return None

        node, depth = current.queue.popleft()

        for neighbor in graph.get_neighbors(node):
            if neighbor in opposite.visited:
                edge = graph.get_edge(node, neighbor)
                if edge:
                    current.edges[(node, neighbor)] = edge
                return neighbor

            if neighbor not in current.visited:
                edge = graph.get_edge(node, neighbor)
                if edge:
                    current.queue.append((neighbor, depth + 1))
                    current.visited.add(neighbor)
                    current.parent[neighbor] = node
                    current.edges[(node, neighbor)] = edge

        return None

    @staticmethod
    def _construct_bidirectional_path(
        intersection: str, forward: SearchState, backward: SearchState
    ) -> List[Edge]:
        """
        Constructs complete path from intersection point in bidirectional search.

        Args:
            intersection: Node where forward and backward searches met
            forward: Forward search state
            backward: Backward search state

        Returns:
            Complete path as list of edges
        """
        path = []

        # Build forward path
        current = intersection
        while current in forward.parent:
            prev = forward.parent[current]
            if prev is not None and (prev, current) in forward.edges:
                path.append(forward.edges[(prev, current)])
            current = prev
        path.reverse()

        # Build backward path
        current = intersection
        while current in backward.parent:
            next_node = backward.parent[current]
            if next_node is not None and (current, next_node) in backward.edges:
                path.append(backward.edges[(current, next_node)])
            current = next_node

        return path

    @staticmethod
    def bidirectional_search(
        graph: Graph,
        start_node: NodeID,
        end_node: NodeID,
        max_depth: Optional[int] = None,
        weight_func: Optional[WeightFunc] = None,
    ) -> PathResult:
        """
        Find path between nodes using bidirectional search.

        Implements bidirectional search starting from both ends simultaneously,
        which can be significantly faster than unidirectional search for large graphs.

        Args:
            graph: The graph instance
            start_node: Starting node ID
            end_node: Target node ID
            max_depth: Maximum search depth (optional)
            weight_func: Optional function for edge weights

        Returns:
            PathResult containing the found path

        Raises:
            NodeNotFoundError: If start_node or end_node doesn't exist
            GraphOperationError: If no path exists between the nodes
        """
        metrics = PerformanceMetrics(operation="bidirectional_search", start_time=time())

        try:
            # Check cache first
            cache_key = PathFinding._get_cache_key(
                start_node, end_node, PathType.BIDIRECTIONAL, weight_func
            )
            cached_result = PathFinding._path_cache.get(cache_key)
            if cached_result is not None:
                metrics.cache_hit = True
                return cached_result

            PathFinding._validate_nodes(graph, start_node, end_node)

            # Initialize forward and backward search states
            forward = SearchState(
                queue=deque([(start_node, 0)]),
                visited={start_node},
                parent={start_node: None},
                edges={},
            )

            backward = SearchState(
                queue=deque([(end_node, 0)]), visited={end_node}, parent={end_node: None}, edges={}
            )

            while forward.queue and backward.queue:
                # Check depth limit
                if max_depth is not None:
                    _, f_depth = forward.queue[0]
                    _, b_depth = backward.queue[0]
                    if f_depth + b_depth > max_depth:
                        break

                # Expand forward search
                intersection = PathFinding._expand_search(graph, forward, backward, weight_func)
                if intersection:
                    path = PathFinding._construct_bidirectional_path(
                        intersection, forward, backward
                    )
                    result = PathFinding._create_path_result(path, weight_func)
                    result.validate()
                    PathFinding._path_cache.put(cache_key, result)
                    metrics.path_length = len(path)
                    return result

                # Expand backward search
                intersection = PathFinding._expand_search(graph, backward, forward, weight_func)
                if intersection:
                    path = PathFinding._construct_bidirectional_path(
                        intersection, forward, backward
                    )
                    result = PathFinding._create_path_result(path, weight_func)
                    result.validate()
                    PathFinding._path_cache.put(cache_key, result)
                    metrics.path_length = len(path)
                    return result

            raise GraphOperationError(f"No path exists between {start_node} and {end_node}")

        finally:
            metrics.end_time = time()

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
        """
        metrics = PerformanceMetrics(operation="shortest_path", start_time=time())

        try:
            # Check cache first
            cache_key = PathFinding._get_cache_key(
                start_node, end_node, PathType.SHORTEST, weight_func
            )
            cached_result = PathFinding._path_cache.get(cache_key)
            if cached_result is not None:
                metrics.cache_hit = True
                return cached_result

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

            # Cache the result
            PathFinding._path_cache.put(cache_key, result)

            metrics.path_length = len(path)
            return result

        finally:
            metrics.end_time = time()

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
        metrics = PerformanceMetrics(operation="all_paths", start_time=time())

        try:
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
                        metrics.path_length = len(path_edges)
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

        finally:
            metrics.end_time = time()

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
    ) -> PathResult | Iterator[PathResult]:
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
            ...     path_type=PathType.BIDIRECTIONAL,
            ...     weight_func=lambda e: e.metadata.weight
            ... )
        """
        if path_type == PathType.SHORTEST:
            return PathFinding.shortest_path(graph, start_node, end_node, weight_func)
        elif path_type == PathType.BIDIRECTIONAL:
            return PathFinding.bidirectional_search(
                graph, start_node, end_node, max_length, weight_func
            )
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

    @staticmethod
    def get_cache_metrics() -> Dict[str, Any]:
        """Get current cache performance metrics."""
        return PathFinding._path_cache.get_metrics()
