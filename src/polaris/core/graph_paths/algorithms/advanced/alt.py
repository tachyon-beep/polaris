"""
A* with Landmarks (ALT) implementation.

This module provides an implementation of the ALT algorithm, which enhances A*
search using triangle inequality with precomputed landmarks. Key features:
- Landmark selection strategies
- Distance precomputation
- Triangle inequality based heuristic
- Memory-efficient implementation
"""

from dataclasses import dataclass
import random
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
class LandmarkDistance:
    """
    Precomputed distances for a landmark.

    Attributes:
        forward: Distances from landmark to all nodes
        backward: Distances from all nodes to landmark
    """

    forward: Dict[str, float]
    backward: Dict[str, float]


class ALTPathFinder(PathFinder):
    """
    A* with Landmarks implementation.

    Features:
    - Maximal separation landmark selection
    - Triangle inequality based heuristic
    - Bidirectional A* support
    - Memory usage monitoring
    """

    def __init__(
        self, graph: Graph, num_landmarks: int = 16, max_memory_mb: Optional[float] = None
    ):
        """
        Initialize with graph and parameters.

        Args:
            graph: Input graph
            num_landmarks: Number of landmarks to select
            max_memory_mb: Optional memory limit in MB
        """
        super().__init__(graph)
        self.memory_manager = MemoryManager(max_memory_mb)
        self.num_landmarks = num_landmarks
        self.landmarks: Dict[str, LandmarkDistance] = {}
        self._preprocessed = False

    def preprocess(self) -> None:
        """
        Preprocess graph to select landmarks and compute distances.

        This performs two main steps:
        1. Select landmarks using maximal separation
        2. Compute distances to/from all landmarks
        """
        print("Starting preprocessing...")

        # Select landmarks
        print("Selecting landmarks...")
        landmarks = self._select_landmarks()
        print(f"Selected {len(landmarks)} landmarks")

        # Compute distances
        print("Computing landmark distances...")
        total = len(landmarks)
        for i, landmark in enumerate(landmarks):
            self.memory_manager.check_memory()

            # Compute forward and backward distances
            forward = self._compute_forward_distances(landmark)
            backward = self._compute_backward_distances(landmark)

            self.landmarks[landmark] = LandmarkDistance(forward=forward, backward=backward)

            # Show progress
            progress = ((i + 1) / total) * 100
            print(f"Computing distances: {progress:.1f}% complete")

        self._preprocessed = True
        print("Preprocessing complete")

    def _select_landmarks(self) -> List[str]:
        """
        Select landmarks using maximal separation method.

        Returns landmarks that are well-distributed throughout the graph.
        """
        nodes = list(self.graph.get_nodes())
        if len(nodes) <= self.num_landmarks:
            return nodes

        landmarks = []
        # Start with a random node
        landmarks.append(random.choice(nodes))

        while len(landmarks) < self.num_landmarks:
            # Find node with maximum distance to current landmarks
            max_distance = -1
            next_landmark = None

            for node in nodes:
                if node in landmarks:
                    continue

                # Compute minimum distance to existing landmarks
                min_dist = float("inf")
                for landmark in landmarks:
                    dist = self._compute_distance(node, landmark)
                    min_dist = min(min_dist, dist)

                # Update if this node is further from landmarks
                if min_dist > max_distance:
                    max_distance = min_dist
                    next_landmark = node

            if next_landmark is None:
                break

            landmarks.append(next_landmark)

        return landmarks

    def _compute_forward_distances(self, landmark: str) -> Dict[str, float]:
        """
        Compute distances from landmark to all nodes.

        Args:
            landmark: Landmark node

        Returns:
            Dictionary mapping node IDs to distances
        """
        distances = {landmark: 0.0}
        pq = [(0.0, landmark)]

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

        return distances

    def _compute_backward_distances(self, landmark: str) -> Dict[str, float]:
        """
        Compute distances from all nodes to landmark.

        Args:
            landmark: Landmark node

        Returns:
            Dictionary mapping node IDs to distances
        """
        distances = {landmark: 0.0}
        pq = [(0.0, landmark)]

        while pq:
            dist, current = heappop(pq)

            for neighbor in self.graph.get_neighbors(current, reverse=True):
                edge = self.graph.get_edge(neighbor, current)
                if not edge:
                    continue

                new_dist = dist + get_edge_weight(edge)
                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    heappush(pq, (new_dist, neighbor))

        return distances

    def _compute_distance(self, source: str, target: str) -> float:
        """
        Compute distance between nodes using Dijkstra's algorithm.

        Args:
            source: Source node
            target: Target node

        Returns:
            Distance between nodes (inf if no path exists)
        """
        distances = {source: 0.0}
        pq = [(0.0, source)]

        while pq:
            dist, current = heappop(pq)

            if current == target:
                return dist

            for neighbor in self.graph.get_neighbors(current):
                edge = self.graph.get_edge(current, neighbor)
                if not edge:
                    continue

                new_dist = dist + get_edge_weight(edge)
                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    heappush(pq, (new_dist, neighbor))

        return float("inf")

    def _compute_heuristic(
        self, node: str, target: str, weight_func: Optional[WeightFunc]
    ) -> float:
        """
        Compute ALT heuristic using triangle inequality.

        Args:
            node: Current node
            target: Target node
            weight_func: Optional weight function

        Returns:
            Lower bound on distance from node to target
        """
        if not self.landmarks:
            return 0.0

        h = 0.0
        for landmark, distances in self.landmarks.items():
            # Forward heuristic
            if node in distances.forward and target in distances.forward:
                h = max(h, abs(distances.forward[target] - distances.forward[node]))

            # Backward heuristic
            if node in distances.backward and target in distances.backward:
                h = max(h, abs(distances.backward[node] - distances.backward[target]))

        return h

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
        Find shortest path using A* with landmarks.

        Args:
            start_node: Starting node ID
            end_node: Target node ID
            max_length: Maximum path length
            max_paths: Not used (always returns single path)
            filter_func: Optional path filter
            weight_func: Optional weight function
            **kwargs: Additional options:
                validate: Whether to validate result (default: True)
                bidirectional: Whether to use bidirectional search (default: True)

        Returns:
            PathResult containing shortest path

        Raises:
            GraphOperationError: If no path exists or graph not preprocessed
        """
        if not self._preprocessed:
            raise GraphOperationError("Graph must be preprocessed before finding paths")

        validate = kwargs.get("validate", True)
        bidirectional = kwargs.get("bidirectional", True)

        if bidirectional:
            return self._bidirectional_alt(
                start_node, end_node, max_length, filter_func, weight_func, validate
            )
        else:
            return self._unidirectional_alt(
                start_node, end_node, max_length, filter_func, weight_func, validate
            )

    def _unidirectional_alt(
        self,
        start_node: str,
        end_node: str,
        max_length: Optional[int],
        filter_func: Optional[Callable[[List[Edge]], bool]],
        weight_func: Optional[WeightFunc],
        validate: bool,
    ) -> PathResult:
        """
        Find path using unidirectional A* with landmarks.

        Args:
            start_node: Starting node
            end_node: Target node
            max_length: Maximum path length
            filter_func: Optional path filter
            weight_func: Optional weight function
            validate: Whether to validate result

        Returns:
            PathResult containing shortest path
        """
        g_score = {start_node: 0.0}
        f_score = {start_node: self._compute_heuristic(start_node, end_node, weight_func)}
        came_from: Dict[str, Tuple[str, Edge]] = {}

        pq = [(f_score[start_node], start_node)]
        path_length = {start_node: 0}

        while pq:
            self.memory_manager.check_memory()

            _, current = heappop(pq)

            if current == end_node:
                break

            # Check max length
            if max_length and path_length[current] >= max_length:
                continue

            for neighbor in self.graph.get_neighbors(current):
                edge = self.graph.get_edge(current, neighbor)
                if not edge:
                    continue

                tentative_g = g_score[current] + get_edge_weight(edge, weight_func)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = (current, edge)
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._compute_heuristic(
                        neighbor, end_node, weight_func
                    )
                    path_length[neighbor] = path_length[current] + 1
                    heappush(pq, (f_score[neighbor], neighbor))

        if end_node not in came_from:
            raise GraphOperationError(f"No path exists between {start_node} and {end_node}")

        # Reconstruct path
        path = []
        current = end_node
        while current in came_from:
            prev, edge = came_from[current]
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
        if validate:
            validate_path(path, self.graph, weight_func, max_length)

        return result

    def _bidirectional_alt(
        self,
        start_node: str,
        end_node: str,
        max_length: Optional[int],
        filter_func: Optional[Callable[[List[Edge]], bool]],
        weight_func: Optional[WeightFunc],
        validate: bool,
    ) -> PathResult:
        """
        Find path using bidirectional A* with landmarks.

        Args:
            start_node: Starting node
            end_node: Target node
            max_length: Maximum path length
            filter_func: Optional path filter
            weight_func: Optional weight function
            validate: Whether to validate result

        Returns:
            PathResult containing shortest path
        """
        # Forward search state
        g_score_f = {start_node: 0.0}
        f_score_f = {start_node: self._compute_heuristic(start_node, end_node, weight_func)}
        came_from_f: Dict[str, Tuple[str, Edge]] = {}
        pq_f = [(f_score_f[start_node], start_node)]
        path_length_f = {start_node: 0}

        # Backward search state
        g_score_b = {end_node: 0.0}
        f_score_b = {end_node: self._compute_heuristic(end_node, start_node, weight_func)}
        came_from_b: Dict[str, Tuple[str, Edge]] = {}
        pq_b = [(f_score_b[end_node], end_node)]
        path_length_b = {end_node: 0}

        # Track best path
        best_dist = float("inf")
        meeting_node = None

        while pq_f and pq_b:
            self.memory_manager.check_memory()

            if pq_f[0][0] + pq_b[0][0] >= best_dist:
                break

            # Forward search
            _, current_f = heappop(pq_f)

            if current_f in g_score_b:
                dist = g_score_f[current_f] + g_score_b[current_f]
                if dist < best_dist:
                    best_dist = dist
                    meeting_node = current_f

            if max_length and path_length_f[current_f] >= max_length:
                continue

            for neighbor in self.graph.get_neighbors(current_f):
                edge = self.graph.get_edge(current_f, neighbor)
                if not edge:
                    continue

                tentative_g = g_score_f[current_f] + get_edge_weight(edge, weight_func)

                if neighbor not in g_score_f or tentative_g < g_score_f[neighbor]:
                    came_from_f[neighbor] = (current_f, edge)
                    g_score_f[neighbor] = tentative_g
                    f_score_f[neighbor] = tentative_g + self._compute_heuristic(
                        neighbor, end_node, weight_func
                    )
                    path_length_f[neighbor] = path_length_f[current_f] + 1
                    heappush(pq_f, (f_score_f[neighbor], neighbor))

            # Backward search
            _, current_b = heappop(pq_b)

            if current_b in g_score_f:
                dist = g_score_f[current_b] + g_score_b[current_b]
                if dist < best_dist:
                    best_dist = dist
                    meeting_node = current_b

            if max_length and path_length_b[current_b] >= max_length:
                continue

            for neighbor in self.graph.get_neighbors(current_b, reverse=True):
                edge = self.graph.get_edge(neighbor, current_b)
                if not edge:
                    continue

                tentative_g = g_score_b[current_b] + get_edge_weight(edge, weight_func)

                if neighbor not in g_score_b or tentative_g < g_score_b[neighbor]:
                    came_from_b[neighbor] = (current_b, edge)
                    g_score_b[neighbor] = tentative_g
                    f_score_b[neighbor] = tentative_g + self._compute_heuristic(
                        neighbor, start_node, weight_func
                    )
                    path_length_b[neighbor] = path_length_b[current_b] + 1
                    heappush(pq_b, (f_score_b[neighbor], neighbor))

        if meeting_node is None:
            raise GraphOperationError(f"No path exists between {start_node} and {end_node}")

        # Reconstruct path
        path = []

        # Forward path
        current = meeting_node
        while current in came_from_f:
            prev, edge = came_from_f[current]
            path.append(edge)
            current = prev
        path.reverse()

        # Backward path
        current = meeting_node
        while current in came_from_b:
            next_node, edge = came_from_b[current]
            path.append(edge)
            current = next_node

        # Apply filter if provided
        if filter_func and not filter_func(path):
            raise GraphOperationError(
                f"No path satisfying filter exists between {start_node} and {end_node}"
            )

        # Create and validate result
        result = create_path_result(path, weight_func)
        if validate:
            validate_path(path, self.graph, weight_func, max_length)

        return result
