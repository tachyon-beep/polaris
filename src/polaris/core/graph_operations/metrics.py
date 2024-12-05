"""Graph metrics calculation functionality."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any
import math

from ..models import Edge


@dataclass
class GraphMetrics:
    """Container for graph metrics results."""

    node_count: int
    edge_count: int
    density: float
    average_degree: float
    clustering_coefficient: float
    connected_components: int
    average_path_length: float  # Changed to float, will be 0.0 for disconnected graphs
    diameter: Optional[float]
    degree_distribution: Dict[str, Tuple[int, int]]
    centrality_measures: Dict[str, Dict[str, float]]


class MetricsCalculator:
    """
    Calculates various graph metrics.

    This class provides functionality to compute different metrics that
    characterize the structure and properties of a graph, such as:
    - Degree distributions
    - Centrality measures
    - Path-based metrics
    - Component analysis
    """

    def __init__(self, edges: List[Edge]):
        """Initialize calculator with graph edges."""
        self.edges = edges
        self.out_adj: Dict[str, Set[str]] = defaultdict(set)
        self.in_adj: Dict[str, Set[str]] = defaultdict(set)
        self.nodes = set()
        self._build_adjacency_lists()

    def _build_adjacency_lists(self) -> None:
        """Build adjacency lists for efficient metric calculation."""
        self.out_adj = defaultdict(set)
        self.in_adj = defaultdict(set)
        self.nodes = set()

        for edge in self.edges:
            self.out_adj[edge.from_entity].add(edge.to_entity)
            self.in_adj[edge.to_entity].add(edge.from_entity)
            self.nodes.add(edge.from_entity)
            self.nodes.add(edge.to_entity)

    def calculate_metrics(self) -> GraphMetrics:
        """Calculate all graph metrics."""
        components = self._find_connected_components()
        avg_path_length = self._calculate_average_path_length()
        diameter = self._calculate_diameter()

        # For disconnected graphs or empty graphs, set appropriate values
        if len(components) != 1:  # Disconnected or empty graph
            avg_path_length = 0.0
            diameter = None
        elif diameter == 0.0:  # Single node or empty graph
            diameter = None

        return GraphMetrics(
            node_count=len(self.nodes),
            edge_count=len(self.edges),
            density=self.get_density(),
            average_degree=self.get_average_degree(),
            clustering_coefficient=self.get_clustering_coefficient(),
            connected_components=len(components),
            average_path_length=avg_path_length,
            diameter=diameter,
            degree_distribution=self.get_degree_distribution(),
            centrality_measures=self.get_centrality_measures(),
        )

    def _find_connected_components(self) -> List[Set[str]]:
        """Find connected components in the graph."""
        components = []
        visited = set()

        def dfs(node: str, component: Set[str]) -> None:
            visited.add(node)
            component.add(node)
            # Consider both outgoing and incoming edges
            for neighbor in self.out_adj[node] | self.in_adj[node]:
                if neighbor not in visited:
                    dfs(neighbor, component)

        for node in self.nodes:
            if node not in visited:
                component = set()
                dfs(node, component)
                components.append(component)

        return components

    def _calculate_average_path_length(self) -> float:
        """Calculate average shortest path length."""
        if len(self.nodes) < 2:
            return 0.0

        total_length = 0
        path_count = 0

        for source in self.nodes:
            for target in self.nodes:
                if source != target:
                    length = self._shortest_path_length(source, target)
                    if length is not None:
                        total_length += length
                        path_count += 1

        return total_length / path_count if path_count > 0 else 0.0

    def _calculate_diameter(self) -> float:
        """Calculate graph diameter (longest shortest path)."""
        if len(self.nodes) < 2:
            return 0.0

        max_length = 0
        found_path = False

        for source in self.nodes:
            for target in self.nodes:
                if source != target:
                    length = self._shortest_path_length(source, target)
                    if length is not None:
                        max_length = max(max_length, length)
                        found_path = True

        return max_length if found_path else 0.0

    def get_degree_distribution(self) -> Dict[str, Tuple[int, int]]:
        """Get in-degree and out-degree for each node."""
        distribution = {}
        for node in self.nodes:
            in_degree = len(self.in_adj[node])
            out_degree = len(self.out_adj[node])
            distribution[node] = (in_degree, out_degree)
        return distribution

    def get_average_degree(self) -> float:
        """Calculate average degree across all nodes."""
        if not self.nodes:
            return 0.0
        total_degree = sum(sum(degrees) for degrees in self.get_degree_distribution().values())
        return total_degree / len(self.nodes)

    def get_density(self) -> float:
        """Calculate graph density (ratio of actual to possible edges)."""
        n = len(self.nodes)
        if n <= 1:
            return 0.0
        possible_edges = n * (n - 1)  # For directed graph
        actual_edges = len(self.edges)
        return actual_edges / possible_edges if possible_edges > 0 else 0.0

    def get_clustering_coefficient(self) -> float:
        """Calculate global clustering coefficient."""
        if not self.nodes:
            return 0.0

        total_coefficient = 0.0
        counted_nodes = 0

        for node in self.nodes:
            neighbors = self.out_adj[node].union(self.in_adj[node])
            if len(neighbors) < 2:
                continue

            possible_connections = len(neighbors) * (len(neighbors) - 1)
            actual_connections = 0

            for neighbor1 in neighbors:
                for neighbor2 in neighbors:
                    if neighbor1 != neighbor2:
                        if (
                            neighbor2 in self.out_adj[neighbor1]
                            or neighbor2 in self.in_adj[neighbor1]
                        ):
                            actual_connections += 1

            if possible_connections > 0:
                total_coefficient += actual_connections / possible_connections
                counted_nodes += 1

        return total_coefficient / counted_nodes if counted_nodes > 0 else 0.0

    def get_centrality_measures(self) -> Dict[str, Dict[str, float]]:
        """Calculate various centrality measures for each node."""
        measures = {}
        for node in self.nodes:
            measures[node] = {
                "degree": len(self.out_adj[node]) + len(self.in_adj[node]),
                "in_degree": len(self.in_adj[node]),
                "out_degree": len(self.out_adj[node]),
                "betweenness": self._calculate_betweenness(node),
                "closeness": self._calculate_closeness(node),
            }
        return measures

    def _calculate_betweenness(self, node: str) -> float:
        """Calculate betweenness centrality for a node."""
        # Simplified betweenness calculation
        if len(self.nodes) < 3:
            return 0.0

        betweenness = 0.0
        for source in self.nodes:
            if source == node:
                continue
            for target in self.nodes:
                if target == node or target == source:
                    continue

                # Count paths through node
                total_paths = 0
                paths_through_node = 0
                for path in self._find_all_paths(source, target, max_length=3):
                    total_paths += 1
                    if node in path[1:-1]:  # Node is in path but not endpoints
                        paths_through_node += 1

                if total_paths > 0:
                    betweenness += paths_through_node / total_paths

        return betweenness

    def _calculate_closeness(self, node: str) -> float:
        """Calculate closeness centrality for a node."""
        if len(self.nodes) < 2:
            return 0.0

        total_distance = 0
        reachable_nodes = 0

        for target in self.nodes:
            if target == node:
                continue

            distance = self._shortest_path_length(node, target)
            if distance is not None:
                total_distance += distance
                reachable_nodes += 1

        if reachable_nodes == 0:
            return 0.0

        return reachable_nodes / total_distance if total_distance > 0 else 0.0

    def _shortest_path_length(self, source: str, target: str) -> Optional[int]:
        """Find length of shortest path between two nodes."""
        if source == target:
            return 0

        visited = {source}
        queue = [(source, 0)]
        while queue:
            node, distance = queue.pop(0)
            for neighbor in self.out_adj[node]:
                if neighbor == target:
                    return distance + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))
        return None

    def _find_all_paths(self, source: str, target: str, max_length: int) -> List[List[str]]:
        """Find all paths between two nodes up to max_length."""

        def dfs(current: str, path: List[str], paths: List[List[str]]) -> None:
            if len(path) > max_length:
                return
            if current == target:
                paths.append(path[:])
                return
            for neighbor in self.out_adj[current]:
                if neighbor not in path:  # Avoid cycles
                    path.append(neighbor)
                    dfs(neighbor, path, paths)
                    path.pop()

        paths = []
        dfs(source, [source], paths)
        return paths
