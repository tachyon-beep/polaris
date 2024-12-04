"""Graph metrics calculation for the knowledge graph.

This module provides comprehensive functionality for calculating and analyzing various
graph metrics that characterize the structure and properties of the knowledge graph.
These metrics provide insights into:
- Graph size and connectivity (node count, edge count, density)
- Network topology (average degree, clustering coefficient)
- Graph components and path characteristics (diameter, average path length)
- Performance optimized calculations for large graphs

The metrics are valuable for:
- Understanding graph complexity and structure
- Analyzing network connectivity patterns
- Identifying bottlenecks or central nodes
- Comparing different graph states or subgraphs
"""

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

from .graph import Graph
from .graph_components import ComponentAnalysis
from .graph_traversal import GraphTraversal


@dataclass
class GraphMetrics:
    """Container for graph metrics.

    This dataclass holds various metrics that characterize the graph's structure
    and properties. Each metric provides different insights into the graph's
    characteristics.

    Attributes:
        node_count (int): Total number of nodes in the graph
        edge_count (int): Total number of edges in the graph
        average_degree (float): Average number of connections per node
        density (float): Ratio of actual to possible edges (0 to 1)
        clustering_coefficient (float): Measure of node clustering (0 to 1)
        connected_components (int): Number of disconnected subgraphs
        diameter (Optional[int]): Maximum shortest path length between any nodes
        average_path_length (float): Average shortest path length between nodes
        degree_distribution (Dict[int, int]): Distribution of node degrees
        centrality_scores (Dict[str, float]): Node centrality measures
    """

    node_count: int
    edge_count: int
    average_degree: float
    density: float
    clustering_coefficient: float
    connected_components: int
    diameter: Optional[int]
    average_path_length: float
    degree_distribution: Dict[int, int]
    centrality_scores: Dict[str, float]


class PathMetricsResult(NamedTuple):
    """Container for path-based metrics results.

    Attributes:
        diameter: Maximum shortest path length, None for empty graphs, 0 for single-node graphs
        avg_path_length: Average shortest path length between nodes
        sample_size: Number of node pairs sampled
        confidence: Confidence level in the approximation
    """

    diameter: Optional[int]
    avg_path_length: float
    sample_size: int
    confidence: float


class MetricsCalculator:
    """Graph metrics calculation.

    This class provides static methods for calculating various graph metrics
    that characterize the structure and properties of the knowledge graph.
    The calculations are implemented as static methods to provide utility-style
    functionality that can be used with any Graph instance without maintaining state.
    """

    # Thresholds for different calculation strategies
    PATH_METRICS_EXACT_THRESHOLD = 1000  # Use exact calculation for graphs smaller than this
    CLUSTERING_EXACT_THRESHOLD = 5000  # Use exact clustering for graphs smaller than this
    MIN_SAMPLE_SIZE = 100  # Minimum number of node pairs to sample
    CONFIDENCE_THRESHOLD = 0.95  # Target confidence level for approximations

    @staticmethod
    def _calculate_basic_metrics(
        graph: Graph,
    ) -> Tuple[int, int, float, float, Dict[int, int]]:
        """Calculate basic graph metrics including degree distribution.

        Args:
            graph (Graph): The graph instance to analyze

        Returns:
            Tuple containing:
            - node_count: Total number of nodes
            - edge_count: Total number of edges
            - average_degree: Average node degree
            - density: Graph density
            - degree_distribution: Distribution of node degrees
        """
        nodes = graph.get_nodes()
        node_count = len(nodes)
        edge_count = graph.get_edge_count()

        if node_count == 0:
            return 0, 0, 0.0, 0.0, {}

        # Calculate degree distribution
        degree_distribution: Dict[int, int] = defaultdict(int)
        for node in nodes:
            degree = graph.get_degree(node)
            degree_distribution[degree] += 1

        # Calculate average degree and density
        average_degree = 2 * edge_count / node_count if node_count > 0 else 0.0

        # For single node graphs, density is 1.0 if it has a self-loop, 0.0 otherwise
        if node_count == 1:
            density = 1.0 if edge_count > 0 else 0.0
        else:
            max_possible_edges = node_count * (node_count - 1)
            density = edge_count / max_possible_edges if max_possible_edges > 0 else 0.0

        return (
            node_count,
            edge_count,
            average_degree,
            density,
            dict(degree_distribution),
        )

    @staticmethod
    def _calculate_clustering_exact(graph: Graph, nodes: Set[str]) -> float:
        """Calculate exact clustering coefficient.

        Args:
            graph (Graph): The graph instance
            nodes (Set[str]): Set of nodes to analyze

        Returns:
            float: Average clustering coefficient
        """
        if not nodes:
            return 0.0

        coefficients = []
        for node in nodes:
            coefficient = ComponentAnalysis.calculate_clustering_coefficient(graph, node)
            if coefficient is not None:
                coefficients.append(coefficient)

        return sum(coefficients) / len(coefficients) if coefficients else 0.0

    @staticmethod
    def _calculate_clustering_approximate(
        graph: Graph, nodes: Set[str], sample_size: int
    ) -> Tuple[float, float]:
        """Calculate approximate clustering coefficient using sampling.

        Args:
            graph (Graph): The graph instance
            nodes (Set[str]): Set of nodes to analyze
            sample_size (int): Number of nodes to sample

        Returns:
            Tuple[float, float]: (clustering coefficient, confidence level)
        """
        if not nodes:
            return 0.0, 1.0

        # Sample nodes
        sample_nodes = random.sample(list(nodes), min(sample_size, len(nodes)))

        coefficients = []
        for node in sample_nodes:
            coefficient = ComponentAnalysis.calculate_clustering_coefficient(graph, node)
            if coefficient is not None:
                coefficients.append(coefficient)

        if not coefficients:
            return 0.0, 1.0

        # Calculate mean and confidence
        mean = sum(coefficients) / len(coefficients)
        if len(coefficients) < 2:
            return mean, 1.0

        # Calculate confidence using standard error
        std_dev = math.sqrt(sum((x - mean) ** 2 for x in coefficients) / (len(coefficients) - 1))
        std_error = std_dev / math.sqrt(len(coefficients))
        confidence = 1.0 - (2 * std_error)

        return mean, confidence

    @staticmethod
    def _calculate_path_metrics_exact(graph: Graph, nodes: Set[str]) -> PathMetricsResult:
        """Calculate exact path metrics using Floyd-Warshall.

        Args:
            graph (Graph): The graph instance
            nodes (Set[str]): Set of nodes to analyze

        Returns:
            PathMetricsResult: Exact path metrics with appropriate diameter values:
                - None for empty graphs or disconnected components
                - 0 for single-node graphs
                - Actual diameter for connected multi-node graphs
        """
        if len(nodes) == 0:
            return PathMetricsResult(
                diameter=None,  # Undefined for empty graphs
                avg_path_length=0.0,
                sample_size=0,
                confidence=1.0,
            )
        elif len(nodes) == 1:
            return PathMetricsResult(
                diameter=0,  # Zero for single-node graphs
                avg_path_length=0.0,
                sample_size=1,
                confidence=1.0,
            )

        # Initialize distances
        distances = {node: {other: float("inf") for other in nodes} for node in nodes}

        # Set distance to self as 0
        for node in nodes:
            distances[node][node] = 0

        # Set direct edge distances
        for node in nodes:
            for neighbor in graph.get_neighbors(node):
                distances[node][neighbor] = 1

        # Floyd-Warshall algorithm
        for k in nodes:
            for i in nodes:
                for j in nodes:
                    if distances[i][k] + distances[k][j] < distances[i][j]:
                        distances[i][j] = distances[i][k] + distances[k][j]

        # Calculate metrics
        max_distance = 0
        total_distance = 0
        path_count = 0
        has_infinite = False

        for i in nodes:
            for j in nodes:
                if i != j:
                    if distances[i][j] == float("inf"):
                        has_infinite = True
                    else:
                        max_distance = max(max_distance, distances[i][j])
                        total_distance += distances[i][j]
                        path_count += 1

        avg_path_length = total_distance / path_count if path_count > 0 else 0.0
        # If there are any infinite distances, the graph is disconnected
        diameter = None if has_infinite else int(max_distance)

        return PathMetricsResult(diameter, avg_path_length, len(nodes), 1.0)

    @staticmethod
    def _calculate_path_metrics_approximate(
        graph: Graph, nodes: Set[str], sample_size: int
    ) -> PathMetricsResult:
        """Calculate approximate path metrics using landmark-based sampling.

        Args:
            graph (Graph): The graph instance
            nodes (Set[str]): Set of nodes to analyze
            sample_size (int): Number of node pairs to sample

        Returns:
            PathMetricsResult: Approximate path metrics with confidence
        """
        if len(nodes) == 0:
            return PathMetricsResult(None, 0.0, 0, 1.0)
        elif len(nodes) == 1:
            return PathMetricsResult(0, 0.0, 1, 1.0)

        # Select landmark nodes
        landmarks = random.sample(list(nodes), min(sample_size, len(nodes)))

        max_distance = 0
        total_distance = 0
        path_count = 0
        distances: List[int] = []
        has_infinite = False

        # Calculate distances from landmarks
        for landmark in landmarks:
            for node, depth in GraphTraversal.bfs(graph, landmark):
                if landmark != node:
                    if depth is None:
                        has_infinite = True
                    else:
                        max_distance = max(max_distance, depth)
                        total_distance += depth
                        path_count += 1
                        distances.append(depth)

        if not distances:
            return PathMetricsResult(None, 0.0, 0, 0.0)

        # Calculate metrics and confidence
        avg_path_length = total_distance / path_count if path_count > 0 else 0.0
        diameter = None if has_infinite else int(max_distance * 1.5)  # Conservative estimate

        # Calculate confidence using standard error
        mean = sum(distances) / len(distances)
        std_dev = math.sqrt(sum((x - mean) ** 2 for x in distances) / (len(distances) - 1))
        std_error = std_dev / math.sqrt(len(distances))
        confidence = 1.0 - (2 * std_error)

        return PathMetricsResult(diameter, avg_path_length, path_count, confidence)

    @staticmethod
    def _calculate_centrality_scores(graph: Graph) -> Dict[str, float]:
        """Calculate degree centrality scores for nodes.

        Args:
            graph (Graph): The graph instance

        Returns:
            Dict[str, float]: Map of node IDs to centrality scores
        """
        nodes = graph.get_nodes()
        if not nodes:
            return {}

        max_possible_degree = len(nodes) - 1
        if max_possible_degree == 0:
            return {node: 0.0 for node in nodes}

        return {node: graph.get_degree(node) / max_possible_degree for node in nodes}

    @staticmethod
    def calculate_metrics(graph: Graph) -> GraphMetrics:
        """Calculate comprehensive graph metrics.

        This method combines all metric calculations to provide a complete analysis
        of the graph's structure and properties. It automatically chooses between
        exact and approximate calculations based on graph size.

        Args:
            graph (Graph): The graph instance to analyze

        Returns:
            GraphMetrics: Object containing all calculated metrics

        Example:
            >>> metrics = MetricsCalculator.calculate_metrics(graph)
            >>> print(f"Average path length: {metrics.average_path_length:.2f}")
            >>> print(f"Clustering coefficient: {metrics.clustering_coefficient:.2f}")
        """
        # Calculate basic metrics
        (
            node_count,
            edge_count,
            average_degree,
            density,
            degree_distribution,
        ) = MetricsCalculator._calculate_basic_metrics(graph)

        nodes = graph.get_nodes()

        # Calculate clustering coefficient
        if node_count < MetricsCalculator.CLUSTERING_EXACT_THRESHOLD:
            clustering_coefficient = MetricsCalculator._calculate_clustering_exact(graph, nodes)
        else:
            sample_size = max(
                MetricsCalculator.MIN_SAMPLE_SIZE,
                int(math.sqrt(node_count)),
            )
            clustering_coefficient, _ = MetricsCalculator._calculate_clustering_approximate(
                graph, nodes, sample_size
            )

        # Find connected components
        components = ComponentAnalysis.find_components(graph)

        # Calculate path metrics
        if node_count < MetricsCalculator.PATH_METRICS_EXACT_THRESHOLD:
            path_metrics = MetricsCalculator._calculate_path_metrics_exact(graph, nodes)
        else:
            sample_size = max(
                MetricsCalculator.MIN_SAMPLE_SIZE,
                int(math.sqrt(node_count)),
            )
            path_metrics = MetricsCalculator._calculate_path_metrics_approximate(
                graph, nodes, sample_size
            )

        # Calculate centrality scores
        centrality_scores = MetricsCalculator._calculate_centrality_scores(graph)

        return GraphMetrics(
            node_count=node_count,
            edge_count=edge_count,
            average_degree=average_degree,
            density=density,
            clustering_coefficient=clustering_coefficient,
            connected_components=len(components),
            diameter=path_metrics.diameter,
            average_path_length=path_metrics.avg_path_length,
            degree_distribution=degree_distribution,
            centrality_scores=centrality_scores,
        )
