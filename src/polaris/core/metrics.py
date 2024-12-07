"""
Graph metric calculation system.

This module provides a flexible strategy pattern for calculating various metrics
on the graph. It supports both built-in metrics and custom metric calculations
through a pluggable strategy system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Set, Tuple

from .graph import Graph
from .models import Edge


class MetricStrategy(Protocol):
    """Protocol for metric calculation strategies."""

    def calculate(self, graph: Graph) -> Dict[str, float]:
        """
        Calculate metrics for the graph.

        Args:
            graph: The graph to analyze

        Returns:
            Dict mapping metric names to their values
        """
        ...


class ClusteringMetricStrategy(MetricStrategy):
    """Calculates clustering-related metrics."""

    def calculate(self, graph: Graph) -> Dict[str, float]:
        """
        Calculate clustering metrics.

        For large graphs (>5000 nodes), uses approximate calculations
        to maintain performance.

        Returns:
            Dict with metrics:
            - average_clustering: Average clustering coefficient
            - global_clustering: Global clustering coefficient
            - max_clustering: Maximum clustering coefficient
        """
        if len(graph.get_nodes()) < 5000:
            return self._calculate_exact(graph)
        return self._calculate_approximate(graph)

    def _calculate_exact(self, graph: Graph) -> Dict[str, float]:
        """Exact calculation for smaller graphs."""
        from .graph_components import ComponentAnalysis

        nodes = graph.get_nodes()
        coefficients = []

        for node in nodes:
            coeff = ComponentAnalysis.calculate_clustering_coefficient(graph, node)
            if coeff is not None:
                coefficients.append(coeff)

        if not coefficients:
            return {"average_clustering": 0.0, "global_clustering": 0.0, "max_clustering": 0.0}

        return {
            "average_clustering": sum(coefficients) / len(coefficients),
            "global_clustering": self._calculate_global_clustering(graph),
            "max_clustering": max(coefficients),
        }

    def _calculate_approximate(self, graph: Graph) -> Dict[str, float]:
        """Approximate calculation for larger graphs."""
        import random

        # Sample 1000 nodes for approximation
        nodes = list(graph.get_nodes())
        sample_size = min(1000, len(nodes))
        sampled_nodes = random.sample(nodes, sample_size)

        from .graph_components import ComponentAnalysis

        coefficients = []

        for node in sampled_nodes:
            coeff = ComponentAnalysis.calculate_clustering_coefficient(graph, node)
            if coeff is not None:
                coefficients.append(coeff)

        if not coefficients:
            return {"average_clustering": 0.0, "global_clustering": 0.0, "max_clustering": 0.0}

        return {
            "average_clustering": sum(coefficients) / len(coefficients),
            "global_clustering": self._calculate_global_clustering(graph),
            "max_clustering": max(coefficients),
        }

    def _calculate_global_clustering(self, graph: Graph) -> float:
        """Calculate global clustering coefficient."""
        total_triplets = 0
        closed_triplets = 0

        for node in graph.get_nodes():
            neighbors = graph.get_neighbors(node)
            if len(neighbors) < 2:
                continue

            # Count potential triplets
            degree = len(neighbors)
            total_triplets += degree * (degree - 1)

            # Count closed triplets
            for n1 in neighbors:
                for n2 in neighbors:
                    if n1 != n2 and graph.has_edge(n1, n2):
                        closed_triplets += 1

        return closed_triplets / total_triplets if total_triplets > 0 else 0.0


class CentralityMetricStrategy(MetricStrategy):
    """Calculates centrality metrics."""

    def calculate(self, graph: Graph) -> Dict[str, float]:
        """
        Calculate centrality metrics.

        Returns:
            Dict with metrics:
            - degree_centrality: Node degree centrality
            - closeness_centrality: Node closeness centrality
            - betweenness_centrality: Node betweenness centrality
        """
        nodes = graph.get_nodes()
        if not nodes:
            return {
                "degree_centrality": 0.0,
                "closeness_centrality": 0.0,
                "betweenness_centrality": 0.0,
            }

        degree_centrality = self._calculate_degree_centrality(graph)
        closeness_centrality = self._calculate_closeness_centrality(graph)
        betweenness_centrality = self._calculate_betweenness_centrality(graph)

        return {
            "degree_centrality": degree_centrality,
            "closeness_centrality": closeness_centrality,
            "betweenness_centrality": betweenness_centrality,
        }

    def _calculate_degree_centrality(self, graph: Graph) -> float:
        """Calculate average degree centrality."""
        nodes = graph.get_nodes()
        max_degree = len(nodes) - 1
        if max_degree == 0:
            return 0.0

        centralities = {node: graph.get_degree(node) / max_degree for node in nodes}
        return sum(centralities.values()) / len(nodes)

    def _calculate_closeness_centrality(self, graph: Graph) -> float:
        """Calculate average closeness centrality."""
        nodes = list(graph.get_nodes())
        if len(nodes) < 2:
            return 0.0

        total_centrality = 0.0
        for node in nodes:
            distances = 0
            reachable = 0

            for target in nodes:
                if target != node:
                    paths = graph.find_paths(node, target)
                    if paths:
                        distances += len(min(paths, key=len)) - 1
                        reachable += 1

            if reachable > 0:
                total_centrality += (reachable / (len(nodes) - 1)) * (
                    reachable / distances if distances > 0 else 0
                )

        return total_centrality / len(nodes)

    def _calculate_betweenness_centrality(self, graph: Graph) -> float:
        """
        Calculate approximate betweenness centrality.

        For large graphs, uses sampling to maintain performance.
        """
        nodes = list(graph.get_nodes())
        if len(nodes) < 3:
            return 0.0

        # For large graphs, sample node pairs
        if len(nodes) > 1000:
            import random

            sample_size = 1000
            node_pairs = [(random.choice(nodes), random.choice(nodes)) for _ in range(sample_size)]
        else:
            node_pairs = [(s, t) for s in nodes for t in nodes if s != t]

        betweenness = {node: 0.0 for node in nodes}

        for s, t in node_pairs:
            paths = graph.find_paths(s, t)
            if not paths:
                continue

            # Find shortest paths
            min_length = len(min(paths, key=len))
            shortest_paths = [p for p in paths if len(p) == min_length]

            # Count intermediate nodes
            for path in shortest_paths:
                for node in path[1:-1]:
                    betweenness[node] += 1 / len(shortest_paths)

        # Normalize
        n = len(nodes)
        scale = (n - 1) * (n - 2) if len(node_pairs) == n * (n - 1) else len(node_pairs)
        if scale > 0:
            for node in betweenness:
                betweenness[node] /= scale

        return sum(betweenness.values()) / len(nodes)


@dataclass
class CompositeMetricCalculator:
    """Manages multiple metric calculation strategies."""

    strategies: Dict[str, MetricStrategy] = field(default_factory=dict)

    def add_strategy(self, name: str, strategy: MetricStrategy) -> None:
        """Add a metric calculation strategy."""
        self.strategies[name] = strategy

    def remove_strategy(self, name: str) -> None:
        """Remove a metric calculation strategy."""
        if name in self.strategies:
            del self.strategies[name]

    def calculate_metrics(self, graph: Graph) -> Dict[str, Dict[str, float]]:
        """
        Calculate all metrics using registered strategies.

        Returns:
            Dict mapping strategy names to their metric results
        """
        return {name: strategy.calculate(graph) for name, strategy in self.strategies.items()}


class GraphWithMetrics(Graph):
    """Graph class with integrated metric calculation."""

    def __init__(self, edges: List[Edge], cache_size: int = 1000, cache_ttl: int = 3600):
        """Initialize graph with metric calculator."""
        super().__init__(edges, cache_size, cache_ttl)
        self.metric_calculator = CompositeMetricCalculator()

        # Add default metric strategies
        self.metric_calculator.add_strategy("clustering", ClusteringMetricStrategy())
        self.metric_calculator.add_strategy("centrality", CentralityMetricStrategy())

    def add_metric_strategy(self, name: str, strategy: MetricStrategy) -> None:
        """Add a custom metric strategy."""
        self.metric_calculator.add_strategy(name, strategy)

    def remove_metric_strategy(self, name: str) -> None:
        """Remove a metric strategy."""
        self.metric_calculator.remove_strategy(name)

    def calculate_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate all registered metrics."""
        return self.metric_calculator.calculate_metrics(self)
