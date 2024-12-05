from typing import Iterator, TypeVar, Tuple, List, Dict, Set
from dataclasses import dataclass, field
from statistics import mean, median
from datetime import datetime
import math

from .graph import Graph
from .models import Edge

T = TypeVar("T")


@dataclass
class StreamingStats:
    """Maintains streaming statistics using Welford's online algorithm."""

    count: int = 0
    _mean: float = 0.0
    _m2: float = 0.0
    _min: float = float("inf")
    _max: float = float("-inf")
    _values: List[float] = field(default_factory=list)

    def update(self, value: float) -> None:
        """Update streaming statistics with a new value."""
        self.count += 1
        self._values.append(value)
        self._min = min(self._min, value)
        self._max = max(self._max, value)

        # Update running mean and variance using Welford's algorithm
        delta = value - self._mean
        self._mean += delta / self.count
        delta2 = value - self._mean
        self._m2 += delta * delta2

    @property
    def mean(self) -> float:
        """Get the current mean value."""
        return self._mean if self.count > 0 else 0.0

    @property
    def variance(self) -> float:
        """Get the current variance."""
        if self.count <= 1:
            return 0.0
        return self._m2 / (self.count - 1)

    @property
    def std_dev(self) -> float:
        """Get the current standard deviation."""
        return math.sqrt(self.variance)

    @property
    def median(self) -> float:
        """Get the current median value."""
        if not self._values:
            return 0.0
        sorted_values = sorted(self._values)
        mid = len(sorted_values) // 2
        if len(sorted_values) % 2 == 0:
            return (sorted_values[mid - 1] + sorted_values[mid]) / 2
        return sorted_values[mid]

    @property
    def min(self) -> float:
        """Get the minimum value seen."""
        return self._min if self._min != float("inf") else 0.0

    @property
    def max(self) -> float:
        """Get the maximum value seen."""
        return self._max if self._max != float("-inf") else 0.0


class StreamingMetricsCalculator:
    """Calculates graph metrics in a streaming fashion."""

    def __init__(self, graph: Graph):
        self.graph = graph
        self.degree_stats = StreamingStats()
        self.path_length_stats = StreamingStats()
        self.clustering_stats = StreamingStats()
        self.betweenness_stats = StreamingStats()

    def process_metrics(self) -> Iterator[Tuple[str, float]]:
        """Process all metrics in a streaming fashion."""
        # Process degree distribution
        yield from self._process_degree_metrics()

        # Process path lengths
        yield from self._process_path_metrics()

        # Process clustering coefficients
        yield from self._process_clustering_metrics()

        # Process betweenness centrality
        yield from self._process_betweenness_metrics()

    def _process_degree_metrics(self) -> Iterator[Tuple[str, float]]:
        """Process degree-related metrics."""
        for node in self.graph.get_nodes():
            degree = float(len(list(self.graph.get_neighbors(node))))
            self.degree_stats.update(degree)
            yield "degree", degree

    def _process_path_metrics(self) -> Iterator[Tuple[str, float]]:
        """Process path-related metrics."""
        nodes = list(self.graph.get_nodes())
        for i, node in enumerate(nodes):
            # Only process a subset of pairs for efficiency
            for target in nodes[i + 1 :]:
                if node != target:
                    paths = self.graph.find_paths(node, target)
                    if paths:
                        min_length = float(min(len(path) - 1 for path in paths))
                        self.path_length_stats.update(min_length)
                        yield "path_length", min_length

    def _process_clustering_metrics(self) -> Iterator[Tuple[str, float]]:
        """Process clustering-related metrics."""
        for node in self.graph.get_nodes():
            coeff = self._calculate_clustering_coefficient(node)
            self.clustering_stats.update(coeff)
            yield "clustering", coeff

    def _process_betweenness_metrics(self) -> Iterator[Tuple[str, float]]:
        """Process betweenness centrality metrics."""
        betweenness = self._calculate_approximate_betweenness()
        for node, centrality in betweenness.items():
            self.betweenness_stats.update(centrality)
            yield "betweenness", centrality

    def _calculate_clustering_coefficient(self, node: str) -> float:
        """Calculate the clustering coefficient for a node."""
        neighbors = set(self.graph.get_neighbors(node))
        if len(neighbors) < 2:
            return 0.0

        possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
        actual_edges = 0.0

        for n1 in neighbors:
            n1_neighbors = set(self.graph.get_neighbors(n1))
            actual_edges += len(neighbors.intersection(n1_neighbors))

        actual_edges /= 2  # Each edge was counted twice
        return actual_edges / possible_edges if possible_edges > 0 else 0.0

    def _calculate_approximate_betweenness(self, sample_size: int = 10) -> Dict[str, float]:
        """Calculate approximate betweenness centrality using sampling."""
        betweenness: Dict[str, float] = {node: 0.0 for node in self.graph.get_nodes()}
        nodes = list(self.graph.get_nodes())
        sample_size = min(sample_size, len(nodes))

        if not nodes:
            return betweenness

        # Sample nodes for efficiency
        import random

        sampled_nodes = random.sample(nodes, sample_size)

        for source in sampled_nodes:
            # Run BFS from each sampled node
            visited: Set[str] = {source}
            queue: List[Tuple[str, List[str]]] = [(source, [source])]

            while queue:
                current, path = queue.pop(0)
                for neighbor in self.graph.get_neighbors(current):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        new_path = path + [neighbor]
                        queue.append((neighbor, new_path))

                        # Update betweenness for nodes in the path
                        for node in path[1:-1]:  # Exclude source and target
                            betweenness[node] += 1.0

        # Normalize values
        max_val = max(betweenness.values())
        if max_val > 0:
            return {k: v / max_val for k, v in betweenness.items()}
        return betweenness

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get a summary of all calculated metrics."""
        return {
            "degree": {
                "mean": self.degree_stats.mean,
                "median": self.degree_stats.median,
                "std_dev": self.degree_stats.std_dev,
                "min": self.degree_stats.min,
                "max": self.degree_stats.max,
            },
            "path_length": {
                "mean": self.path_length_stats.mean,
                "median": self.path_length_stats.median,
                "std_dev": self.path_length_stats.std_dev,
                "min": self.path_length_stats.min,
                "max": self.path_length_stats.max,
            },
            "clustering": {
                "mean": self.clustering_stats.mean,
                "median": self.clustering_stats.median,
                "std_dev": self.clustering_stats.std_dev,
                "min": self.clustering_stats.min,
                "max": self.clustering_stats.max,
            },
            "betweenness": {
                "mean": self.betweenness_stats.mean,
                "median": self.betweenness_stats.median,
                "std_dev": self.betweenness_stats.std_dev,
                "min": self.betweenness_stats.min,
                "max": self.betweenness_stats.max,
            },
        }
