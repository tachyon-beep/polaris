"""
Composite graph implementation combining all enhanced features.

This module provides a comprehensive graph implementation that combines
all the enhanced features:
- State management with change notifications
- Edge validation
- Metric calculation
- Flexible traversal
- Path caching with invalidation
"""

from typing import Dict, List, Optional, Set, Tuple, Union, Iterator, Callable

from .graph import Graph, GraphEvent, GraphStateListener
from .models import Edge
from .validation import ValidationStrategy, CyclicDependencyValidator, DuplicateEdgeValidator
from .validation import SelfLoopValidator, BidirectionalEdgeValidator, EdgeValidator
from .metrics import (
    CompositeMetricCalculator,
    ClusteringMetricStrategy,
    CentralityMetricStrategy,
    MetricStrategy,
)
from .traversal import GraphIterator
from .graph_paths.cache import PathCache
from .graph_paths.models import PathResult


class CompositeGraph(Graph):
    """
    Feature-rich graph implementation combining all enhancements.

    This class integrates all the enhanced features into a single implementation:
    - State management with change notifications
    - Edge validation with pluggable validators
    - Metric calculation with multiple strategies
    - Flexible traversal algorithms
    - Path caching with time-based invalidation

    Example:
        >>> graph = CompositeGraph([
        ...     Edge(from_entity="A", to_entity="B", relation_type=RelationType.CONNECTS_TO)
        ... ])
        >>> graph.add_metric_strategy("custom", CustomMetricStrategy())
        >>> metrics = graph.calculate_metrics()
        >>> for node, depth in graph.traverse("A", strategy="bfs"):
        ...     print(f"Node {node} at depth {depth}")
    """

    def __init__(self, edges: List[Edge], cache_size: int = 1000, cache_ttl: int = 3600):
        """
        Initialize composite graph.

        Args:
            edges: Initial edges
            cache_size: Maximum cache size
            cache_ttl: Cache time-to-live in seconds
        """
        super().__init__(edges, cache_size, cache_ttl)

        # Initialize validation
        self.validation_strategy = ValidationStrategy()
        self._setup_default_validators()

        # Initialize metrics
        self.metric_calculator = CompositeMetricCalculator()
        self._setup_default_metrics()

        # Initialize caching
        self._path_cache = PathCache()

    def _setup_default_validators(self) -> None:
        """Set up default edge validators."""
        self.validation_strategy.add_validator(CyclicDependencyValidator())
        self.validation_strategy.add_validator(DuplicateEdgeValidator())
        self.validation_strategy.add_validator(SelfLoopValidator())
        self.validation_strategy.add_validator(BidirectionalEdgeValidator())

    def _setup_default_metrics(self) -> None:
        """Set up default metric strategies."""
        self.metric_calculator.add_strategy("clustering", ClusteringMetricStrategy())
        self.metric_calculator.add_strategy("centrality", CentralityMetricStrategy())

    def add_edge(self, edge: Edge) -> None:
        """
        Add an edge with validation.

        Args:
            edge: Edge to add

        Raises:
            ValidationError: If edge fails validation
        """
        # Validate edge
        valid, errors = self.validation_strategy.validate(edge, self)
        if not valid:
            from .exceptions import ValidationError

            raise ValidationError(f"Edge validation failed: {'; '.join(errors)}")

        # Add edge and invalidate cache
        super().add_edge(edge)
        self._path_cache.clear()  # Clear entire cache when graph changes

    def find_paths(
        self, from_node: str, to_node: str, max_depth: Optional[int] = None
    ) -> List[List[str]]:
        """Find paths between nodes using cache."""
        cache_key = f"{from_node}|{to_node}|{max_depth}"
        cached_result = self._path_cache.get(cache_key)

        if cached_result is not None:
            return [[node for node in path] for path in cached_result.nodes]

        paths = super().find_paths(from_node, to_node, max_depth)

        # Convert paths to PathResult for caching
        edges = []
        total_weight = 0.0  # Default weight since we don't track weights in paths
        for path in paths:
            path_edges = []
            for i in range(len(path) - 1):
                edge = self.get_edge(path[i], path[i + 1])
                if edge:
                    path_edges.append(edge)
            edges.extend(path_edges)

        path_result = PathResult(path=edges, total_weight=total_weight)
        self._path_cache.put(cache_key, path_result)

        return paths

    def add_validator(self, validator: EdgeValidator) -> None:
        """Add a custom edge validator."""
        self.validation_strategy.add_validator(validator)

    def add_metric_strategy(self, name: str, strategy: MetricStrategy) -> None:
        """Add a custom metric calculation strategy."""
        self.metric_calculator.add_strategy(name, strategy)

    def remove_metric_strategy(self, name: str) -> None:
        """Remove a metric calculation strategy."""
        self.metric_calculator.remove_strategy(name)

    def calculate_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate all registered metrics."""
        return self.metric_calculator.calculate_metrics(self)

    def iterator(self, start_node: str, strategy: str = "bfs") -> GraphIterator:
        """
        Get an iterator for graph traversal.

        Args:
            start_node: Starting node
            strategy: Traversal strategy ('bfs', 'dfs', 'bidirectional', 'topological')

        Returns:
            Appropriate iterator instance
        """
        if not self.has_node(start_node):
            raise ValueError(f"Start node '{start_node}' not found in graph")

        from .traversal import BFSIterator, DFSIterator, BiDirectionalIterator, TopologicalIterator

        strategies = {
            "bfs": BFSIterator,
            "dfs": DFSIterator,
            "bidirectional": BiDirectionalIterator,
            "topological": TopologicalIterator,
        }

        if strategy not in strategies:
            raise ValueError(
                f"Unknown traversal strategy '{strategy}'. "
                f"Must be one of: {', '.join(strategies.keys())}"
            )

        return strategies[strategy](self, start_node)

    def traverse(
        self,
        start_node: str,
        strategy: str = "bfs",
        max_depth: Optional[int] = None,
        filter_func: Optional[Callable[[str], bool]] = None,
    ) -> Iterator[Tuple[str, int]]:
        """
        Traverse the graph with options.

        Args:
            start_node: Starting node
            strategy: Traversal strategy
            max_depth: Maximum depth
            filter_func: Optional function that takes a node ID and returns bool

        Returns:
            Iterator yielding (node_id, depth) tuples
        """
        iterator = self.iterator(start_node, strategy)

        for node, depth in iterator:
            if max_depth is not None and depth > max_depth:
                break
            if filter_func is None or filter_func(node):
                yield node, depth

    def get_cache_stats(self) -> Dict[str, Union[int, float]]:
        """Get path cache statistics."""
        return self._path_cache.get_metrics()
