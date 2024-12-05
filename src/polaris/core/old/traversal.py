"""
Graph traversal system using iterator pattern.

This module provides flexible traversal strategies for the graph using
the iterator pattern. It supports different traversal algorithms and
allows for custom traversal implementations.
"""

from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, Iterator, List, Optional, Set, Tuple, Callable

from .graph import Graph


class GraphIterator(ABC):
    """Base class for graph traversal iterators."""

    def __init__(self, graph: Graph, start_node: str):
        """
        Initialize iterator.

        Args:
            graph: The graph to traverse
            start_node: Starting node for traversal
        """
        self.graph = graph
        self.start = start_node
        self.visited: Set[str] = set()

    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[str, int]]:
        """
        Get iterator for traversal.

        Returns:
            Iterator yielding tuples of (node_id, depth)
        """
        pass


class BFSIterator(GraphIterator):
    """Breadth-first traversal iterator."""

    def __iter__(self) -> Iterator[Tuple[str, int]]:
        """
        Traverse graph in breadth-first order.

        Yields:
            Tuples of (node_id, depth) in BFS order
        """
        if not self.graph.has_node(self.start):
            return

        queue = deque([(self.start, 0)])
        self.visited.add(self.start)

        while queue:
            node, depth = queue.popleft()
            yield node, depth

            # Add unvisited neighbors to queue
            for neighbor in sorted(self.graph.get_neighbors(node)):
                if neighbor not in self.visited:
                    self.visited.add(neighbor)
                    queue.append((neighbor, depth + 1))


class DFSIterator(GraphIterator):
    """Depth-first traversal iterator."""

    def __iter__(self) -> Iterator[Tuple[str, int]]:
        """
        Traverse graph in depth-first order.

        Yields:
            Tuples of (node_id, depth) in DFS order
        """
        if not self.graph.has_node(self.start):
            return

        stack = [(self.start, 0, iter(sorted(self.graph.get_neighbors(self.start))))]
        self.visited.add(self.start)
        yield self.start, 0

        while stack:
            node, depth, neighbors = stack[-1]
            try:
                neighbor = next(neighbors)
                if neighbor not in self.visited:
                    self.visited.add(neighbor)
                    yield neighbor, depth + 1
                    stack.append(
                        (neighbor, depth + 1, iter(sorted(self.graph.get_neighbors(neighbor))))
                    )
            except StopIteration:
                stack.pop()


class BiDirectionalIterator(GraphIterator):
    """Bidirectional traversal iterator."""

    def __iter__(self) -> Iterator[Tuple[str, int]]:
        """
        Traverse graph in both forward and reverse directions.

        Yields:
            Tuples of (node_id, depth) in bidirectional order
        """
        if not self.graph.has_node(self.start):
            return

        forward_queue = deque([(self.start, 0)])
        reverse_queue = deque([(self.start, 0)])
        self.visited.add(self.start)
        yield self.start, 0

        while forward_queue or reverse_queue:
            # Process forward direction
            if forward_queue:
                node, depth = forward_queue.popleft()
                for neighbor in sorted(self.graph.get_neighbors(node)):
                    if neighbor not in self.visited:
                        self.visited.add(neighbor)
                        yield neighbor, depth + 1
                        forward_queue.append((neighbor, depth + 1))

            # Process reverse direction
            if reverse_queue:
                node, depth = reverse_queue.popleft()
                for neighbor in sorted(self.graph.get_neighbors(node, reverse=True)):
                    if neighbor not in self.visited:
                        self.visited.add(neighbor)
                        yield neighbor, depth + 1
                        reverse_queue.append((neighbor, depth + 1))


class TopologicalIterator(GraphIterator):
    """Topological sort traversal iterator."""

    def __iter__(self) -> Iterator[Tuple[str, int]]:
        """
        Traverse graph in topological order.

        Yields:
            Tuples of (node_id, depth) in topological order
        """
        if not self.graph.has_node(self.start):
            return

        # Calculate in-degree for each node
        in_degree: Dict[str, int] = {}
        for node in self.graph.get_nodes():
            in_degree[node] = len(self.graph.get_neighbors(node, reverse=True))

        # Initialize queue with nodes having no incoming edges
        queue = deque([(node, 0) for node in in_degree if in_degree[node] == 0])

        while queue:
            node, depth = queue.popleft()
            if node not in self.visited:
                self.visited.add(node)
                yield node, depth

                # Decrease in-degree of neighbors and add to queue if in-degree becomes 0
                for neighbor in self.graph.get_neighbors(node):
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append((neighbor, depth + 1))


class GraphWithTraversal(Graph):
    """Graph class with integrated traversal support."""

    def iterator(self, start_node: str, strategy: str = "bfs") -> GraphIterator:
        """
        Get an iterator for traversing the graph.

        Args:
            start_node: Starting node for traversal
            strategy: Traversal strategy ('bfs', 'dfs', 'bidirectional', or 'topological')

        Returns:
            Appropriate iterator instance

        Raises:
            ValueError: If strategy is not recognized
        """
        if not self.has_node(start_node):
            raise ValueError(f"Start node '{start_node}' not found in graph")

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
        Traverse the graph with optional depth limit and filtering.

        Args:
            start_node: Starting node for traversal
            strategy: Traversal strategy to use
            max_depth: Maximum depth to traverse (None for no limit)
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
