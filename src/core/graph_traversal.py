"""
Graph traversal algorithms for the knowledge graph.

This module provides efficient implementations of fundamental graph traversal
algorithms:
- Breadth-first search (BFS) for level-by-level exploration
- Depth-first search (DFS) for deep exploration
- Memory-efficient iterative DFS implementation

The traversal algorithms support:
- Maximum depth limits
- Custom entity filtering
- Generator-based iteration for memory efficiency
- Depth tracking for each visited entity
- Configurable traversal strategies
"""

from collections import deque
from enum import Enum
from typing import Callable, Generator, Optional, Tuple

from .exceptions import NodeNotFoundError
from .graph import Graph

FilterFunc = Callable[[str], bool]


class TraversalStrategy(Enum):
    """
    Enumeration of available traversal strategies.

    Attributes:
        RECURSIVE_DFS: Traditional recursive depth-first search
        ITERATIVE_DFS: Memory-efficient iterative depth-first search
        BFS: Breadth-first search
    """

    RECURSIVE_DFS = "recursive_dfs"
    ITERATIVE_DFS = "iterative_dfs"
    BFS = "bfs"


class GraphTraversal:
    """
    Optimized graph traversal algorithms for knowledge graphs.

    This class provides static methods implementing fundamental graph traversal
    algorithms optimized for memory efficiency and flexibility. The traversals
    are implemented as generators to allow for efficient memory usage when
    processing large graphs.

    The algorithms support optional depth limits and entity filtering, making
    them suitable for various graph analysis tasks.
    """

    @staticmethod
    def _validate_start_node(graph: Graph, start_entity: str) -> None:
        """
        Validate that the start node exists in the graph.

        Args:
            graph (Graph): The graph instance
            start_entity (str): Starting entity ID

        Raises:
            NodeNotFoundError: If start_entity doesn't exist in the graph
        """
        if not graph.has_node(start_entity):
            raise NodeNotFoundError(f"Start node '{start_entity}' not found in the graph")

    @staticmethod
    def _check_filter(entity: str, filter_func: Optional[FilterFunc]) -> bool:
        """
        Check if an entity passes the filter function.

        Args:
            entity (str): Entity ID to check
            filter_func (Optional[FilterFunc]): Optional filter function

        Returns:
            bool: True if entity passes filter or no filter provided
        """
        return filter_func is None or filter_func(entity)

    @staticmethod
    def bfs(
        graph: Graph,
        start_entity: str,
        max_depth: Optional[int] = None,
        filter_func: Optional[FilterFunc] = None,
        chunk_size: int = 1000,
    ) -> Generator[Tuple[str, int], None, None]:
        """
        Memory-efficient breadth-first traversal of the graph.

        Performs a level-by-level traversal of the graph, visiting all entities
        at the current depth before moving to the next depth level. Uses chunked
        processing to maintain memory efficiency for large graphs.

        If a filter function is provided and the start node does not pass the filter,
        no traversal will occur and no nodes will be yielded. This ensures that
        traversal only proceeds from valid starting points.

        Args:
            graph (Graph): The graph instance to traverse
            start_entity (str): Starting entity ID for traversal
            max_depth (Optional[int]): Maximum traversal depth (None for unlimited)
            filter_func (Optional[FilterFunc]): Optional function to filter entities
            chunk_size (int): Size of chunks for processing large levels

        Yields:
            Tuple[str, int]: Pairs of (entity_id, depth) for each visited entity
                           that passes the filter

        Raises:
            NodeNotFoundError: If the start_entity does not exist in the graph

        Example:
            >>> def filter_func(entity_id: str) -> bool:
            ...     return entity_id > "C"  # Only include nodes after 'C'
            >>> for entity, depth in GraphTraversal.bfs(
            ...     graph, "D",  # D passes filter, traversal proceeds
            ...     max_depth=3,
            ...     filter_func=filter_func
            ... ):
            ...     print(f"Found {entity} at depth {depth}")
        """
        GraphTraversal._validate_start_node(graph, start_entity)

        # If start node doesn't pass filter, no traversal occurs
        if not GraphTraversal._check_filter(start_entity, filter_func):
            return

        visited = {start_entity}
        current_level = deque([(start_entity, 0)])
        next_level = deque()

        # Start node already passed filter check above
        yield start_entity, 0

        while current_level and (max_depth is None or current_level[0][1] < max_depth):
            # Process current level in chunks
            chunk_count = 0
            while current_level and chunk_count < chunk_size:
                current_entity, depth = current_level.popleft()
                chunk_count += 1

                neighbors = sorted(graph.get_neighbors(current_entity))
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.append((neighbor, depth + 1))
                        if GraphTraversal._check_filter(neighbor, filter_func):
                            yield neighbor, depth + 1

            # If current level is empty, swap with next level
            if not current_level and next_level:
                current_level, next_level = next_level, current_level

    @staticmethod
    def dfs(
        graph: Graph,
        start_entity: str,
        max_depth: Optional[int] = None,
        filter_func: Optional[FilterFunc] = None,
        strategy: TraversalStrategy = TraversalStrategy.ITERATIVE_DFS,
    ) -> Generator[Tuple[str, int], None, None]:
        """
        Configurable depth-first traversal of the graph.

        Performs a deep traversal of the graph using either recursive or iterative
        strategy. The iterative strategy is more memory-efficient and recommended
        for large graphs or deep traversals.

        Args:
            graph (Graph): The graph instance to traverse
            start_entity (str): Starting entity ID for traversal
            max_depth (Optional[int]): Maximum traversal depth (None for unlimited)
            filter_func (Optional[FilterFunc]): Optional function to filter entities
            strategy (TraversalStrategy): Traversal strategy to use

        Yields:
            Tuple[str, int]: Pairs of (entity_id, depth) for each visited entity

        Raises:
            NodeNotFoundError: If the start_entity does not exist in the graph
        """
        GraphTraversal._validate_start_node(graph, start_entity)

        if not GraphTraversal._check_filter(start_entity, filter_func):
            return

        if strategy == TraversalStrategy.RECURSIVE_DFS:
            yield from GraphTraversal._recursive_dfs(graph, start_entity, max_depth, filter_func)
        else:  # ITERATIVE_DFS
            yield from GraphTraversal._iterative_dfs(graph, start_entity, max_depth, filter_func)

    @staticmethod
    def _recursive_dfs(
        graph: Graph,
        start_entity: str,
        max_depth: Optional[int],
        filter_func: Optional[FilterFunc],
    ) -> Generator[Tuple[str, int], None, None]:
        """
        Recursive implementation of depth-first search.

        Note: This implementation may hit recursion limits on deep graphs.
        Consider using iterative_dfs for such cases.

        Args:
            graph (Graph): The graph instance
            start_entity (str): Starting entity ID
            max_depth (Optional[int]): Maximum depth limit
            filter_func (Optional[FilterFunc]): Optional filter function

        Yields:
            Tuple[str, int]: Pairs of (entity_id, depth)
        """
        visited = set()

        def dfs_helper(entity: str, depth: int) -> Generator[Tuple[str, int], None, None]:
            if max_depth is not None and depth > max_depth:
                return

            visited.add(entity)
            yield entity, depth

            neighbors = sorted(graph.get_neighbors(entity))
            for neighbor in neighbors:
                if neighbor not in visited:
                    if GraphTraversal._check_filter(neighbor, filter_func):
                        yield from dfs_helper(neighbor, depth + 1)

        yield from dfs_helper(start_entity, 0)

    @staticmethod
    def _iterative_dfs(
        graph: Graph,
        start_entity: str,
        max_depth: Optional[int],
        filter_func: Optional[FilterFunc],
    ) -> Generator[Tuple[str, int], None, None]:
        """
        Memory-efficient iterative implementation of depth-first search.

        This implementation uses a stack instead of recursion, making it
        suitable for deep graphs where recursive DFS might hit stack limits.

        Args:
            graph (Graph): The graph instance
            start_entity (str): Starting entity ID
            max_depth (Optional[int]): Maximum depth limit
            filter_func (Optional[FilterFunc]): Optional filter function

        Yields:
            Tuple[str, int]: Pairs of (entity_id, depth)
        """
        visited = set()
        stack = [(start_entity, 0, iter(sorted(graph.get_neighbors(start_entity))))]

        # Process start entity
        visited.add(start_entity)
        yield start_entity, 0

        while stack:
            _, depth, neighbors = stack[-1]

            if max_depth is not None and depth >= max_depth:
                stack.pop()
                continue

            # Try to get next neighbor
            try:
                neighbor = next(neighbors)
                if neighbor not in visited:
                    if GraphTraversal._check_filter(neighbor, filter_func):
                        visited.add(neighbor)
                        yield neighbor, depth + 1
                        stack.append(
                            (
                                neighbor,
                                depth + 1,
                                iter(sorted(graph.get_neighbors(neighbor))),
                            )
                        )
            except StopIteration:
                stack.pop()

    @staticmethod
    def traverse(
        graph: Graph,
        start_entity: str,
        strategy: TraversalStrategy = TraversalStrategy.BFS,
        max_depth: Optional[int] = None,
        filter_func: Optional[FilterFunc] = None,
    ) -> Generator[Tuple[str, int], None, None]:
        """
        Generic traversal method supporting multiple strategies.

        This method provides a unified interface for graph traversal,
        allowing the caller to specify the desired traversal strategy.

        Args:
            graph (Graph): The graph instance to traverse
            start_entity (str): Starting entity ID for traversal
            strategy (TraversalStrategy): Traversal strategy to use
            max_depth (Optional[int]): Maximum traversal depth
            filter_func (Optional[FilterFunc]): Optional filter function

        Yields:
            Tuple[str, int]: Pairs of (entity_id, depth)
        """
        if strategy == TraversalStrategy.BFS:
            yield from GraphTraversal.bfs(graph, start_entity, max_depth, filter_func)
        else:
            yield from GraphTraversal.dfs(graph, start_entity, max_depth, filter_func, strategy)
