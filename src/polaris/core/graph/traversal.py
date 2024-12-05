"""
Graph traversal and path finding algorithms.

This module provides various algorithms for traversing graphs and finding paths
between nodes. It separates traversal logic from the core graph structure,
allowing for different traversal strategies to be implemented and used
interchangeably.
"""

from typing import Dict, List, Optional, Protocol, Set
from .base import BaseGraph


class PathFinder(Protocol):
    """Protocol defining interface for path finding algorithms."""

    def find_paths(
        self, graph: BaseGraph, start: str, end: str, max_depth: Optional[int] = None
    ) -> List[List[str]]:
        """
        Find paths between two nodes in the graph.

        Args:
            graph (BaseGraph): The graph to traverse
            start (str): Starting node
            end (str): Target node
            max_depth (Optional[int]): Maximum path length to consider

        Returns:
            List[List[str]]: List of paths, where each path is a list of node IDs
        """
        ...


class DFSPathFinder:
    """
    Depth-first search implementation of path finding.

    This class implements a depth-first search algorithm for finding all possible
    paths between two nodes in a graph, up to a specified maximum depth.
    """

    def find_paths(
        self, graph: BaseGraph, start: str, end: str, max_depth: Optional[int] = None
    ) -> List[List[str]]:
        """
        Find all paths between two nodes using depth-first search.

        This implementation uses recursive DFS to explore all possible paths
        between the start and end nodes, respecting the maximum depth limit
        if specified.

        Args:
            graph (BaseGraph): The graph to traverse
            start (str): Starting node
            end (str): Target node
            max_depth (Optional[int]): Maximum path length to consider

        Returns:
            List[List[str]]: List of paths, where each path is a list of node IDs
        """
        if not graph.has_node(start) or not graph.has_node(end):
            return []

        paths: List[List[str]] = []
        visited: Set[str] = set()

        def dfs(current: str, target: str, path: List[str], depth: int) -> None:
            """
            Recursive DFS helper function.

            Args:
                current (str): Current node being explored
                target (str): Target node we're trying to reach
                path (List[str]): Current path being built
                depth (int): Current depth in the search
            """
            if max_depth is not None and depth > max_depth:
                return
            if current == target:
                paths.append(path.copy())
                return

            visited.add(current)
            for neighbor in graph.get_neighbors(current):
                if neighbor not in visited:
                    path.append(neighbor)
                    dfs(neighbor, target, path, depth + 1)
                    path.pop()
            visited.remove(current)

        dfs(start, end, [start], 0)
        return paths


class BFSPathFinder:
    """
    Breadth-first search implementation of path finding.

    This class implements a breadth-first search algorithm for finding the
    shortest paths between two nodes in a graph.
    """

    def find_paths(
        self, graph: BaseGraph, start: str, end: str, max_depth: Optional[int] = None
    ) -> List[List[str]]:
        """
        Find shortest paths between two nodes using breadth-first search.

        This implementation uses iterative BFS to find the shortest paths
        between nodes, making it more suitable for finding optimal paths
        in unweighted graphs.

        Args:
            graph (BaseGraph): The graph to traverse
            start (str): Starting node
            end (str): Target node
            max_depth (Optional[int]): Maximum path length to consider

        Returns:
            List[List[str]]: List of shortest paths
        """
        if not graph.has_node(start) or not graph.has_node(end):
            return []

        # Queue of (node, path) pairs
        queue: List[tuple[str, List[str]]] = [(start, [start])]
        visited: Set[str] = {start}
        paths: List[List[str]] = []
        min_depth: Optional[int] = None

        while queue:
            current, path = queue.pop(0)
            current_depth = len(path) - 1

            # If we found a path, only look for others of the same length
            if min_depth is not None and current_depth > min_depth:
                break

            if max_depth is not None and current_depth > max_depth:
                continue

            if current == end:
                paths.append(path)
                min_depth = current_depth
                continue

            for neighbor in graph.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path))

        return paths


class BiDirectionalPathFinder:
    """
    Bidirectional search implementation of path finding.

    This class implements a bidirectional search strategy that explores
    from both the start and end nodes simultaneously, potentially finding
    paths more efficiently in large graphs.
    """

    def find_paths(
        self, graph: BaseGraph, start: str, end: str, max_depth: Optional[int] = None
    ) -> List[List[str]]:
        """
        Find paths between two nodes using bidirectional search.

        This implementation searches from both directions simultaneously,
        which can be more efficient than unidirectional search in many cases.

        Args:
            graph (BaseGraph): The graph to traverse
            start (str): Starting node
            end (str): Target node
            max_depth (Optional[int]): Maximum path length to consider

        Returns:
            List[List[str]]: List of paths found
        """
        if not graph.has_node(start) or not graph.has_node(end):
            return []

        # Forward and backward search frontiers
        forward: Dict[str, List[str]] = {start: [start]}
        backward: Dict[str, List[str]] = {end: [end]}
        paths: List[List[str]] = []

        def extend_path(
            current_paths: Dict[str, List[str]],
            other_paths: Dict[str, List[str]],
            reverse: bool = False,
        ) -> bool:
            """Extend paths in one direction and check for intersections."""
            new_paths: Dict[str, List[str]] = {}
            found_intersection = False

            for node, path in current_paths.items():
                neighbors = graph.get_neighbors(node, reverse=reverse)
                for neighbor in neighbors:
                    if max_depth is not None and len(path) >= max_depth:
                        continue

                    new_path = path + [neighbor]
                    if neighbor in other_paths:
                        # Found an intersection
                        other_path = other_paths[neighbor]
                        if reverse:
                            full_path = new_path + other_path[1:]
                        else:
                            full_path = other_path[:-1] + new_path
                        paths.append(full_path)
                        found_intersection = True
                    elif neighbor not in current_paths:
                        new_paths[neighbor] = new_path

            current_paths.update(new_paths)
            return found_intersection

        max_iterations = max_depth if max_depth is not None else float("inf")
        iteration = 0

        while forward and backward and iteration < max_iterations:
            if len(forward) <= len(backward):
                if extend_path(forward, backward):
                    break
            else:
                if extend_path(backward, forward, reverse=True):
                    break
            iteration += 1

        return paths
