"""
Utility functions for path finding operations.

This module provides helper functions and classes used across different path finding
algorithms, including:
- Path result creation and validation
- Weight calculation
- Memory management
- Performance monitoring
- Priority queue with decrease-key operation
- Path state tracking
"""

import os
import time
import math
import psutil
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Callable, Generator, Any
from heapq import heappush, heappop

from ..graph import Graph
from ..models import Edge
from .models import PathResult, PathValidationError

# Type alias for weight functions
WeightFunc = Callable[[Edge], float]


@dataclass(frozen=True)
class PathState:
    """
    Memory-efficient immutable path state.

    Uses __slots__ for memory optimization and implements custom hash
    function for better performance in sets/dicts.

    Attributes:
        node: Current node ID
        prev_edge: Previous edge in path
        prev_state: Previous path state
        depth: Current path depth
        total_weight: Total path weight so far
    """

    __slots__ = ("node", "prev_edge", "prev_state", "depth", "total_weight")

    node: str
    prev_edge: Optional[Edge]
    prev_state: Optional["PathState"]
    depth: int
    total_weight: float

    def __hash__(self) -> int:
        """Custom hash combining node and state chain."""
        hash_val = hash(self.node)
        if self.prev_state:
            hash_val ^= hash(self.prev_state)
        return hash_val

    def get_path(self) -> List[Edge]:
        """Efficiently reconstruct path from state chain."""
        path = []
        current = self
        while current.prev_edge:
            path.append(current.prev_edge)
            current = current.prev_state
            if not current:
                break
        path.reverse()
        return path

    def get_visited(self) -> Set[str]:
        """Efficiently get set of visited nodes."""
        visited = set()
        current = self
        while current:
            visited.add(current.node)
            current = current.prev_state
        return visited


@dataclass
class PathMetrics:
    """
    Container for path analysis metrics.

    Attributes:
        length: Number of edges in path
        total_weight: Total path weight
        unique_nodes: Number of unique nodes
        has_cycles: Whether path contains cycles
        max_edge_weight: Maximum edge weight
        min_edge_weight: Minimum edge weight
    """

    length: int
    total_weight: float
    unique_nodes: int
    has_cycles: bool
    max_edge_weight: float
    min_edge_weight: float

    @classmethod
    def from_path(cls, path: List[Edge], weight_func: Optional[WeightFunc] = None) -> "PathMetrics":
        """Create metrics from path."""
        if not path:
            return cls(0, 0.0, 0, False, 0.0, 0.0)

        weights = [weight_func(e) if weight_func else getattr(e, "weight", 1.0) for e in path]
        nodes = set()
        for edge in path:
            nodes.add(edge.from_entity)
            nodes.add(edge.to_entity)

        return cls(
            length=len(path),
            total_weight=sum(weights),
            unique_nodes=len(nodes),
            has_cycles=len(nodes) < len(path) + 1,
            max_edge_weight=max(weights),
            min_edge_weight=min(weights),
        )


def get_memory_usage() -> int:
    """
    Get current memory usage in bytes using psutil.

    Returns:
        Current memory usage in bytes
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def get_edge_weight(edge: Edge, weight_func: Optional[WeightFunc] = None) -> float:
    """
    Get weight of an edge using optional weight function.

    Args:
        edge: Edge to get weight for
        weight_func: Optional function to calculate custom weight

    Returns:
        Weight of the edge

    Raises:
        ValueError: If weight function returns non-positive value

    Example:
        >>> weight = get_edge_weight(edge, lambda e: e.metadata.weight)
    """
    if weight_func is None:
        return 1.0

    weight = weight_func(edge)
    if weight <= 0:
        raise ValueError("Edge weight must be positive")
    return weight


def calculate_path_weight(path: List[Edge], weight_func: Optional[WeightFunc] = None) -> float:
    """
    Calculate total weight of a path.

    Args:
        path: List of edges forming the path
        weight_func: Optional function to calculate custom weights

    Returns:
        Total weight of the path

    Example:
        >>> total = calculate_path_weight(path, lambda e: e.metadata.weight)
    """
    if not path:
        return 0.0

    if weight_func is None:
        # Default to weight of 1.0 per edge
        return float(len(path))

    return sum(weight_func(edge) for edge in path)


def create_path_result(
    path: List[Edge], weight_func: Optional[WeightFunc], graph: Optional[Graph] = None
) -> PathResult:
    """
    Create PathResult from path.

    Creates a PathResult object containing the path and its properties,
    calculating total weight using provided weight function.

    Args:
        path: List of edges forming the path
        weight_func: Optional function to calculate custom weights
        graph: Optional graph instance for validation

    Returns:
        PathResult object containing the path and its properties

    Example:
        >>> result = create_path_result(path, lambda e: e.metadata.weight, graph)
        >>> print(f"Path length: {len(result)}")
        >>> print(f"Total weight: {result.total_weight}")
    """
    total_weight = calculate_path_weight(path, weight_func)
    result = PathResult(path=path, total_weight=total_weight)
    if graph:
        result.validate(graph, weight_func)
    return result


class PriorityQueue:
    """
    Priority queue with decrease-key operation.

    This implementation provides efficient decrease-key operation which is
    crucial for algorithms like Dijkstra's and A*.

    Example:
        >>> pq = PriorityQueue()
        >>> pq.add_or_update("A", 5.0)
        >>> pq.add_or_update("A", 3.0)  # Updates priority
        >>> priority, item = pq.pop()
        >>> print(f"Got {item} with priority {priority}")
    """

    def __init__(self):
        self._queue: List[Tuple[float, int, str]] = []
        self._entry_finder: Dict[str, Tuple[float, int]] = {}
        self._counter = 0

    def add_or_update(self, item: str, priority: float) -> None:
        """Add new item or update existing item's priority."""
        if item in self._entry_finder:
            old_priority, old_count = self._entry_finder[item]
            if priority >= old_priority:
                return  # Don't update if new priority is worse
            # Mark old entry as invalid
            self._entry_finder[item] = (float("inf"), old_count)

        entry = (priority, self._counter, item)
        self._entry_finder[item] = (priority, self._counter)
        self._counter += 1
        heappush(self._queue, entry)

    def pop(self) -> Optional[Tuple[float, str]]:
        """Remove and return lowest priority item."""
        while self._queue:
            priority, count, item = heappop(self._queue)
            stored_priority, stored_count = self._entry_finder.get(item, (None, None))
            if stored_priority == priority and stored_count == count:
                del self._entry_finder[item]
                return (priority, item)
        return None

    def empty(self) -> bool:
        """Return True if queue is empty."""
        return len(self._entry_finder) == 0

    def __len__(self) -> int:
        """Return number of valid items in queue."""
        return len(self._entry_finder)


class MemoryManager:
    """
    Memory management utilities for graph algorithms.

    Provides tools to monitor and manage memory usage during
    path finding operations.

    Example:
        >>> with MemoryManager(max_mb=1000) as mm:
        ...     # Your memory-intensive code here
        ...     mm.check_memory()  # Raises if limit exceeded
    """

    def __init__(self, max_memory_mb: Optional[float] = None):
        self.max_memory = max_memory_mb * 1024 * 1024 if max_memory_mb else None
        self.start_memory = get_memory_usage()
        self._peak_memory = self.start_memory

    def check_memory(self) -> None:
        """Check if memory usage exceeds limit."""
        if not self.max_memory:
            return

        current = get_memory_usage()
        self._peak_memory = max(self._peak_memory, current)

        if current - self.start_memory > self.max_memory:
            raise MemoryError(
                f"Memory usage {current/1024/1024:.1f}MB exceeds "
                f"limit of {self.max_memory/1024/1024:.1f}MB"
            )

    @property
    def peak_memory_mb(self) -> float:
        """Get peak memory usage in MB."""
        return self._peak_memory / 1024 / 1024

    @contextmanager
    def monitor_allocation(self, label: str = "") -> Generator[None, None, None]:
        """Context manager to monitor memory allocation."""
        start = get_memory_usage()
        yield
        end = get_memory_usage()
        allocated = end - start
        if allocated > 1024 * 1024:  # Log if over 1MB
            print(f"Memory allocated for {label}: {allocated/1024/1024:.1f}MB")


def validate_path(
    path: List[Edge],
    graph: Graph,
    weight_func: Optional[WeightFunc] = None,
    max_length: Optional[int] = None,
    allow_cycles: bool = False,
    weight_epsilon: float = 1e-9,
) -> None:
    """
    Validate path with comprehensive checks.

    Args:
        path: List of edges to validate
        graph: Graph instance for validation
        weight_func: Optional custom weight function
        max_length: Optional maximum path length
        allow_cycles: Whether to allow cycles in path
        weight_epsilon: Precision for weight comparisons

    Raises:
        PathValidationError: If validation fails
    """
    if not path:
        return

    # Check length constraints
    if max_length and len(path) > max_length:
        raise PathValidationError(f"Path length {len(path)} exceeds maximum {max_length}")

    # Check node connectivity
    for i in range(len(path) - 1):
        if path[i].to_entity != path[i + 1].from_entity:
            raise PathValidationError(
                f"Path discontinuity between edges {i} and {i+1}: "
                f"{path[i].to_entity} != {path[i + 1].from_entity}"
            )

    # Check for cycles if not allowed
    if not allow_cycles:
        visited = set()
        # Add starting node
        visited.add(path[0].from_entity)

        # Check each node in the path
        for edge in path:
            # Check if the next node creates a cycle
            if edge.to_entity in visited:
                raise PathValidationError(f"Cycle detected in path")
            visited.add(edge.to_entity)

    # Verify edges exist in graph
    for edge in path:
        if not graph.has_edge(edge.from_entity, edge.to_entity):
            raise PathValidationError(f"Edge {edge.from_entity} -> {edge.to_entity} not in graph")

    # Verify weights if function provided
    if weight_func:
        try:
            for edge in path:
                weight = weight_func(edge)
                if math.isnan(weight) or math.isinf(weight):
                    raise PathValidationError(
                        f"Invalid weight {weight} for edge "
                        f"{edge.from_entity} -> {edge.to_entity}"
                    )
        except Exception as e:
            raise PathValidationError(f"Weight calculation error: {str(e)}")


@contextmanager
def timer(label: str = "") -> Generator[None, None, None]:
    """Context manager for timing operations."""
    start = time.perf_counter()
    yield
    duration = time.perf_counter() - start
    if label:
        print(f"{label}: {duration*1000:.1f}ms")


def estimate_memory_usage(path: List[Edge]) -> int:
    """Estimate memory usage of path in bytes."""
    edge_size = 64  # Approximate Edge object size
    return 24 + (edge_size * len(path))  # List overhead + edges
