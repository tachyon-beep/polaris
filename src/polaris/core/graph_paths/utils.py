"""
Utility functions for path finding operations.
"""

import os
import time
import math
import psutil
import logging
import gc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Callable, Generator, Any
from heapq import heappush, heappop
from functools import wraps

from polaris.core.graph import Graph
from polaris.core.models import Edge
from polaris.core.graph_paths.models import PathResult, PathValidationError
from polaris.core.graph_paths.types import WeightFunc, allow_negative_weights

# Configure logging
logger = logging.getLogger(__name__)

# Constants
EPSILON = 1e-10  # Floating point comparison tolerance
MAX_QUEUE_SIZE = 100000  # Maximum size for priority queues
PATH_COST_EXCEEDED_MSG = "Path cost exceeded maximum value"


def get_edge_weight(edge: Edge, weight_func: Optional[WeightFunc] = None) -> float:
    """Get weight of an edge using optional weight function."""
    if weight_func is None:
        return 1.0

    try:
        weight = weight_func(edge)
        if not isinstance(weight, (int, float)):
            raise ValueError("Weight must be numeric")
        if math.isnan(weight) or math.isinf(weight):
            raise ValueError("Edge weight must be finite number")
        if weight == 0:  # Zero weights are never allowed
            raise ValueError("Edge weight must be non-zero")
        # Only enforce positive weights for standard weight functions
        # Negative weights are allowed for maximization via minimization
        if not getattr(weight_func, "_allow_negative", False) and weight <= 0:
            raise ValueError("Edge weight must be positive")
        return float(weight)
    except Exception as e:
        # Preserve original error message
        raise ValueError(str(e))


def is_better_cost(new_cost: float, old_cost: float) -> bool:
    """Compare costs with floating point tolerance.

    For positive weights (minimization):
        Returns True if new_cost < old_cost
        Example: new=2, old=3 returns True because 2 < 3

    For negative weights (maximization):
        Returns True if new_cost < old_cost (more negative is better)
        Example: new=-3, old=-2 returns True because -3 < -2
    """
    assert isinstance(new_cost, float) and isinstance(old_cost, float), "Costs must be floats"
    # Always use the same comparison - for both positive and negative weights,
    # we want the smaller value (more negative = better for negative weights)
    return (new_cost - old_cost) < -EPSILON


def calculate_path_weight(path: List[Edge], weight_func: Optional[WeightFunc] = None) -> float:
    """Calculate total weight of a path."""
    if not path:
        return 0.0

    total = 0.0
    for edge in path:
        weight = get_edge_weight(edge, weight_func)
        new_total = total + weight
        if math.isinf(new_total) or math.isnan(new_total):
            raise ValueError("Path cost overflow")
        total = new_total
    return total


def create_path_result(
    path: List[Edge], weight_func: Optional[WeightFunc], graph: Optional[Graph] = None
) -> PathResult:
    """Create PathResult from path."""
    total_weight = calculate_path_weight(path, weight_func)
    result = PathResult(path=path, total_weight=total_weight)
    if graph:
        validate_path(result.path, graph, weight_func)
    return result


def validate_path(
    path: List[Edge],
    graph: Graph,
    weight_func: Optional[WeightFunc] = None,
    max_length: Optional[int] = None,
    allow_cycles: bool = False,
    weight_epsilon: float = EPSILON,
) -> None:
    """Validate path with comprehensive checks."""
    if not path:
        return

    # Validate parameters
    if max_length is not None:
        if not isinstance(max_length, int):
            raise TypeError("max_length must be an integer")
        if max_length < 0:
            raise ValueError("max_length must be non-negative")
        if len(path) > max_length:
            raise PathValidationError(f"Path length {len(path)} exceeds maximum {max_length}")

    if not isinstance(weight_epsilon, (int, float)):
        raise TypeError("weight_epsilon must be a numeric value")
    if weight_epsilon <= 0:
        raise ValueError("weight_epsilon must be positive")

    # Validate edge types
    if not all(isinstance(edge, Edge) for edge in path):
        raise TypeError("path must contain only Edge objects")

    # Check existence of all nodes
    if not graph.has_node(path[0].from_entity):
        raise PathValidationError(f"Starting node {path[0].from_entity} not in graph")

    for edge in path:
        if not graph.has_node(edge.to_entity):
            raise PathValidationError(f"Node {edge.to_entity} not in graph")

    # Check node connectivity
    for i in range(len(path) - 1):
        if path[i].to_entity != path[i + 1].from_entity:
            raise PathValidationError(
                f"Path discontinuity between edges {i} and {i+1}: "
                f"{path[i].to_entity} != {path[i + 1].from_entity}"
            )

    # Check for self-loops and cycles
    path_visited = set()
    path_visited.add(path[0].from_entity)
    for edge in path:
        if edge.from_entity == edge.to_entity:
            raise PathValidationError(f"Self-loop detected at node {edge.from_entity}")
        if not allow_cycles and edge.to_entity in path_visited:
            raise PathValidationError(f"Cycle detected at node {edge.to_entity}")
        path_visited.add(edge.to_entity)

    # Verify edges exist in graph
    for edge in path:
        if not graph.has_edge(edge.from_entity, edge.to_entity):
            raise PathValidationError(
                f"Edge from {edge.from_entity} to {edge.to_entity} not found in graph"
            )

    # Verify weights if function provided
    if weight_func:
        try:
            total_weight = 0.0
            for edge in path:
                weight = weight_func(edge)
                if not isinstance(weight, (int, float)):
                    raise PathValidationError(
                        f"Invalid weight type for edge {edge.from_entity}->{edge.to_entity}"
                    )
                if math.isnan(weight) or math.isinf(weight):
                    raise PathValidationError(PATH_COST_EXCEEDED_MSG)
                if weight == 0:
                    raise PathValidationError("Edge weight must be non-zero")
                new_total = total_weight + weight
                if math.isinf(new_total) or math.isnan(new_total):
                    raise PathValidationError(PATH_COST_EXCEEDED_MSG)
                total_weight = new_total
        except Exception as e:
            if "Edge weight must be finite number" in str(e):
                raise PathValidationError(PATH_COST_EXCEEDED_MSG)
            raise PathValidationError(f"Weight calculation error: {str(e)}")


@contextmanager
def timer(label: str = "") -> Generator[None, None, None]:
    """Context manager for timing operations."""
    start = time.perf_counter()
    yield
    duration = time.perf_counter() - start
    if label:
        print(f"{label}: {duration*1000:.1f}ms")


@dataclass
class PathState:
    """Memory-efficient immutable path state."""

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


class PriorityQueue:
    """Priority queue with decrease-key and peek functionality."""

    def __init__(self, maxsize: int = MAX_QUEUE_SIZE):
        self._queue: List[Tuple[float, int, str]] = []
        self._entry_finder: Dict[str, Tuple[float, int]] = {}
        self._counter = 0  # Unique counter to break ties
        self._maxsize = maxsize

    def add_or_update(self, item: str, priority: float) -> None:
        if item in self._entry_finder:
            old_priority, old_count = self._entry_finder[item]
            # Only update if new priority is lower (better)
            if not is_better_cost(priority, old_priority):
                return
            # Mark old entry as invalid
            self._entry_finder[item] = (float("inf"), old_count)

        # Add new entry
        entry = (priority, self._counter, item)
        self._entry_finder[item] = (priority, self._counter)
        heappush(self._queue, entry)
        self._counter += 1

    def pop(self) -> Optional[Tuple[float, str]]:
        """Remove and return the item with the lowest priority."""
        while self._queue:
            priority, count, item = heappop(self._queue)
            stored_priority, stored_count = self._entry_finder.get(item, (None, None))
            if stored_priority == priority and stored_count == count:
                del self._entry_finder[item]
                return (priority, item)
        return None

    def empty(self) -> bool:
        """Return True if the queue is empty."""
        return len(self._entry_finder) == 0

    def __len__(self) -> int:
        """Return the number of valid items in the queue."""
        return len(self._entry_finder)


class MemoryManager:
    """Memory management utilities for graph algorithms."""

    def __init__(self, max_memory_mb: Optional[float] = None):
        """Initialize memory manager."""
        # Force garbage collection at start
        gc.collect()

        self.max_memory = max_memory_mb * 1024 * 1024 if max_memory_mb else None
        self.start_memory = get_memory_usage()
        self._peak_memory = self.start_memory
        self._last_check = time.time()
        self._check_interval = 0.1  # Check memory every 100ms

    def check_memory(self) -> None:
        """Check if memory usage exceeds limit."""
        current_time = time.time()
        if current_time - self._last_check < self._check_interval:
            return

        self._last_check = current_time
        if not self.max_memory:
            return

        current = get_memory_usage()
        self._peak_memory = max(self._peak_memory, current)

        if current - self.start_memory > self.max_memory:
            # Try to reclaim memory
            gc.collect()
            gc.collect()  # Second collection for cyclic references
            current = get_memory_usage()

            if current - self.start_memory > self.max_memory:
                raise MemoryError(
                    f"Memory usage {current/1024/1024:.1f}MB exceeds "
                    f"limit of {self.max_memory/1024/1024:.1f}MB"
                )

    @property
    def peak_memory_mb(self) -> float:
        """Get peak memory usage in MB."""
        return self._peak_memory / 1024 / 1024

    def reset_peak_memory(self) -> None:
        """Reset peak memory tracking."""
        gc.collect()
        gc.collect()  # Double collection for better cleanup
        self._peak_memory = get_memory_usage()
        self.start_memory = self._peak_memory
        self._last_check = time.time()


def get_memory_usage() -> int:
    """Get current memory usage in bytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss
