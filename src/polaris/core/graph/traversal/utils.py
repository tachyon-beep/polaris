"""
Utility functions for path finding operations.
"""

import gc
import logging
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Dict, Generator, List, Optional, Set, Tuple, TYPE_CHECKING, Iterator

import psutil  # type: ignore # Missing stubs

from .path_models import PathResult, PathValidationError
from .types import WeightFunc, allow_negative_weights  # Re-export allow_negative_weights
from polaris.core.models import Edge

if TYPE_CHECKING:
    from polaris.core.graph import Graph

# Configure logging
logger = logging.getLogger(__name__)

# Constants
EPSILON = 1e-10  # Floating point comparison tolerance
MAX_QUEUE_SIZE = 100000  # Maximum size for priority queues
PATH_COST_EXCEEDED_MSG = "Path cost exceeded maximum value"


def get_edge_weight(edge: Edge, weight_func: Optional[WeightFunc] = None) -> float:
    """Get weight of an edge using optional weight function."""
    if weight_func is None:
        return edge.metadata.weight

    try:
        weight = weight_func(edge)
        if not isinstance(weight, (int, float)):
            raise ValueError("Weight must be numeric")
        if math.isnan(weight) or math.isinf(weight):
            raise ValueError("Edge weight must be finite number")
        if weight == 0:  # Zero weights are never allowed
            raise ValueError("Edge weight must be non-zero")

        # For functions marked with allow_negative_weights, we don't check sign
        if not getattr(weight_func, "_allow_negative", False) and weight <= 0:
            raise ValueError("Edge weight must be positive")

        return float(weight)
    except Exception as e:
        raise ValueError(str(e))


def is_better_cost(new_cost: float, old_cost: float) -> bool:
    """Compare costs with floating point tolerance.

    For both standard weights and inverse weights (1/impact_score),
    we want to minimize the total weight. In the case of inverse weights,
    this corresponds to maximizing the total impact score.

    Example with inverse weights (1/impact_score):
    Path A->B->C->E: impact scores 0.8, 0.7, 0.4
        - Inverse weights = 1/0.8 + 1/0.7 + 1/0.4 ≈ 5.18

    Path A->D->E: impact scores 0.5, 0.3
        - Inverse weights = 1/0.5 + 1/0.3 ≈ 5.33

    A->B->C->E is better because 5.18 < 5.33
    This corresponds to higher total impact (1.9 > 0.8)

    Args:
        new_cost: The cost of the new path
        old_cost: The cost of the existing path

    Returns:
        True if new_cost is better than old_cost
    """
    assert isinstance(new_cost, float) and isinstance(old_cost, float), "Costs must be floats"
    # For both standard and inverse weights, smaller total is better
    # Use relative comparison for floating point numbers
    # This handles both small and large differences appropriately


def is_better_cost(new_cost: float, old_cost: float) -> bool:
    relative_diff = abs(new_cost - old_cost) / max(abs(new_cost), abs(old_cost))
    if relative_diff < EPSILON:
        return True  # When costs are very close, prefer the new path
    return new_cost < old_cost


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
    path: List[Edge], weight_func: Optional[WeightFunc], graph: Optional["Graph"] = None
) -> PathResult:
    """Create PathResult from path."""
    total_weight = calculate_path_weight(path, weight_func)
    # Debug: Print the path being created
    reconstructed_path = [f"{e.from_entity}->{e.to_entity}" for e in path]
    print(f"Creating PathResult with path: {reconstructed_path}, Total weight: {total_weight}")
    # Create result without validation - let the caller handle validation if needed
    return PathResult(path=path, total_weight=total_weight)


def validate_path(
    path: List[Edge],
    graph: "Graph",
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

    # Check node connectivity only if there's more than one edge
    if len(path) > 1:
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
        path: List[Edge] = []
        current: Optional[PathState] = self
        while current is not None and current.prev_edge is not None:
            path.append(current.prev_edge)
            current = current.prev_state
        path.reverse()
        return path

    def get_visited(self) -> Set[str]:
        """Efficiently get set of visited nodes."""
        visited: Set[str] = set()
        current: Optional[PathState] = self
        while current is not None:
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
        """Add a new item or update priority of existing item."""
        if item in self._entry_finder:
            old_priority, old_count = self._entry_finder[item]
            # Only update if new priority is better
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

    def peek(self) -> Optional[Tuple[float, str]]:
        """Return the item with lowest priority without removing it."""
        while self._queue:
            priority, count, item = self._queue[0]  # Look at top item
            stored_priority, stored_count = self._entry_finder.get(item, (None, None))
            if stored_priority == priority and stored_count == count:
                return (priority, item)
            # Remove invalid entry
            heappop(self._queue)
        return None

    def empty(self) -> bool:
        """Return True if the queue is empty."""
        return len(self._entry_finder) == 0

    def __len__(self) -> int:
        """Return the number of valid items in the queue."""
        return len(self._entry_finder)

    def __iter__(self) -> Iterator[Tuple[float, str]]:
        """Return iterator over queue items in priority order."""
        # Get all valid items
        valid_items: List[Tuple[float, str]] = []
        while not self.empty():
            item = self.pop()
            if item is not None:
                valid_items.append(item)

        # Restore items to queue
        for priority, node in valid_items:
            self.add_or_update(node, priority)

        return iter(valid_items)


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
    mem_info = process.memory_info()
    return int(mem_info.rss)  # Explicitly convert to int for type safety
