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

from ..graph import Graph
from ..models import Edge
from .models import PathResult, PathValidationError

# Configure logging
logger = logging.getLogger(__name__)

# Type alias for weight functions
WeightFunc = Callable[[Edge], float]

# Constants
EPSILON = 1e-10  # Floating point comparison tolerance
MAX_QUEUE_SIZE = 100000  # Maximum size for priority queues


@dataclass(frozen=True)
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


@dataclass
class PathMetrics:
    """Container for path analysis metrics."""

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
    """Get current memory usage in bytes using psutil."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def is_better_cost(new_cost: float, old_cost: float) -> bool:
    """Compare costs with floating point tolerance."""
    return (old_cost - new_cost) > EPSILON


def get_edge_weight(edge: Edge, weight_func: Optional[WeightFunc] = None) -> float:
    """Get weight of an edge using optional weight function."""
    if weight_func is None:
        return 1.0

    try:
        weight = weight_func(edge)
        if not isinstance(weight, (int, float)):
            raise ValueError("Weight must be numeric")
        if weight <= 0 or math.isnan(weight) or math.isinf(weight):
            raise ValueError("Edge weight must be positive finite number")
        return float(weight)
    except Exception as e:
        raise ValueError(
            f"Invalid edge weight for edge {edge.from_entity}->{edge.to_entity}: {str(e)}"
        )


def calculate_path_weight(path: List[Edge], weight_func: Optional[WeightFunc] = None) -> float:
    """Calculate total weight of a path."""
    if not path:
        return 0.0

    try:
        if weight_func is None:
            return float(len(path))

        total = 0.0
        for edge in path:
            weight = get_edge_weight(edge, weight_func)
            new_total = total + weight
            if math.isinf(new_total) or math.isnan(new_total):
                raise ValueError("Path cost overflow")
            total = new_total
        return total
    except OverflowError:
        raise ValueError("Path cost exceeded maximum value")


def create_path_result(
    path: List[Edge], weight_func: Optional[WeightFunc], graph: Optional[Graph] = None
) -> PathResult:
    """Create PathResult from path."""
    total_weight = calculate_path_weight(path, weight_func)
    result = PathResult(path=path, total_weight=total_weight)
    if graph:
        result.validate(graph, weight_func)
    return result


class PriorityQueue:
    """Priority queue with decrease-key and peek functionality."""

    def __init__(self, maxsize: int = MAX_QUEUE_SIZE):
        self._queue: List[Tuple[float, int, str]] = []
        self._entry_finder: Dict[str, Tuple[float, int]] = {}
        self._counter = 0  # Unique counter to break ties
        self._maxsize = maxsize

    def add_or_update(self, item: str, priority: float) -> None:
        """Add a new item or update the priority of an existing item."""
        if len(self._entry_finder) >= self._maxsize:
            raise ValueError(f"Queue size exceeded limit of {self._maxsize}")

        logger.debug(f"PriorityQueue: Adding/updating '{item}' with priority {priority}")
        if item in self._entry_finder:
            old_priority, old_count = self._entry_finder[item]
            if not is_better_cost(priority, old_priority):
                logger.debug(f"  Keeping existing priority {old_priority} (better than {priority})")
                return
            # Mark the old entry as invalid
            self._entry_finder[item] = (float("inf"), old_count)

        # Add the new entry
        entry = (priority, self._counter, item)
        self._entry_finder[item] = (priority, self._counter)
        heappush(self._queue, entry)
        self._counter += 1
        logger.debug(f"  Current queue: {[f'({p}, {i}, {item})' for p, i, item in self._queue]}")

    def pop(self) -> Optional[Tuple[float, str]]:
        """Remove and return the item with the lowest priority."""
        while self._queue:
            priority, count, item = heappop(self._queue)
            stored_priority, stored_count = self._entry_finder.get(item, (None, None))
            if stored_priority == priority and stored_count == count:
                del self._entry_finder[item]
                logger.debug(f"Popped '{item}' with priority {priority}")
                return (priority, item)
            logger.debug(f"Skipping invalidated entry for '{item}' with priority {priority}")
        logger.debug("PriorityQueue is empty")
        return None

    def peek_priority(self) -> float:
        """
        Peek at the lowest priority value in the queue without removing it.
        Returns:
            The lowest priority value, or float('inf') if the queue is empty.
        """
        while self._queue:
            priority, count, item = self._queue[0]
            stored_priority, stored_count = self._entry_finder.get(item, (None, None))
            if stored_priority == priority and stored_count == count:
                logger.debug(f"Peeked priority: {priority} for item '{item}'")
                return priority
            # Remove invalid entries
            heappop(self._queue)
            logger.debug(f"Removed invalid entry during peek: '{item}'")
        logger.debug("PriorityQueue is empty during peek")
        return float("inf")

    def empty(self) -> bool:
        """Return True if the queue is empty."""
        is_empty = len(self._entry_finder) == 0
        logger.debug(f"PriorityQueue empty: {is_empty}")
        return is_empty

    def __len__(self) -> int:
        """Return the number of valid items in the queue."""
        return len(self._entry_finder)


class MemoryManager:
    """Memory management utilities for graph algorithms."""

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
            gc.collect()  # Try to reclaim memory
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
        self._peak_memory = get_memory_usage()

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
    weight_epsilon: float = EPSILON,
) -> None:
    """Validate path with comprehensive checks."""
    if not path:
        return

    # Check length constraints
    if max_length and len(path) > max_length:
        raise PathValidationError(f"Path length {len(path)} exceeds maximum {max_length}")

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

    # Check for cycles if not allowed
    if not allow_cycles:
        path_visited = set()
        path_visited.add(path[0].from_entity)
        for edge in path:
            if edge.to_entity in path_visited:
                raise PathValidationError(f"Cycle detected at node {edge.to_entity}")
            path_visited.add(edge.to_entity)

    # Verify edges exist in graph
    for edge in path:
        if not graph.has_edge(edge.from_entity, edge.to_entity):
            raise PathValidationError(f"Edge {edge.from_entity} -> {edge.to_entity} not in graph")

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
                    raise PathValidationError(
                        f"Invalid weight {weight} for edge "
                        f"{edge.from_entity} -> {edge.to_entity}"
                    )
                if weight <= 0:
                    raise PathValidationError(
                        f"Non-positive weight {weight} for edge "
                        f"{edge.from_entity} -> {edge.to_entity}"
                    )
                new_total = total_weight + weight
                if math.isinf(new_total) or math.isnan(new_total):
                    raise PathValidationError("Path cost overflow")
                total_weight = new_total
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
