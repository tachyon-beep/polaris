"""Type definitions for graph path finding."""

from enum import Enum
from typing import Callable, List, Optional, Union
from functools import wraps

from ..models import Edge


class PathType(Enum):
    """Enumeration of path finding types."""

    SHORTEST = "shortest"  # Supports negative weights with Bellman-Ford
    ALL = "all"  # Supports negative weights
    ALL_PATHS = "all_paths"  # For backward compatibility
    BIDIRECTIONAL = "bidirectional"  # Supports negative weights
    DIJKSTRA = "dijkstra"  # Positive weights only
    BELLMAN_FORD = "bellman_ford"  # Supports negative weights
    A_STAR = "a_star"  # Positive weights only
    FILTERED = "filtered"  # For filtered path finding


def allow_negative_weights(func: Callable[[Edge], float]) -> Callable[[Edge], float]:
    """
    Decorator to mark weight functions that allow negative weights.

    This decorator is used to indicate that a weight function intentionally returns
    negative weights, typically for maximization problems solved via minimization.
    Without this decorator, negative weights will raise a ValueError.

    Note: Not all algorithms support negative weights. Supported algorithms:
    - Bellman-Ford (shortest_path)
    - All Paths
    - Bidirectional Search

    Unsupported algorithms (require positive weights):
    - Dijkstra's Algorithm
    - A* and ALT
    - Contraction Hierarchies

    Example:
        @allow_negative_weights
        def weight_func(edge: Edge) -> float:
            return -edge.impact_score  # Negate to find path with maximum impact

        # This weight function can be used with path finding algorithms
        # that support negative weights to find paths with maximum total impact
        path = PathFinding.shortest_path(graph, "A", "B", weight_func=weight_func)

    Args:
        func: The weight function to mark as allowing negative weights.
            Must take an Edge parameter and return a float.

    Returns:
        The decorated weight function.
    """
    func._allow_negative = True  # type: ignore
    return func


# Type alias for weight functions
WeightFunc = Callable[[Edge], float]

# Type alias for path filter functions
PathFilter = Callable[[List[Edge]], bool]

# Type alias for path result
PathResult = Union[List[Edge], None]

# Type alias for path length
PathLength = Optional[int]
