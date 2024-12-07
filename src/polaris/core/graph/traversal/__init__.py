"""
Graph traversal algorithms and path finding functionality.
"""

import warnings
from typing import Any, Iterator, List, Optional, Type, Union

from .algorithms.all_paths import AllPathsFinder
from .algorithms.bidirectional import BidirectionalFinder
from .algorithms.shortest_path import ShortestPathFinder
from .base import PathFinder, PathFinding
from .cache import PathCache
from .path_models import PathResult, PathValidationError, PerformanceMetrics
from .types import PathFilter, PathType, WeightFunc, allow_negative_weights
from .utils import calculate_path_weight, create_path_result, get_edge_weight

# Constants
DEFAULT_MAX_PATH_LENGTH = 100

# Re-export types and classes
__all__ = [
    "PathType",
    "WeightFunc",
    "PathFilter",
    "PathResult",
    "PathValidationError",
    "PerformanceMetrics",
    "allow_negative_weights",
    "DEFAULT_MAX_PATH_LENGTH",
    "PathFinder",
    "PathFinding",
    "AllPathsFinder",
    "BidirectionalFinder",
    "ShortestPathFinder",
    "PathCache",
    "calculate_path_weight",
    "create_path_result",
    "get_edge_weight",
]


def deprecated_import_warning():
    warnings.warn(
        "Importing from polaris.core.graph_paths is deprecated. "
        "Use polaris.core.graph.traversal instead.",
        DeprecationWarning,
        stacklevel=2,
    )
