"""Path finding algorithm implementations."""

from typing import Iterator, List, Optional, Union

from ..path_models import PathResult
from ..types import PathFilter, WeightFunc
from .all_paths import AllPathsFinder
from .bidirectional import BidirectionalFinder
from .shortest_path import ShortestPathFinder

# Constants
DEFAULT_MAX_PATH_LENGTH = 100

__all__ = [
    "DEFAULT_MAX_PATH_LENGTH",
    "PathResult",
    "WeightFunc",
    "PathFilter",
    "AllPathsFinder",
    "BidirectionalFinder",
    "ShortestPathFinder",
]
