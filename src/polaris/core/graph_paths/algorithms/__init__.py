"""Path finding algorithm implementations."""

from typing import List, Optional, Union, Iterator

from polaris.core.graph_paths.models import PathResult
from polaris.core.graph_paths.types import WeightFunc, PathFilter

# Constants
DEFAULT_MAX_PATH_LENGTH = 100

__all__ = [
    "DEFAULT_MAX_PATH_LENGTH",
    "PathResult",
    "WeightFunc",
    "PathFilter",
]
