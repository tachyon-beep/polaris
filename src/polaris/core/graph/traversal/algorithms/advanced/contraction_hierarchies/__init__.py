"""
Contraction Hierarchies implementation for fast path queries.

This module provides an implementation of Contraction Hierarchies, a preprocessing-based
speedup technique for shortest path queries.
"""

from typing import Callable, List, Optional, TYPE_CHECKING

from polaris.core.exceptions import GraphOperationError
from polaris.core.graph.traversal.base import PathFinder
from polaris.core.graph.traversal.path_models import PathResult
from polaris.core.graph.traversal.utils import WeightFunc
from polaris.core.models import Edge
from .models import ContractionState
from .preprocessor import ContractionPreprocessor
from .finder import ContractionPathFinder
from .storage import ContractionStorage

if TYPE_CHECKING:
    from polaris.core.graph import Graph


class ContractionHierarchies(PathFinder[PathResult]):
    """
    Contraction Hierarchies implementation for fast path queries.

    Features:
    - Preprocessing-based speedup technique
    - Efficient shortcut creation
    - Witness path search optimization
    - Memory-efficient path reconstruction
    """

    def __init__(self, graph: "Graph", max_memory_mb: Optional[float] = None):
        """Initialize with graph and optional memory limit."""
        super().__init__(graph)
        self.storage = ContractionStorage(max_memory_mb)
        self._state: Optional[ContractionState] = None
        self._preprocessed = False

    @property
    def state(self) -> ContractionState:
        """Get algorithm state, raising error if not preprocessed."""
        if not self._preprocessed or self._state is None:
            raise GraphOperationError("Graph must be preprocessed before accessing state")
        return self._state

    def preprocess(self) -> None:
        """
        Preprocess graph to build contraction hierarchy.

        This builds a hierarchy by contracting nodes in order of importance,
        adding shortcuts as necessary to preserve shortest paths.
        """
        preprocessor = ContractionPreprocessor(self.graph, self.storage)
        self._state = preprocessor.preprocess()
        self._preprocessed = True

    def find_path(
        self,
        start_node: str,
        end_node: str,
        max_length: Optional[int] = None,
        max_paths: Optional[int] = None,
        filter_func: Optional[Callable[[List[Edge]], bool]] = None,
        weight_func: Optional[WeightFunc] = None,
        **kwargs,
    ) -> PathResult:
        """
        Find shortest path using contraction hierarchies.

        Args:
            start_node: Starting node ID
            end_node: Target node ID
            max_length: Maximum path length
            max_paths: Not used (always returns single path)
            filter_func: Optional path filter
            weight_func: Optional weight function
            **kwargs: Additional options:
                validate: Whether to validate result (default: True)

        Returns:
            PathResult containing shortest path

        Raises:
            GraphOperationError: If no path exists or graph not preprocessed
        """
        if not self._preprocessed or self._state is None:
            raise GraphOperationError("Graph must be preprocessed before finding paths")

        finder = ContractionPathFinder(self.graph, self._state, self.storage)
        return finder.find_path(
            start_node,
            end_node,
            max_length=max_length,
            filter_func=filter_func,
            weight_func=weight_func,
            **kwargs,
        )
