"""
Contraction Hierarchies implementation.

This module provides the main interface for the Contraction Hierarchies
algorithm, including preprocessing and path finding functionality.
"""

from typing import List, Optional, Callable, TYPE_CHECKING

from polaris.core.exceptions import GraphOperationError
from polaris.core.graph.traversal.path_models import PathResult
from polaris.core.graph.traversal.utils import WeightFunc
from polaris.core.models import Edge
from .finder import ContractionPathFinder
from .models import ContractionState
from .preprocessor import ContractionPreprocessor
from .storage import ContractionStorage

if TYPE_CHECKING:
    from polaris.core.graph import Graph


class ContractionHierarchies:
    """
    Contraction Hierarchies implementation.

    This class provides the main interface for using Contraction Hierarchies,
    including preprocessing the graph and finding shortest paths.
    """

    def __init__(self, graph: "Graph"):
        """
        Initialize Contraction Hierarchies.

        Args:
            graph: Graph to build hierarchy on
        """
        self.graph: "Graph" = graph
        self._state: Optional[ContractionState] = None  # Explicitly initialize as None
        self.storage: ContractionStorage = ContractionStorage(self._state or ContractionState())
        self.preprocessor: ContractionPreprocessor = ContractionPreprocessor(
            self.graph, self.storage
        )

    @property
    def state(self) -> ContractionState:
        """Get current algorithm state."""
        if self._state is None:
            raise GraphOperationError("Graph must be preprocessed before accessing state.")
        return self._state

    def preprocess(self) -> None:
        """Preprocess graph to build contraction hierarchy."""
        # Clear existing state when preprocessing
        self.storage.clear()
        self._state = self.preprocessor.preprocess()  # Assign new state

    def find_path(
        self,
        start_node: str,
        end_node: str,
        max_length: Optional[int] = None,
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
            filter_func: Optional path filter
            weight_func: Optional weight function
            **kwargs: Additional options:
                validate: Whether to validate result (default: True)
                allow_cycles: Whether to allow cycles in path (default: True)

        Returns:
            PathResult containing shortest path

        Raises:
            GraphOperationError: If no path exists or graph not preprocessed
        """
        finder = ContractionPathFinder(self.graph, self.state, self.storage)
        return finder.find_path(
            start_node=start_node,
            end_node=end_node,
            max_length=max_length,
            filter_func=filter_func,
            weight_func=weight_func,
            allow_cycles=kwargs.get("allow_cycles", True),  # Default to allowing cycles
            validate=kwargs.get("validate", True),
        )
