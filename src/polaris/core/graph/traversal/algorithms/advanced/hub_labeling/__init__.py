"""
Hub Labeling algorithm implementation.

This module provides the main interface for the Hub Labeling algorithm,
which enables fast shortest path queries by precomputing distance labels.
"""

from typing import List, Optional, TYPE_CHECKING

from polaris.core.exceptions import GraphOperationError
from polaris.core.models import Edge
from polaris.core.graph.traversal.path_models import PathResult
from .finder import HubLabelPathFinder
from .preprocessor import HubLabelPreprocessor
from .storage import HubLabelStorage
from .models import HubLabelState

if TYPE_CHECKING:
    from polaris.core.graph import Graph


class HubLabeling:
    """
    Hub Labeling algorithm implementation.

    This class provides the main interface for using Hub Labeling:
    1. Preprocess the graph to compute distance labels
    2. Use labels to quickly answer shortest path queries
    """

    def __init__(self, graph: "Graph"):
        """
        Initialize Hub Labeling algorithm.

        Args:
            graph: Graph to run algorithm on
        """
        self.graph = graph
        self.storage = HubLabelStorage()
        self.state = HubLabelState()  # Initialize state immediately

    def preprocess(self) -> None:
        """
        Preprocess graph to compute hub labels.

        This computes forward and backward labels for each node using
        a pruned labeling approach.
        """
        preprocessor = HubLabelPreprocessor(self.graph, self.storage)
        preprocessor.preprocess()
        self.state = preprocessor.state  # Update state with preprocessed labels

    def find_path(
        self,
        start_node: str,
        end_node: str,
    ) -> PathResult:
        """
        Find shortest path between nodes using hub labels.

        Args:
            start_node: Starting node
            end_node: Target node

        Returns:
            PathResult containing path edges and total weight

        Raises:
            GraphOperationError: If no path exists or preprocessing hasn't been run
        """
        if not self.state.get_nodes():  # Check if preprocessing has been run
            raise GraphOperationError("Must run preprocessing before finding paths")

        finder = HubLabelPathFinder(self.graph, self.state)
        return finder.find_path(start_node, end_node)
