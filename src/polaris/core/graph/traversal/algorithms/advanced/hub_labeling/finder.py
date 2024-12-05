"""
Path finding implementation for Hub Labeling algorithm.

This module handles the query phase of the Hub Labeling algorithm,
which uses precomputed labels to quickly find shortest paths.
"""

from typing import List, Optional, TYPE_CHECKING

from polaris.core.exceptions import GraphOperationError
from polaris.core.models import Edge
from polaris.core.graph.traversal.path_models import PathResult
from .models import HubLabelState
from .utils import compute_distance, reconstruct_path

if TYPE_CHECKING:
    from polaris.core.graph import Graph


class HubLabelPathFinder:
    """
    Path finder using Hub Labeling.

    This class handles the query phase, which uses precomputed
    labels to quickly find shortest paths between nodes.
    """

    def __init__(self, graph: "Graph", state: Optional[HubLabelState] = None):
        """
        Initialize path finder.

        Args:
            graph: Graph to find paths in
            state: Preprocessed hub label state
        """
        self.graph = graph
        self.state = state or HubLabelState()

    def find_path(
        self,
        start_node: str,
        end_node: str,
    ) -> PathResult:
        """
        Find shortest path between nodes.

        Args:
            start_node: Starting node
            end_node: Target node

        Returns:
            PathResult containing path edges and total weight

        Raises:
            GraphOperationError: If no path exists
        """
        path = self._find_path(start_node, end_node)
        total_weight = sum(edge.metadata.weight for edge in path)
        return PathResult(path=path, total_weight=total_weight)

    def _find_path(
        self,
        start_node: str,
        end_node: str,
    ) -> List[Edge]:
        """
        Find shortest path using hub labels.

        Args:
            start_node: Starting node
            end_node: Target node

        Returns:
            List of edges forming shortest path

        Raises:
            GraphOperationError: If no path exists
        """
        # Check if path exists using hub labels
        distance = compute_distance(start_node, end_node, self.state)
        if distance is None:
            raise GraphOperationError(f"No path exists between {start_node} and {end_node}")

        # Reconstruct path
        path = reconstruct_path(start_node, end_node, self.state, self.graph)
        if not path:
            raise GraphOperationError(
                f"Failed to reconstruct path between {start_node} and {end_node}"
            )

        return path
