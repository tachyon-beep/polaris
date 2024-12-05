"""
Path finding implementation for Hub Labeling algorithm.

This module handles the query phase of the Hub Labeling algorithm,
which uses precomputed labels to quickly find shortest paths.
"""

from typing import List, Optional, TYPE_CHECKING

from polaris.core.exceptions import GraphOperationError
from polaris.core.models import Edge
from polaris.core.graph.traversal.path_models import PathResult
from polaris.core.graph.traversal.utils import get_edge_weight
from .models import HubLabelState
from .utils import (
    compute_distance,
    reconstruct_path,
    validate_path_continuity,
    validate_path_distance,
)

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

        Raises:
            ValueError: If graph is None
        """
        if graph is None:
            raise ValueError("Graph cannot be None")

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
            GraphOperationError: If no path exists or path is invalid
            ValueError: If input nodes are invalid
        """
        # Validate input nodes
        if not isinstance(start_node, str) or not start_node:
            raise ValueError("Invalid start node")
        if not isinstance(end_node, str) or not end_node:
            raise ValueError("Invalid end node")

        # Check if nodes exist in graph
        if start_node not in self.graph.get_nodes():
            raise GraphOperationError(f"Start node {start_node} not found in graph")
        if end_node not in self.graph.get_nodes():
            raise GraphOperationError(f"End node {end_node} not found in graph")

        # Check if preprocessing has been done
        if not self.state.get_nodes():
            raise GraphOperationError("No labels found. Run preprocessing before finding paths.")

        # Handle self-loops
        if start_node == end_node:
            return PathResult(path=[], total_weight=0.0)

        path = self._find_path(start_node, end_node)
        total_weight = sum(get_edge_weight(edge) for edge in path)

        # Validate result
        if not validate_path_continuity(path):
            raise GraphOperationError(f"Path between {start_node} and {end_node} is not continuous")

        expected_distance = compute_distance(start_node, end_node, self.state)
        if expected_distance is None:
            raise GraphOperationError(f"No valid path exists between {start_node} and {end_node}")

        if not validate_path_distance(path, expected_distance):
            raise GraphOperationError(
                f"Path distance {total_weight} does not match expected {expected_distance}"
            )

        return PathResult(
            path=path,
            total_weight=total_weight,
        )

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
            GraphOperationError: If no path exists or path reconstruction fails
        """
        # Try direct edge first
        direct_edge = self.graph.get_edge(start_node, end_node)
        if direct_edge:
            direct_dist = get_edge_weight(direct_edge)
            min_hub_dist = compute_distance(start_node, end_node, self.state)
            if min_hub_dist is None or direct_dist <= min_hub_dist:
                return [direct_edge]

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

    def validate_state(self) -> None:
        """
        Validate internal state consistency.

        Raises:
            GraphOperationError: If state is invalid
        """
        try:
            self.state.validate_state()
        except ValueError as e:
            raise GraphOperationError(f"Invalid hub label state: {str(e)}")
