import logging
from functools import lru_cache
from time import time
from typing import Any, Dict, List, Optional, Set, Tuple, TypedDict, Union, cast

from polaris.core.exceptions import GraphOperationError
from polaris.core.graph_paths.base import PathFinder
from polaris.core.graph_paths.models import PathResult, PerformanceMetrics
from polaris.core.graph_paths.types import PathFilter, WeightFunc
from polaris.core.graph_paths.utils import (
    MAX_QUEUE_SIZE,
    MemoryManager,
    PathState,
    PriorityQueue,
    create_path_result,
    get_edge_weight,
    is_better_cost,
    validate_path,
)
from polaris.core.models import Edge

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture detailed logs
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


class SearchStates(TypedDict):
    forward_queue: PriorityQueue
    forward_distances: Dict[str, float]
    forward_states: Dict[str, PathState]
    forward_explored: Set[str]
    backward_queue: PriorityQueue
    backward_distances: Dict[str, float]
    backward_states: Dict[str, PathState]
    backward_explored: Set[str]


class BidirectionalFinder(PathFinder[PathResult]):
    """
    Enhanced Bidirectional search implementation.

    This implementation supports both positive and negative weights through the
    allow_negative_weights decorator. When using negative weights (typically for
    maximization problems), the algorithm will find the path with the minimum
    total weight, which corresponds to the maximum value when weights are negated.

    Example:
        @allow_negative_weights
        def weight_func(edge: Edge) -> float:
            return -edge.impact_score  # Negate to find path with maximum impact

        # Find path with maximum total impact score
        path = PathFinding.bidirectional_search(graph, "A", "B", weight_func=weight_func)
    """

    def __init__(self, graph: Any, max_memory_mb: Optional[float] = None):
        """Initialize finder with optional memory limit."""
        super().__init__(graph)
        self.memory_manager = MemoryManager(max_memory_mb)
        logger.debug(
            "BidirectionalFinder initialized with graph: %s and max_memory_mb: %s",
            graph,
            max_memory_mb,
        )

    @lru_cache(maxsize=None)
    def _cached_get_edge_weight(
        self, from_entity: str, to_entity: str, weight_func: Optional[WeightFunc]
    ) -> float:
        """
        Cache the computation of edge weights to improve performance.

        Args:
            from_entity (str): Source node of the edge
            to_entity (str): Target node of the edge
            weight_func (Optional[WeightFunc]): Function to compute the weight of the edge.

        Returns:
            float: Weight of the edge.
        """
        edge = self.graph.get_edge(from_entity, to_entity)
        if not edge:
            logger.error("Edge from %s to %s not found in the graph.", from_entity, to_entity)
            raise GraphOperationError(f"Edge from {from_entity} to {to_entity} not found.")
        return get_edge_weight(edge, weight_func)

    def find_path(
        self,
        start_node: str,
        end_node: str,
        max_length: Optional[int] = None,
        max_paths: Optional[int] = None,
        filter_func: Optional[PathFilter] = None,
        weight_func: Optional[WeightFunc] = None,
        **kwargs,
    ) -> Optional[PathResult]:
        """Find path using bidirectional search."""
        logger.debug(
            "Starting find_path with start_node=%s, end_node=%s, max_length=%s, max_paths=%s",
            start_node,
            end_node,
            max_length,
            max_paths,
        )
        self.validate_nodes(start_node, end_node)
        metrics = self._initialize_metrics()
        max_depth = self._determine_max_depth(max_length, kwargs)

        try:
            if special_case_result := self._handle_special_case(
                max_depth, start_node, end_node, weight_func
            ):
                logger.debug("Handled special case with result: %s", special_case_result)
                return special_case_result

            search_states = self._initialize_search_states(start_node, end_node)
            best_path_info = self._search(
                start_node, end_node, max_depth, weight_func, metrics, search_states
            )

            metrics.end_time = time()
            logger.debug("Search completed with metrics: %s", metrics)

            if not best_path_info:
                error_msg = f"No path exists between {start_node} and {end_node}"
                logger.error(error_msg)
                raise GraphOperationError(error_msg)

            return self._reconstruct_and_validate_path(
                best_path_info, start_node, end_node, max_depth, filter_func, weight_func
            )

        except GraphOperationError as goe:
            logger.error("Graph operation error: %s", goe)
            raise
        except Exception as e:
            logger.exception("An unexpected error occurred during path finding.")
            raise GraphOperationError(f"An unexpected error occurred: {e}") from e
        finally:
            self.memory_manager.reset_peak_memory()
            logger.debug("Memory manager reset after path finding.")

    def _initialize_metrics(self) -> PerformanceMetrics:
        """Initialize performance metrics."""
        metrics = PerformanceMetrics(operation="bidirectional", start_time=time(), nodes_explored=0)
        logger.debug("PerformanceMetrics initialized: %s", metrics)
        return metrics

    def _determine_max_depth(
        self, max_length: Optional[int], kwargs: Dict[str, Any]
    ) -> Optional[int]:
        """Determine the maximum depth from max_length or kwargs."""
        max_depth = kwargs.get("max_depth", max_length)
        if max_depth is not None:
            if not isinstance(max_depth, int):
                error_msg = "max_depth must be an integer"
                logger.error(error_msg)
                raise TypeError(error_msg)
            if max_depth <= 0:
                error_msg = "max_depth must be positive"
                logger.error(error_msg)
                raise ValueError(error_msg)
            logger.debug("Maximum search depth set to: %d", max_depth)
        return max_depth

    def _handle_special_case(
        self, max_depth: Optional[int], start: str, end: str, weight_func: Optional[WeightFunc]
    ) -> Optional[PathResult]:
        """
        Handle the special case where max_depth is 1.

        Returns a direct path if exists, otherwise raises an error.
        """
        if max_depth == 1:
            logger.debug("Handling special case for max_depth=1")
            edge = self.graph.get_edge(start, end)
            if not edge:
                error_msg = f"No path of length <= 1 exists between {start} and {end}"
                logger.error(error_msg)
                raise GraphOperationError(error_msg)
            result = create_path_result([edge], weight_func)
            logger.debug("Direct path found for max_depth=1: %s", result)
            return result
        return None

    def _initialize_search_states(self, start: str, end: str) -> SearchStates:
        """Initialize forward and backward search states."""
        logger.debug("Initializing search states for bidirectional search.")
        forward_queue = PriorityQueue(maxsize=MAX_QUEUE_SIZE)
        forward_queue.add_or_update(start, 0.0)
        forward_distances = {start: 0.0}
        forward_states = {start: PathState(start, None, None, 0, 0.0)}
        forward_explored = set()

        backward_queue = PriorityQueue(maxsize=MAX_QUEUE_SIZE)
        backward_queue.add_or_update(end, 0.0)
        backward_distances = {end: 0.0}
        backward_states = {end: PathState(end, None, None, 0, 0.0)}
        backward_explored = set()

        search_states: SearchStates = {
            "forward_queue": forward_queue,
            "forward_distances": forward_distances,
            "forward_states": forward_states,
            "forward_explored": forward_explored,
            "backward_queue": backward_queue,
            "backward_distances": backward_distances,
            "backward_states": backward_states,
            "backward_explored": backward_explored,
        }

        logger.debug("Search states initialized.")
        return search_states

    def _search(
        self,
        start: str,
        end: str,
        max_depth: Optional[int],
        weight_func: Optional[WeightFunc],
        metrics: PerformanceMetrics,
        states: SearchStates,
    ) -> Optional[Tuple[str, PathState, PathState, float]]:
        """Perform the bidirectional search."""
        best_total_dist = float("inf")
        best_meeting_node: Optional[str] = None
        best_forward_state: Optional[PathState] = None
        best_backward_state: Optional[PathState] = None

        # Initialize nodes_explored if None
        if metrics.nodes_explored is None:
            metrics.nodes_explored = 0

        logger.debug("Beginning bidirectional search loop.")
        while not (states["forward_queue"].empty() and states["backward_queue"].empty()):
            self.memory_manager.check_memory()
            metrics.nodes_explored += 1
            logger.debug("Nodes explored so far: %d", metrics.nodes_explored)

            # Forward search step
            forward_result = self._process_search_step(
                current_queue=states["forward_queue"],
                current_distances=states["forward_distances"],
                current_states=states["forward_states"],
                current_explored=states["forward_explored"],
                other_distances=states["backward_distances"],
                other_states=states["backward_states"],
                other_explored=states["backward_explored"],
                is_forward=True,
                max_depth=max_depth,
                weight_func=weight_func,
            )

            if forward_result:
                node, state, other_state, total_dist = forward_result
                logger.debug("Forward search met at node %s with total_dist %f", node, total_dist)
                if is_better_cost(total_dist, best_total_dist):
                    if self._is_valid_meeting(node, state, other_state, max_depth):
                        best_total_dist = total_dist
                        best_meeting_node = node
                        best_forward_state = state
                        best_backward_state = other_state
                        logger.debug(
                            "New best meeting node: %s with total_dist: %f", node, total_dist
                        )
                        # Early termination if an optimal path is found
                        if self._can_terminate_early(
                            states["forward_queue"], states["backward_queue"], best_total_dist
                        ):
                            logger.debug("Early termination condition met.")
                            return (
                                best_meeting_node,
                                cast(PathState, best_forward_state),
                                cast(PathState, best_backward_state),
                                best_total_dist,
                            )

            # Backward search step
            backward_result = self._process_search_step(
                current_queue=states["backward_queue"],
                current_distances=states["backward_distances"],
                current_states=states["backward_states"],
                current_explored=states["backward_explored"],
                other_distances=states["forward_distances"],
                other_states=states["forward_states"],
                other_explored=states["forward_explored"],
                is_forward=False,
                max_depth=max_depth,
                weight_func=weight_func,
            )

            if backward_result:
                node, state, other_state, total_dist = backward_result
                logger.debug("Backward search met at node %s with total_dist %f", node, total_dist)
                if is_better_cost(total_dist, best_total_dist):
                    if self._is_valid_meeting(node, other_state, state, max_depth):
                        best_total_dist = total_dist
                        best_meeting_node = node
                        best_forward_state = other_state
                        best_backward_state = state
                        logger.debug(
                            "New best meeting node: %s with total_dist: %f", node, total_dist
                        )
                        # Early termination if an optimal path is found
                        if self._can_terminate_early(
                            states["forward_queue"], states["backward_queue"], best_total_dist
                        ):
                            logger.debug("Early termination condition met.")
                            return (
                                best_meeting_node,
                                cast(PathState, best_forward_state),
                                cast(PathState, best_backward_state),
                                best_total_dist,
                            )

        if best_meeting_node and best_forward_state and best_backward_state:
            logger.debug(
                "Best meeting node found: %s with total_dist: %f",
                best_meeting_node,
                best_total_dist,
            )
            return (
                best_meeting_node,
                best_forward_state,
                best_backward_state,
                best_total_dist,
            )
        logger.debug("No meeting node found after search completion.")
        return None

    def _process_search_step(
        self,
        current_queue: PriorityQueue,
        current_distances: Dict[str, float],
        current_states: Dict[str, PathState],
        current_explored: Set[str],
        other_distances: Dict[str, float],
        other_states: Dict[str, PathState],
        other_explored: Set[str],
        is_forward: bool,
        max_depth: Optional[int],
        weight_func: Optional[WeightFunc],
    ) -> Optional[Tuple[str, PathState, PathState, float]]:
        """
        Process a single search step for either forward or backward search.

        Returns information about a potential meeting node if found.
        """
        if current_queue.empty():
            logger.debug("Current queue is empty; no nodes to process.")
            return None

        current = current_queue.pop()
        if not current:
            logger.debug("No node retrieved from the queue.")
            return None
        current_dist, current_node = current
        logger.debug("Processing node %s with current_dist %f", current_node, current_dist)

        if current_node in current_explored:
            logger.debug("Node %s already explored; skipping.", current_node)
            return None
        current_explored.add(current_node)

        current_state = current_states[current_node]

        # Check for meeting point
        if current_node in other_distances:
            total_dist = current_dist + other_distances[current_node]
            other_state = other_states[current_node]
            logger.debug(
                "Meeting point found at node %s with total_dist %f", current_node, total_dist
            )
            return (current_node, current_state, other_state, total_dist)

        # Expand neighbors
        if max_depth is None or current_state.depth < max_depth:
            neighbors = self.graph.get_neighbors(current_node, reverse=not is_forward)
            logger.debug("Expanding neighbors for node %s: %s", current_node, neighbors)
            for neighbor in neighbors:
                if neighbor in current_state.get_visited():
                    logger.debug("Neighbor %s already visited in current path; skipping.", neighbor)
                    continue

                edge = (
                    self.graph.get_edge(neighbor, current_node)
                    if not is_forward
                    else self.graph.get_edge(current_node, neighbor)
                )
                if not edge:
                    logger.debug(
                        "No edge found between %s and %s; skipping.", current_node, neighbor
                    )
                    continue

                # Get edge endpoints for weight calculation
                from_entity = current_node if is_forward else neighbor
                to_entity = neighbor if is_forward else current_node

                try:
                    edge_weight = self._cached_get_edge_weight(from_entity, to_entity, weight_func)
                    new_dist = current_dist + edge_weight
                    new_depth = current_state.depth + 1

                    if neighbor not in current_distances or is_better_cost(
                        new_dist, current_distances[neighbor]
                    ):
                        current_distances[neighbor] = new_dist
                        current_states[neighbor] = PathState(
                            neighbor, edge, current_state, new_depth, new_dist
                        )
                        current_queue.add_or_update(neighbor, new_dist)
                        logger.debug(
                            "Updated neighbor %s with new_dist %f and new_depth %d",
                            neighbor,
                            new_dist,
                            new_depth,
                        )
                except ValueError as e:
                    if "Edge weight must be finite number" in str(e):
                        error_msg = "Path cost exceeded maximum value"
                        logger.error(error_msg)
                        raise GraphOperationError(error_msg) from e
                    logger.warning(
                        "Invalid edge weight for edge from %s to %s: %s", from_entity, to_entity, e
                    )
                    continue

        return None

    def _is_valid_meeting(
        self,
        node: str,
        forward_state: PathState,
        backward_state: PathState,
        max_depth: Optional[int],
    ) -> bool:
        """
        Check if the meeting at the node forms a valid path without cycles
        and respects the max_depth constraint.
        """
        total_depth = forward_state.depth + backward_state.depth
        logger.debug("Validating meeting at node %s with total_depth %d", node, total_depth)
        if max_depth is not None and total_depth > max_depth:
            logger.debug("Meeting node %s exceeds max_depth %d", node, max_depth)
            return False

        # Check for cycles in combined path
        forward_nodes = forward_state.get_visited()
        backward_nodes = backward_state.get_visited()

        # Only overlap at the meeting node is allowed
        if len(forward_nodes & backward_nodes) > 1:
            logger.debug("Meeting node %s introduces a cycle in the path.", node)
            return False

        return True

    def _reconstruct_and_validate_path(
        self,
        best_path_info: Tuple[str, PathState, PathState, float],
        start: str,
        end: str,
        max_depth: Optional[int],
        filter_func: Optional[PathFilter],
        weight_func: Optional[WeightFunc],
    ) -> Optional[PathResult]:
        """
        Reconstruct the path from the best meeting node and validate it.
        """
        node, forward_state, backward_state, _ = best_path_info  # Replace 'total_dist' with '_'
        logger.debug("Reconstructing path from meeting node %s", node)

        # Get forward path from start to meeting node
        forward_path = forward_state.get_path()

        # Get backward path from meeting node to end
        backward_path = backward_state.get_path()

        # For backward path edges, we need to find the forward edges
        forward_backward_path = []
        for edge in backward_path:
            # Get the edge in the forward direction
            forward_edge = self.graph.get_edge(edge.to_entity, edge.from_entity)
            if not forward_edge:
                # Try to get the edge in the original direction
                forward_edge = self.graph.get_edge(edge.from_entity, edge.to_entity)
            if not forward_edge:
                logger.error(
                    "Failed to find edge between %s and %s in either direction",
                    edge.from_entity,
                    edge.to_entity,
                )
                raise GraphOperationError(
                    f"Failed to find edge between {edge.from_entity} and {edge.to_entity}"
                )
            forward_backward_path.append(forward_edge)

        # Combine paths
        path = forward_path + forward_backward_path
        logger.debug("Combined path: %s", path)

        # Check if path exceeds max_depth
        if max_depth is not None and len(path) > max_depth:
            error_msg = f"No path of length <= {max_depth} exists between {start} and {end}"
            logger.error(error_msg)
            raise GraphOperationError(error_msg)

        # Apply filter if provided
        if filter_func:
            if not filter_func(path):
                logger.debug("Path does not satisfy the filter function.")
                return None
            logger.debug("Path satisfies the filter function.")

        result = create_path_result(path, weight_func)
        logger.debug("Path result created: %s", result)
        return result

    def _can_terminate_early(
        self, forward_queue: PriorityQueue, backward_queue: PriorityQueue, best_total_dist: float
    ) -> bool:
        """
        Determine if the search can terminate early based on current queues and best_total_dist.

        Args:
            forward_queue (PriorityQueue): The forward search priority queue.
            backward_queue (PriorityQueue): The backward search priority queue.
            best_total_dist (float): The current best total distance found.

        Returns:
            bool: True if the search can terminate early, False otherwise.
        """
        if forward_queue.empty() or backward_queue.empty():
            logger.debug("One of the queues is empty; cannot terminate early.")
            return False

        # Get minimum distances from queues
        forward_next = forward_queue.pop()
        backward_next = backward_queue.pop()

        if not forward_next or not backward_next:
            return False

        forward_min, forward_node = forward_next
        backward_min, backward_node = backward_next

        # Re-add the items back to the queues
        forward_queue.add_or_update(forward_node, forward_min)
        backward_queue.add_or_update(backward_node, backward_min)

        logger.debug(
            "Forward queue min distance: %f, Backward queue min distance: %f",
            forward_min,
            backward_min,
        )

        # If the sum of the minimum distances in both queues is greater than or equal to the best_total_dist, terminate
        if forward_min + backward_min >= best_total_dist:
            logger.debug(
                "Early termination condition met: %f + %f >= %f",
                forward_min,
                backward_min,
                best_total_dist,
            )
            return True

        return False
