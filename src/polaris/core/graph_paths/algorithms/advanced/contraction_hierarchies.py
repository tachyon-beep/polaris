"""
Contraction Hierarchies implementation for fast path queries.

This module provides an implementation of Contraction Hierarchies, a preprocessing-based
speedup technique for shortest path queries. Key features:
- Node contraction with shortcut creation
- Witness path search
- Importance-based node ordering
- Memory-efficient path reconstruction
"""

from dataclasses import dataclass
import time
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Callable
from heapq import heappush, heappop

from polaris.core.exceptions import GraphOperationError
from polaris.core.models import Edge, EdgeMetadata
from polaris.core.graph import Graph
from polaris.core.enums import RelationType
from polaris.core.graph_paths.base import PathFinder
from polaris.core.graph_paths.models import PathResult
from polaris.core.graph_paths.utils import (
    WeightFunc,
    MemoryManager,
    get_edge_weight,
    create_path_result,
    validate_path,
)


# Use CONNECTS_TO temporarily for shortcuts
SHORTCUT_TYPE = RelationType.CONNECTS_TO


@dataclass(frozen=True)
class Shortcut:
    """Represents a shortcut edge in the contraction hierarchy."""

    edge: Edge
    via_node: str
    lower_edge: Edge
    upper_edge: Edge


class ContractionHierarchies(PathFinder[PathResult]):
    """
    Contraction Hierarchies implementation for fast path queries.

    Features:
    - Preprocessing-based speedup technique
    - Efficient shortcut creation
    - Witness path search optimization
    - Memory-efficient path reconstruction
    """

    def __init__(self, graph: Graph, max_memory_mb: Optional[float] = None):
        """Initialize with graph and optional memory limit."""
        super().__init__(graph)
        self.memory_manager = MemoryManager(max_memory_mb)
        self.node_level: Dict[str, int] = {}
        self.shortcuts: Dict[Tuple[str, str], Shortcut] = {}
        self.contracted_neighbors: Dict[str, Set[str]] = {}
        self._preprocessed = False

    def _calculate_node_importance(self, node: str) -> float:
        """
        Calculate importance of a node for contraction ordering.

        Uses multiple factors:
        - Edge difference (shortcuts needed - original edges)
        - Number of contracted neighbors
        - Current level in hierarchy
        """
        edge_difference = self._count_shortcuts(node) - len(list(self.graph.get_neighbors(node)))
        contracted_neighbors = len(self.contracted_neighbors.get(node, set()))
        level = self.node_level.get(node, 0)

        # Combine factors with appropriate weights
        return 5 * edge_difference + 3 * contracted_neighbors + 2 * level

    def preprocess(self) -> None:
        """
        Preprocess graph to build contraction hierarchy.

        This builds a hierarchy by contracting nodes in order of importance,
        adding shortcuts as necessary to preserve shortest paths.
        """
        # Calculate initial node ordering
        node_importance = {
            node: self._calculate_node_importance(node) for node in self.graph.get_nodes()
        }
        pq = [(importance, node) for node, importance in node_importance.items()]

        level = 0
        total_nodes = len(self.graph.get_nodes())
        last_progress = 0
        progress_interval = 5  # Show progress every 5%
        start_time = time.time()

        while pq:
            self.memory_manager.check_memory()
            importance, node = heappop(pq)

            if node not in self.node_level:
                # Contract node
                shortcuts = self._contract_node(node)
                self.node_level[node] = level
                level += 1

                # Show progress
                progress = (level / total_nodes) * 100
                if progress - last_progress >= progress_interval:
                    elapsed = time.time() - start_time
                    remaining = (elapsed / level) * (total_nodes - level)
                    print(f"Preprocessing: {progress:.1f}% complete, " f"ETA: {remaining:.1f}s")
                    last_progress = progress

                # Update importance of affected nodes
                affected = set()
                for u, v in shortcuts:
                    affected.add(u)
                    affected.add(v)

                for affected_node in affected:
                    if affected_node not in self.node_level:
                        new_importance = self._calculate_node_importance(affected_node)
                        heappush(pq, (new_importance, affected_node))

        self._preprocessed = True
        total_time = time.time() - start_time
        print(
            f"Preprocessing complete in {total_time:.1f}s, "
            f"created {len(self.shortcuts)} shortcuts"
        )

    def _contract_node(self, node: str) -> List[Tuple[str, str]]:
        """
        Contract a node and add necessary shortcuts.

        Args:
            node: Node to contract

        Returns:
            List of (u, v) pairs where shortcuts were added
        """
        shortcuts = []
        incoming = self.graph.get_neighbors(node, reverse=True)
        outgoing = self.graph.get_neighbors(node)

        for u in incoming:
            if u in self.node_level:  # Skip already contracted nodes
                continue

            for v in outgoing:
                if v in self.node_level or u == v:
                    continue

                # Check if shortcut is necessary
                lower_edge = self.graph.get_edge(u, node)
                upper_edge = self.graph.get_edge(node, v)
                if not lower_edge or not upper_edge:
                    continue

                shortcut_weight = get_edge_weight(lower_edge) + get_edge_weight(upper_edge)

                # Check if shortcut is necessary using witness search
                if self._is_shortcut_necessary(u, v, node, shortcut_weight):
                    # Calculate impact score based on original edges
                    impact_score = min(lower_edge.impact_score, upper_edge.impact_score)

                    # Create shortcut edge
                    now = datetime.now()
                    shortcut_edge = Edge(
                        from_entity=u,
                        to_entity=v,
                        relation_type=SHORTCUT_TYPE,
                        metadata=EdgeMetadata(
                            created_at=now,
                            last_modified=now,
                            confidence=min(
                                lower_edge.metadata.confidence, upper_edge.metadata.confidence
                            ),
                            source="contraction_hierarchies",
                            weight=shortcut_weight,
                        ),
                        impact_score=impact_score,
                        context=f"Shortcut via {node}",
                    )

                    self.shortcuts[(u, v)] = Shortcut(
                        edge=shortcut_edge,
                        via_node=node,
                        lower_edge=lower_edge,
                        upper_edge=upper_edge,
                    )

                    shortcuts.append((u, v))
                    if u not in self.contracted_neighbors:
                        self.contracted_neighbors[u] = set()
                    self.contracted_neighbors[u].add(v)

        return shortcuts

    def _is_shortcut_necessary(self, u: str, v: str, via: str, shortcut_weight: float) -> bool:
        """
        Determine if shortcut is necessary using witness search.

        Args:
            u: Source node
            v: Target node
            via: Node being contracted
            shortcut_weight: Weight of potential shortcut

        Returns:
            True if shortcut is necessary, False if witness path exists
        """
        MAX_WITNESS_SEARCH_STEPS = 50  # Limit witness search

        distances = {u: 0.0}
        pq = [(0.0, u)]
        visited = set()
        steps = 0

        while pq and steps < MAX_WITNESS_SEARCH_STEPS:
            dist, node = heappop(pq)

            if node == v:
                return dist >= shortcut_weight

            if node in visited:
                continue

            visited.add(node)
            steps += 1

            for neighbor in self.graph.get_neighbors(node):
                if neighbor == via:  # Don't go through contracted node
                    continue

                edge = self.graph.get_edge(node, neighbor)
                if not edge:
                    continue

                new_dist = dist + get_edge_weight(edge)
                if new_dist < shortcut_weight:  # Early termination
                    if neighbor not in distances or new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        heappush(pq, (new_dist, neighbor))

        return True  # No witness path found

    def _count_shortcuts(self, node: str) -> int:
        """Count number of shortcuts needed when contracting node."""
        shortcuts = set()
        incoming = self.graph.get_neighbors(node, reverse=True)
        outgoing = self.graph.get_neighbors(node)

        for u in incoming:
            if u in self.node_level:
                continue
            for v in outgoing:
                if v in self.node_level or u == v:
                    continue
                shortcuts.add((u, v))

        return len(shortcuts)

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
        if not self._preprocessed:
            raise GraphOperationError("Graph must be preprocessed before finding paths")

        validate = kwargs.get("validate", True)

        # Initialize forward and backward searches
        forward_distances: Dict[str, float] = {start_node: 0.0}
        backward_distances: Dict[str, float] = {end_node: 0.0}

        forward_pq = [(0.0, start_node)]
        backward_pq = [(0.0, end_node)]

        forward_predecessors: Dict[str, Tuple[str, Edge]] = {}
        backward_predecessors: Dict[str, Tuple[str, Edge]] = {}

        # Track best path
        best_dist = float("inf")
        meeting_node = None

        # Bidirectional search
        while forward_pq and backward_pq:
            self.memory_manager.check_memory()

            if forward_pq[0][0] + backward_pq[0][0] >= best_dist:
                break

            # Forward search
            dist, node = heappop(forward_pq)
            if node in backward_distances:
                total_dist = dist + backward_distances[node]
                if total_dist < best_dist:
                    best_dist = total_dist
                    meeting_node = node

            self._expand_search(
                node, dist, forward_distances, forward_predecessors, forward_pq, True, weight_func
            )

            # Backward search
            dist, node = heappop(backward_pq)
            if node in forward_distances:
                total_dist = dist + forward_distances[node]
                if total_dist < best_dist:
                    best_dist = total_dist
                    meeting_node = node

            self._expand_search(
                node,
                dist,
                backward_distances,
                backward_predecessors,
                backward_pq,
                False,
                weight_func,
            )

        if meeting_node is None:
            raise GraphOperationError(f"No path exists between {start_node} and {end_node}")

        # Reconstruct path
        path = self._reconstruct_path(meeting_node, forward_predecessors, backward_predecessors)

        # Apply filter if provided
        if filter_func and not filter_func(path):
            raise GraphOperationError(
                f"No path satisfying filter exists between {start_node} and {end_node}"
            )

        # Create and validate result
        result = create_path_result(path, weight_func)
        if validate:
            validate_path(path, self.graph, weight_func, max_length)

        return result

    def _expand_search(
        self,
        node: str,
        dist: float,
        distances: Dict[str, float],
        predecessors: Dict[str, Tuple[str, Edge]],
        pq: List[Tuple[float, str]],
        is_forward: bool,
        weight_func: Optional[WeightFunc],
    ) -> None:
        """Expand search in one direction."""
        # Get neighbors including shortcuts
        neighbors = set(self.graph.get_neighbors(node, reverse=not is_forward))
        if is_forward:
            shortcuts = [(v, shortcut) for (u, v), shortcut in self.shortcuts.items() if u == node]
        else:
            shortcuts = [(u, shortcut) for (u, v), shortcut in self.shortcuts.items() if v == node]

        # Process regular edges and shortcuts
        for neighbor in neighbors:
            edge = self.graph.get_edge(
                node if is_forward else neighbor, neighbor if is_forward else node
            )
            if edge:
                self._process_edge(
                    node, neighbor, edge, dist, distances, predecessors, pq, weight_func
                )

        # Process shortcuts
        for neighbor, shortcut in shortcuts:
            self._process_edge(
                node, neighbor, shortcut.edge, dist, distances, predecessors, pq, weight_func
            )

    def _process_edge(
        self,
        node: str,
        neighbor: str,
        edge: Edge,
        dist: float,
        distances: Dict[str, float],
        predecessors: Dict[str, Tuple[str, Edge]],
        pq: List[Tuple[float, str]],
        weight_func: Optional[WeightFunc],
    ) -> None:
        """Process a single edge in the search."""
        edge_weight = get_edge_weight(edge, weight_func)
        new_dist = dist + edge_weight

        if neighbor not in distances or new_dist < distances[neighbor]:
            distances[neighbor] = new_dist
            predecessors[neighbor] = (node, edge)
            heappush(pq, (new_dist, neighbor))

    def _reconstruct_path(
        self,
        meeting_node: str,
        forward_predecessors: Dict[str, Tuple[str, Edge]],
        backward_predecessors: Dict[str, Tuple[str, Edge]],
    ) -> List[Edge]:
        """
        Reconstruct complete path from meeting point.

        Handles path unpacking through shortcuts.
        """
        path = []

        # Forward path
        current = meeting_node
        while current in forward_predecessors:
            prev, edge = forward_predecessors[current]
            if edge.relation_type == SHORTCUT_TYPE:
                # Unpack shortcut
                shortcut = self.shortcuts[(prev, current)]
                path.extend(self._unpack_shortcut(shortcut))
            else:
                path.append(edge)
            current = prev

        path.reverse()

        # Backward path
        current = meeting_node
        while current in backward_predecessors:
            next_node, edge = backward_predecessors[current]
            if edge.relation_type == SHORTCUT_TYPE:
                # Unpack shortcut
                shortcut = self.shortcuts[(current, next_node)]
                path.extend(self._unpack_shortcut(shortcut))
            else:
                path.append(edge)
            current = next_node

        return path

    def _unpack_shortcut(self, shortcut: Shortcut) -> List[Edge]:
        """Recursively unpack a shortcut into its constituent edges."""
        path = []

        # Add lower edge (recursively unpack if it's a shortcut)
        if shortcut.lower_edge.relation_type == SHORTCUT_TYPE:
            lower_shortcut = self.shortcuts[
                (shortcut.lower_edge.from_entity, shortcut.lower_edge.to_entity)
            ]
            path.extend(self._unpack_shortcut(lower_shortcut))
        else:
            path.append(shortcut.lower_edge)

        # Add upper edge (recursively unpack if it's a shortcut)
        if shortcut.upper_edge.relation_type == SHORTCUT_TYPE:
            upper_shortcut = self.shortcuts[
                (shortcut.upper_edge.from_entity, shortcut.upper_edge.to_entity)
            ]
            path.extend(self._unpack_shortcut(upper_shortcut))
        else:
            path.append(shortcut.upper_edge)

        return path
