"""
Contraction Hierarchies implementation for fast path queries.

This module provides an implementation of Contraction Hierarchies, a preprocessing-based
speedup technique for shortest path queries. Key features:
- Node contraction with shortcut creation
- Witness path search
- Importance-based node ordering
- Memory-efficient path reconstruction
"""

import time
from dataclasses import dataclass
from datetime import datetime
from heapq import heappop, heappush
from typing import Callable, Dict, List, Optional, Set, Tuple

from polaris.core.enums import RelationType
from polaris.core.exceptions import GraphOperationError
from polaris.core.graph import Graph
from polaris.core.graph_paths.base import PathFinder
from polaris.core.graph_paths.models import PathResult
from polaris.core.graph_paths.utils import (
    MemoryManager,
    WeightFunc,
    create_path_result,
    get_edge_weight,
    validate_path,
)
from polaris.core.models import Edge, EdgeMetadata

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
        - Node degree (in + out)
        """
        shortcuts_needed = self._count_shortcuts(node)
        original_edges = len(list(self.graph.get_neighbors(node)))
        contracted_neighbors = len(self.contracted_neighbors.get(node, set()))
        level = self.node_level.get(node, 0)

        # Calculate importance based on multiple factors
        importance = (
            5 * shortcuts_needed  # Weight the cost of adding shortcuts
            + 2 * contracted_neighbors  # Consider neighborhood complexity
            + level  # Preserve hierarchy levels
            + original_edges  # Base importance on connectivity
        )
        return importance

    def preprocess(self) -> None:
        """
        Preprocess graph to build contraction hierarchy.

        This builds a hierarchy by contracting nodes in order of importance,
        adding shortcuts as necessary to preserve shortest paths.
        """
        # Calculate initial node ordering
        nodes = list(self.graph.get_nodes())
        node_importance = {node: self._calculate_node_importance(node) for node in nodes}
        pq = [(importance, node) for node, importance in node_importance.items()]

        level = 0
        total_nodes = len(nodes)
        last_progress = 0
        progress_interval = 5  # Show progress every 5%
        start_time = time.time()

        # Contract nodes in order of increasing importance
        while pq:
            self.memory_manager.check_memory()
            _, node = heappop(pq)

            if node not in self.node_level:
                # Contract node
                shortcuts = self._contract_node(node)
                self.node_level[node] = level
                level += 1

                # Debug: Print contracted node and shortcuts
                print(
                    f"Contracted node: {node}, Level: {self.node_level[node]}, Shortcuts added: {shortcuts}"
                )

                # Show progress
                progress = (level / total_nodes) * 100
                if progress - last_progress >= progress_interval:
                    elapsed = time.time() - start_time
                    remaining = (elapsed / level) * (total_nodes - level)
                    print(f"Preprocessing: {progress:.1f}% complete, ETA: {remaining:.1f}s")
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
        incoming = sorted(
            list(self.graph.get_neighbors(node, reverse=True))
        )  # Sort for determinism
        outgoing = sorted(list(self.graph.get_neighbors(node)))  # Sort for determinism

        # Consider all pairs of incoming and outgoing edges
        for u in incoming:
            # Don't skip contracted nodes - we need to consider all paths
            for v in outgoing:
                if u == v:  # Skip self-loops
                    continue

                # Check if shortcut is necessary
                lower_edge = self.graph.get_edge(u, node)
                upper_edge = self.graph.get_edge(node, v)
                if not lower_edge or not upper_edge:
                    continue

                shortcut_weight = lower_edge.metadata.weight + upper_edge.metadata.weight

                # Check if shortcut is necessary using witness search
                if self._is_shortcut_necessary(u, v, node, shortcut_weight):
                    # Calculate impact score based on original edges
                    impact_score = min(lower_edge.impact_score, upper_edge.impact_score)

                    # Create shortcut edge with correct weight
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
                            weight=shortcut_weight,  # Use sum of component weights
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
        # Initialize distances and priority queue
        distances = {u: 0.0}
        pq = [(0.0, u)]
        visited = set()

        while pq:
            dist, node = heappop(pq)

            if node == v:
                # Found a witness path that's better than the shortcut
                return dist > shortcut_weight - 1e-10  # Allow for floating point error

            if node in visited or node == via:  # Don't go through contracted node
                continue

            visited.add(node)

            # Sort neighbors for deterministic behavior
            neighbors = sorted(self.graph.get_neighbors(node))
            for neighbor in neighbors:
                if neighbor == via:  # Don't go through contracted node
                    continue

                edge = self.graph.get_edge(node, neighbor)
                if not edge:
                    continue

                new_dist = dist + edge.metadata.weight
                if new_dist >= shortcut_weight + 1e-10:  # Early termination with epsilon
                    continue

                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    heappush(pq, (new_dist, neighbor))

        # No witness path found within weight limit
        return True

    def _count_shortcuts(self, node: str) -> int:
        """Count number of shortcuts needed when contracting node."""
        shortcuts = set()
        incoming = sorted(
            list(self.graph.get_neighbors(node, reverse=True))
        )  # Sort for determinism
        outgoing = sorted(list(self.graph.get_neighbors(node)))  # Sort for determinism

        for u in incoming:
            for v in outgoing:
                if u == v:
                    continue
                # Only count if shortcut would be necessary
                lower_edge = self.graph.get_edge(u, node)
                upper_edge = self.graph.get_edge(node, v)
                if lower_edge and upper_edge:
                    shortcut_weight = get_edge_weight(lower_edge) + get_edge_weight(upper_edge)
                    if self._is_shortcut_necessary(u, v, node, shortcut_weight):
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

        # Try all possible paths if filter is provided
        if filter_func:
            # First try direct path
            try:
                path = self._find_path(start_node, end_node, weight_func)
                if filter_func(path):
                    result = create_path_result(path, weight_func)
                    if validate:
                        validate_path(path, self.graph, weight_func, max_length)
                    return result
            except GraphOperationError:
                pass

            # Try alternative paths through different nodes
            # Sort nodes for deterministic behavior
            nodes = sorted(self.graph.get_nodes())
            for node in nodes:
                if node == start_node or node == end_node:
                    continue
                try:
                    # Find path through this node
                    path1 = self._find_path(start_node, node, weight_func)
                    path2 = self._find_path(node, end_node, weight_func)
                    path = path1 + path2
                    if filter_func(path):
                        result = create_path_result(path, weight_func)
                        if validate:
                            validate_path(path, self.graph, weight_func, max_length)
                        return result
                except GraphOperationError:
                    continue

            raise GraphOperationError(
                f"No path satisfying filter exists between {start_node} and {end_node}"
            )

        # No filter, find shortest path
        path = self._find_path(start_node, end_node, weight_func)
        result = create_path_result(path, weight_func)
        if validate:
            validate_path(path, self.graph, weight_func, max_length)
        return result

    def _find_path(
        self,
        start_node: str,
        end_node: str,
        weight_func: Optional[WeightFunc],
    ) -> List[Edge]:
        """Find shortest path without filter."""
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

        # Track visited nodes to ensure we don't miss any paths
        forward_visited = set()
        backward_visited = set()

        # Bidirectional search
        while forward_pq or backward_pq:
            self.memory_manager.check_memory()

            # Forward search
            if forward_pq:
                dist, node = heappop(forward_pq)
                if node in forward_visited:
                    continue

                forward_visited.add(node)

                if node in backward_distances:
                    total_dist = dist + backward_distances[node]
                    if total_dist < best_dist:
                        best_dist = total_dist
                        meeting_node = node
                        print(
                            f"Meeting node found (forward): {meeting_node} with total_dist: {best_dist}"
                        )

                if dist <= best_dist:
                    self._expand_search(
                        node,
                        dist,
                        forward_distances,
                        forward_predecessors,
                        forward_pq,
                        True,
                        weight_func,
                    )

            # Backward search
            if backward_pq:
                dist, node = heappop(backward_pq)
                if node in backward_visited:
                    continue

                backward_visited.add(node)

                if node in forward_distances:
                    total_dist = dist + forward_distances[node]
                    if total_dist < best_dist:
                        best_dist = total_dist
                        meeting_node = node
                        print(
                            f"Meeting node found (backward): {meeting_node} with total_dist: {best_dist}"
                        )

                if dist <= best_dist:
                    self._expand_search(
                        node,
                        dist,
                        backward_distances,
                        backward_predecessors,
                        backward_pq,
                        False,
                        weight_func,
                    )

            # Debug: Print current best_dist and meeting_node
            print(f"Current best_dist: {best_dist}, Meeting node: {meeting_node}")

            if (not forward_pq or forward_pq[0][0] >= best_dist) and (
                not backward_pq or backward_pq[0][0] >= best_dist
            ):
                break

        if meeting_node is None:
            raise GraphOperationError(f"No path exists between {start_node} and {end_node}")

        # Reconstruct path
        forward_path = []
        backward_path = []

        # Forward path
        current = meeting_node
        while current in forward_predecessors:
            prev, edge = forward_predecessors[current]
            if edge.context and "Shortcut" in edge.context:
                # Unpack shortcut
                shortcut = self.shortcuts.get((prev, current))
                if shortcut:
                    forward_path.extend(reversed(self._unpack_shortcut(shortcut)))
                else:
                    print(f"Shortcut not found for edge: {prev} -> {current}")
            else:
                forward_path.append(edge)
            current = prev

        # Backward path
        current = meeting_node
        while current in backward_predecessors:
            next_node, edge = backward_predecessors[current]
            if edge.context and "Shortcut" in edge.context:
                # Unpack shortcut
                shortcut = self.shortcuts.get((current, next_node))
                if shortcut:
                    backward_path.extend(self._unpack_shortcut(shortcut))
                else:
                    print(f"Shortcut not found for edge: {current} -> {next_node}")
            else:
                # Create forward edge from backward edge
                backward_edge = edge
                forward_edge = Edge(
                    from_entity=backward_edge.to_entity,
                    to_entity=backward_edge.from_entity,
                    relation_type=backward_edge.relation_type,
                    metadata=EdgeMetadata(
                        created_at=backward_edge.metadata.created_at,
                        last_modified=backward_edge.metadata.last_modified,
                        confidence=backward_edge.metadata.confidence,
                        source=backward_edge.metadata.source,
                        weight=backward_edge.metadata.weight,
                    ),
                    impact_score=backward_edge.impact_score,
                    context=backward_edge.context,
                )
                backward_path.append(forward_edge)
            current = next_node

        # Combine paths in correct order
        forward_path.reverse()  # Reverse to get correct order
        path = forward_path + backward_path

        # Debug: Print the reconstructed path and total weight
        reconstructed_path = [f"{edge.from_entity}->{edge.to_entity}" for edge in path]
        total_weight = sum(edge.metadata.weight for edge in path)
        print(f"Reconstructed path: {reconstructed_path}")
        print(f"Total weight: {total_weight}")

        return path

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
        neighbors = set()
        # Get regular neighbors
        neighbors.update(self.graph.get_neighbors(node, reverse=not is_forward))
        # Get shortcut neighbors
        if is_forward:
            neighbors.update(v for (u, v), _ in self.shortcuts.items() if u == node)
        else:
            neighbors.update(u for (u, v), _ in self.shortcuts.items() if v == node)

        # Sort neighbors for deterministic behavior
        neighbors = sorted(neighbors)

        # Process all edges (regular and shortcuts)
        for neighbor in neighbors:
            # Only expand to nodes of higher or equal level in the hierarchy
            if neighbor in self.node_level and self.node_level[neighbor] < self.node_level[node]:
                continue

            # Try regular edge first
            edge = None
            if is_forward:
                edge = self.graph.get_edge(node, neighbor)
            else:
                edge = self.graph.get_edge(neighbor, node)
                if edge:
                    # For backward search, create forward edge
                    edge = Edge(
                        from_entity=edge.to_entity,
                        to_entity=edge.from_entity,
                        relation_type=edge.relation_type,
                        metadata=EdgeMetadata(
                            created_at=edge.metadata.created_at,
                            last_modified=edge.metadata.last_modified,
                            confidence=edge.metadata.confidence,
                            source=edge.metadata.source,
                            weight=edge.metadata.weight,
                        ),
                        impact_score=edge.impact_score,
                        context=edge.context,
                    )

            # If no regular edge, try shortcut
            if not edge:
                if is_forward:
                    shortcut = self.shortcuts.get((node, neighbor))
                    if shortcut:
                        edge = shortcut.edge
                else:
                    shortcut = self.shortcuts.get((neighbor, node))
                    if shortcut:
                        # Create forward edge from backward shortcut
                        edge = Edge(
                            from_entity=shortcut.edge.to_entity,
                            to_entity=shortcut.edge.from_entity,
                            relation_type=shortcut.edge.relation_type,
                            metadata=EdgeMetadata(
                                created_at=shortcut.edge.metadata.created_at,
                                last_modified=shortcut.edge.metadata.last_modified,
                                confidence=shortcut.edge.metadata.confidence,
                                source=shortcut.edge.metadata.source,
                                weight=shortcut.edge.metadata.weight,
                            ),
                            impact_score=shortcut.edge.impact_score,
                            context=shortcut.edge.context,
                        )

            if edge:
                self._process_edge(
                    node, neighbor, edge, dist, distances, predecessors, pq, weight_func
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
        edge_weight = get_edge_weight(edge, weight_func) if weight_func else edge.metadata.weight
        new_dist = dist + edge_weight

        # For equal distances, prefer lexicographically smaller paths
        if (
            neighbor not in distances
            or new_dist < distances[neighbor]
            or (
                new_dist == distances[neighbor]
                and edge.to_entity < predecessors[neighbor][1].to_entity
            )
        ):
            distances[neighbor] = new_dist
            predecessors[neighbor] = (node, edge)
            heappush(pq, (new_dist, neighbor))

    def _unpack_shortcut(self, shortcut: Shortcut) -> List[Edge]:
        """Recursively unpack a shortcut into its constituent edges."""
        path = []

        # Add lower edge (recursively unpack if it's a shortcut)
        if shortcut.lower_edge.context and "Shortcut" in shortcut.lower_edge.context:
            lower_shortcut = self.shortcuts[
                (shortcut.lower_edge.from_entity, shortcut.lower_edge.to_entity)
            ]
            path.extend(self._unpack_shortcut(lower_shortcut))
        else:
            path.append(shortcut.lower_edge)

        # Add upper edge (recursively unpack if it's a shortcut)
        if shortcut.upper_edge.context and "Shortcut" in shortcut.upper_edge.context:
            upper_shortcut = self.shortcuts[
                (shortcut.upper_edge.from_entity, shortcut.upper_edge.to_entity)
            ]
            path.extend(self._unpack_shortcut(upper_shortcut))
        else:
            path.append(shortcut.upper_edge)

        return path
