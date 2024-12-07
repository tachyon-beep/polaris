"""
Hub Labeling preprocessing implementation.

This module handles the preprocessing phase of the Hub Labeling algorithm,
which computes distance labels for each node in the graph.
"""

import math
import time
from typing import Dict, Optional, TYPE_CHECKING

from polaris.core.graph.traversal.utils import get_edge_weight
from .models import HubLabel, HubLabelState
from .storage import HubLabelStorage
from .utils import calculate_node_importance, should_prune_label

if TYPE_CHECKING:
    from polaris.core.graph import Graph


class HubLabelPreprocessor:
    """
    Preprocessor for Hub Labeling algorithm.

    This class handles the preprocessing phase, which:
    1. Computes node importance scores
    2. Orders nodes by importance
    3. Computes forward and backward distance labels
    """

    def __init__(self, graph: "Graph", storage: Optional[HubLabelStorage] = None):
        """
        Initialize preprocessor.

        Args:
            graph: Graph to preprocess
            storage: Optional storage for persisting labels
        """

        if not graph or not graph.get_nodes():
            raise ValueError("Graph cannot be empty")

        self.graph = graph
        self.storage = storage
        self.state = HubLabelState()  # type: ignore
        self._start_time = 0.0

    def preprocess(self, verbose: bool = True) -> None:
        """
        Run preprocessing phase.

        This computes distance labels for all nodes in the graph.
        The labels allow for fast distance queries at query time.

        Args:
            verbose: Whether to print progress information
        """
        if verbose:
            print("\nStarting preprocessing...")

        self._start_time = time.time()

        # Calculate node ordering
        hub_order = self._compute_hub_order(verbose)

        # Process nodes in order
        nodes = sorted(hub_order.keys(), key=lambda x: hub_order[x])
        for node in nodes:
            if verbose:
                print(f"\nProcessing node {node}:")

            # Initialize labels for this node
            self.state.get_forward_labels(node).add_label(
                HubLabel(hub=node, distance=0.0, first_hop=None)
            )
            self.state.get_backward_labels(node).add_label(
                HubLabel(hub=node, distance=0.0, first_hop=None)
            )

            # Compute forward labels (paths from this node to more important hubs)
            self._compute_forward_labels(node, hub_order, verbose)

            # Compute backward labels (paths to this node from less important nodes)
            self._compute_backward_labels(node, hub_order, verbose)

            # Save labels if storage is available
            if self.storage is not None:
                self.storage.save_labels(
                    node,
                    self.state.get_forward_labels(node),
                    self.state.get_backward_labels(node),
                )

        # Add direct path labels
        self._add_direct_path_labels(verbose)

        # Add transitive path labels
        self._add_transitive_path_labels(verbose)

        if verbose:
            duration_ms = (time.time() - self._start_time) * 1000
            avg_labels = self._compute_average_labels()
            print(
                f"\nPreprocessing complete in {duration_ms:.1f}ms",
                f"average labels per node: {avg_labels:.1f}",
            )

    def _compute_hub_order(self, verbose: bool) -> Dict[str, int]:
        """
        Compute ordering of nodes based on importance.

        More important nodes get lower order numbers and become hubs.
        This is critical for performance - good hub selection reduces
        the total number of labels needed.

        Args:
            verbose: Whether to print progress information

        Returns:
            Map of node to order value (lower = more important)
        """
        # Check if graph is complete
        if self._is_complete_graph():
            if verbose:
                print("\nDetected complete graph, using minimal hub set")
            return self._compute_complete_graph_order()

        # Calculate importance scores
        scores: Dict[str, float] = {}
        for node in self.graph.get_nodes():
            scores[node] = calculate_node_importance(node, self.graph, {})

        if verbose:
            print("\nImportance scores:")
            for node, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                print(f"  {node}: {score}")

        # Assign order numbers (0 = most important)
        hub_order = {
            node: i
            for i, (node, _) in enumerate(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        }

        if verbose:
            print("\nHub ordering:")
            for node, order in sorted(hub_order.items(), key=lambda x: x[1]):
                print(f"  {node}: order={order}")

        return hub_order

    def _is_complete_graph(self) -> bool:
        """Check if graph is complete."""
        n = len(self.graph.get_nodes())
        expected_edges = (n * (n - 1)) // 2  # Complete graph has n*(n-1)/2 edges
        actual_edges = len(set((e.from_entity, e.to_entity) for e in self.graph.get_edges()))
        return actual_edges >= expected_edges

    def _compute_complete_graph_order(self) -> Dict[str, int]:
        """Compute optimized hub order for complete graph."""
        nodes = list(self.graph.get_nodes())
        n = len(nodes)
        # Select approximately sqrt(n) hubs
        num_hubs = int(math.sqrt(n))
        hub_nodes = nodes[:num_hubs]
        # Assign higher importance to hubs
        return {node: (0 if node in hub_nodes else 1) for node in nodes}

    def _compute_forward_labels(self, node: str, hub_order: Dict[str, int], verbose: bool) -> None:
        """
        Compute forward labels with pruning.

        Args:
            node: Node to compute labels for
            hub_order: Map of node to order value
            verbose: Whether to print progress information
        """
        forward_labels = self.state.get_forward_labels(node)

        # Process all more important hubs
        for hub in sorted(hub_order.keys(), key=lambda x: hub_order[x]):
            if hub_order[hub] >= hub_order[node]:
                continue

            # Try direct edge first
            edge = self.graph.get_edge(node, hub)
            if edge:
                distance = get_edge_weight(edge)
                if verbose:
                    print(f"    Adding forward label: {node} -> {hub} (dist={distance})")
                forward_labels.add_label(HubLabel(hub=hub, distance=distance, first_hop=edge))
                continue

            # Try paths through neighbors
            min_dist = float("inf")
            best_hop = None

            for neighbor in self.graph.get_neighbors(node):
                neighbor_edge = self.graph.get_edge(node, neighbor)
                if not neighbor_edge:
                    continue

                neighbor_labels = self.state.get_forward_labels(neighbor)
                if not neighbor_labels:
                    continue

                hub_label = neighbor_labels.get_label(hub)
                if hub_label:
                    total_dist = get_edge_weight(neighbor_edge) + hub_label.distance
                    if total_dist < min_dist:
                        min_dist = total_dist
                        best_hop = neighbor_edge

            # Add label if we found a valid path
            if best_hop and min_dist < float("inf"):
                if verbose:
                    print(f"    Adding forward label: {node} -> {hub} (dist={min_dist})")
                forward_labels.add_label(HubLabel(hub=hub, distance=min_dist, first_hop=best_hop))

    def _compute_backward_labels(
        self,
        node: str,
        hub_order: Dict[str, int],
        verbose: bool,
    ) -> None:
        """
        Compute backward labels for a node.

        These labels represent shortest paths to this node from less
        important nodes.

        Args:
            node: Node to compute labels for
            hub_order: Map of node to order value
            verbose: Whether to print progress information
        """
        # Get existing labels
        backward_labels = self.state.get_backward_labels(node)

        # Try paths through existing hubs
        for source in hub_order:
            if hub_order[source] <= hub_order[node]:
                continue  # Only consider more important nodes

            # Check direct edge
            edge = self.graph.get_edge(source, node)
            if edge:
                distance = get_edge_weight(edge)
                if not should_prune_label(source, node, distance, self.state):
                    if verbose:
                        print(f"    Adding backward label: {node} <- {source} (dist={distance})")
                    backward_labels.add_label(
                        HubLabel(hub=source, distance=distance, first_hop=edge)
                    )

            # Check paths through neighbors
            for neighbor in self.graph.get_neighbors(node, reverse=True):
                neighbor_labels = self.state.get_backward_labels(neighbor)
                neighbor_edge = self.graph.get_edge(neighbor, node)
                if neighbor_edge and neighbor_labels:
                    for label in neighbor_labels.labels:
                        if hub_order[label.hub] < hub_order[source]:
                            total_dist = get_edge_weight(neighbor_edge) + label.distance
                            if not should_prune_label(source, node, total_dist, self.state):
                                if verbose:
                                    print(
                                        f"    Adding backward label: {node} <- {source} "
                                        f"(dist={total_dist})"
                                    )
                                backward_labels.add_label(
                                    HubLabel(
                                        hub=source,
                                        distance=total_dist,
                                        first_hop=neighbor_edge,
                                    )
                                )

    def _add_direct_path_labels(self, verbose: bool) -> None:
        """
        Add labels for direct paths.

        This ensures we can reconstruct paths between directly connected nodes.

        Args:
            verbose: Whether to print progress information
        """
        # For each edge in the graph
        for edge in self.graph.get_edges():
            from_node = edge.from_entity
            to_node = edge.to_entity
            distance = get_edge_weight(edge)

            # Add forward label
            forward_labels = self.state.get_forward_labels(from_node)
            if not forward_labels.get_label(to_node):
                if verbose:
                    print(
                        f"    Adding direct forward label: {from_node} -> {to_node} "
                        f"(dist={distance})"
                    )
                forward_labels.add_label(HubLabel(hub=to_node, distance=distance, first_hop=edge))

            # Add backward label
            backward_labels = self.state.get_backward_labels(to_node)
            if not backward_labels.get_label(from_node):
                if verbose:
                    print(
                        f"    Adding direct backward label: {to_node} <- {from_node} "
                        f"(dist={distance})"
                    )
                backward_labels.add_label(
                    HubLabel(hub=from_node, distance=distance, first_hop=edge)
                )

    def _add_transitive_path_labels(self, verbose: bool) -> None:
        """
        Add labels for transitive paths.

        This ensures we can reconstruct paths that require multiple hops.

        Args:
            verbose: Whether to print progress information
        """
        # For each node
        for source in self.graph.get_nodes():
            # Get all reachable nodes through forward edges
            visited = {source}
            queue = [(source, 0.0, None)]  # (node, distance, first_hop)

            while queue:
                current, dist, first_hop = queue.pop(0)

                # Add forward label from source to current
                if current != source:
                    forward_labels = self.state.get_forward_labels(source)
                    if not forward_labels.get_label(current):
                        if verbose:
                            print(
                                f"    Adding transitive forward label: {source} -> {current} "
                                f"(dist={dist})"
                            )
                        forward_labels.add_label(
                            HubLabel(hub=current, distance=dist, first_hop=first_hop)
                        )
                    # Add corresponding backward label
                    backward_labels = self.state.get_backward_labels(current)
                    if not backward_labels.get_label(source):
                        if verbose:
                            print(
                                f"    Adding transitive backward label: {current} <- {source} "
                                f"(dist={dist})"
                            )
                        backward_labels.add_label(
                            HubLabel(hub=source, distance=dist, first_hop=first_hop)
                        )

                # Explore neighbors
                for neighbor in self.graph.get_neighbors(current):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        edge = self.graph.get_edge(current, neighbor)
                        if edge:
                            new_dist = dist + get_edge_weight(edge)
                            new_first_hop = first_hop if first_hop else edge
                            queue.append((neighbor, new_dist, new_first_hop))

    def _compute_average_labels(self) -> float:
        """
        Compute average number of labels per node.

        Returns:
            Average number of labels per node
        """
        total_labels = 0
        num_nodes = len(self.graph.get_nodes())
        for node in self.graph.get_nodes():
            total_labels += len(self.state.get_forward_labels(node).labels)
            total_labels += len(self.state.get_backward_labels(node).labels)
        return total_labels / num_nodes if num_nodes > 0 else 0.0
