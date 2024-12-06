"""
Storage management for Contraction Hierarchies algorithm.

This module handles memory management and data storage for the Contraction
Hierarchies implementation, ensuring efficient use of memory resources.
"""

from typing import Dict, Optional, Set, Tuple
import threading
import time
from collections import defaultdict

from polaris.core.exceptions import GraphOperationError
from polaris.core.graph.traversal.utils import MemoryManager
from .models import ContractionState, Shortcut


class ContractionStorage:
    """
    Manages storage and memory for Contraction Hierarchies algorithm.

    Features:
    - Memory usage monitoring
    - Efficient data structure storage
    - State management
    - Thread safety
    - Performance monitoring
    """

    def __init__(self, max_memory_mb: Optional[float] = None):
        """
        Initialize storage manager.

        Args:
            max_memory_mb: Optional memory limit in MB
        """
        self.memory_manager = MemoryManager(max_memory_mb)
        self._state = ContractionState()
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        self._performance_metrics = defaultdict(list)
        self._shortcut_index = {}  # Cache for shortcut lookups
        self._neighbor_index = defaultdict(set)  # Cache for contracted neighbor lookups

    def _record_operation_time(self, operation: str, duration: float) -> None:
        """Record the duration of an operation."""
        with self._lock:
            self._performance_metrics[operation].append(duration)

    def _update_indices(self, shortcut: Shortcut) -> None:
        """Update internal indices when adding a shortcut."""
        key = (shortcut.edge.from_entity, shortcut.edge.to_entity)
        self._shortcut_index[key] = shortcut

        # Update neighbor indices
        self._neighbor_index[shortcut.edge.from_entity].add(shortcut.edge.to_entity)
        self._neighbor_index[shortcut.edge.to_entity].add(shortcut.edge.from_entity)

    def check_memory(self) -> None:
        """Check if memory usage is within limits."""
        start_time = time.time()
        try:
            self.memory_manager.check_memory()
        finally:
            self._record_operation_time("memory_check", time.time() - start_time)

    def get_state(self) -> ContractionState:
        """
        Get current algorithm state.

        Returns:
            Current ContractionState
        """
        with self._lock:
            return self._state

    def add_shortcut(self, shortcut: Shortcut) -> None:
        """
        Add a shortcut to storage.

        Args:
            shortcut: Shortcut to store

        Raises:
            GraphOperationError: If memory limit is exceeded
        """
        start_time = time.time()
        try:
            self.check_memory()
            with self._lock:
                self._state.add_shortcut(shortcut)
                self._update_indices(shortcut)
        except Exception as e:
            raise GraphOperationError(f"Failed to add shortcut: {str(e)}")
        finally:
            self._record_operation_time("add_shortcut", time.time() - start_time)

    def get_shortcut(self, from_node: str, to_node: str) -> Optional[Shortcut]:
        """
        Get shortcut between nodes if it exists.

        Args:
            from_node: Source node
            to_node: Target node

        Returns:
            Shortcut if it exists, None otherwise
        """
        start_time = time.time()
        try:
            with self._lock:
                # Try cache first
                key = (from_node, to_node)
                if key in self._shortcut_index:
                    return self._shortcut_index[key]
                # Fall back to state if not in cache
                return self._state.get_shortcut(from_node, to_node)
        finally:
            self._record_operation_time("get_shortcut", time.time() - start_time)

    def set_node_level(self, node: str, level: int) -> None:
        """
        Set contraction level for a node.

        Args:
            node: Node ID
            level: Contraction level

        Raises:
            GraphOperationError: If memory limit is exceeded
        """
        start_time = time.time()
        try:
            self.check_memory()
            with self._lock:
                self._state.set_node_level(node, level)
        except Exception as e:
            raise GraphOperationError(f"Failed to set node level: {str(e)}")
        finally:
            self._record_operation_time("set_node_level", time.time() - start_time)

    def get_node_level(self, node: str) -> int:
        """
        Get contraction level of a node.

        Args:
            node: Node ID

        Returns:
            Contraction level (0 if not contracted)
        """
        start_time = time.time()
        try:
            with self._lock:
                return self._state.get_node_level(node)
        finally:
            self._record_operation_time("get_node_level", time.time() - start_time)

    def get_contracted_neighbors(self, node: str) -> Set[str]:
        """
        Get contracted neighbors of a node.

        Args:
            node: Node ID

        Returns:
            Set of contracted neighbor node IDs
        """
        start_time = time.time()
        try:
            with self._lock:
                # Try cache first
                if node in self._neighbor_index:
                    return self._neighbor_index[node].copy()
                # Fall back to state if not in cache
                return self._state.get_contracted_neighbors(node)
        finally:
            self._record_operation_time("get_contracted_neighbors", time.time() - start_time)

    def get_shortcuts(self) -> Dict[Tuple[str, str], Shortcut]:
        """
        Get all shortcuts.

        Returns:
            Dictionary mapping (from_node, to_node) to Shortcut
        """
        start_time = time.time()
        try:
            with self._lock:
                return self._state.shortcuts.copy()
        finally:
            self._record_operation_time("get_shortcuts", time.time() - start_time)

    def clear(self) -> None:
        """Clear all stored data."""
        start_time = time.time()
        try:
            with self._lock:
                self._state = ContractionState()
                self._shortcut_index.clear()
                self._neighbor_index.clear()
                self._performance_metrics.clear()
        finally:
            self._record_operation_time("clear", time.time() - start_time)

    def get_performance_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance metrics for storage operations.

        Returns:
            Dictionary mapping operation names to their statistics
        """
        import statistics

        with self._lock:
            metrics = {}
            for operation, times in self._performance_metrics.items():
                if times:
                    metrics[operation] = {
                        "avg": statistics.mean(times),
                        "min": min(times),
                        "max": max(times),
                        "std": statistics.stdev(times) if len(times) > 1 else 0,
                        "count": len(times),
                    }
            return metrics

    def get_memory_usage(self) -> Dict[str, int]:
        """
        Get current memory usage statistics.

        Returns:
            Dictionary with memory usage details
        """
        with self._lock:
            return {
                "shortcuts": len(self._state.shortcuts),
                "node_levels": len(self._state.node_level),
                "shortcut_index": len(self._shortcut_index),
                "neighbor_index": len(self._neighbor_index),
                "total_metrics": sum(len(times) for times in self._performance_metrics.values()),
            }
