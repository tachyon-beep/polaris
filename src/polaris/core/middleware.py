from abc import ABC, abstractmethod
from typing import Callable, Dict, Any, List, Optional, Iterable
from functools import wraps
import time
import logging
from datetime import datetime

from .graph import Graph
from .models import Edge


class GraphMiddleware(ABC):
    """Base class for all graph middleware components."""

    @abstractmethod
    def process(self, context: Dict[str, Any], next_middleware: Callable) -> Any:
        """Process the operation with middleware logic.

        Args:
            context: Operation context containing relevant data
            next_middleware: Next middleware in the chain to execute

        Returns:
            Result from the middleware chain execution
        """
        pass


class LoggingMiddleware(GraphMiddleware):
    """Middleware for logging graph operations."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def process(self, context: Dict[str, Any], next_middleware: Callable) -> Any:
        operation = context.get("operation", "unknown")
        self.logger.info(f"Starting {operation} with context: {context}")
        try:
            result = next_middleware(context)
            self.logger.info(f"Completed {operation}")
            return result
        except Exception as e:
            self.logger.error(f"Failed {operation}: {str(e)}")
            raise


class PerformanceMiddleware(GraphMiddleware):
    """Middleware for monitoring operation performance."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def process(self, context: Dict[str, Any], next_middleware: Callable) -> Any:
        start_time = time.perf_counter()
        result = next_middleware(context)
        duration = time.perf_counter() - start_time
        self.logger.info(
            f"Operation {context.get('operation', 'unknown')} " f"took {duration:.3f} seconds"
        )
        return result


class ValidationMiddleware(GraphMiddleware):
    """Middleware for validating graph operations."""

    def process(self, context: Dict[str, Any], next_middleware: Callable) -> Any:
        operation = context.get("operation", "")

        if operation == "add_edge":
            edge = context.get("edge")
            if not edge:
                raise ValueError("Edge is required for add_edge operation")
            if not isinstance(edge, Edge):
                raise TypeError(f"Invalid edge type: {type(edge)}")

        elif operation == "remove_edge":
            from_node = context.get("from_node")
            to_node = context.get("to_node")
            if not from_node or not to_node:
                raise ValueError(
                    "Both from_node and to_node are required for remove_edge operation"
                )

        return next_middleware(context)


class GraphWithMiddleware(Graph):
    """Graph implementation that supports middleware processing."""

    def __init__(self, edges: Optional[Iterable[Edge]] = None):
        # Convert Iterable to List before passing to parent
        super().__init__(list(edges) if edges is not None else [])
        self.middleware: List[GraphMiddleware] = []

    def add_middleware(self, middleware: GraphMiddleware) -> None:
        """Add a middleware to the processing chain."""
        self.middleware.append(middleware)

    def _execute_with_middleware(self, context: Dict[str, Any], operation: Callable) -> Any:
        """Execute an operation through the middleware chain."""

        def execute_middleware(index: int, ctx: Dict[str, Any]) -> Any:
            if index >= len(self.middleware):
                return operation(ctx)
            return self.middleware[index].process(ctx, lambda c: execute_middleware(index + 1, c))

        return execute_middleware(0, context)

    def add_edge(self, edge: Edge) -> None:
        """Add an edge with middleware processing."""
        context = {"operation": "add_edge", "edge": edge, "timestamp": datetime.now()}
        self._execute_with_middleware(context, lambda ctx: super().add_edge(ctx["edge"]))

    def remove_edge(self, from_node: str, to_node: str) -> None:
        """Remove an edge with middleware processing."""
        context = {
            "operation": "remove_edge",
            "from_node": from_node,
            "to_node": to_node,
            "timestamp": datetime.now(),
        }
        self._execute_with_middleware(
            context, lambda ctx: super().remove_edge(ctx["from_node"], ctx["to_node"])
        )

    def get_nodes(self):
        """Get nodes with middleware processing."""
        context = {"operation": "get_nodes", "timestamp": datetime.now()}
        return self._execute_with_middleware(context, lambda ctx: super().get_nodes())

    def get_edges(self):
        """Get edges with middleware processing."""
        context = {"operation": "get_edges", "timestamp": datetime.now()}
        return self._execute_with_middleware(context, lambda ctx: super().get_edges())

    def get_neighbors(self, node: str, reverse: bool = False):
        """Get node neighbors with middleware processing."""
        context = {
            "operation": "get_neighbors",
            "node": node,
            "reverse": reverse,
            "timestamp": datetime.now(),
        }
        return self._execute_with_middleware(
            context, lambda ctx: super().get_neighbors(ctx["node"], ctx["reverse"])
        )
