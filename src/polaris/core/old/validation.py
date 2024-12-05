"""
Edge validation system for the graph.

This module provides a flexible validation strategy pattern for validating edges
in the graph. It allows for pluggable validators that can enforce different rules
and constraints on edges.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Protocol, Tuple

from .models import Edge
from .graph import Graph
from .exceptions import ValidationError


class EdgeValidator(Protocol):
    """Protocol for edge validators."""

    def validate(self, edge: Edge, graph: Graph) -> bool:
        """
        Validate an edge in the context of a graph.

        Args:
            edge: The edge to validate
            graph: The graph context for validation

        Returns:
            bool: True if validation passes, False otherwise
        """
        ...


@dataclass
class ValidationStrategy:
    """Strategy for validating edges using multiple validators."""

    validators: List[EdgeValidator] = field(default_factory=list)

    def add_validator(self, validator: EdgeValidator) -> None:
        """Add a validator to the strategy."""
        self.validators.append(validator)

    def validate(self, edge: Edge, graph: Graph) -> Tuple[bool, List[str]]:
        """
        Validate an edge using all registered validators.

        Args:
            edge: The edge to validate
            graph: The graph context for validation

        Returns:
            Tuple containing:
            - bool: True if all validations pass
            - List[str]: List of error messages from failed validations
        """
        errors = []
        for validator in self.validators:
            if not validator.validate(edge, graph):
                errors.append(f"Validation failed: {validator.__class__.__name__}")
        return not errors, errors


class CyclicDependencyValidator(EdgeValidator):
    """Validates that an edge does not create a cyclic dependency."""

    def validate(self, edge: Edge, graph: Graph) -> bool:
        """
        Check if adding this edge would create a cycle.

        Args:
            edge: The edge to validate
            graph: The graph to check for cycles in

        Returns:
            bool: True if no cycle would be created, False otherwise
        """
        from .enums import RelationType

        if edge.relation_type == RelationType.DEPENDS_ON:
            # Check if there's already a path from target back to source
            paths = graph.find_paths(edge.to_entity, edge.from_entity)
            return len(paths) == 0
        return True


class DuplicateEdgeValidator(EdgeValidator):
    """Validates that an edge is not a duplicate."""

    def validate(self, edge: Edge, graph: Graph) -> bool:
        """
        Check if this edge already exists.

        Args:
            edge: The edge to validate
            graph: The graph to check for duplicates in

        Returns:
            bool: True if no duplicate exists, False otherwise
        """
        existing = graph.get_edge(edge.from_entity, edge.to_entity)
        if existing is None:
            return True
        return (
            existing.relation_type != edge.relation_type
            or existing.metadata.bidirectional != edge.metadata.bidirectional
        )


class SelfLoopValidator(EdgeValidator):
    """Validates that an edge does not create a self-loop."""

    def validate(self, edge: Edge, graph: Graph) -> bool:
        """
        Check if this edge creates a self-loop.

        Args:
            edge: The edge to validate
            graph: The graph context (unused)

        Returns:
            bool: True if not a self-loop, False otherwise
        """
        return edge.from_entity != edge.to_entity


class BidirectionalEdgeValidator(EdgeValidator):
    """Validates bidirectional edge constraints."""

    def validate(self, edge: Edge, graph: Graph) -> bool:
        """
        Check if bidirectional edge constraints are satisfied.

        For bidirectional edges, verifies that:
        1. If this edge is bidirectional, no opposing edge exists
        2. If an opposing edge exists, it has matching bidirectional flag

        Args:
            edge: The edge to validate
            graph: The graph context

        Returns:
            bool: True if bidirectional constraints are satisfied
        """
        opposing = graph.get_edge(edge.to_entity, edge.from_entity)
        if opposing is None:
            return True
        return edge.metadata.bidirectional == opposing.metadata.bidirectional


class GraphWithValidation(Graph):
    """Graph class with edge validation support."""

    def __init__(self, edges: List[Edge], cache_size: int = 1000, cache_ttl: int = 3600):
        """Initialize graph with validation strategy."""
        super().__init__(edges, cache_size, cache_ttl)
        self.validation_strategy = ValidationStrategy()

        # Add default validators
        self.validation_strategy.add_validator(CyclicDependencyValidator())
        self.validation_strategy.add_validator(DuplicateEdgeValidator())
        self.validation_strategy.add_validator(SelfLoopValidator())
        self.validation_strategy.add_validator(BidirectionalEdgeValidator())

    def add_validator(self, validator: EdgeValidator) -> None:
        """Add a custom validator to the strategy."""
        self.validation_strategy.add_validator(validator)

    def add_edge(self, edge: Edge) -> None:
        """Add an edge with validation."""
        valid, errors = self.validation_strategy.validate(edge, self)
        if not valid:
            raise ValidationError(f"Edge validation failed: {'; '.join(errors)}")
        super().add_edge(edge)

    def add_edges_batch(self, edges: List[Edge]) -> None:
        """Add multiple edges with validation."""
        with self.transaction():
            for edge in edges:
                self.add_edge(edge)  # This will validate each edge
