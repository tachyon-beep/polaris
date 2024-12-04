"""
Data Integrity Validation Components for Polaris Graph Database

This module provides comprehensive data integrity validation for nodes and edges
in the Polaris graph database. It ensures that all data structures maintain their
integrity and conform to expected formats and constraints.

The module implements validation for:
- Node and edge metadata
- Attribute dictionaries
- Metric values
- DateTime fields
- Confidence scores
- Impact scores

These validations help maintain data quality and consistency throughout the system.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from ...core.exceptions import ValidationError
from ...core.models import Edge, EdgeMetadata, Node, NodeMetadata
from .base import ValidationResult


class DataIntegrityValidator:
    """
    Validator for ensuring data integrity of nodes and edges.

    This class provides static methods for validating various aspects of nodes
    and edges, including their metadata, attributes, and metrics. It helps ensure
    that all data structures maintain their integrity and conform to expected formats.
    """

    @staticmethod
    def _validate_metadata_datetime(dt: Any, field_name: str) -> str:
        """
        Validate a metadata datetime field.

        Args:
            dt: Value to validate as datetime
            field_name: Name of the field being validated

        Returns:
            str: Error message if validation fails, empty string if validation passes
        """
        if not isinstance(dt, datetime):
            return f"{field_name} must be a datetime"
        return ""

    @staticmethod
    def _validate_attributes(attributes: Optional[Dict[str, Any]]) -> List[str]:
        """
        Validate a dictionary of attributes.

        Ensures that all attribute keys are non-empty strings and values are not None.

        Args:
            attributes: Dictionary of attributes to validate

        Returns:
            List[str]: List of validation error messages
        """
        errors = []
        if not attributes:
            return errors

        for key, value in attributes.items():
            if not isinstance(key, str) or not key.strip():
                errors.append("Invalid attribute key")
            if value is None:
                errors.append(f"Attribute value cannot be None for key: {key}")
        return errors

    @staticmethod
    def _validate_metrics(metrics: Optional[Dict[str, Any]]) -> List[str]:
        """
        Validate node metrics.

        Ensures that all metric keys are non-empty strings and values are numeric.

        Args:
            metrics: Dictionary of metrics to validate

        Returns:
            List[str]: List of validation error messages
        """
        errors = []
        if not metrics:
            return errors

        for key, value in metrics.items():
            if not isinstance(key, str) or not key.strip():
                errors.append(f"Invalid metric key: {key}")
            if not isinstance(value, (int, float)):
                errors.append(f"Metric value must be numeric for key: {key}")
            elif not isinstance(value, bool) and not (float("-inf") < value < float("inf")):
                errors.append("Metric value must be finite")
        return errors

    @staticmethod
    def _validate_node_metadata(metadata: NodeMetadata) -> List[str]:
        """
        Validate node metadata.

        Ensures that node metadata contains valid datetime fields and version number.

        Args:
            metadata: NodeMetadata instance to validate

        Returns:
            List[str]: List of validation error messages
        """
        errors = []

        if not isinstance(metadata, NodeMetadata):
            return ["Invalid metadata type"]

        dt_error = DataIntegrityValidator._validate_metadata_datetime(
            metadata.created_at, "Created at"
        )
        if dt_error:
            errors.append(dt_error)

        dt_error = DataIntegrityValidator._validate_metadata_datetime(
            metadata.last_modified, "Last modified"
        )
        if dt_error:
            errors.append(dt_error)

        if metadata.version < 1:
            errors.append("Version must be positive")

        return errors

    @staticmethod
    def _validate_edge_metadata(metadata: EdgeMetadata) -> List[str]:
        """
        Validate edge metadata.

        Ensures that edge metadata contains valid datetime fields and confidence score.

        Args:
            metadata: EdgeMetadata instance to validate

        Returns:
            List[str]: List of validation error messages
        """
        errors = []

        if not isinstance(metadata, EdgeMetadata):
            return ["Invalid metadata type"]

        dt_error = DataIntegrityValidator._validate_metadata_datetime(
            metadata.created_at, "Created at"
        )
        if dt_error:
            errors.append(dt_error)

        dt_error = DataIntegrityValidator._validate_metadata_datetime(
            metadata.last_modified, "Last modified"
        )
        if dt_error:
            errors.append(dt_error)

        if not 0 <= metadata.confidence <= 1:
            errors.append("Confidence must be between 0 and 1")

        return errors

    @staticmethod
    def validate_node_integrity(node: Node) -> ValidationResult:
        """
        Validate node data integrity.

        Performs comprehensive validation of a node including:
        - Basic node properties
        - Node metadata
        - Node attributes
        - Node metrics

        Args:
            node: Node instance to validate

        Returns:
            ValidationResult containing validation details and any errors or warnings

        Raises:
            ValidationError: If validation fails
        """
        try:
            errors = []
            warnings = []

            # Basic validation
            if not node.name:
                errors.append("Node name is required")

            # Metadata validation
            errors.extend(DataIntegrityValidator._validate_node_metadata(node.metadata))

            # Attributes validation
            errors.extend(DataIntegrityValidator._validate_attributes(node.attributes))

            # Metrics validation
            errors.extend(DataIntegrityValidator._validate_metrics(node.metrics))

            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                context={"node_id": node.name},
            )
        except ValueError as e:
            raise ValidationError(str(e))

    @staticmethod
    def validate_edge_integrity(edge: Edge) -> ValidationResult:
        """
        Validate edge data integrity.

        Performs comprehensive validation of an edge including:
        - Basic edge properties
        - Edge metadata
        - Impact score
        - Edge attributes

        Args:
            edge: Edge instance to validate

        Returns:
            ValidationResult containing validation details and any errors or warnings

        Raises:
            ValidationError: If validation fails
        """
        try:
            errors = []
            warnings = []

            # Basic validation
            if not edge.from_entity:
                errors.append("Source node is required")
            if not edge.to_entity:
                errors.append("Target node is required")

            # Metadata validation
            errors.extend(DataIntegrityValidator._validate_edge_metadata(edge.metadata))

            # Impact score validation
            if not 0 <= edge.impact_score <= 1:
                errors.append("Impact score must be between 0 and 1")

            # Attributes validation
            errors.extend(DataIntegrityValidator._validate_attributes(edge.attributes))

            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                context={
                    "from_node": edge.from_entity,
                    "to_node": edge.to_entity,
                },
            )
        except ValueError as e:
            raise ValidationError(str(e))
