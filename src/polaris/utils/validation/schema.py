"""
Schema Validation Components for Polaris Graph Database

This module provides JSON schema-based validation for nodes and edges
in the Polaris graph database. It supports:
- Registration of JSON schemas for different node types
- Registration of JSON schemas for different edge types
- Validation of nodes and edges against their registered schemas
- Comprehensive validation reporting

The schema validation system ensures that nodes and edges conform to
their expected structure and data types as defined in JSON schemas.
"""

from typing import Any, Dict, TYPE_CHECKING

from jsonschema import ValidationError as JsonSchemaError
from jsonschema import validate as json_validate

from ...core.enums import EntityType, RelationType
from .base import ValidationResult

if TYPE_CHECKING:
    from ...core.models import Edge, Node


class SchemaValidator:
    """
    JSON Schema-based validator for nodes and edges.

    This class manages JSON schemas for different types of nodes and edges,
    allowing for schema-based validation of these objects. It supports registering
    different schemas for different types and provides methods for validating
    instances against their registered schemas.

    Attributes:
        node_schemas (Dict[EntityType, Dict[str, Any]]): Dictionary mapping node
            types to their JSON schemas
        edge_schemas (Dict[RelationType, Dict[str, Any]]): Dictionary mapping
            edge types to their JSON schemas
    """

    def __init__(self):
        """
        Initialize an empty schema validator.

        Creates a new validator with no registered schemas. Schemas can be
        registered using the register_node_schema and register_edge_schema
        methods.
        """
        self.node_schemas: Dict[EntityType, Dict[str, Any]] = {}
        self.edge_schemas: Dict[RelationType, Dict[str, Any]] = {}

    def register_node_schema(self, entity_type: EntityType, schema: Dict[str, Any]) -> None:
        """
        Register a JSON schema for a node type.

        Associates a JSON schema with a specific node type. This schema will
        be used to validate nodes of this type.

        Args:
            entity_type: Type of node this schema applies to
            schema: JSON schema definition as a dictionary

        Example:
            >>> validator = SchemaValidator()
            >>> schema = {
            ...     "type": "object",
            ...     "properties": {
            ...         "name": {"type": "string"},
            ...         "age": {"type": "integer", "minimum": 0}
            ...     },
            ...     "required": ["name"]
            ... }
            >>> validator.register_node_schema(EntityType.PERSON, schema)
        """
        self.node_schemas[entity_type] = schema

    def register_edge_schema(self, relation_type: RelationType, schema: Dict[str, Any]) -> None:
        """
        Register a JSON schema for an edge type.

        Associates a JSON schema with a specific edge type. This schema will
        be used to validate edges of this type.

        Args:
            relation_type: Type of edge this schema applies to
            schema: JSON schema definition as a dictionary

        Example:
            >>> validator = SchemaValidator()
            >>> schema = {
            ...     "type": "object",
            ...     "properties": {
            ...         "from_entity": {"type": "string"},
            ...         "to_entity": {"type": "string"},
            ...         "impact_score": {"type": "number", "minimum": 0, "maximum": 1}
            ...     },
            ...     "required": ["from_entity", "to_entity"]
            ... }
            >>> validator.register_edge_schema(RelationType.DEPENDS_ON, schema)
        """
        self.edge_schemas[relation_type] = schema

    def validate_node(self, node: "Node") -> ValidationResult:
        """
        Validate a node against its registered schema.

        Validates the node against the JSON schema registered for its type.
        If no schema is registered for the node type, a warning is included
        in the validation result.

        Args:
            node: Node instance to validate

        Returns:
            ValidationResult containing validation details and any errors or warnings

        Example:
            >>> validator = SchemaValidator()
            >>> # ... register schema for EntityType.PERSON ...
            >>> node = Node(name="John", entity_type=EntityType.PERSON)
            >>> result = validator.validate_node(node)
            >>> print(result.is_valid)
            True
        """
        errors = []
        warnings = []

        try:
            schema = self.node_schemas.get(node.entity_type)
            if schema:
                json_validate(instance=node.__dict__, schema=schema)
            else:
                warnings.append(f"No schema registered for node type: {node.entity_type}")

        except JsonSchemaError as e:
            errors.append(f"Schema validation failed: {str(e)}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            context={"node_type": node.entity_type.name},
        )

    def validate_edge(self, edge: "Edge") -> ValidationResult:
        """
        Validate an edge against its registered schema.

        Validates the edge against the JSON schema registered for its type.
        If no schema is registered for the edge type, a warning is included
        in the validation result.

        Args:
            edge: Edge instance to validate

        Returns:
            ValidationResult containing validation details and any errors or warnings

        Example:
            >>> validator = SchemaValidator()
            >>> # ... register schema for RelationType.DEPENDS_ON ...
            >>> edge = Edge(
            ...     from_entity="service_a",
            ...     to_entity="service_b",
            ...     relation_type=RelationType.DEPENDS_ON
            ... )
            >>> result = validator.validate_edge(edge)
            >>> print(result.is_valid)
            True
        """
        errors = []
        warnings = []

        try:
            schema = self.edge_schemas.get(edge.relation_type)
            if schema:
                json_validate(instance=edge.__dict__, schema=schema)
            else:
                warnings.append(f"No schema registered for edge type: {edge.relation_type}")

        except JsonSchemaError as e:
            errors.append(f"Schema validation failed: {str(e)}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            context={"edge_type": edge.relation_type.name},
        )
