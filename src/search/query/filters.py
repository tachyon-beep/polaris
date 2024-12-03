"""
Filter models for search queries in Polaris.

This module provides filter models that enable fine-grained control over search results.
It supports a wide range of filtering operations including:
- Equality and inequality comparisons
- Numeric range filters
- String pattern matching
- List membership checks
- Field existence validation
- Type checking
- Regular expression matching
- Semantic similarity filtering
"""

from dataclasses import dataclass
from typing import Any, Dict

from ...core.exceptions import QueryError

# Storage query constants
REGEX_KEY = "$regex"
OPTIONS_KEY = "$options"
CASE_INSENSITIVE = "i"


@dataclass
class SearchFilter:
    """
    Filter criteria for search queries.

    This class defines individual filter conditions that can be applied to search queries.
    It supports a wide range of operators for different types of filtering needs.

    Attributes:
        field (str): The field to apply the filter on
        operator (str): The type of filter operation to perform
        value (Any): The value to filter against

    Supported Operators:
        Equality:
            - "eq": Equal to
            - "ne": Not equal to

        Numeric:
            - "gt": Greater than
            - "gte": Greater than or equal to
            - "lt": Less than
            - "lte": Less than or equal to

        List:
            - "in": Value in list
            - "nin": Value not in list

        String:
            - "contains": String contains value
            - "startswith": String starts with value
            - "endswith": String ends with value

        Special:
            - "exists": Field existence check
            - "type": Type check
            - "regex": Regular expression match
            - "near": Semantic similarity
    """

    field: str
    operator: str  # "eq", "gt", "lt", "contains", "in", etc.
    value: Any

    def validate(self) -> None:
        """
        Validate filter configuration.

        Ensures the filter operator is one of the supported types and the
        configuration is valid for the chosen operator.

        Raises:
            QueryError: If the filter configuration is invalid
        """
        valid_operators = {
            "eq",
            "ne",  # Equality operators
            "gt",
            "gte",  # Greater than operators
            "lt",
            "lte",  # Less than operators
            "in",
            "nin",  # List membership operators
            "contains",  # String contains
            "startswith",  # String starts with
            "endswith",  # String ends with
            "exists",  # Field existence check
            "type",  # Type check
            "regex",  # Regular expression match
            "near",  # Semantic similarity
        }
        if self.operator not in valid_operators:
            raise QueryError(f"Invalid operator: {self.operator}")

    def to_storage_filter(self) -> Dict[str, Any]:
        """
        Convert filter to storage-compatible format.

        Transforms the filter into a format that can be understood by the storage
        backend. Each operator type is converted to its corresponding storage
        query format.

        Returns:
            Dictionary containing the storage-formatted filter condition

        Raises:
            QueryError: If the operator is not supported
        """
        if self.operator == "eq":
            return {self.field: self.value}
        elif self.operator == "ne":
            return {self.field: {"$ne": self.value}}
        elif self.operator in {"gt", "gte", "lt", "lte"}:
            return {self.field: {f"${self.operator}": self.value}}
        elif self.operator in {"in", "nin"}:
            return {self.field: {f"${self.operator}": list(self.value)}}
        elif self.operator == "contains":
            return {
                self.field: {
                    REGEX_KEY: f".*{self.value}.*",
                    OPTIONS_KEY: CASE_INSENSITIVE,
                }
            }
        elif self.operator == "startswith":
            return {self.field: {REGEX_KEY: f"^{self.value}", OPTIONS_KEY: CASE_INSENSITIVE}}
        elif self.operator == "endswith":
            return {self.field: {REGEX_KEY: f"{self.value}$", OPTIONS_KEY: CASE_INSENSITIVE}}
        elif self.operator == "exists":
            return {self.field: {"$exists": self.value}}
        elif self.operator == "type":
            return {self.field: {"$type": self.value}}
        elif self.operator == "regex":
            return {self.field: {REGEX_KEY: self.value, OPTIONS_KEY: CASE_INSENSITIVE}}
        elif self.operator == "near":
            return {
                self.field: {
                    "$near": self.value,
                    "$maxDistance": self.value.get("distance", 1.0),
                }
            }
        else:
            raise QueryError(f"Unsupported operator: {self.operator}")
