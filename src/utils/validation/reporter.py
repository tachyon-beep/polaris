"""
Validation Reporter Components for Polaris Graph Database

This module provides components for formatting and outputting validation results
in various formats. It supports:
- Human-readable string formatting
- Dictionary conversion
- JSON serialization

The reporter components help in presenting validation results in a clear and
consistent manner, making it easier to understand validation outcomes and
debug validation issues.
"""

import json
from typing import Any, Dict

from .base import ValidationResult


class ValidationReporter:
    """
    Reporter for formatting and outputting validation results.

    This class provides static methods for converting ValidationResult instances
    into various formats suitable for different use cases, such as human-readable
    output, dictionary representation, or JSON serialization.
    """

    @staticmethod
    def format_result(result: ValidationResult) -> str:
        """
        Format a validation result as a human-readable string.

        Creates a formatted string representation of the validation result,
        including errors, warnings, and context information. The output is
        structured for easy reading with appropriate indentation and sections.

        Args:
            result: ValidationResult instance to format

        Returns:
            str: Formatted string representation of the validation result

        Example:
            >>> result = ValidationResult(False, ["Invalid type"], ["Check value"], {"field": "age"})
            >>> print(ValidationReporter.format_result(result))
            Validation failed with the following errors:
              - Invalid type

            Warnings:
              - Check value

            Context:
              field: age
        """
        lines = []

        if not result.is_valid:
            lines.append("Validation failed with the following errors:")
            for error in result.errors:
                lines.append(f"  - {error}")

        if result.warnings:
            lines.append("\nWarnings:")
            for warning in result.warnings:
                lines.append(f"  - {warning}")

        if result.context:
            lines.append("\nContext:")
            for key, value in result.context.items():
                lines.append(f"  {key}: {value}")

        if not lines:
            lines.append("Validation passed successfully")

        return "\n".join(lines)

    @staticmethod
    def to_dict(result: ValidationResult) -> Dict[str, Any]:
        """
        Convert a validation result to a dictionary.

        Transforms the ValidationResult instance into a dictionary format,
        suitable for serialization or further processing.

        Args:
            result: ValidationResult instance to convert

        Returns:
            Dict[str, Any]: Dictionary representation of the validation result

        Example:
            >>> result = ValidationResult(True, [], [], {"id": 123})
            >>> ValidationReporter.to_dict(result)
            {
                'is_valid': True,
                'errors': [],
                'warnings': [],
                'context': {'id': 123}
            }
        """
        return {
            "is_valid": result.is_valid,
            "errors": result.errors,
            "warnings": result.warnings,
            "context": result.context,
        }

    @staticmethod
    def to_json(result: ValidationResult) -> str:
        """
        Convert a validation result to JSON.

        Serializes the ValidationResult instance to a JSON string with proper
        formatting and indentation.

        Args:
            result: ValidationResult instance to convert

        Returns:
            str: JSON string representation of the validation result

        Example:
            >>> result = ValidationResult(True, [], [], {"id": 123})
            >>> print(ValidationReporter.to_json(result))
            {
              "is_valid": true,
              "errors": [],
              "warnings": [],
              "context": {
                "id": 123
              }
            }
        """
        return json.dumps(ValidationReporter.to_dict(result), indent=2)
