"""
Validation System for Polaris Graph Database

This module provides a comprehensive validation system for ensuring data integrity,
schema compliance, and custom validation rules in the Polaris graph database.

Key Components:
- ValidationResult: Container for validation outcomes including errors and warnings
- ValidationRule: Base class for implementing validation rules
- SchemaValidator: JSON schema-based validation for entities and relations
- DataIntegrityValidator: Ensures data consistency and integrity
- ValidationRuleSet: Manages collections of validation rules
- ValidationReporter: Formats and outputs validation results

The validation system supports:
- Type checking and required field validation
- Numeric range validation
- Regular expression pattern matching
- Custom validation functions
- JSON schema validation
- Data integrity checks for entities and relations
- Flexible rule sets for complex validation scenarios
"""

from .base import (
    CustomRule,
    RangeRule,
    RegexRule,
    RequiredRule,
    TypeRule,
    ValidationResult,
    ValidationRule,
)
from .integrity import DataIntegrityValidator
from .reporter import ValidationReporter
from .ruleset import ValidationRuleSet
from .schema import SchemaValidator

__all__ = [
    "ValidationResult",
    "ValidationRule",
    "RequiredRule",
    "TypeRule",
    "RangeRule",
    "RegexRule",
    "CustomRule",
    "SchemaValidator",
    "DataIntegrityValidator",
    "ValidationRuleSet",
    "ValidationReporter",
]
