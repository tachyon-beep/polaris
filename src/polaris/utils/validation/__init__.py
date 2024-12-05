"""
Validation package for Polaris Graph Database.

This package provides validation utilities and rules for ensuring data integrity
and type safety throughout the system.
"""

from .base import (
    ValidationResult,
    ValidationRule,
    RequiredRule,
    TypeRule,
    RangeRule,
    RegexRule,
    CustomRule,
    DataclassRule,
    validate_dataclass,
)

__all__ = [
    "ValidationResult",
    "ValidationRule",
    "RequiredRule",
    "TypeRule",
    "RangeRule",
    "RegexRule",
    "CustomRule",
    "DataclassRule",
    "validate_dataclass",
]
