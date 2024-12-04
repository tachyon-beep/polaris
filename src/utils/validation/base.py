"""
Base Validation Components for Polaris Graph Database

This module provides the foundational validation components used throughout the validation system.
It includes the core ValidationResult class for reporting validation outcomes and a hierarchy
of ValidationRule classes for implementing different types of validation logic.

The module implements a flexible and extensible validation framework that supports:
- Basic validation results with errors, warnings, and context
- Required field validation
- Type checking
- Numeric range validation
- Regular expression pattern matching
- Custom validation functions

These components form the building blocks of more complex validation scenarios
implemented throughout the Polaris system.
"""

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union


@dataclass
class ValidationResult:
    """
    Container for validation results providing comprehensive validation outcome details.

    Attributes:
        is_valid (bool): Whether the validation passed successfully
        errors (List[str]): List of validation error messages
        warnings (List[str]): List of validation warning messages
        context (Optional[Dict[str, Any]]): Additional context about the validation
    """

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    context: Optional[Dict[str, Any]] = None


class ValidationRule:
    """
    Base class for all validation rules in the system.

    This class defines the interface that all validation rules must implement.
    Subclasses should override the validate() method to implement specific
    validation logic.

    Attributes:
        error_message (str): Message to display when validation fails
    """

    def __init__(self, error_message: str):
        """
        Initialize a validation rule.

        Args:
            error_message: Message to display when validation fails
        """
        self.error_message = error_message

    def validate(self, value: Any) -> bool:
        """
        Validate a value against the rule.

        Args:
            value: Value to validate

        Returns:
            bool: True if validation passes, False otherwise

        Raises:
            NotImplementedError: If the subclass doesn't implement this method
        """
        raise NotImplementedError("Validation rules must implement validate()")


class RequiredRule(ValidationRule):
    """
    Rule for validating required fields.

    This rule ensures that a value is not None and, if it's a string,
    that it's not empty after stripping whitespace.
    """

    def validate(self, value: Any) -> bool:
        """
        Validate that a value is present and non-empty.

        Args:
            value: Value to validate

        Returns:
            bool: True if the value is present and non-empty, False otherwise
        """
        if isinstance(value, str):
            return bool(value.strip())
        return value is not None


class TypeRule(ValidationRule):
    """
    Rule for type checking values.

    This rule ensures that a value is of the expected type or types.

    Attributes:
        expected_type: Single type or tuple of types to check against
    """

    def __init__(self, expected_type: Union[Type, Tuple[Type, ...]], error_message: str):
        """
        Initialize a type validation rule.

        Args:
            expected_type: Type or tuple of types to check against
            error_message: Message to display when validation fails
        """
        super().__init__(error_message)
        self.expected_type = expected_type

    def validate(self, value: Any) -> bool:
        """
        Validate that a value is of the expected type.

        Args:
            value: Value to validate

        Returns:
            bool: True if the value is of the expected type, False otherwise
        """
        return isinstance(value, self.expected_type)


class RangeRule(ValidationRule):
    """
    Rule for validating numeric ranges.

    This rule ensures that a numeric value falls within a specified range.
    Either min_value or max_value can be None to create an open-ended range.

    Attributes:
        min_value (Optional[float]): Minimum allowed value
        max_value (Optional[float]): Maximum allowed value
    """

    def __init__(
        self,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        error_message: str = "",
    ):
        """
        Initialize a range validation rule.

        Args:
            min_value: Minimum allowed value, or None for no minimum
            max_value: Maximum allowed value, or None for no maximum
            error_message: Message to display when validation fails
        """
        super().__init__(error_message)
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, value: Any) -> bool:
        """
        Validate that a value falls within the specified range.

        Args:
            value: Value to validate

        Returns:
            bool: True if the value is within range, False otherwise
        """
        if not isinstance(value, (int, float)):
            return False
        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False
        return True


class RegexRule(ValidationRule):
    """
    Rule for regex pattern matching.

    This rule ensures that a value matches a specified regular expression pattern.

    Attributes:
        pattern: Compiled regular expression pattern
    """

    def __init__(self, pattern: str, error_message: str):
        """
        Initialize a regex validation rule.

        Args:
            pattern: Regular expression pattern string
            error_message: Message to display when validation fails
        """
        super().__init__(error_message)
        self.pattern = re.compile(pattern)

    def validate(self, value: str) -> bool:
        """
        Validate that a value matches the regex pattern.

        Args:
            value: Value to validate

        Returns:
            bool: True if the value matches the pattern, False otherwise
        """
        return bool(self.pattern.match(str(value)))


class CustomRule(ValidationRule):
    """
    Rule for custom validation functions.

    This rule allows for arbitrary validation logic to be implemented
    through a callable function.

    Attributes:
        validator_func: Custom validation function that returns a boolean
    """

    def __init__(self, validator_func: Callable[[Any], bool], error_message: str):
        """
        Initialize a custom validation rule.

        Args:
            validator_func: Function that takes a value and returns True if valid
            error_message: Message to display when validation fails
        """
        super().__init__(error_message)
        self.validator_func = validator_func

    def validate(self, value: Any) -> bool:
        """
        Validate a value using the custom validation function.

        Args:
            value: Value to validate

        Returns:
            bool: True if validation passes, False otherwise
        """
        return self.validator_func(value)
