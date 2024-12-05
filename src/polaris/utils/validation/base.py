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
- Dataclass field validation

These components form the building blocks of more complex validation scenarios
implemented throughout the Polaris system.
"""

import re
from dataclasses import dataclass, fields
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)


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


class DataclassRule(ValidationRule):
    """
    Rule for validating dataclass fields.

    This rule ensures that fields in a dataclass instance match their type hints
    and validates nested dataclass instances recursively.

    Attributes:
        dataclass_type: The dataclass type to validate against
    """

    def __init__(self, dataclass_type: Type, error_message: str = ""):
        """
        Initialize a dataclass validation rule.

        Args:
            dataclass_type: The dataclass type to validate against
            error_message: Message to display when validation fails
        """
        super().__init__(error_message or f"Invalid value for {dataclass_type.__name__}")
        self.dataclass_type = dataclass_type
        self.type_hints = get_type_hints(dataclass_type)

    def _validate_type(self, value: Any, expected_type: Type) -> bool:
        """Validate a value against its expected type."""
        origin = get_origin(expected_type)

        # Handle Any type
        if expected_type is Any:
            return True

        # Handle Optional types
        if origin is Union:
            args = get_args(expected_type)
            if type(None) in args and value is None:
                return True
            # Get the non-None type for Optional
            expected_type = next(t for t in args if t is not type(None))
            origin = get_origin(expected_type)

        # Handle None value for non-Optional type
        if value is None and origin is not Union:
            return False

        # Handle datetime specially
        if expected_type is datetime:
            return isinstance(value, datetime)

        # Handle container types
        if origin is list:
            if not isinstance(value, list):
                return False
            args = get_args(expected_type)
            if not args:  # Handle List without type parameter
                return True
            elem_type = args[0]
            return all(self._validate_type(item, elem_type) for item in value)
        elif origin is dict:
            if not isinstance(value, dict):
                return False
            args = get_args(expected_type)
            if not args:  # Handle Dict without type parameters
                return True
            if len(args) != 2:  # Dict should have exactly 2 type parameters
                return True
            key_type, val_type = args
            return all(
                self._validate_type(k, key_type) and self._validate_type(v, val_type)
                for k, v in value.items()
            )
        elif origin is tuple:
            if not isinstance(value, tuple):
                return False
            args = get_args(expected_type)
            if not args:  # Handle Tuple without type parameters
                return True
            if len(args) != len(value):
                return False
            return all(self._validate_type(val, typ) for val, typ in zip(value, args))
        # Handle basic types
        elif origin is not None:
            try:
                return isinstance(value, origin)
            except TypeError:
                return True
        else:
            try:
                return isinstance(value, expected_type)
            except TypeError:
                # Handle Any and other special types that don't work with isinstance
                return True

    def validate(self, value: Any) -> bool:
        """
        Validate a dataclass instance.

        Args:
            value: Dataclass instance to validate

        Returns:
            bool: True if validation passes, False otherwise
        """
        if not isinstance(value, self.dataclass_type):
            return False

        for field_name, field_type in self.type_hints.items():
            field_value = getattr(value, field_name)
            if not self._validate_type(field_value, field_type):
                return False

        return True


def validate_dataclass(cls: Type[Any]) -> Type[Any]:
    """
    Decorator that adds runtime type checking to dataclass fields.

    Args:
        cls: The dataclass to validate

    Returns:
        The decorated class with type validation

    Example:
        >>> @validate_dataclass
        ... @dataclass
        ... class Example:
        ...     name: str
        ...     count: int
    """
    original_post_init = getattr(cls, "__post_init__", None)

    def validated_post_init(self):
        """Validate all fields after initialization."""
        try:
            # Call original __post_init__ first to allow field validation
            if original_post_init:
                original_post_init(self)
        except ValueError as e:
            # Re-raise the original validation error
            raise ValueError(str(e))

        # Then validate types
        validator = DataclassRule(cls)
        if not validator.validate(self):
            raise TypeError(f"Invalid field types in {cls.__name__}")

    cls.__post_init__ = validated_post_init
    return cls
