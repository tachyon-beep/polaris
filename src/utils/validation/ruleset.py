"""
Rule Set Validation Components for Polaris Graph Database

This module provides components for managing and applying collections of validation
rules. It allows for:
- Grouping multiple validation rules by field
- Applying multiple rules to a single field
- Validating dictionaries of data against rule sets
- Collecting and reporting validation results

The rule set system provides a flexible way to define and apply validation rules
to complex data structures while maintaining clear organization and reporting.
"""

from typing import Any, Dict, List

from .base import ValidationResult, ValidationRule


class ValidationRuleSet:
    """
    Collection of validation rules organized by field.

    This class manages groups of validation rules that can be applied to
    dictionaries of data. It allows multiple rules to be associated with
    each field and provides methods for adding rules and validating data.

    Attributes:
        rules (Dict[str, List[ValidationRule]]): Dictionary mapping field names
            to lists of validation rules
    """

    def __init__(self):
        """
        Initialize an empty validation rule set.

        Creates a new rule set with no rules. Rules can be added using the
        add_rule method.
        """
        self.rules: Dict[str, List[ValidationRule]] = {}

    def add_rule(self, field: str, rule: ValidationRule) -> None:
        """
        Add a validation rule for a field.

        Associates a validation rule with a specific field. Multiple rules
        can be added for the same field, and they will be applied in the
        order they were added.

        Args:
            field: Name of the field to validate
            rule: ValidationRule instance to apply to the field

        Example:
            >>> rule_set = ValidationRuleSet()
            >>> rule_set.add_rule('age', RangeRule(0, 120, "Age must be between 0 and 120"))
            >>> rule_set.add_rule('age', RequiredRule("Age is required"))
        """
        if field not in self.rules:
            self.rules[field] = []
        self.rules[field].append(rule)

    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate data against all rules in the rule set.

        Applies all validation rules to their respective fields in the provided
        data dictionary. Rules for each field are applied in the order they
        were added.

        Args:
            data: Dictionary of field values to validate

        Returns:
            ValidationResult containing validation details and any errors

        Example:
            >>> rule_set = ValidationRuleSet()
            >>> rule_set.add_rule('age', RangeRule(0, 120, "Age must be between 0 and 120"))
            >>> result = rule_set.validate({'age': 150})
            >>> print(result.is_valid)
            False
            >>> print(result.errors)
            ['age: Age must be between 0 and 120']
        """
        errors = []
        warnings = []

        for field, rules in self.rules.items():
            value = data.get(field)
            for rule in rules:
                if not rule.validate(value):
                    errors.append(f"{field}: {rule.error_message}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            context={"validated_fields": list(self.rules.keys())},
        )
