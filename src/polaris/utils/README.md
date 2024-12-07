# Polaris Utilities

This directory contains utility modules and components used throughout the Polaris graph database system. The utilities provide essential functionality for data validation, integrity checking, and schema validation.

## Directory Structure

```
utils/
├── __init__.py          # Main utilities module
├── validation/          # Validation subsystem
│   ├── __init__.py     # Validation package initialization
│   ├── base.py         # Core validation components
│   ├── integrity.py    # Data integrity validation
│   ├── reporter.py     # Validation result reporting
│   ├── ruleset.py      # Validation rule management
│   └── schema.py       # JSON schema validation
```

## Core Components

### Validation System

The validation system provides a comprehensive framework for ensuring data integrity and schema compliance throughout the Polaris graph database. It consists of several interconnected components:

#### Base Validation (`validation/base.py`)
- `ValidationResult`: Container for validation outcomes including errors and warnings
- `ValidationRule`: Base class for implementing validation rules
- Built-in rule types:
  - `RequiredRule`: Validates required fields
  - `TypeRule`: Performs type checking
  - `RangeRule`: Validates numeric ranges
  - `RegexRule`: Pattern matching using regular expressions
  - `CustomRule`: Support for custom validation functions

#### Data Integrity (`validation/integrity.py`)
- `DataIntegrityValidator`: Ensures data consistency for entities and relations
- Validates:
  - Entity and relation metadata
  - Attribute dictionaries
  - Metric values
  - DateTime fields
  - Confidence scores
  - Impact scores

#### Validation Reporting (`validation/reporter.py`)
- `ValidationReporter`: Formats and outputs validation results
- Supported formats:
  - Human-readable string output
  - Dictionary representation
  - JSON serialization
- Includes context information for debugging

#### Rule Sets (`validation/ruleset.py`)
- `ValidationRuleSet`: Manages collections of validation rules
- Features:
  - Group multiple rules by field
  - Apply multiple rules to single fields
  - Validate dictionaries of data
  - Collect and report validation results

#### Schema Validation (`validation/schema.py`)
- `SchemaValidator`: JSON schema-based validation
- Capabilities:
  - Register schemas for different entity types
  - Register schemas for different relation types
  - Validate entities against schemas
  - Validate relations against schemas
  - Schema-based structural validation

## Usage Examples

### Basic Validation

```python
from polaris.utils.validation.base import RequiredRule, ValidationResult

# Create a validation rule
rule = RequiredRule("Field is required")

# Validate a value
result = ValidationResult(
    is_valid=rule.validate("test"),
    errors=[],
    warnings=[]
)
```

### Data Integrity Validation

```python
from polaris.utils.validation.integrity import DataIntegrityValidator

# Validate an entity
result = DataIntegrityValidator.validate_entity_integrity(entity)

# Validate a relation
result = DataIntegrityValidator.validate_relation_integrity(relation)
```

### Schema Validation

```python
from polaris.utils.validation.schema import SchemaValidator

# Create a schema validator
validator = SchemaValidator()

# Register a schema for an entity type
validator.register_entity_schema(EntityType.PERSON, schema_dict)

# Validate an entity
result = validator.validate_entity(entity)
```

### Rule Sets

```python
from polaris.utils.validation.ruleset import ValidationRuleSet
from polaris.utils.validation.base import RangeRule, RequiredRule

# Create a rule set
rule_set = ValidationRuleSet()

# Add multiple rules for a field
rule_set.add_rule('age', RequiredRule("Age is required"))
rule_set.add_rule('age', RangeRule(0, 120, "Age must be between 0 and 120"))

# Validate data
result = rule_set.validate({'age': 25})
```

### Validation Reporting

```python
from polaris.utils.validation.reporter import ValidationReporter

# Format result as string
formatted = ValidationReporter.format_result(result)

# Convert to dictionary
dict_result = ValidationReporter.to_dict(result)

# Convert to JSON
json_result = ValidationReporter.to_json(result)
```

## Best Practices

1. **Rule Organization**
   - Group related validation rules in rule sets
   - Use descriptive error messages
   - Consider validation order when adding multiple rules

2. **Schema Management**
   - Register schemas during application initialization
   - Keep schemas in separate configuration files
   - Version control your schemas

3. **Error Handling**
   - Always check ValidationResult.is_valid
   - Handle both errors and warnings appropriately
   - Include context information for debugging

4. **Performance**
   - Cache ValidationRuleSet instances when possible
   - Use schema validation for structural checks
   - Use data integrity validation for semantic checks

## Contributing

When adding new utility functions or validation components:

1. Follow the existing documentation patterns
2. Add comprehensive docstrings with examples
3. Include type hints for all functions and methods
4. Update this README with any new components or examples
5. Ensure backward compatibility when modifying existing utilities
