"""
Enumerations for entity and relation types in the knowledge graph.

This module defines the core enumeration types used throughout the system to classify
and categorize different entities and their relationships. It provides a structured
way to define and maintain the taxonomy of the knowledge graph.

The enumerations are organized into three main categories:
- EntityType: Defines all possible types of nodes in the knowledge graph
- RelationType: Defines all possible types of edges/relationships between entities
- DomainContext: Defines the high-level domains for specialized operations
"""

from enum import Enum


class EntityType(Enum):
    """
    Enumeration of all possible entity types in the knowledge graph.

    Entity types are organized into several categories:
    - Code and Development: Represents software development concepts
    - Game Design: Represents game development and mechanics
    - Systems Engineering: Represents system architecture and requirements
    - Analysis and Intelligence: Represents data analysis and insights
    - Documentation: Represents various forms of documentation
    """

    # Code and Development Entities
    CODE_MODULE = "code_module"  # A module/file of code
    CODE_FUNCTION = "code_function"  # A function definition
    CODE_CLASS = "code_class"  # A class definition
    CODE_PATTERN = "code_pattern"  # A design pattern implementation
    LIBRARY = "library"  # An external library or package
    API = "api"  # An API endpoint or interface

    # Game Design Entities
    GAME_MECHANIC = "game_mechanic"  # Core gameplay mechanic
    GAME_SYSTEM = "game_system"  # Major game system
    GAME_RESOURCE = "game_resource"  # In-game resource
    GAME_COMPONENT = "game_component"  # Game component or element
    GAME_PATTERN = "game_pattern"  # Game design pattern
    PLAYER_BEHAVIOR = "player_behavior"  # Player interaction pattern

    # Systems Engineering Entities
    SYSTEM = "system"  # Complete system
    SUBSYSTEM = "subsystem"  # Part of larger system
    COMPONENT = "component"  # System component
    INTERFACE = "interface"  # Component interface
    REQUIREMENT = "requirement"  # System requirement
    CONSTRAINT = "constraint"  # System constraint
    STAKEHOLDER = "stakeholder"  # System stakeholder

    # Analysis and Intelligence Entities
    DATA_SOURCE = "data_source"  # Source of data
    DATASET = "dataset"  # Collection of data
    METRIC = "metric"  # Measurement or KPI
    INSIGHT = "insight"  # Derived understanding
    HYPOTHESIS = "hypothesis"  # Proposed explanation
    EVIDENCE = "evidence"  # Supporting data
    CORRELATION = "correlation"  # Related patterns

    # Documentation Entities
    DOCUMENT = "document"  # General document
    SPEC = "specification"  # Detailed specification
    STANDARD = "standard"  # Industry standard
    PROCEDURE = "procedure"  # Process description
    REFERENCE = "reference"  # Reference material
    EXAMPLE = "example"  # Example or sample
    USE_CASE = "use_case"  # Usage scenario


class RelationType(Enum):
    """
    Enumeration of all possible relationship types between entities in the knowledge graph.

    Relations are organized into several categories:
    - Code Relations: Represents relationships between code entities
    - Game Design Relations: Represents interactions between game elements
    - Systems Relations: Represents system component connections
    - Analysis Relations: Represents analytical relationships
    - Documentation Relations: Represents documentation connections
    """

    # Code Relations
    IMPORTS = "imports"  # Module import relationship
    INHERITS_FROM = "inherits_from"  # Class inheritance
    CALLS = "calls"  # Function invocation
    IMPLEMENTS = "implements"  # Interface implementation
    DEPENDS_ON = "depends_on"  # Dependency relationship
    TESTED_BY = "tested_by"  # Test coverage

    # Game Design Relations
    INTERACTS_WITH = "interacts_with"  # Mechanic interaction
    GENERATES = "generates"  # Resource generation
    CONSUMES = "consumes"  # Resource consumption
    BALANCES = "balances"  # Game balance relationship
    COUNTERS = "counters"  # Counter mechanism
    EXTENDS = "extends"  # Mechanic extension

    # Systems Relations
    COMPOSED_OF = "composed_of"  # Component composition
    CONNECTS_TO = "connects_to"  # System connection
    PROVIDES = "provides"  # Service provision
    REQUIRES = "requires"  # Dependency requirement
    CONSTRAINS = "constrains"  # Constraint relationship
    VALIDATES = "validates"  # Validation relationship

    # Analysis Relations
    DERIVED_FROM = "derived_from"  # Data derivation
    CORRELATES_WITH = "correlates_with"  # Statistical correlation
    SUPPORTS = "supports"  # Evidence support
    CONTRADICTS = "contradicts"  # Contradictory relationship
    CONFIRMS = "confirms"  # Confirmation relationship
    SOURCES = "sources"  # Data sourcing

    # Documentation Relations
    REFERENCES = "references"  # Document reference
    DESCRIBES = "describes"  # Description relationship
    EXEMPLIFIES = "exemplifies"  # Example relationship
    SUPERSEDES = "supersedes"  # Version supersession
    RELATES_TO = "relates_to"  # General relation


class DomainContext(Enum):
    """
    Domain contexts for specialized queries and analysis.

    These contexts define the high-level domains under which entities and relations
    can be organized and analyzed. Each context provides a specific lens through
    which to view and process the knowledge graph data.

    Attributes:
        CODE: Software development and code analysis context
        GAME_DESIGN: Game development and design context
        SYSTEMS: Systems engineering and architecture context
        ANALYSIS: Data analysis and intelligence context
        DOCUMENTATION: Documentation and knowledge management context
    """

    CODE = "code"
    GAME_DESIGN = "game_design"
    SYSTEMS = "systems"
    ANALYSIS = "analysis"
    DOCUMENTATION = "documentation"
