"""
Semantic-specific query model for Polaris.

This module provides specialized query functionality for semantic search operations.
It extends the base search query model with semantic-specific features including:
- Embedding model selection
- Similarity thresholds
- Context window configuration
- Semantic operator support
- Field-specific semantic search
- Temporal boosting
- Similar result inclusion
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from ...core.exceptions import QueryError
from .base import SearchQuery


@dataclass
class SemanticQuery(SearchQuery):
    """
    Specialized query for semantic search operations.

    This class extends the base SearchQuery with semantic-specific functionality,
    enabling natural language understanding and similarity-based search.

    Attributes:
        embedding_model (str): Name of the embedding model to use
        similarity_threshold (float): Minimum similarity score for matches (0-1)
        context_window (int): Number of surrounding items to include for context
        semantic_operators (List[str]): List of semantic operations to apply
        semantic_fields (List[str]): Fields to include in semantic analysis
        boost_recent (bool): Whether to boost scores of recent items
        include_similar (bool): Whether to include similar items in results

    The query supports various semantic operations:
    - Text embedding generation
    - Similarity scoring
    - Contextual analysis
    - Semantic field selection
    - Temporal relevance boosting
    - Similar item discovery

    Example semantic operators:
    - "expand_synonyms": Include synonymous terms
    - "include_related": Include related concepts
    - "fuzzy_match": Enable fuzzy matching
    - "concept_expansion": Expand to related concepts
    """

    embedding_model: str = "default"
    similarity_threshold: float = 0.7
    context_window: int = 5
    semantic_operators: List[str] = field(default_factory=list)
    semantic_fields: List[str] = field(
        default_factory=lambda: ["name", "documentation", "observations"]
    )
    boost_recent: bool = False
    include_similar: bool = False

    def validate(self) -> None:
        """
        Validate semantic query configuration.

        This method extends the base query validation with semantic-specific checks:
        - Similarity threshold is between 0 and 1
        - Context window is positive
        - At least one semantic field is specified
        - Semantic operators are valid
        - Embedding model is available

        Raises:
            QueryError: If any validation check fails
        """
        super().validate()
        if self.similarity_threshold < 0 or self.similarity_threshold > 1:
            raise QueryError("Similarity threshold must be between 0 and 1")
        if self.context_window < 1:
            raise QueryError("Context window must be positive")
        if not self.semantic_fields:
            raise QueryError("At least one semantic field must be specified")

    def to_storage_query(self) -> Dict[str, Any]:
        """
        Convert semantic query to storage-compatible format.

        This method extends the base query conversion with semantic-specific parameters:
        - Embedding model configuration
        - Similarity threshold
        - Semantic field selection
        - Semantic operator configuration
        - Temporal boosting settings

        Returns:
            Dictionary containing storage-formatted query parameters suitable
            for execution against the semantic search backend

        Example:
            A semantic query might be converted to:
            {
                "$semanticSearch": {
                    "fields": ["name", "documentation"],
                    "threshold": 0.7,
                    "model": "default",
                    "context_window": 5,
                    "operators": ["expand_synonyms"],
                    "boost_recent": true,
                    "include_similar": false
                },
                ...additional base query parameters
            }
        """
        query = super().to_storage_query()

        # Add semantic search specific parameters
        query["$semanticSearch"] = {
            "fields": self.semantic_fields,
            "threshold": self.similarity_threshold,
            "model": self.embedding_model,
            "context_window": self.context_window,
            "operators": self.semantic_operators,
            "boost_recent": self.boost_recent,
            "include_similar": self.include_similar,
        }

        return query
