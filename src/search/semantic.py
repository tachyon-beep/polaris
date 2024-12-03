"""
Semantic search capabilities for the knowledge graph.

This module implements semantic search functionality using vector embeddings and 
similarity matching. It provides a search engine that can understand natural language
queries and find semantically related nodes in the knowledge graph.

Key features:
- Vector embedding generation for text
- Cosine similarity matching
- Semantic operator support (synonyms, related concepts)
- Caching of embeddings for performance
- Concurrent processing of search candidates
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, cast

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..core.exceptions import QueryError, StorageError
from ..core.models import Node
from ..infrastructure.cache import LRUCache
from ..infrastructure.storage import StorageService
from .query import SearchFilter, SearchResults, SemanticQuery

logger = logging.getLogger(__name__)


class SemanticSearchEngine:
    """
    Engine for performing semantic searches across the knowledge graph.

    This class handles the core semantic search functionality, including:
    - Embedding generation for queries and nodes
    - Similarity calculation and ranking
    - Result filtering and pagination
    - Semantic operator application

    Attributes:
        storage (StorageService): Service for accessing node storage
        embedding_service (Any): Service for generating text embeddings
        cache (Optional[LRUCache[np.ndarray]]): Cache for storing embeddings
        executor (ThreadPoolExecutor): Executor for parallel processing
    """

    def __init__(
        self,
        storage: StorageService,
        embedding_service: Any,  # Replace with actual embedding service type
        cache: Optional[LRUCache[np.ndarray]] = None,
        max_workers: int = 4,
    ):
        """
        Initialize the semantic search engine.

        Args:
            storage: Service for accessing node storage
            embedding_service: Service for generating text embeddings
            cache: Optional cache for storing embeddings
            max_workers: Maximum number of concurrent workers for processing
        """
        self.storage = storage
        self.embedding_service = embedding_service
        self.cache = cache
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def search(self, query: SemanticQuery) -> SearchResults:
        """
        Perform semantic search based on the provided query.

        This method handles the complete semantic search workflow:
        1. Query validation
        2. Embedding generation
        3. Candidate retrieval
        4. Similarity ranking
        5. Result pagination

        Args:
            query: SemanticQuery object containing search parameters

        Returns:
            SearchResults containing matched items and metadata

        Raises:
            QueryError: If search execution fails
            StorageError: If storage operations fail
        """
        start_time = datetime.now()

        try:
            # Validate query
            query.validate()

            if not query.query_text:
                raise QueryError("Query text is required for semantic search")

            # Generate embeddings for query text
            query_embedding = await self._get_embedding(query.query_text)

            # Retrieve candidate nodes
            candidates = await self._get_candidates(query)

            # Calculate similarities and rank results
            results = await self._rank_results(
                candidates, query_embedding, query.similarity_threshold
            )

            # Apply pagination
            paginated_results = self._paginate_results(results, query.page, query.page_size)

            execution_time = (datetime.now() - start_time).total_seconds()

            return SearchResults(
                items=paginated_results,
                total=len(results),
                page=query.page,
                page_size=query.page_size,
                execution_time=execution_time,
                query=query,
            )

        except StorageError as e:
            logger.error(f"Storage error during semantic search: {str(e)}")
            raise QueryError(f"Storage error during semantic search: {str(e)}")
        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            raise QueryError(f"Failed to execute semantic search: {str(e)}")

    async def _get_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding vector for input text.

        This method handles embedding generation with caching support:
        1. Check cache for existing embedding
        2. Generate new embedding if needed
        3. Cache new embedding
        4. Return embedding vector

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as numpy array

        Raises:
            QueryError: If embedding generation fails
        """
        try:
            cache_key = f"emb:{text}"

            # Check cache first if available
            if self.cache:
                cached_embedding = self.cache.get(cache_key)
                if cached_embedding is not None:
                    return cached_embedding

            # Generate embedding
            embedding = await self.embedding_service.embed_text(text)

            # Cache result if cache service is available
            if self.cache:
                self.cache.put(cache_key, embedding)

            return embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise QueryError(f"Embedding generation failed: {str(e)}")

    async def _get_candidates(self, query: SemanticQuery) -> List[Node]:
        """
        Retrieve candidate nodes based on query parameters.

        This method handles candidate retrieval:
        1. Convert query filters to storage format
        2. Apply semantic operators if specified
        3. Retrieve matching nodes from storage

        Args:
            query: Semantic query containing filter parameters

        Returns:
            List of candidate nodes

        Raises:
            StorageError: If storage operations fail
        """
        try:
            # Convert List[SearchFilter] to Dict[str, Any]
            filters: Dict[str, Any] = {}
            if query.filters:
                for filter in query.filters:
                    filters.update(filter.to_storage_filter())

            # Add semantic operator filters if specified
            if query.semantic_operators:
                await self._apply_semantic_operators(filters, query.semantic_operators)

            # Get candidates from storage
            candidates = await self.storage.list_nodes(filters=filters)

            return candidates

        except StorageError as e:
            logger.error(f"Failed to retrieve candidates: {str(e)}")
            raise StorageError(f"Candidate retrieval failed: {str(e)}")

    async def _rank_results(
        self,
        candidates: List[Node],
        query_embedding: np.ndarray,
        similarity_threshold: float,
    ) -> List[Dict[str, Any]]:
        """
        Rank candidates based on semantic similarity.

        This method handles result ranking:
        1. Process candidates in parallel
        2. Calculate similarity scores
        3. Filter by threshold
        4. Sort by similarity

        Args:
            candidates: List of candidate nodes
            query_embedding: Query embedding vector
            similarity_threshold: Minimum similarity threshold

        Returns:
            Ranked list of results with similarity scores
        """
        try:
            ranked_results = []

            # Process candidates in parallel
            async def process_candidate(candidate: Node) -> Optional[Dict[str, Any]]:
                # Generate embedding for candidate
                candidate_text = self._get_searchable_text(candidate)
                candidate_embedding = await self._get_embedding(candidate_text)

                # Calculate similarity
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1), candidate_embedding.reshape(1, -1)
                )[0][0]

                if similarity >= similarity_threshold:
                    return {**asdict(candidate), "similarity_score": float(similarity)}
                return None

            # Process candidates concurrently
            tasks = [process_candidate(candidate) for candidate in candidates]
            results = await asyncio.gather(*tasks)

            # Filter None results and sort by similarity
            ranked_results = [r for r in results if r is not None]
            ranked_results.sort(key=lambda x: x["similarity_score"], reverse=True)

            return ranked_results

        except Exception as e:
            logger.error(f"Failed to rank results: {str(e)}")
            raise QueryError(f"Result ranking failed: {str(e)}")

    def _paginate_results(
        self, results: List[Dict[str, Any]], page: int, page_size: int
    ) -> List[Dict[str, Any]]:
        """
        Apply pagination to ranked results.

        Args:
            results: List of ranked results
            page: Page number (1-based)
            page_size: Number of results per page

        Returns:
            Paginated subset of results
        """
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        return results[start_idx:end_idx]

    async def _apply_semantic_operators(
        self, filters: Dict[str, Any], operators: List[str]
    ) -> None:
        """
        Apply semantic operators to modify search filters.

        This method enhances search filters with semantic operations:
        - Synonym expansion
        - Related node inclusion
        - Other semantic enrichments

        Args:
            filters: Current filter dictionary
            operators: List of semantic operators to apply
        """
        for operator in operators:
            if operator == "expand_synonyms":
                await self._expand_synonym_filters(filters)
            elif operator == "include_related":
                await self._include_related_filters(filters)
            # Add more semantic operators as needed

    def _get_searchable_text(self, node: Node) -> str:
        """
        Extract searchable text content from node.

        This method combines various node fields into a single searchable text:
        - Node name
        - Documentation
        - Observations
        - Examples
        - Tags

        Args:
            node: Node to extract text from

        Returns:
            Concatenated searchable text
        """
        searchable_parts = [
            node.name,
            node.documentation or "",
            " ".join(node.observations),
            " ".join(node.examples),
            " ".join(node.metadata.tags),
        ]
        return " ".join(filter(None, searchable_parts))

    async def _expand_synonym_filters(self, filters: Dict[str, Any]) -> None:
        """
        Expand search filters with synonymous terms.

        This method modifies filters to include synonyms of search terms,
        improving recall for semantically similar content.

        Args:
            filters: Filter dictionary to modify
        """
        if "name" in filters:
            synonyms = await self.embedding_service.get_synonyms(filters["name"])
            if synonyms:
                filters["$or"] = [
                    {"name": filters["name"]},
                    {"name": {"$in": synonyms}},
                ]
                del filters["name"]

    async def _include_related_filters(self, filters: Dict[str, Any]) -> None:
        """
        Include related nodes in search filters.

        This method expands node type filters to include related types,
        improving search coverage across related concepts.

        Args:
            filters: Filter dictionary to modify
        """
        if "entity_type" in filters:
            related_types = await self.embedding_service.get_related_types(filters["entity_type"])
            if related_types:
                filters["entity_type"] = {"$in": [filters["entity_type"]] + related_types}
