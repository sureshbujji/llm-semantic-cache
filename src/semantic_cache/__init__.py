"""Semantic cache for LLM calls: skip the API when a similar prompt was seen before."""

from .cache import CacheEntry, SemanticCache
from .decorators import semantic_cached
from .embeddings import EmbeddingProvider, HashingEmbedder

__all__ = ["CacheEntry", "SemanticCache", "semantic_cached",
           "EmbeddingProvider", "HashingEmbedder"]
__version__ = "0.1.0"
