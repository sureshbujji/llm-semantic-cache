"""Drop-in decorator: cache any text-in/text-out LLM call by prompt similarity."""
from __future__ import annotations

import functools
from typing import Any, Callable, Dict, Optional

from .cache import SemanticCache
from .embeddings import EmbeddingProvider


def semantic_cached(
    fn: Optional[Callable[..., str]] = None,
    *,
    cache: Optional[SemanticCache] = None,
    embedder: Optional[EmbeddingProvider] = None,
    similarity_threshold: float = 0.97,
    ttl_seconds: Optional[float] = 3600,
    max_size: int = 1000,
    prompt_arg: str = "prompt",
) -> Callable[..., str]:
    """Wrap an LLM call so repeated/similar prompts hit the cache.

    Usage:
        cache = SemanticCache()
        @semantic_cached(cache=cache)
        def ask_llm(prompt: str) -> str: ...
    """
    shared = cache or SemanticCache(
        embedder=embedder,
        similarity_threshold=similarity_threshold,
        ttl_seconds=ttl_seconds,
        max_size=max_size,
    )

    def decorator(func: Callable[..., str]) -> Callable[..., str]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> str:
            prompt = kwargs.get(prompt_arg)
            if prompt is None and args:
                prompt = args[0]
            if not isinstance(prompt, str):
                raise TypeError(f"prompt must be str, got {type(prompt).__name__}")
            hit, _score, _entry = shared.get(prompt)
            if hit is not None:
                return hit
            response = func(*args, **kwargs)
            shared.put(prompt, response)
            return response

        wrapper.cache = shared  # type: ignore[attr-defined]
        return wrapper

    return decorator(fn) if fn is not None else decorator
