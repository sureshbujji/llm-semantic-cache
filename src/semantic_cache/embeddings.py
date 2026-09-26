"""Embedding providers.

The cache only needs fixed-length vectors with cosine similarity.
Ship a dependency-free default (HashingEmbedder) so the cache works
offline, and let users plug in any real embedding model.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import List, Protocol


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> List[float]: ...


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _char_ngrams(text: str, n: int = 3) -> List[str]:
    text = re.sub(r"\s+", " ", text.lower().strip())
    padded = f"  {text}  "
    return [padded[i:i + n] for i in range(len(padded) - n + 1)] or [text]


class HashingEmbedder:
    """Deterministic, dependency-free embedding via feature hashing.

    Good enough for near-duplicate prompt detection without API keys.
    Not a semantic embedding — swap in a real model for meaning-level matches.
    """

    def __init__(self, dim: int = 512, ngram: int = 3):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.ngram = ngram

    def _bucket(self, token: str) -> int:
        h = hashlib.md5(token.encode()).digest()
        return int.from_bytes(h[:4], "little") % self.dim

    def embed(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise ValueError("text must be non-empty")
        vec = [0.0] * self.dim
        for tok in _TOKEN_RE.findall(text.lower()):
            vec[self._bucket("w:" + tok)] += 1.0
        for ng in _char_ngrams(text, self.ngram):
            vec[self._bucket("c:" + ng)] += 0.5
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
