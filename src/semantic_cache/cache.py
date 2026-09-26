"""Thread-safe semantic cache with TTL, LRU eviction, stats, and JSON persistence."""
from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .embeddings import EmbeddingProvider, HashingEmbedder


@dataclass
class CacheEntry:
    prompt: str
    response: str
    vector: List[float]
    created_at: float = field(default_factory=time.time)
    last_hit_at: float = field(default_factory=time.time)
    hits: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


def _cosine(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # vectors are L2-normalized


class SemanticCache:
    """Cache LLM responses keyed by prompt similarity.

    get() returns (response, similarity, entry) on hit, else (None, best_score, None).
    """

    def __init__(
        self,
        embedder: Optional[EmbeddingProvider] = None,
        similarity_threshold: float = 0.97,
        ttl_seconds: Optional[float] = 3600,
        max_size: int = 1000,
    ):
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in (0, 1]")
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.embedder = embedder or HashingEmbedder()
        self.threshold = similarity_threshold
        self.ttl = ttl_seconds
        self.max_size = max_size
        self._entries: List[CacheEntry] = []
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._tokens_saved = 0

    # -- core API ---------------------------------------------------------
    def put(self, prompt: str, response: str,
            metadata: Optional[Dict[str, Any]] = None) -> CacheEntry:
        vector = self.embedder.embed(prompt)
        entry = CacheEntry(prompt=prompt, response=response, vector=vector,
                           metadata=metadata or {})
        with self._lock:
            self._evict_expired_locked()
            self._entries.append(entry)
            # LRU: drop least-recently-hit entries past max_size
            while len(self._entries) > self.max_size:
                oldest = min(self._entries, key=lambda e: e.last_hit_at)
                self._entries.remove(oldest)
        return entry

    def get(self, prompt: str) -> Tuple[Optional[str], float, Optional[CacheEntry]]:
        vector = self.embedder.embed(prompt)
        with self._lock:
            self._evict_expired_locked()
            best, best_score = None, -1.0
            for entry in self._entries:
                score = _cosine(vector, entry.vector)
                if score > best_score:
                    best, best_score = entry, score
            if best is not None and best_score >= self.threshold:
                now = time.time()
                best.hits += 1
                best.last_hit_at = now
                self._hits += 1
                self._tokens_saved += _approx_tokens(best.prompt) + _approx_tokens(best.response)
                return best.response, best_score, best
            self._misses += 1
            return None, best_score, None

    def clear(self) -> int:
        with self._lock:
            n = len(self._entries)
            self._entries.clear()
            return n

    # -- observability ----------------------------------------------------
    @property
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "entries": len(self._entries),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": (self._hits / total) if total else 0.0,
                "approx_tokens_saved": self._tokens_saved,
            }

    # -- persistence ------------------------------------------------------
    def save(self, path: str) -> None:
        with self._lock:
            payload = {
                "threshold": self.threshold,
                "ttl": self.ttl,
                "max_size": self.max_size,
                "entries": [
                    {"prompt": e.prompt, "response": e.response,
                     "vector": e.vector, "created_at": e.created_at,
                     "last_hit_at": e.last_hit_at, "hits": e.hits,
                     "metadata": e.metadata}
                    for e in self._entries
                ],
            }
        with open(path, "w") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str,
             embedder: Optional[EmbeddingProvider] = None) -> "SemanticCache":
        with open(path) as f:
            payload = json.load(f)
        cache = cls(embedder=embedder,
                    similarity_threshold=payload.get("threshold", 0.97),
                    ttl_seconds=payload.get("ttl"),
                    max_size=payload.get("max_size", 1000))
        for raw in payload.get("entries", []):
            cache._entries.append(CacheEntry(**raw))
        return cache

    # -- internals --------------------------------------------------------
    def _evict_expired_locked(self) -> None:
        if self.ttl is None:
            return
        cutoff = time.time() - self.ttl
        self._entries = [e for e in self._entries if e.created_at >= cutoff]


def _approx_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))
