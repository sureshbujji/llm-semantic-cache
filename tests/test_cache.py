import math
import os
import tempfile
import threading
import time

import pytest

from semantic_cache import HashingEmbedder, SemanticCache, semantic_cached
from semantic_cache.cache import _approx_tokens, _cosine


@pytest.fixture
def cache():
    return SemanticCache(similarity_threshold=0.97, ttl_seconds=60, max_size=100)


# --- embedder ---------------------------------------------------------------
def test_embedder_deterministic():
    e = HashingEmbedder()
    assert e.embed("hello world") == e.embed("hello world")


def test_embedder_normalized():
    vec = HashingEmbedder(dim=128).embed("some prompt")
    assert len(vec) == 128
    assert math.isclose(sum(v * v for v in vec), 1.0, rel_tol=1e-6)


def test_embedder_similar_texts_score_high():
    e = HashingEmbedder()
    a = e.embed("What is the capital of France?")
    b = e.embed("What is the capital of France? ")
    assert _cosine(a, b) > 0.99


def test_embedder_different_texts_score_lower():
    e = HashingEmbedder()
    a = e.embed("What is the capital of France?")
    b = e.embed("Explain quantum entanglement in detail")
    assert _cosine(a, b) < _cosine(a, e.embed("What is the capital of France?"))


def test_embedder_rejects_empty():
    with pytest.raises(ValueError):
        HashingEmbedder().embed("   ")


def test_embedder_rejects_bad_dim():
    with pytest.raises(ValueError):
        HashingEmbedder(dim=0)


# --- basic put/get -----------------------------------------------------------
def test_exact_prompt_hits(cache):
    cache.put("hello", "hi there")
    resp, score, entry = cache.get("hello")
    assert resp == "hi there"
    assert score >= 0.97
    assert entry.hits == 1


def test_unseen_prompt_misses(cache):
    cache.put("hello", "hi there")
    resp, score, entry = cache.get("something completely different xyz")
    assert resp is None
    assert entry is None


def test_whitespace_variant_hits(cache):
    cache.put("What is 2+2?", "4")
    resp, _score, _e = cache.get("What is 2+2? ")
    assert resp == "4"


def test_miss_returns_best_score_below_threshold(cache):
    cache.put("aaa bbb", "r1")
    resp, score, _e = cache.get("zzz yyy qqq")
    assert resp is None
    assert score < 0.97


def test_empty_cache_miss(cache):
    resp, score, entry = cache.get("anything")
    assert resp is None and entry is None
    assert score == -1.0


def test_put_returns_entry_with_metadata(cache):
    entry = cache.put("p", "r", metadata={"model": "x"})
    assert entry.metadata == {"model": "x"}
    assert entry.prompt == "p"


def test_invalid_threshold_rejected():
    with pytest.raises(ValueError):
        SemanticCache(similarity_threshold=1.5)
    with pytest.raises(ValueError):
        SemanticCache(similarity_threshold=0.0)


def test_invalid_max_size_rejected():
    with pytest.raises(ValueError):
        SemanticCache(max_size=0)


# --- TTL ---------------------------------------------------------------------
def test_expired_entries_are_invisible():
    c = SemanticCache(ttl_seconds=0.05)
    c.put("hello", "hi")
    time.sleep(0.08)
    resp, _s, _e = c.get("hello")
    assert resp is None


def test_no_ttl_means_never_expires():
    c = SemanticCache(ttl_seconds=None)
    c.put("hello", "hi")
    assert c.get("hello")[0] == "hi"


# --- eviction -----------------------------------------------------------------
def test_max_size_evicts_lru():
    c = SemanticCache(max_size=2, similarity_threshold=0.999)
    c.put("prompt alpha one", "r1")
    c.put("prompt beta two", "r2")
    c.get("prompt alpha one")          # refresh r1
    c.put("prompt gamma three", "r3")  # evicts r2 (least recently hit)
    assert c.get("prompt alpha one")[0] == "r1"
    assert c.get("prompt gamma three")[0] == "r3"
    assert c.get("prompt beta two")[0] is None


def test_clear_empties_cache(cache):
    cache.put("a", "b")
    assert cache.clear() == 1
    assert cache.get("a")[0] is None


# --- stats --------------------------------------------------------------------
def test_stats_hit_rate_and_tokens(cache):
    cache.put("hello world", "hi there friend")
    cache.get("hello world")   # hit
    cache.get("nope nope")     # miss
    s = cache.stats
    assert s["hits"] == 1
    assert s["misses"] == 1
    assert s["hit_rate"] == pytest.approx(0.5)
    assert s["approx_tokens_saved"] == _approx_tokens("hello world") + _approx_tokens("hi there friend")
    assert s["entries"] == 1


def test_stats_zero_division_safe():
    assert SemanticCache().stats["hit_rate"] == 0.0


# --- persistence ----------------------------------------------------------------
def test_save_and_load_roundtrip(cache):
    cache.put("hello", "hi", metadata={"m": 1})
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
        path = f.name
    try:
        cache.save(path)
        loaded = SemanticCache.load(path)
        resp, score, entry = loaded.get("hello")
        assert resp == "hi"
        assert entry.metadata == {"m": 1}
        assert loaded.threshold == cache.threshold
    finally:
        os.unlink(path)


# --- thread safety ---------------------------------------------------------------
def test_concurrent_put_get_no_crash():
    c = SemanticCache(max_size=50)
    def worker(i):
        for j in range(20):
            c.put(f"prompt {i} {j}", f"resp {j}")
            c.get(f"prompt {i} {j}")
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert c.stats["entries"] <= 50


# --- decorator ---------------------------------------------------------------------
def test_decorator_caches_second_call():
    calls = []
    cache = SemanticCache()

    @semantic_cached(cache=cache)
    def ask(prompt: str) -> str:
        calls.append(prompt)
        return f"answer:{prompt}"

    assert ask("hello") == "answer:hello"
    assert ask("hello") == "answer:hello"
    assert len(calls) == 1


def test_decorator_miss_calls_function_each_time():
    calls = []

    @semantic_cached(similarity_threshold=0.9999)
    def ask(prompt: str) -> str:
        calls.append(prompt)
        return "x"

    ask("alpha")
    ask("beta completely different")
    assert len(calls) == 2


def test_decorator_reads_prompt_kwarg():
    seen = []

    @semantic_cached()
    def ask(prompt: str = "", other: int = 0) -> str:
        seen.append(prompt)
        return "ok"

    assert ask(prompt="hi", other=1) == "ok"
    assert ask(prompt="hi", other=2) == "ok"  # cache hit despite other kwarg
    assert len(seen) == 1


def test_decorator_rejects_non_string_prompt():
    @semantic_cached()
    def ask(prompt):
        return "x"

    with pytest.raises(TypeError):
        ask(123)


def test_decorator_exposes_cache():
    @semantic_cached()
    def ask(prompt: str) -> str:
        return "x"

    ask("hello")
    assert isinstance(ask.cache, SemanticCache)
    assert ask.cache.stats["hits"] == 0
    ask("hello")
    assert ask.cache.stats["hits"] == 1


def test_decorator_preserves_function_name():
    @semantic_cached()
    def ask(prompt: str) -> str:
        """docstring"""
        return "x"

    assert ask.__name__ == "ask"


def test_custom_embedder_pluggable():
    class Constant:
        def embed(self, text):
            return [1.0]

    c = SemanticCache(embedder=Constant(), similarity_threshold=0.5)
    c.put("a", "b")
    assert c.get("zzz")[0] == "b"  # constant vectors always match
