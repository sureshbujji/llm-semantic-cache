# llm-semantic-cache

A tiny, dependency-free **semantic cache for LLM calls**: skip the API (and the
bill) when a similar prompt was already answered.

```python
from semantic_cache import SemanticCache, semantic_cached

cache = SemanticCache(similarity_threshold=0.97, ttl_seconds=3600, max_size=1000)

@semantic_cached(cache=cache)
def ask_llm(prompt: str) -> str:
    return my_llm_client.complete(prompt)   # only called on cache miss

ask_llm("What is the capital of France?")   # miss -> calls the model
ask_llm("What is the capital of France? ")  # hit  -> instant, $0
print(cache.stats)
# {'entries': 1, 'hits': 1, 'misses': 1, 'hit_rate': 0.5, 'approx_tokens_saved': 14}
```

## Why

LLM apps repeat themselves: retries, paraphrased user questions, dev loops.
A semantic cache cuts latency, cost, and rate-limit pressure with one decorator.

## Features

- **Similarity matching** — cosine similarity over embeddings, configurable threshold
- **Pluggable embeddings** — ships with a dependency-free `HashingEmbedder`
  (deterministic, offline); bring your own model via the `EmbeddingProvider` protocol
- **TTL + LRU eviction** — stale entries expire, hot entries stay
- **Stats** — hits, misses, hit rate, approximate tokens saved
- **Persistence** — `save()` / `load()` the cache as JSON
- **Thread-safe** — safe to share across request handlers

## Install

Zero runtime dependencies:

```bash
pip install -e .
```

## Run the tests

```bash
pytest -q
```

## Notes

`HashingEmbedder` is a feature-hashed character/word n-gram vector — great for
near-duplicate detection with no API key. For meaning-level matches (paraphrases),
plug in a real embedding model.
