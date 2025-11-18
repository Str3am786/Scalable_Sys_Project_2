import re, json
from ..cache.memo import ttl_lru_cache

def _norm_query(q: str) -> str:
    q = re.sub(r"\s+", " ", q.strip().lower())
    return q

@ttl_lru_cache(maxsize=1024, ttl_seconds=3600)
def retrieve_ids(index_version: str, query: str, k: int = 20) -> tuple[str, ...]:
    """
    Pure function wrapper around your actual vector/graph lookups.
    Include index_version so cache invalidates when you rebuild embeddings/graph.
    """
    qn = _norm_query(query)
    # ... perform expensive search here ...
    ids: list[str] = expensive_search(index_version, qn, k)
    return tuple(ids)
