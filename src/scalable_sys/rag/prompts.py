import json
from ..cache.memo import ttl_lru_cache

@ttl_lru_cache(maxsize=1024)
def render(template_id: str, variables: dict) -> str:
    # load template by id (or pass the template text directly)
    # produce a final string
    return compile_and_fill(template_id, variables)
