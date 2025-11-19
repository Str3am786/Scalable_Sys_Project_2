from functools import lru_cache, wraps
import time
from typing import Callable, Any, Tuple

def ttl_lru_cache(maxsize: int = 256, ttl_seconds: int = 0):
    """
    LRU with optional TTL. ttl_seconds=0 -> plain LRU.
    Cache key is whatever your function uses (be careful to make args hashable).
    """
    def deco(func: Callable):
        cached_func = lru_cache(maxsize=maxsize)(func)
        expiries = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = time.time()
            if ttl_seconds > 0:
                exp = expiries.get(key)
                if exp is None or now > exp:
                    # bust this specific key
                    try:
                        cached_func.cache_pop(*key[0], **dict(key[1]))
                    except Exception:
                        pass
                    expiries[key] = now + ttl_seconds
            return cached_func(*args, **kwargs)
        wrapper.cache_clear = cached_func.cache_clear  # expose for tests/admin
        return wrapper
    return deco
