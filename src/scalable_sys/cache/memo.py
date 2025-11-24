import time
import functools
from collections import OrderedDict
import logging
from datetime import datetime
import os

os.makedirs("./results", exist_ok=True)

logging.basicConfig(
    filename="./results/cache_log.txt",
    level=logging.INFO,
    format="[%(asctime)s] - %(message)s",
)

def write(value: str):
    logging.info(value)



def ttl_lru_cache(maxsize: int = 128, ttl_seconds: int = 0, test: bool = False):
    """
    LRU Cache decorator with Time-to-Live (TTL) support.
    If ttl_seconds <= 0, entries never expire (standard LRU).
    """
    def decorator(func):
        # The cache stores keys mapped to (result, timestamp)
        cache = OrderedDict()
                    

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a hashable key from args and kwargs
            # Note: In your usage, you are passing a single string key, which is safe.
            key = (args, tuple(sorted(kwargs.items())))

            # 1. Check if key exists
            if key in cache:
                result, timestamp = cache[key]
                write(f"HIT key: {key}")

                # 2. Check TTL (if enabled)
                if ttl_seconds > 0 and (time.time() - timestamp) > ttl_seconds:
                    # Expired: remove and fall through to re-compute
                    write(f"Expired key: {key}")
                    del cache[key]
                else:
                    # Hit: Move to end (Mark as recently used)
                    cache.move_to_end(key)
                    return result

            # 3. Compute result
            result = func(*args, **kwargs)

            # 4. Store in cache
            cache[key] = (result, time.time())
            write(f"Store Key {key} in cache")

            # 5. Enforce Size Limit (Pruning)
            if len(cache) > maxsize:
                removed = cache.popitem(last=False)  # FIFO removal (removes the first/oldest item)
                write(f"Max space reached -> Removed {removed}")

            return result

        def clear_cache():
            cache.clear()

        wrapper.cache_clear = clear_cache
        return wrapper

    return decorator