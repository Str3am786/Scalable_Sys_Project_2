import json
from .base import LLM
from ..cache.memo import ttl_lru_cache

def _norm_prompt_key(prompt: str) -> str:
    # normalize whitespace to improve hit rate; adjust to your data
    return " ".join(prompt.split())

class CachedLLM(LLM):
    def __init__(self, inner: LLM, maxsize: int = 256, ttl_seconds: int = 0):
        self.inner = inner
        self._gen = self._make_cached_generate(maxsize, ttl_seconds)

    def _make_cached_generate(self, maxsize: int, ttl_seconds: int):
        @ttl_lru_cache(maxsize=maxsize, ttl_seconds=ttl_seconds)
        def _cached_call(key: str) -> str:
            payload = json.loads(key)
            return self.inner.generate(
                prompt=payload["prompt"],
                max_tokens=payload["max_tokens"],
                temperature=payload["temperature"],
            )
        return _cached_call

    def generate(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0) -> str:
        # include parameters that affect determinism in the key
        key = json.dumps({
            "backend": type(self.inner).__name__,
            "model": getattr(self.inner, "model", None),      # llama-server model name
            "model_path": getattr(getattr(self.inner, "llm", None), "model_path", None),
            "prompt": _norm_prompt_key(prompt),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }, sort_keys=True)
        return self._gen(key)

    def stream(self, *args, **kwargs):
        # usually skip caching for streams
        yield from self.inner.stream(*args, **kwargs)
