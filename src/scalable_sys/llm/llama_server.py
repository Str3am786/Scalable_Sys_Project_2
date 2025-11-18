import os
from typing import Iterable
from .base import LLM

class LlamaServer(LLM):
    def __init__(self, base_url: str = "http://localhost:8000/v1", api_key: str = "sk-noauth", model: str = "local"):
        # llama.cpp server uses an OpenAI-like API; any non-empty key usually works
        self.client = None
        self.model = model

    def generate(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0) -> str:
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return r.choices[0].message.content

    def stream(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0) -> Iterable[str]:
        with self.client.chat.completions.stream(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        ) as s:
            for event in s:
                if event.type == "token":
                    yield event.token
