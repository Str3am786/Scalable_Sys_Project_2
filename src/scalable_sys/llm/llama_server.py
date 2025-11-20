from typing import Iterable
from openai import OpenAI   # <-- this is the client you want
from .base import LLM

class LlamaServer(LLM):
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "sk-noauth",
        model: str = "local",          # or whatever alias you used
    ):
        # llama.cpp server exposes an OpenAI-compatible API
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model = model

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return r.choices[0].message.content

    def stream(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Iterable[str]:
        # streaming style for openai>=1.0
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
