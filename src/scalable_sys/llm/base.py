from abc import ABC, abstractmethod
from typing import Iterable, Optional

class LLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0) -> str:
        ...

    def stream(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0) -> Iterable[str]:
        # default non-streaming fallback
        yield self.generate(prompt, max_tokens=max_tokens, temperature=temperature)
