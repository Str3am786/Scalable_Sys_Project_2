from typing import Iterable
from llama_cpp import Llama
from .base import LLM

class LlamaInProc(LLM):
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,   # set >0 if you have GPU support
        verbose: bool = False
    ):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )

    def generate(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0) -> str:
        out = self.llm(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=[]
        )
        return out["choices"][0]["text"]

    def stream(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0) -> Iterable[str]:
        for token in self.llm(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        ):
            yield token["choices"][0]["text"]
