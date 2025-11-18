import os, yaml
from dataclasses import dataclass
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent
@dataclass
class LLMConfig:
    backend: str
    model_path: str
    server_base_url: str
    server_api_key: str
    server_model: str
    n_ctx: int
    n_gpu_layers: int

def load_config(path: str = "config.yaml") -> LLMConfig:
    with open(BASE_PATH / path) as f:
        cfg = yaml.safe_load(f)["llm"]
    return LLMConfig(
        backend=cfg["backend"],
        model_path=cfg.get("model_path", ""),
        server_base_url=cfg.get("server", {}).get("base_url", "http://localhost:8000/v1"),
        server_api_key=cfg.get("server", {}).get("api_key", "sk-noauth"),
        server_model=cfg.get("server", {}).get("model", "local"),
        n_ctx=cfg.get("params", {}).get("n_ctx", 4096),
        n_gpu_layers=cfg.get("params", {}).get("n_gpu_layers", 0),
    )


# if __name__ == "__main__":
#     print()

