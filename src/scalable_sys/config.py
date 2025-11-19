# src/scalable_sys/config.py
import os, yaml
from dataclasses import dataclass
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent           # .../src/scalable_sys
PROJECT_ROOT = BASE_PATH.parent.parent                # .../Scalable_Sys_Project_2

@dataclass
class LLMConfig:
    backend: str
    model_path: str
    server_base_url: str
    server_api_key: str
    server_model: str
    n_ctx: int
    n_gpu_layers: int
    # RAG-specific config
    rag_db_path: str = "data/nobel.kuzu"
    rag_use_exemplars: bool = True
    rag_use_self_refine: bool = True
    rag_use_postprocess: bool = True
    rag_cache_text2cypher: bool = True
    rag_cache_maxsize: int = 256
    rag_cache_ttl_seconds: int = 0

def load_config(path: str = "config.yaml") -> LLMConfig:
    with open(BASE_PATH / path) as f:
        cfg = yaml.safe_load(f)["llm"]

    rag_cfg = cfg.get("rag", {})

    # Resolve DB path relative to PROJECT_ROOT by default
    raw_rag_db_path = rag_cfg.get("db_path", "data/nobel.kuzu")
    rag_db_path = Path(raw_rag_db_path)
    if not rag_db_path.is_absolute():
        rag_db_path = PROJECT_ROOT / rag_db_path
    rag_db_path_str = str(rag_db_path)

    server_base_url = os.getenv("LLM_BASE_URL", cfg.get("server", {}).get("base_url", "http://localhost:8000/v1"))
    server_api_key = os.getenv("LLM_API_KEY", cfg.get("server", {}).get("api_key", "sk-noauth"))

    return LLMConfig(
        backend=cfg["backend"],
        model_path=cfg.get("model_path", ""),
        server_base_url=server_base_url,
        server_api_key=server_api_key,
        server_model=cfg.get("server", {}).get("model", "local"),
        n_ctx=cfg.get("params", {}).get("n_ctx", 4096),
        n_gpu_layers=cfg.get("params", {}).get("n_gpu_layers", 0),
        rag_db_path=rag_db_path_str,
        rag_use_exemplars=rag_cfg.get("use_exemplars", True),
        rag_use_self_refine=rag_cfg.get("use_self_refine", True),
        rag_use_postprocess=rag_cfg.get("use_postprocess", True),
        rag_cache_text2cypher=rag_cfg.get("cache_text2cypher", True),
        rag_cache_maxsize=rag_cfg.get("cache_maxsize", 256),
        rag_cache_ttl_seconds=rag_cfg.get("cache_ttl_seconds", 0),
    )
