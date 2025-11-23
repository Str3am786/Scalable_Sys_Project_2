import os
import yaml
from dataclasses import dataclass
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_PATH.parent.parent

@dataclass
class LLMConfig:
    # Infrastructure from .env
    server_base_url: str
    server_api_key: str
    server_model: str
    
    # Behavior from config.yaml
    backend: str
    rag_db_path: str
    rag_use_exemplars: bool
    rag_use_self_refine: bool
    rag_use_postprocess: bool
    rag_cache_text2cypher: bool
    rag_cache_maxsize: int
    rag_cache_ttl_seconds: int
    
    # Evaluation / judge model
    evaluation_model : str
    evaluation_url : str
    evaluation_key : str
    evaluation_prompt : str


def load_config(yaml_path: str = "config.yaml") -> LLMConfig:
    # Load Behavioral Config from YAML
    with open(BASE_PATH / yaml_path) as f:
        file_cfg = yaml.safe_load(f)["llm"]

    rag_cfg = file_cfg.get("rag", {})
    evaluation = file_cfg.get("evaluation", {})

    # Base LLM from .env
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    base_key = os.getenv("LLM_API_KEY", "ollama")
    base_model = os.getenv("LLM_MODEL", "llama3.1")

    # Resolve DB path relative to project root
    raw_db_path = rag_cfg.get("db_path", "data/nobel.kuzu")
    db_path = Path(raw_db_path)
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path

    # Load Infrastructure from Environment
    return LLMConfig(
        server_base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
        server_api_key=os.getenv("LLM_API_KEY", "ollama"),
        server_model=os.getenv("LLM_MODEL", "llama3.1"),

        backend=file_cfg.get("backend", "rag"),
        rag_db_path=str(db_path),
        rag_use_exemplars=rag_cfg.get("use_exemplars", True),
        rag_use_self_refine=rag_cfg.get("use_self_refine", True),
        rag_use_postprocess=rag_cfg.get("use_postprocess", True),
        rag_cache_text2cypher=rag_cfg.get("cache_text2cypher", True),
        rag_cache_maxsize=rag_cfg.get("cache_maxsize", 256),
        rag_cache_ttl_seconds=rag_cfg.get("cache_ttl_seconds", 0),
        
        # Use different judge llm for evaluation if set, fallback to base llm
        evaluation_model=os.getenv("EVAL_LLM_MODEL") or os.getenv("LLM_MODEL", "llama3.1"),
        evaluation_url=os.getenv("EVAL_LLM_BASE_URL") or os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
        evaluation_key=os.getenv("EVAL_LLM_API_KEY") or os.getenv("LLM_API_KEY", "ollama"),

        # Get LLM-as-a-judge prompt from yaml
        evaluation_prompt = evaluation.get("prompt", "")
    )