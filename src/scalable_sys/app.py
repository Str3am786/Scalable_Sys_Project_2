import argparse

from src.scalable_sys.config import load_config
from src.scalable_sys.llm.llama_server import LlamaServer
from src.scalable_sys.llm.cached import CachedLLM
from src.scalable_sys.rag.graph_rag import GraphRAG


def _make_base_llm(cfg):
    """Plain LLM factory independent of RAG."""

    if cfg.backend in ("server", "rag"):
        # RAG reuses the same underlying server model
        return LlamaServer(
            base_url=cfg.server_base_url,
            api_key=cfg.server_api_key,
            model=cfg.server_model,
        )
    else:
        raise ValueError(f"Unknown backend: {cfg.backend}")


def get_llm(use_cache: bool = False):
    """
    Top-level LLM factory.
    Returns:
      - plain LLM  (server/inproc)  when backend != "rag"
      - GraphRAG (DSPy)            when backend == "rag"
    """
    cfg = load_config()
    base_llm = _make_base_llm(cfg)

    if cfg.backend == "rag":
        # Instantiate the DSPy-based GraphRAG pipeline
        llm = GraphRAG(
            llm=base_llm,
            db_path=cfg.rag_db_path,
            use_exemplars=cfg.rag_use_exemplars,
            use_self_refine=cfg.rag_use_self_refine,
            use_postprocess=cfg.rag_use_postprocess,
            cache_text2cypher=use_cache,
            cache_maxsize=cfg.rag_cache_maxsize,
            cache_ttl_seconds=cfg.rag_cache_ttl_seconds,
        )

    else:
        llm = base_llm

    if use_cache and cfg.backend != "rag":
        # Optionally cache whole-LLM completions (Task 2 extra)
        llm = CachedLLM(
            inner=llm,
            maxsize=256,
            ttl_seconds=0,
        )

    return llm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True, help="User question or prompt")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable outer LLM cache (Text2Cypher cache still controlled via config.yaml).",
    )
    args = parser.parse_args()

    # For benchmarking, toggle config.llm.backend between:
    #   - "server" (plain LLM baseline)
    #   - "rag"    (Graph-RAG backend)
    llm = get_llm(use_cache=not args.no_cache)

    print(llm.generate(args.prompt))


if __name__ == "__main__":
    main()