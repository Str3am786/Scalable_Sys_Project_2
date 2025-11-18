import argparse
from src.scalable_sys.config import load_config
from src.scalable_sys.llm.llama_inproc import LlamaInProc
from src.scalable_sys.llm.llama_server import LlamaServer
from src.scalable_sys.rag.graph_rag import GraphRAG
from .llm.llama_server import LlamaServer



# llm = get_llm()  # inproc or server
# if cfg.cache.enable_llm:
#     llm = CachedLLM(llm, maxsize=cfg.cache.llm_maxsize, ttl_seconds=cfg.cache.llm_ttl)
def get_llm():
    c = load_config()
    if c.backend == "inproc":
        return LlamaInProc(model_path=c.model_path, n_ctx=c.n_ctx, n_gpu_layers=c.n_gpu_layers)
    elif c.backend == "server":
        return LlamaServer(base_url=c.server_base_url, api_key=c.server_api_key, model=c.server_model)
    elif c.backend == "rag":
        llm = LlamaServer(base_url=c.server_base_url, api_key=c.server_api_key, model=c.server_model)
        #TODO
        return GraphRAG(llm=llm, db_path="")
    else:
        raise ValueError(f"Unknown backend: {c.backend}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()

    llm = get_llm()
    print(llm.generate(args.prompt))

if __name__ == "__main__":
    main()