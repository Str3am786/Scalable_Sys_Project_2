import argparse
from .config import load_config
from .llm.llama_inproc import LlamaInProc
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