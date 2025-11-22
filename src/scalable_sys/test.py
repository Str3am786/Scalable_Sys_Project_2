import argparse

from src.scalable_sys.config import load_config
from src.scalable_sys.llm.llama_server import LlamaServer
from src.scalable_sys.llm.cached import CachedLLM
from src.scalable_sys.rag.graph_rag import GraphRAG

from datetime import datetime


import json
from typing import List, Dict, Any
import os
from .rag.prompts import EVALUATION_PROMPT


def make_evaluation_model(cfg):
    """Evaluation model and prompt."""
    


    return LlamaServer(
        base_url=cfg.evaluation_url,
        api_key=cfg.evaluation_key,
        model=cfg.evaluation_model,
    ), cfg.evaluation_prompt
    

def get_evaluation_model():
    
    cfg = load_config()
    evaluation_llm = make_evaluation_model(cfg)
    return evaluation_llm

def record_test(filename : str, result)-> None:

    filename = f"./data/{filename}"
    if not os.path.exists(filename):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({"results": [result]}, f, indent=2)
        return

    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["results"].append(result)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        

def load_test_set(path: str = "./data/test_input.json") -> List[Dict[str, Any]]:

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "tests" not in data or not isinstance(data["tests"], list):
        raise ValueError("JSON must contain a top-level 'tests' array")

    cleaned_tests = []
    
    for i, entry in enumerate(data["tests"]):
        if not all(k in entry for k in ("question", "cypher", "expected")):
            raise ValueError(
                f"Entry {i} is missing required fields (question, cypher, expected)"
            )

        if not isinstance(entry["expected"], list):
            raise ValueError(
                f"'expected' must be a list for entry {i}: {entry['expected']}"
            )

        cleaned_tests.append({
            "question": entry["question"],
            "cypher": entry["cypher"],
            "expected": entry["expected"]
        })

    return cleaned_tests



def _make_prediction_llm(cfg):
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


def get_llm(use_cache: bool = False, rag: bool = False):
    """
    Top-level LLM factory.
    Returns:
      - plain LLM  (server/inproc)  when backend != "rag"
      - GraphRAG (DSPy)            when backend == "rag"
    """
    cfg = load_config()
    base_llm = _make_prediction_llm(cfg)
    
    if rag:
        # Instantiate the DSPy-based GraphRAG pipeline
        print("GRAPH")
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
        print("BASE")
        llm = base_llm

    if use_cache and rag!=True:
        # Optionally cache whole-LLM completions
        print("BASEEE")
        llm = CachedLLM(
            inner=llm,
            maxsize=256,
            ttl_seconds=0,
        )

    return llm


def main():
    print("START")

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True, help="User question or prompt")
    
    parser.add_argument(
        "--rag",
        action= "store_true",
        help="Rag pipeline or single model"
    )
    
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable outer LLM cache (Text2Cypher cache still controlled via config.yaml).",
    )
    
    parser.add_argument(
        "--complete-test",
        action= "store_true",
        help="Launch Complete Project Test"
    )
    
    parser.add_argument(
        "--test",
        action= "store_true",
        help="Launch Single Test"
    )

    args = parser.parse_args()
    
    
    use_cache = not args.no_cache
    print("RAG: ",args.rag)
    
    if args.complete_test:
        print("TEST COMPLETO")
        
        test_set = load_test_set()
        test_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        evaluation_model, evaluation_script = get_evaluation_model()
        compare = ["data/2025-11-22_17-08-53_rag_True_cache_True.json","data/2025-11-22_17-08-53_rag_False_cache_True.json"] 
        compare = []
    
        for i in range(2):
            for j in range(2):
                rag = bool(i)
                use_cache = bool(j)
                llm = get_llm(use_cache=use_cache, rag = rag)
                filename = f"{test_datetime}_rag_{rag}_cache_{use_cache}.json"
                if j == 1:
                    compare.append(f"data/{filename}")
                
                for test in test_set:
                    prompt = test["question"]
                    cypher = test["cypher"]
                    true = test["expected"]
                    if rag:
                        answer, stats, c  = llm.generate(prompt)
                    
                        result = {
                            "question": prompt,
                            "cypher": cypher,
                            "answer": answer,
                            "stats": stats
                        }
                    else:
                        start = datetime.now()
                        answer = llm.generate(prompt)
                        end = datetime.now()
                        delta = end - start
                                        
                        result = {
                            "question": prompt,
                            "answer": answer,
                            "time": str(delta),
                        }
                    record_test(filename,result)
            
        with open(compare[0], "r", encoding="utf-8") as f:
            data_plain = json.load(f)["results"]
        
        with open(compare[1], "r", encoding="utf-8") as f:
            data_rag = json.load(f)["results"]
            print(data_rag)
                
        with open("data/test_input.json", "r", encoding="utf-8") as f:
            data_true = json.load(f)["tests"]
            
        
        for i in range(len(data_true["tests"])):
            true_answer = data_true[i]
            rag_answer = data_rag[i]
            plain_answer = data_plain[i] 
            
            evaluation_input = (
                evaluation_script
                + "\nRAG_ANSWER:\n" + json.dumps(rag_answer)
                + "\nPLAIN_MODEL_ANSWER:\n" + json.dumps(plain_answer)
                + "\nGROUND_TRUTH:\n" + json.dumps(true_answer)
            )
            
            print(evaluation_input)
            
            # Generate evaluation (assuming evaluation_model is already defined)
            evaluation_result = evaluation_model.generate(evaluation_input)
            print("Evaluation Result:\n", evaluation_result)
            
    elif args.test:
        # Just one test of the 4 -- Maybe we can drop it
        print("Partial")
        
        #llm = get_llm(use_cache=use_cache, rag = args.rag)
        print(llm.generate(args.prompt))      
    
    else:

        print(f"Cache: {use_cache}")
        llm = get_llm(use_cache=use_cache, rag = args.rag)
        start = datetime.now()
        
        print(llm.generate(args.prompt))      
        
        end = datetime.now()
        delta = end - start
        print(f"time: {delta} seconds")

        
    # For benchmarking, toggle config.llm.backend between:
    #   - "server" (plain LLM baseline)
    #   - "rag"    (Graph-RAG backend)



if __name__ == "__main__":
    main()