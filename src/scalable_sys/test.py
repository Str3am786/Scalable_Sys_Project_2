import argparse

from src.scalable_sys.config import load_config
from src.scalable_sys.llm.llama_server import LlamaServer
from src.scalable_sys.llm.cached import CachedLLM
from src.scalable_sys.rag.graph_rag import GraphRAG

from datetime import datetime


import json
from typing import List, Dict, Any
import os

TEST = {
    "0": { "rag" : False,"cache": False},
    "1": { "rag" : False,"cache": True},
    "2": { "rag" : True,"cache": False},
    "3": { "rag" : True,"cache": True},
}

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

    if use_cache and rag!=True:
        # Optionally cache whole-LLM completions
        llm = CachedLLM(
            inner=llm,
            maxsize=256,
            ttl_seconds=0,
        )

    return llm, cfg.rag_cache_maxsize, cfg.rag_cache_ttl_seconds


def test(rag: bool, cache: bool,input_file : str= "./data/test_input.json", out_folder : str = "results") -> str:
    
    test_set = load_test_set(input_file)
    test_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    llm = get_llm(use_cache=cache, rag = rag)
    
    filepath = f"./{out_folder}/{test_datetime}_rag_{rag}_with_cache_{cache}.json"
    counter = 1
    
    for test in test_set:
        
        print(f"Starting test for question n° {counter}")
        counter += 1
        prompt = test["question"]
        if rag:
            answer, stats, c , all_tested_cyphers = llm.generate(prompt)
            result = {
                "question": prompt,
                "cypher": c,
                "all_cyphers" : all_tested_cyphers,
                "answer": answer,
                "stats": stats
            }
        else:
            start = datetime.now()
            answer = llm.generate(prompt)
            end = datetime.now()
            delta = (end - start).total_seconds()  
            
            result = {
                "question": prompt,
                "answer": answer,
                "time": round(delta,2),
            }
            
        record_test(filepath,result)
        
    return filepath


def cache_test(input_file : str= "./data/test_input.json", out_folder : str = "results") -> str:
    
    test_set = load_test_set(input_file)
    test_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    llm ,cache_maxsize,cache_ttl_seconds= get_llm(use_cache=True, rag = True)
    
    print(f"CACHE PARAMETERS: Size = {cache_maxsize}, Time to Live = {cache_ttl_seconds}")
    
    filepath = f"./{out_folder}/{test_datetime}_cache_test.json"
    counter = 1
    
    for test in test_set:
        
        print(f"Starting test for question n° {counter}")
        counter += 1
        prompt = test["question"]
        answer, stats, c , all_tested_cyphers = llm.generate(prompt)
        result = {
            "question": prompt,
            "answer": answer,
            "stats": stats
        }            
        record_test(filepath,result)
                
    return filepath

def evaluate_test(filepath : str) -> str:
    
    evaluation_model, evaluation_script = get_evaluation_model()

    # read test results
    with open(filepath, "r", encoding="utf-8") as f:
        data_test = json.load(f)["results"]
    # read ground truth data    
    with open("data/test_input.json", "r", encoding="utf-8") as f:
        data_true = json.load(f)["tests"]
        
    output_filepath =  open("./results/accuracy.txt","a")
        
    for i in range(len(data_true)):
        print(f"Evaluating response n° {i}")
        question =  data_true[i]["question"]
        true_answer = data_true[i]["expected"]
        test_answer = data_test[i]["answer"]
                                
        evaluation_input_prompt = (
            evaluation_script
            + "\nQUESTION:\n" + json.dumps(question)
            + "\nMODEL_ANSWER:\n" + json.dumps(test_answer)
            + "\nGROUND_TRUTH:\n" + json.dumps(true_answer)
        )
        # Generate evaluation (assuming evaluation_model is already defined)
        evaluation_results = evaluation_model.generate(evaluation_input_prompt)
        output_filepath.write(evaluation_input_prompt+"\n"+evaluation_results+"\n\n\n",)
    
    return "./results/accuracy.txt"

def main():

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
        "--single-test",
        type=str,
        choices=["0", "1", "2", "3"],
        help="Select which test to run"
    )
    
    parser.add_argument(
        "--caching-test",
        action="store_true",
        help="Disable outer LLM cache (Text2Cypher cache still controlled via config.yaml).",
    )
    args = parser.parse_args()
    use_cache = not args.no_cache
    
    if args.complete_test:
        
        test_filpaths = [] 
        print("------------Starting Complete Test---------------------")
        for i in range(len(TEST)):
            # Test all possible combination
            print(f"Test {i}: RAG = {TEST[str(i)]["rag"]}, CACHE = {TEST[str(i)]["cache"]}")
            test_filepath = test(**TEST[str(i)])
            test_filpaths.append(test_filepath)
            print(f"Conlcluded Test {i}. You can find the result in {test_filepath}") 
        
        print("------------Starting Evaluation of Tests Answers---------------------")
        evaluation_res_plain = evaluate_test(test_filpaths[0])
        print(f"Conlclude Evaluation of Test without RAG. You can find the result in {evaluation_res_plain}") 

        evaluation_res_rag = evaluate_test(test_filepath[1])
        print(f"Conlclude Evaluation of Test with RAG. You can find the result in {evaluation_res_rag}") 

    elif args.single_test:
        
        print("------------Starting Single Test---------------------")
        print(f"Test {args.single_test}: RAG = {TEST[args.single_test]["rag"]}, CACHE = {TEST[args.single_test]["cache"]}")
        test_filepath = test(**TEST[str(args.single_test)])
        print(f"Conlcluded Test {args.single_test}. You can find the result in {test_filepath}") 
        
        print("------------Starting Evaluation of Tests Answers---------------------")
        evaluation_res_test = evaluate_test(test_filepath)
        print(f"Conlcluded Evaluation of Test {args.single_test}. You can find the result in {evaluation_res_test}") 
        
        
    elif args.caching_test: 
        
        print("------------Starting Cache Test---------------------")
        test_filepath = cache_test(input_file="./data/test_cache.json")
        print(f"Conlcluded Test. You can find the result in {test_filepath}") 
        
        
    else:

        llm = get_llm(use_cache=use_cache, rag = args.rag)        
        print(llm.generate(args.prompt))      

    # For benchmarking, toggle config.llm.backend between:
    #   - "server" (plain LLM baseline)
    #   - "rag"    (Graph-RAG backend)



if __name__ == "__main__":
    main()