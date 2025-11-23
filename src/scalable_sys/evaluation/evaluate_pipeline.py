import argparse
import re
import json
import os
from datetime import datetime
from typing import List, Dict, Any

from src.scalable_sys.config import load_config
from src.scalable_sys.llm.llama_server import LlamaServer
from src.scalable_sys.llm.cached import CachedLLM
from src.scalable_sys.rag.graph_rag import GraphRAG

from src.scalable_sys.evaluation.analyze_results import generate_comparison_table

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

def record_test(filename : str, result) -> None:
    os.makedirs("./data", exist_ok=True)
    
    filepath = f"./data/{filename}"
    if not os.path.exists(filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"results": [result]}, f, indent=2)
        return

    with open(filepath, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = {"results": []}

    data["results"].append(result)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_test_set(path: str = "./data/test_input.json") -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Test set not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "tests" not in data or not isinstance(data["tests"], list):
        raise ValueError("JSON must contain a top-level 'tests' array")

    cleaned_tests = []
    for i, entry in enumerate(data["tests"]):
        # Basic validation
        if "question" not in entry:
            continue 
            
        cleaned_tests.append({
            "question": entry["question"],
            "cypher": entry.get("cypher", ""),
            "expected": entry.get("expected", [])
        })

    return cleaned_tests

def _make_prediction_llm(cfg):
    """Plain LLM factory independent of RAG."""
    if cfg.backend in ("server", "rag"):
        return LlamaServer(
            base_url=cfg.server_base_url,
            api_key=cfg.server_api_key,
            model=cfg.server_model,
        )
    else:
        raise ValueError(f"Unknown backend: {cfg.backend}")

def get_llm(use_cache: bool = False, rag: bool = False):
    """Top-level LLM factory."""
    cfg = load_config()
    base_llm = _make_prediction_llm(cfg)
    
    if rag:
        print(f"Initializing GraphRAG (Cache={use_cache})")
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
        print("Initializing Base LLM")
        llm = base_llm

    # Outer cache for the Plain LLM or the RAG answer generation
    if use_cache and not rag:
        print("Enabling Outer Cache")
        llm = CachedLLM(
            inner=llm,
            maxsize=256,
            ttl_seconds=0,
        )
    return llm

def extract_score(text: str) -> int:
    """Helper to extract score from model output text like 'Score : 8'"""
    try:
        match = re.search(r'Score\s*:\s*(\d+)', text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    except:
        pass
    return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=False, help="User question or prompt for single test")
    parser.add_argument("--rag", action="store_true", help="Use RAG pipeline for single test")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--complete-test", action="store_true", help="Run full evaluation suite")
    parser.add_argument("--test", action="store_true", help="Run single prompt test")

    args = parser.parse_args()
    
    use_cache = not args.no_cache
    
    if args.complete_test:
        print("=== STARTING FULL EVALUATION PIPELINE ===")
        
        test_set = load_test_set()
        test_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        evaluation_model, evaluation_script = get_evaluation_model()
        
        # Files we will compare at the end
        file_plain = f"data/{test_datetime}_plain.json"
        file_rag = f"data/{test_datetime}_rag.json"
        
        # Run Plain LLM Baseline
        print("\n--- Phase 1: Running Plain LLM Baseline ---")
        llm_plain = get_llm(use_cache=True, rag=False)
        for t in test_set:
            start = datetime.now()
            ans = llm_plain.generate(t["question"])
            delta = (datetime.now() - start).total_seconds()
            
            record_test(f"{test_datetime}_plain.json", {
                "question": t["question"],
                "answer": ans,
                "time": str(delta)
            })
            print(f"Processed: {t['question'][:30]}...")

        # Run RAG Pipeline
        print("\n--- Phase 2: Running GraphRAG Pipeline ---")
        llm_rag = get_llm(use_cache=True, rag=True)
        for t in test_set:
            ans, stats, cypher, all_cyphers = llm_rag.generate(t["question"])
            
            record_test(f"{test_datetime}_rag.json", {
                "question": t["question"],
                "answer": ans,
                "cypher": cypher,
                "stats": stats
            })
            print(f"Processed: {t['question'][:30]}...")

        # Evaluation using LLM-as-a-Judge
        print("\n--- Phase 3: LLM-as-a-Judge Evaluation ---")
        
        # Read back results
        with open(file_plain, "r", encoding="utf-8") as f:
            results_plain = json.load(f)["results"]
        with open(file_rag, "r", encoding="utf-8") as f:
            results_rag = json.load(f)["results"]
            
        plain_scores = {}
        rag_scores = {}
        
        # Logging files
        path_log_plain = "data/accuracy_plain.txt"
        path_log_rag = "data/accuracy_rag.txt"

        for i in range(len(test_set)):
            question = test_set[i]["question"]
            ground_truth = test_set[i]["expected"]
            
            ans_p = results_plain[i]["answer"]
            ans_r = results_rag[i]["answer"]
            
            # Construct Prompts
            prompt_p = f"{evaluation_script}\nQUESTION:\n{json.dumps(question)}\nMODEL_ANSWER:\n{json.dumps(ans_p)}\nGROUND_TRUTH:\n{json.dumps(ground_truth)}"
            prompt_r = f"{evaluation_script}\nQUESTION:\n{json.dumps(question)}\nMODEL_ANSWER:\n{json.dumps(ans_r)}\nGROUND_TRUTH:\n{json.dumps(ground_truth)}"
            
            # Generate Scores
            eval_p = evaluation_model.generate(prompt_p)
            eval_r = evaluation_model.generate(prompt_r)
            
            plain_scores[question] = extract_score(eval_p)
            rag_scores[question] = extract_score(eval_r)
            
            # Append to text logs
            with open(path_log_plain, "a", encoding="utf-8") as f:
                f.write(f"{prompt_p}\n{eval_p}\n\n\n")
            with open(path_log_rag, "a", encoding="utf-8") as f:
                f.write(f"{prompt_r}\n{eval_r}\n\n\n")

        # Generate Summary Report
        print("\n--- Phase 4: Generating Report ---")
        report_md, stats = generate_comparison_table(plain_scores, rag_scores)
        
        summary_path = f"data/summary_{test_datetime}.md"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(report_md)
            
        print(f"Done! Report saved to {summary_path}")
        print(f"Plain Average: {stats['plain_avg']:.2f}")
        print(f"RAG Average:   {stats['rag_avg']:.2f}")

    else:
        # Single Prompt Mode
        if not args.prompt:
            print("Error: --prompt is required for single test mode")
            return

        llm = get_llm(use_cache=use_cache, rag=args.rag)
        start = datetime.now()
        
        if args.rag:
            answer, stats, cypher, _ = llm.generate(args.prompt)
            print(f"\nAnswer: {answer}")
            print(f"Cypher: {cypher}")
            print(f"Stats: {stats}")
        else:
            answer = llm.generate(args.prompt)
            print(f"\nAnswer: {answer}")
            
        print(f"Time: {(datetime.now() - start).total_seconds()}s")

if __name__ == "__main__":
    main()
