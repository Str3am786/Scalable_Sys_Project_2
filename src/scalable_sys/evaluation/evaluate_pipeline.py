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

from .analyze_cache_results import summary_cache_log, get_cache_results, create_report


CACHE_TEST = {
    "0": False,
    "1": True,
    "2": "BOTH"
}

def get_project_root() -> str:
    """Returns the absolute path to the project root."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, "../../../"))

def make_evaluation_model(cfg):
    return LlamaServer(
        base_url=cfg.evaluation_url,
        api_key=cfg.evaluation_key,
        model=cfg.evaluation_model,
    ), cfg.evaluation_prompt

def get_evaluation_model():
    cfg = load_config()
    evaluation_llm = make_evaluation_model(cfg)
    return evaluation_llm

def record_test(filepath: str, result) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
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

def record_evaluation(filepath: str, result: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data = {"evaluations": []}
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, dict):
                    data = content
                    if "evaluations" not in data:
                        data["evaluations"] = []
        except json.JSONDecodeError:
            pass
    data["evaluations"].append(result)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_test_set(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Test set not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "tests" not in data or not isinstance(data["tests"], list):
        raise ValueError("JSON must contain a top-level 'tests' array")
    cleaned_tests = []
    for i, entry in enumerate(data["tests"]):
        if "question" not in entry:
            continue 
        cleaned_tests.append({
            "question": entry["question"],
            "cypher": entry.get("cypher", ""),
            "expected": entry.get("expected", [])
        })
    return cleaned_tests

def _make_prediction_llm(cfg):
    if cfg.backend in ("server", "rag"):
        return LlamaServer(
            base_url=cfg.server_base_url,
            api_key=cfg.server_api_key,
            model=cfg.server_model,
        )
    else:
        raise ValueError(f"Unknown backend: {cfg.backend}")

def get_llm(use_cache: bool = False, rag: bool = False):
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
    if use_cache and not rag:
        print("Enabling Outer Cache")
        llm = CachedLLM(inner=llm, maxsize=256, ttl_seconds=0)
    return llm

def extract_score(text: str) -> int:
    try:
        match = re.search(r'Score\s*:\s*(\d+)', text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    except:
        pass
    return 0

def cache_test(
    input_file: str = "./data/test_input.json",
    out_folder: str = "results/cache_test",
    use_cache: bool = False,
    llm=None,
    label: str = "",
) -> str:
    test_set = load_test_set(input_file)
    test_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if llm is None:
        llm = get_llm(use_cache=use_cache, rag=True)

    if use_cache:
        cache_maxsize,cache_ttl_seconds = get_cache_parameters()
        print(f"CACHE PARAMETERS: Size = {cache_maxsize}, Time to Live = {cache_ttl_seconds}")
        base_name = "Cache_test"

    else:
        print("NO CACHE USED")
        base_name = "NO_Cache_test"
    
    suffix = f"_{label}" if label else ""
    filepath = f"./{out_folder}/{base_name}{suffix}.json"

    counter = 1
    for test in test_set:
        
        print(f"Starting test for question n° {counter}")
        counter += 1
        prompt = test["question"]
        answer, stats, c , all_tested_cyphers, results = llm.generate(prompt)
        result = {
            "question": prompt,
            "answer": answer,
            "stats": stats
        }
        record_test(filepath,result)
                
    return filepath


def get_cache_parameters() -> List[int]:
    cfg = load_config()
    return cfg.rag_cache_maxsize, cfg.rag_cache_ttl_seconds
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=False, help="User question for single test")
    parser.add_argument("--rag", action="store_true", help="Use RAG pipeline")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--complete-test", action="store_true", help="Run full suite")
    parser.add_argument("--judge-only", action="store_true", help="Run evaluation on existing files")
    parser.add_argument("--plain-file", type=str, help="Path to existing plain LLM results")
    parser.add_argument("--rag-file", type=str, help="Path to existing RAG results")
    parser.add_argument("--test", action="store_true", help="Run single prompt test")
    parser.add_argument(
        "--caching-test",
        type=str,
        choices=["0", "1", "2", "3"],
        help="Select which test to run",
    )
    
    args = parser.parse_args()
    use_cache = not args.no_cache
    
    PROJECT_ROOT = get_project_root()
    TEST_INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "test_input.json")
    
    if args.caching_test: 
        
        if args.caching_test == "3":
            run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            out_folder = os.path.join("results", "cache_eval", run_id)
            os.makedirs(out_folder, exist_ok=True)

            llm = get_llm(use_cache=True, rag=True)

            # COLD run: cache starts empty
            print("------------Starting COLD cache test---------------------")
            cold_filepath = cache_test(
                input_file="./data/test_cache_warm_start.json",
                out_folder=out_folder,
                use_cache=True,
                llm=llm,
                label="cold",
            )
            cache_maxsize, cache_ttl_seconds = get_cache_parameters()
            create_report(cold_filepath, cache_maxsize, cache_ttl_seconds, used_cache=True)
            print(f"Concluded cold test. You can find the result in {cold_filepath}")

            # WARM run: same LLM instance, cache already populated
            print("------------Starting WARM cache test---------------------")
            warm_filepath = cache_test(
                input_file="./data/test_cache_warm_start.json",
                out_folder=out_folder,
                use_cache=True,
                llm=llm,
                label="warm",
            )
            create_report(warm_filepath, cache_maxsize, cache_ttl_seconds, used_cache=True)
            print(f"Concluded warm test. You can find the result in {warm_filepath}")

            print(f"FINISHED, CHECK FOLDER {out_folder} for cold/warm results")
            return

        if args.caching_test == "2":
            c = [True,False]
            print("YOU CHOOSED BOTH configurations:")
        else:
            c = [CACHE_TEST[args.caching_test]]
        
        for type_test in c:
            
            print("------------Starting Test---------------------")
            print(f"CACHE: {type_test}")
            test_filepath = cache_test(input_file="./data/test_cache.json",use_cache = type_test)
            print(f"Conlcluded Test. You can find the result in {test_filepath}") 
            cache_maxsize,cache_ttl_seconds = get_cache_parameters()
            create_report(test_filepath, cache_maxsize, cache_ttl_seconds,used_cache=type_test)
            print(f"Conlcluded Test. You can find the result in {test_filepath}") 
        
        print("FINISHED, CHECK FOLDER cache_test for results")
        
    
    if args.complete_test or args.judge_only:
        print("=== STARTING EVALUATION PIPELINE ===")
        
        cfg = load_config() 

        test_set = load_test_set(TEST_INPUT_PATH)
        evaluation_model, evaluation_script = get_evaluation_model()

        DIR_RAW = os.path.join(PROJECT_ROOT, "results", "rag_eval", "raw")
        DIR_GRADED = os.path.join(PROJECT_ROOT, "results", "rag_eval", "graded")
        DIR_SUMMARY = os.path.join(PROJECT_ROOT, "results", "summary")

        if args.judge_only:
            if not args.plain_file or not args.rag_file:
                print("Error: --judge-only requires --plain-file and --rag-file")
                return
            
            print(f"\n[Mode: Judge Only] Loading existing results...")
            file_plain_raw = args.plain_file
            file_rag_raw = args.rag_file
            
            test_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_JUDGE")
        else:
            test_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
            # Save RAW answers to results/rag_eval/raw/
            file_plain_raw = os.path.join(DIR_RAW, f"{test_datetime}_plain.json")
            file_rag_raw = os.path.join(DIR_RAW, f"{test_datetime}_rag.json")
            
            # --- Running Plain LLM Baseline ---
            print(f"\n--- Running Plain LLM Baseline -> {file_plain_raw} ---")
            llm_plain = get_llm(use_cache=True, rag=False)
            for t in test_set:
                start = datetime.now()
                ans = llm_plain.generate(t["question"])
                delta = (datetime.now() - start).total_seconds()
                record_test(file_plain_raw, {
                    "question": t["question"],
                    "answer": ans,
                    "time": str(delta)
                })
                print(f"Processed: {t['question'][:30]}...")

            # --- Running GraphRAG Pipeline ---
            print(f"\n--- Running GraphRAG Pipeline -> {file_rag_raw} ---")
            llm_rag = get_llm(use_cache=True, rag=True)
            for t in test_set:
                ans, stats, cypher, all_cyphers, context = llm_rag.generate(t["question"])
                record_test(file_rag_raw, {
                    "question": t["question"],
                    "answer": ans,
                    "cypher": cypher,
                    "stats": stats,
                    "context": context
                })
                print(f"Processed: {t['question']}...")

        print("\n--- LLM-as-a-Judge Evaluation ---")
        
        with open(file_plain_raw, "r", encoding="utf-8") as f:
            results_plain = json.load(f)["results"]
        with open(file_rag_raw, "r", encoding="utf-8") as f:
            results_rag = json.load(f)["results"]
            
        plain_scores = {}
        rag_scores = {}
        
        # Save GRADED answers to results/rag_eval/graded/
        file_plain_graded = os.path.join(DIR_GRADED, f"{test_datetime}_plain_graded.json")
        file_rag_graded = os.path.join(DIR_GRADED, f"{test_datetime}_rag_graded.json")

        if len(results_plain) != len(test_set) or len(results_rag) != len(test_set):
            print("Warning: Result file length mismatch.")

        for i in range(len(test_set)):
            question = test_set[i]["question"]
            ground_truth = test_set[i]["expected"]
            try:
                ans_p = results_plain[i]["answer"]
                ans_r = results_rag[i]["answer"]

                cypher_r = results_rag[i].get("cypher", "")
                context_r = results_rag[i].get("context", [])

            except IndexError:
                continue
            
            prompt_p = f"{evaluation_script}\nQUESTION:\n{json.dumps(question)}\nMODEL_ANSWER:\n{json.dumps(ans_p)}\nGROUND_TRUTH:\n{json.dumps(ground_truth)}"
            prompt_r = f"{evaluation_script}\nQUESTION:\n{json.dumps(question)}\nMODEL_ANSWER:\n{json.dumps(ans_r)}\nGROUND_TRUTH:\n{json.dumps(ground_truth)}"
            
            print(f"Evaluating Question {i+1}/{len(test_set)}...")
            eval_p = evaluation_model.generate(prompt_p)
            eval_r = evaluation_model.generate(prompt_r)
            
            score_p = extract_score(eval_p)
            score_r = extract_score(eval_r)
            
            plain_scores[question] = score_p
            rag_scores[question] = score_r
            
            record_evaluation(file_plain_graded, {
                "question": question,
                "model_answer": ans_p,
                "ground_truth": ground_truth,
                "judge_output": eval_p,
                "score": score_p
            })
            record_evaluation(file_rag_graded, {
                "question": question,
                "model_answer": ans_r,
                "generated_cypher": cypher_r,
                "fetched_context": context_r,
                "ground_truth": ground_truth,
                "judge_output": eval_r,
                "score": score_r
            })

        print("\n--- Generating Report ---")
        report_md, stats = generate_comparison_table(
            plain_scores, 
            rag_scores,
            base_model=cfg.server_model,
            judge_model=cfg.evaluation_model
        )
        
        summary_path = os.path.join(DIR_SUMMARY, f"{test_datetime}_summary.md")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(report_md)
            
        print(f"Report saved to {summary_path}")
        print(f"Plain Average: {stats['plain_avg']:.2f}")
        print(f"RAG Average:   {stats['rag_avg']:.2f}")
    else:
        if not args.prompt:
            print("Error: --prompt required for single test")
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