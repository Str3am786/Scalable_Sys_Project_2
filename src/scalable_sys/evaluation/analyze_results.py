import json
import argparse
import os
from typing import List, Dict, Any, Tuple
from statistics import mean

def get_project_root() -> str:
    """Returns the absolute path to the project root."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, "../../../"))

def load_evaluation_json(filepath: str) -> Dict[str, Dict[str, Any]]:
    """
    Parses the evaluation JSON file into a rich dictionary.
    Returns: {Question: {'score': int, 'cypher': str, 'answer': str}}
    """
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} not found.")
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = {}
        entries = data.get("evaluations", [])
        for entry in entries:
            q = entry.get("question")
            s = entry.get("score")
            if q is not None and s is not None:
                results[q] = {
                    "score": int(s),
                    "cypher": entry.get("generated_cypher", "N/A"),
                    "answer": entry.get("model_answer", "")
                }
        return results
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {filepath}")
        return {}
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return {}

def generate_comparison_table(
    plain_data: Dict[str, Any],  # Fixed Variable Name
    rag_data: Dict[str, Any],    # Fixed Variable Name
    base_model: str = "Unknown",
    judge_model: str = "Unknown"
) -> Tuple[str, Dict[str, float]]:
    
    all_questions = sorted(list(set(plain_data.keys()) | set(rag_data.keys())))
    
    if not all_questions:
        return "No results found.", {"count": 0, "plain_avg": 0, "rag_avg": 0, "plain_pct": 0, "rag_pct": 0}

    md_table = "| ID | Question | Plain LLM Score | RAG Score | Difference |\n"
    md_table += "|---|---|:---:|:---:|:---:|\n"
    
    plain_values = []
    rag_values = []
    
    for idx, q in enumerate(all_questions):
        # Handle both Integer scores (from pipeline) and Dictionary objects (from file load)
        p_obj = plain_data.get(q, 0)
        r_obj = rag_data.get(q, 0)

        if isinstance(p_obj, dict):
            p_score = p_obj.get("score", 0)
        else:
            p_score = p_obj

        if isinstance(r_obj, dict):
            r_score = r_obj.get("score", 0)
        else:
            r_score = r_obj

        if not isinstance(p_score, (int, float)): p_score = 0
        if not isinstance(r_score, (int, float)): r_score = 0

        plain_values.append(p_score)
        rag_values.append(r_score)
        
        diff = r_score - p_score
        diff_str = "="
        if diff > 0:
            diff_str = f"+{diff} (RAG)"
        elif diff < 0:
            diff_str = f"{diff} (Plain)"
        
        q_display = (q[:75] + '..') if len(q) > 75 else q
        q_display = q_display.replace("|", "&#124;")
        md_table += f"| {idx+1} | {q_display} | {p_score} | {r_score} | {diff_str} |\n"

    total_questions = len(plain_values)
    max_possible_points = total_questions * 10
    plain_sum = sum(plain_values)
    rag_sum = sum(rag_values)
    plain_pct = (plain_sum / max_possible_points * 100) if max_possible_points > 0 else 0
    rag_pct = (rag_sum / max_possible_points * 100) if max_possible_points > 0 else 0

    stats = {
        "plain_avg": mean(plain_values) if plain_values else 0,
        "rag_avg": mean(rag_values) if rag_values else 0,
        "plain_pct": plain_pct,
        "rag_pct": rag_pct,
        "count": total_questions
    }
    
    summary = "\n### Summary Statistics\n\n"
    summary += f"- **Base Model**: {base_model}\n"
    summary += f"- **Judge Model**: {judge_model}\n"
    summary += f"- **Total Questions**: {stats['count']}\n"
    summary += f"- **Plain LLM Accuracy**: {stats['plain_pct']:.1f}% ({plain_sum}/{max_possible_points})\n"
    summary += f"- **RAG Accuracy**: {stats['rag_pct']:.1f}% ({rag_sum}/{max_possible_points})\n"
    
    delta = stats['rag_pct'] - stats['plain_pct']
    improvement = "Improved" if delta > 0 else "Regressed"
    summary += f"- **Net Impact**: {improvement} by {abs(delta):.1f}%\n"

    full_report = "# Evaluation Report\n\n" + summary + "\n### Detailed Results\n\n" + md_table
    return full_report, stats

def main():
    PROJECT_ROOT = get_project_root()

    default_plain = os.path.join(PROJECT_ROOT, "results/rag_eval/graded/latest_plain_graded.json")
    default_rag = os.path.join(PROJECT_ROOT, "results/rag_eval/graded/latest_rag_graded.json")
    default_output = os.path.join(PROJECT_ROOT, "results/summary/evaluation_summary.md")

    parser = argparse.ArgumentParser(description="Generate summary table from evaluation JSONs.")
    parser.add_argument("--plain", default=default_plain, help="Path to plain LLM evaluation JSON")
    parser.add_argument("--rag", default=default_rag, help="Path to RAG evaluation JSON")
    parser.add_argument("--output", default=default_output, help="Output markdown file")

    parser.add_argument("--base-model", default="Unknown", help="Name of base model")
    parser.add_argument("--judge-model", default="Unknown", help="Name of judge model")
    
    args = parser.parse_args()
    
    print(f"Reading plain logs from: {args.plain}")
    plain_data = load_evaluation_json(args.plain)
    print(f"Reading RAG logs from: {args.rag}")
    rag_data = load_evaluation_json(args.rag)
    
    if not plain_data and not rag_data:
        print("No data found. Exiting.")
        return

    print("Generating report...")
    report, stats = generate_comparison_table(
        plain_data, 
        rag_data, 
        base_model=args.base_model, 
        judge_model=args.judge_model
    )
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
        
    print(f"\nReport written to {args.output}")
    print("-" * 30)
    print(f"Total Questions: {stats['count']}")
    print(f"Plain Accuracy:  {stats['plain_pct']:.1f}%")
    print(f"RAG Accuracy:    {stats['rag_pct']:.1f}%")
    print("-" * 30)

    # === DEBUG OUTPUT ===
    print("\n=== LOW SCORING RAG ANSWERS (DEBUG) ===")
    for q, val in rag_data.items():
        # Handle simple vs rich dict
        score = val.get("score", 0) if isinstance(val, dict) else val
        cypher = val.get("cypher", "N/A") if isinstance(val, dict) else "N/A"
        
        if isinstance(score, int) and score < 10:
            print(f"Q: {q}")
            print(f"Score: {score}")
            print(f"Cypher: {cypher}")
            print("-" * 30)

if __name__ == "__main__":
    main()