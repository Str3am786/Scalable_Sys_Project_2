import re
import json
import argparse
from typing import List, Dict, Any, Tuple
from statistics import mean
import os

def parse_evaluation_file(filepath: str) -> Dict[str, int]:
    """
    Parses a raw evaluation text file.
    Returns a dictionary mapping {Question: Score}.
    """
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} not found.")
        return {}

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = content.split("\n\n\n")
    
    results = {}
    
    # Regex to find the Question and the Score
    question_pattern = re.compile(r'QUESTION:\s*\n(".*?")', re.DOTALL)
    score_pattern = re.compile(r'Score\s*:\s*(\d+)', re.IGNORECASE)

    for block in blocks:
        if not block.strip():
            continue
            
        q_match = question_pattern.search(block)
        s_match = score_pattern.search(block)
        
        if q_match and s_match:
            try:
                question_text = json.loads(q_match.group(1))
                score = int(s_match.group(1))
                results[question_text] = score
            except Exception as e:
                # print(f"Error parsing block: {e}")
                continue

    return results

def generate_comparison_table(plain_scores: Dict[str, int], rag_scores: Dict[str, int]) -> Tuple[str, Dict[str, float]]:
    """
    Generates a Markdown table and calculates percentage statistics.
    """
    
    all_questions = sorted(list(set(plain_scores.keys()) | set(rag_scores.keys())))
    
    # Header
    md_table = "| ID | Question | Plain LLM Score | RAG Score | Difference |\n"
    md_table += "|---|---|:---:|:---:|:---:|\n"
    
    plain_values = []
    rag_values = []
    
    for idx, q in enumerate(all_questions):
        p_score = plain_scores.get(q, 0) # Default to 0 if missing for safety
        r_score = rag_scores.get(q, 0)
        
        plain_values.append(p_score)
        rag_values.append(r_score)
        
        diff = r_score - p_score
        
        diff_str = "="
        if diff > 0:
            diff_str = f"+{diff} (RAG)"
        elif diff < 0:
            diff_str = f"{diff} (Plain)"
        
        q_display = (q[:75] + '..') if len(q) > 75 else q
        md_table += f"| {idx+1} | {q_display} | {p_score} | {r_score} | {diff_str} |\n"

    # --- PERCENTAGE CALCULATION ---
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
    summary += f"- **Total Questions**: {stats['count']}\n"
    summary += f"- **Plain LLM Accuracy**: {stats['plain_pct']:.1f}% ({plain_sum}/{max_possible_points})\n"
    summary += f"- **RAG Accuracy**: {stats['rag_pct']:.1f}% ({rag_sum}/{max_possible_points})\n"
    
    delta = stats['rag_pct'] - stats['plain_pct']
    improvement = "Improved" if delta > 0 else "Regressed"
    summary += f"- **Net Impact**: {improvement} by {abs(delta):.1f}%\n"

    full_report = "# Evaluation Report\n\n" + summary + "\n### Detailed Results\n\n" + md_table
    return full_report, stats

def main():
    parser = argparse.ArgumentParser(description="Generate summary table from accuracy logs.")
    parser.add_argument("--plain", default="data/accuracy_plain.txt", help="Path to plain LLM logs")
    parser.add_argument("--rag", default="data/accuracy_rag.txt", help="Path to RAG logs")
    parser.add_argument("--output", default="data/evaluation_summary.md", help="Output markdown file")
    
    args = parser.parse_args()
    
    print(f"Reading plain logs from: {args.plain}")
    plain_data = parse_evaluation_file(args.plain)
    
    print(f"Reading RAG logs from: {args.rag}")
    rag_data = parse_evaluation_file(args.rag)
    
    if not plain_data and not rag_data:
        print("No data found. Exiting.")
        return

    print("Generating report...")
    report, stats = generate_comparison_table(plain_data, rag_data)
    
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
        
    print(f"\nReport written to {args.output}")
    print("-" * 30)
    print(f"Total Questions: {stats['count']}")
    print(f"Plain Accuracy:  {stats['plain_pct']:.1f}%")
    print(f"RAG Accuracy:    {stats['rag_pct']:.1f}%")
    print("-" * 30)

if __name__ == "__main__":
    main()