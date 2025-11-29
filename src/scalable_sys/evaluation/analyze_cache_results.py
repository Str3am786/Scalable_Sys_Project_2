import re
from typing import Dict, List
import json
import pandas as pd
from datetime import datetime
import os

def summary_cache_log(log_file: str = "./results/cache_test/cache_log.txt") -> Dict[str, List[str]]:

    # Regex patterns for each log event
    patterns = {
        "store":   re.compile(r"Store Key \(\((.*?)\)\)"),
        "hit":     re.compile(r"HIT key: \(\((.*?)\)\)"),
        "expired": re.compile(r"Expired key: \(\((.*?)\)\)"),
        "removed": re.compile(r"Removed \(\((.*?)\)\),"),
    }

    # Stats container
    stats = {event: [] for event in patterns}

    # Load log text

    with open(log_file, "r", encoding="utf-8") as f:
        text = f.read()
    # Parse line by line
    for line in text.splitlines():
        for event, regex in patterns.items():
            m = regex.search(line)
            if m:
                stats[event].append(m.group(1))
    return stats


def get_cache_results(filepath: str):

    # Read the JSON file
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    data = data["results"]
    d = []
    for entry in data:
        d.append(entry["stats"])
    
    df = pd.DataFrame(d)
    return df


def create_report(filepath : str, cache_dim: int, cache_ttl: int, used_cache: bool):

    # ---- Load JSON results ----
    df = get_cache_results(filepath)

    total = df["total"].sum()
    text2cypher_total = df["text2cypher"].sum()
    answer_gen_total = df["answer_gen"].sum()

    avg_total = df["total"].mean()
    avg_text2cypher = df["text2cypher"].mean()
    avg_answer_gen = df["answer_gen"].mean()

    
    if used_cache:
        log = summary_cache_log()

        total_hits = len(log["hit"])
        total_removed = len(log["removed"])
        total_expired = len(log["expired"])
        total_store = len(log["store"])

    report_lines = []

    report_lines.append(f"# Performance\n")

    report_lines.append("## Cache Configuration\n")
    if used_cache:
        report_lines.append(f"- **Cache Dimension:** `{cache_dim} bytes`")
        report_lines.append(f"- **Cache TTL:** `{cache_ttl}` seconds\n")
    else:
        report_lines.append(f"- **Cache Deactivated**")

    report_lines.append("## Execution Statistics\n")
    report_lines.append("| Metric | Sum | Average |")
    report_lines.append("|--------|-----|---------|")
    report_lines.append(f"| Total Time | `{total:.2f}` | `{avg_total:.2f}` |")
    report_lines.append(f"| Text2cypher Time | `{text2cypher_total:.2f}` | `{avg_text2cypher:.2f}` |")
    report_lines.append(f"| Answer_gen Time | `{answer_gen_total:.2f}` | `{avg_answer_gen:.2f}` |")
    report_lines.append("")

    if used_cache:
        report_lines.append("## Cache Log Summary\n")
        report_lines.append("| Event | Count |")
        report_lines.append("|-------|-------|")
        report_lines.append(f"| Stored Key | `{total_store}` |")
        report_lines.append(f"| Hits | `{total_hits}` |")
        report_lines.append(f"| Removed Key | `{total_removed}` |")
        report_lines.append(f"| Expired Key | `{total_expired}` |")
        report_lines.append("")

    report_lines.append("## 📝 Raw Results Summary\n")
    report_lines.append("```json")
    report_lines.append(df.to_json(orient="records", indent=2))
    report_lines.append("```\n")

    # ---- Write Markdown file ----
    report_path = os.path.splitext(filepath)[0] + "_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))


