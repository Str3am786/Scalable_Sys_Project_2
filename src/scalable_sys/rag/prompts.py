# src/scalable_sys/rag/prompts.py
from __future__ import annotations

from typing import List, Dict
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---- Few-shot exemplars for Text2Cypher ----

EXEMPLARS: list[dict] = [
    {
        "question": "Which scholars won the Nobel Prize in Physics?",
        "cypher": (
            "MATCH (s:Scholar)-[:WON]->(p:Prize) "
            "WHERE toLower(p.category) = 'physics' "
            "RETURN s.knownName AS scholar, p.awardYear AS award_year "
            "ORDER BY p.awardYear"
        ),
    },
    {
        "question": "Which scholars won Nobel Prizes in Physics and were affiliated with the University of Cambridge?",
        "cypher": (
            "MATCH (s:Scholar)-[:WON]->(p:Prize), "
            "      (s)-[:AFFILIATED_WITH]->(i:Institution) "
            "WHERE toLower(p.category) = 'physics' "
            "  AND toLower(i.name) CONTAINS 'university of cambridge' "
            "RETURN s.knownName AS scholar, p.awardYear AS award_year, i.name AS institution "
            "ORDER BY p.awardYear"
        ),
    },
    {
        "question": "Who won the Nobel Prize in Physics in 2001?",
        "cypher": (
            "MATCH (s:Scholar)-[:WON]->(p:Prize) "
            "WHERE toLower(p.category) = 'physics' AND p.awardYear = 2001 "
            "RETURN s.knownName AS winner, p.category AS category, p.awardYear AS award_year"
        ),
    },
    {
        "question": "Which institutions are located in the United Kingdom?",
        "cypher": (
            "MATCH (i:Institution)-[:IS_LOCATED_IN]->(ci:City)-[:IS_CITY_IN]->(co:Country) "
            "WHERE toLower(co.name) = 'united kingdom' "
            "RETURN i.name AS institution, ci.name AS city"
        ),
    },
    {
        "question": "Which Nobel laureates were born in the United States?",
        "cypher": (
            "MATCH (s:Scholar)-[:BORN_IN]->(ci:City)-[:IS_CITY_IN]->(co:Country) "
            "WHERE toLower(co.name) = 'united states' "
            "RETURN s.knownName AS scholar, ci.name AS city"
        ),
    },
    {
        "question": "Which Nobel laureates are from Germany?",
        "cypher": (
            "MATCH (s:Scholar)-[:BORN_IN]->(ci:City)-[:IS_CITY_IN]->(co:Country) "
            "WHERE toLower(co.name) = 'germany' "
            "RETURN s.knownName AS laureate, co.name AS country"
        )
    },
]

EXEMPLAR_QUESTIONS: List[str] = [ex["question"] for ex in EXEMPLARS]

_tfidf_vectorizer = TfidfVectorizer()
_EXEMPLAR_MATRIX = _tfidf_vectorizer.fit_transform(EXEMPLAR_QUESTIONS)


def select_exemplars(question: str, k: int = 3) -> List[Dict]:
    """Return top-k exemplar dicts most similar to the question."""
    if not EXEMPLARS:
        return []

    query_vec = _tfidf_vectorizer.transform([question])
    sims = cosine_similarity(query_vec, _EXEMPLAR_MATRIX)[0]

    k = min(k, len(EXEMPLARS))
    top_indices = sims.argsort()[::-1][:k]

    # Debug logging – remove or gate with a flag if needed
    print(f"\n=== Selected exemplars for question: {question!r} ===")
    for rank, idx in enumerate(top_indices, start=1):
        ex_q = EXEMPLARS[idx]["question"]
        print(f"{rank}. sim={sims[idx]:.3f}  Q: {ex_q}")

    return [EXEMPLARS[i] for i in top_indices]


def format_exemplars_for_prompt(exemplars: List[Dict]) -> str:
    if not exemplars:
        return ""
    parts = []
    for ex in exemplars:
        parts.append(f"Q: {ex['question']}\nCypher: {ex['cypher']}")
    return "\n\n".join(parts)


# ---- Rule-based post-processing (Task 1) ----

def _extract_labelled_vars(query: str) -> dict[str, str]:
    """
    Return mapping var -> label from patterns like (s:Scholar).
    """
    pattern = re.compile(r"\((\w+):(\w+)\)")
    return {var: label for var, label in pattern.findall(query)}


def _fix_name_properties(query: str) -> str:
    """
    Replace .name on :Scholar/:Prize with knownName/category.
    """
    var_labels = _extract_labelled_vars(query)

    for var, label in var_labels.items():
        if label == "Scholar":
            query = query.replace(f"{var}.name", f"{var}.knownName")
        elif label == "Prize":
            query = query.replace(f"{var}.name", f"{var}.category")
    return query


def _enforce_lowercase_string_comparisons(query: str) -> str:
    """
    Wrap string comparisons in toLower(...), lowercase literals.
    """
    # Equality: x.prop = 'Value'
    pattern_eq = re.compile(r"(\w+\.\w+)\s*=\s*'([^']*)'", flags=re.IGNORECASE)

    def repl_eq(match):
        prop = match.group(1)
        lit = match.group(2)
        if prop.strip().lower().startswith("tolower("):
            return match.group(0)
        return f"toLower({prop}) = '{lit.lower()}'"

    query = pattern_eq.sub(repl_eq, query)

    # CONTAINS: x.prop CONTAINS 'Value'
    pattern_contains = re.compile(r"(\w+\.\w+)\s+CONTAINS\s+'([^']*)'", flags=re.IGNORECASE)

    def repl_contains(match):
        prop = match.group(1)
        lit = match.group(2)
        if prop.strip().lower().startswith("tolower("):
            return match.group(0)
        return f"toLower({prop}) CONTAINS '{lit.lower()}'"

    query = pattern_contains.sub(repl_contains, query)

    return query


def postprocess_cypher(query: str) -> str:
    """
    Apply rule-based post-processing to the generated Cypher.
    """
    if not query:
        return query

    cleaned = query.strip()
    cleaned = _fix_name_properties(cleaned)
    cleaned = _enforce_lowercase_string_comparisons(cleaned)
    return cleaned
