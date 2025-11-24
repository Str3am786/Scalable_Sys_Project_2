# src/scalable_sys/rag/prompts.py
from __future__ import annotations

from typing import List, Dict
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

EVALUATION_PROMPT = (
    "You are an impartial evaluator. Your task is to compare two answers to the same question:\n\n"
    "1) A RAG-based answer (Answer A)\n"
    "2) A plain LLM answer without retrieval (Answer B)\n\n"
    "You must evaluate which answer is better according to the following criteria:\n\n"
    "1. Correctness – factual accuracy, logical consistency, and alignment with the question.\n"
    "2. Grounding – whether the answer relies on verifiable information, avoids hallucinations, and is supported by provided facts if present.\n"
    "3. Completeness – whether it fully answers all components of the question.\n"
    "4. Clarity – readability, coherence, and organization of the response.\n"
    "5. Safety – absence of fabricated citations, misinformation, or overconfident claims.\n\n"
    "Instructions:\n"
    "- Read the question.\n"
    "- Read both answers completely.\n"
    "- Score each answer from 1 to 10 for each criterion.\n"
    "- Provide a brief justification for each score.\n"
    "- Provide a final verdict: “Answer A is better”, “Answer B is better”, or “Both are equivalent”.\n"
    "- Be strict about hallucinations and factual errors.\n\n"
    "Output Format (strict):\n"
    "{\n"
    "  \"scores\": {\n"
    "    \"answer_a\": {\n"
    "      \"correctness\": <1-10>,\n"
    "      \"grounding\": <1-10>,\n"
    "      \"completeness\": <1-10>,\n"
    "      \"clarity\": <1-10>,\n"
    "      \"safety\": <1-10>\n"
    "    },\n"
    "    \"answer_b\": {\n"
    "      \"correctness\": <1-10>,\n"
    "      \"grounding\": <1-10>,\n"
    "      \"completeness\": <1-10>,\n"
    "      \"clarity\": <1-10>,\n"
    "      \"safety\": <1-10>\n"
    "    }\n"
    "  },\n"
    "  \"justification\": \"<3–6 sentences summarizing reasoning>\",\n"
    "  \"winner\": \"<A | B | tie>\"\n"
    "}\n\n"
    "Content to evaluate:\n"
    "Question: {{QUESTION}}\n\n"
    "Answer A (RAG): {{RAG_ANSWER}}\n\n"
    "Answer B (Plain LLM): {{LLM_ANSWER}}\n"
)

# ---- Few-shot exemplars for Text2Cypher ----

EXEMPLARS: list[dict] = [
    # Basic Retrieval
    {
        "question": "Which scholars have won the Nobel Prize in Physics?",
        "cypher": (
            "MATCH (s:Scholar)-[:WON]->(p:Prize) "
            "WHERE toLower(p.category) = 'physics' "
            "RETURN s.knownName AS scholar, p.awardYear AS award_year "
            "ORDER BY p.awardYear"
        ),
    },

    # Filtering (Ranges & Specific Dates)
    {
        "question": "Who won the Nobel Prize in Physics in 2003?",
        "cypher": (
            "MATCH (s:Scholar)-[:WON]->(p:Prize) "
            "WHERE toLower(p.category) = 'physics' AND p.awardYear = 2003 "
            "RETURN s.knownName AS winner, p.category AS category, p.awardYear AS award_year"
        ),
    },
    {
        "question": "List all female Nobel laureates.",
        "cypher": (
            "MATCH (s:Scholar)-[:WON]->(p:Prize) "
            "WHERE toLower(s.gender) = 'female' "
            "RETURN s.knownName, p.category, p.awardYear"
        ),
    },
    {
        "question": "Who won the Chemistry prize between 1950 and 1960?",
        "cypher": (
            "MATCH (s:Scholar)-[:WON]->(p:Prize) "
            "WHERE toLower(p.category) = 'chemistry' "
            "AND p.awardYear >= 1950 AND p.awardYear <= 1960 "
            "RETURN s.knownName, p.awardYear"
        ),
    },

    # Geography: Birthplace
    {
        "question": "Which Nobel laureates were born in the United States?",
        "cypher": (
            "MATCH (s:Scholar)-[:BORN_IN]->(ci:City)-[:IS_CITY_IN]->(co:Country) "
            "WHERE toLower(co.name) = 'united states' "
            "RETURN s.knownName AS scholar, ci.name AS city"
        ),
    },
    {
        "question": "Which Nobel laureates were born in Germany?",
        "cypher": (
            "MATCH (s:Scholar)-[:BORN_IN]->(ci:City)-[:IS_CITY_IN]->(co:Country) "
            "WHERE toLower(co.name) = 'germany' "
            "RETURN s.knownName AS scholar, ci.name AS birth_city, co.name AS birth_country"
        ),
    },

    # Geography: Affiliation/Work
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
        "question": "Which institutions are located in the United Kingdom?",
        "cypher": (
            "MATCH (i:Institution)-[:IS_LOCATED_IN]->(ci:City)-[:IS_CITY_IN]->(co:Country) "
            "WHERE toLower(co.name) = 'united kingdom' "
            "RETURN i.name AS institution, ci.name AS city"
        ),
    },
    {
        "question": "Which laureates worked at institutions in France?",
        "cypher": (
            "MATCH (s:Scholar)-[:AFFILIATED_WITH]->(i:Institution)-[:IS_LOCATED_IN]->(ci:City)-[:IS_CITY_IN]->(co:Country) "
            "WHERE toLower(co.name) = 'france' "
            "RETURN s.knownName AS scholar, i.name AS institution"
        ),
    },

    # Mentorship & Lineage
    {
        "question": "Who is the academic grandfather of Richard Feynman?",
        "cypher": (
            "MATCH (grand_mentor:Scholar)-[:MENTORED]->(mentor:Scholar)-[:MENTORED]->(laureate:Scholar) "
            "WHERE toLower(laureate.knownName) CONTAINS 'richard feynman' "
            "RETURN grand_mentor.knownName"
        ),
    },
    {
        "question": "Who are the mentors of scholars who won the Medicine prize in 2015?",
        "cypher": (
            "MATCH (mentor:Scholar)-[:MENTORED]->(laureate:Scholar) "
            "MATCH (laureate)-[:WON]->(p:Prize) "
            "WHERE toLower(p.category) = 'medicine' AND p.awardYear = 2015 "
            "RETURN mentor.knownName AS mentor, laureate.knownName AS winner"
        ),
    },

    # Birth + Workplace
    {
        "question": "List Physics laureates who were born in Germany but were affiliated with an institution in the USA.",
        "cypher": (
            "MATCH (s:Scholar)-[:WON]->(p:Prize) "
            "WHERE toLower(p.category) = 'physics' "
            "MATCH (s)-[:BORN_IN]->(:City)-[:IS_CITY_IN]->(birth_country:Country) "
            "WHERE toLower(birth_country.name) = 'germany' "
            "MATCH (s)-[:AFFILIATED_WITH]->(:Institution)-[:IS_LOCATED_IN]->(:City)-[:IS_CITY_IN]->(work_country:Country) "
            "WHERE toLower(work_country.name) = 'usa' "
            "RETURN DISTINCT s.knownName"
        ),
    },
    {
        "question": "Which institutions in the United Kingdom have hosted Chemistry laureates?",
        "cypher": (
            "MATCH (i:Institution)-[:IS_LOCATED_IN]->(:City)-[:IS_CITY_IN]->(c:Country) "
            "WHERE toLower(c.name) = 'united kingdom' "
            "MATCH (s:Scholar)-[:AFFILIATED_WITH]->(i) "
            "MATCH (s)-[:WON]->(p:Prize) "
            "WHERE toLower(p.category) = 'chemistry' "
            "RETURN DISTINCT i.name"
        ),
    },

    # Aggregation (Counting)
    {
        "question": "How many Nobel laureates were born in Poland?",
        "cypher": (
            "MATCH (s:Scholar)-[:BORN_IN]->(:City)-[:IS_CITY_IN]->(c:Country) "
            "WHERE toLower(c.name) = 'poland' "
            "RETURN count(s) as laureate_count"
        )
    },
    {
        "question": "How many Chemistry laureates are there?",
        "cypher": (
            "MATCH (s:Scholar)-[:WON]->(p:Prize) "
            "WHERE toLower(p.category) = 'chemistry' "
            "RETURN count(s) as chemistry_laureates"
        )
    },
    {
        "question": "Which scholars affiliated with an institution in France died between the years 1945 and 1955?",
        "cypher": (
            "MATCH (s:Scholar)-[:AFFILIATED_WITH]->(i:Institution)-[:IS_LOCATED_IN]->(c:City)-[:IS_CITY_IN]->(co:Country) "
            "WHERE toLower(co.name) CONTAINS 'france' "
            "AND s.deathDate >= '1945-01-01' AND s.deathDate <= '1955-12-31' "
            "RETURN DISTINCT s.knownName AS Scholarname, i.name AS InstitutionName"
        )
    },

    # Limits (First, Last, Oldest)
    {
        "question": "Who was the first person to win the Nobel Prize in Physics?",
        "cypher": (
            "MATCH (s:Scholar)-[:WON]->(p:Prize) "
            "WHERE toLower(p.category) = 'physics' "
            "RETURN s.knownName, p.awardYear "
            "ORDER BY p.awardYear ASC LIMIT 1"
        )
    },
    {
        "question": "List the 5 most recent Economics laureates.",
        "cypher": (
            "MATCH (s:Scholar)-[:WON]->(p:Prize) "
            "WHERE toLower(p.category) = 'economics' "
            "RETURN s.knownName, p.awardYear "
            "ORDER BY p.awardYear DESC LIMIT 5"
        )
    },
    # Shared nobel
    {
        "question": "Which scholars won a prize with a 1/3 share portion?",
        "cypher": (
            "MATCH (s:Scholar)-[r:WON]->(p:Prize) "
            "WHERE r.portion = '1/3' "
            "RETURN s.knownName"
        ),
    },
]

EXEMPLAR_QUESTIONS: List[str] = [ex["question"] for ex in EXEMPLARS]

_tfidf_vectorizer = TfidfVectorizer()
_EXEMPLAR_MATRIX = _tfidf_vectorizer.fit_transform(EXEMPLAR_QUESTIONS)


def select_exemplars(question: str, k: int = 3, threshold: float = 0.35) -> List[Dict]:
    """
    Return top-k exemplar dicts most similar to the question.
    Filters out exemplars with similarity < threshold to avoid noise.
    """
    if not EXEMPLARS:
        return []

    query_vec = _tfidf_vectorizer.transform([question])
    sims = cosine_similarity(query_vec, _EXEMPLAR_MATRIX)[0]

    # Filter indices by threshold
    valid_indices = [i for i, sim in enumerate(sims) if sim >= threshold]

    # If nothing meets the threshold, fallback to at least the single best match 
    # Unless it's truly terrible (< 0.1)
    if not valid_indices:
        best_idx = sims.argmax()
        if sims[best_idx] < 0.1:
            return []
        valid_indices = [best_idx]

    # Sort valid indices by score descending
    valid_indices.sort(key=lambda i: sims[i], reverse=True)
    
    # Take top k
    top_indices = valid_indices[:k]

    # Debug logging
    print(f"\n=== Selected exemplars for question: {question!r} ===")
    if not top_indices:
        print("  (No exemplars met the similarity threshold)")
    
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
