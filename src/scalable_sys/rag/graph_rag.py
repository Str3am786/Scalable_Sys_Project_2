# src/scalable_sys/rag/graph_rag.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import kuzu

from ..llm.base import LLM
from ..cache.memo import ttl_lru_cache
from .prompts import (
    select_exemplars,
    format_exemplars_for_prompt,
    postprocess_cypher,
)


@dataclass
class KuzuSchema:
    nodes: list[dict]
    edges: list[dict]


def _load_schema(conn: kuzu.Connection) -> KuzuSchema:
    """Read Kuzu schema from DB."""
    # Nodes
    resp = conn.execute("CALL SHOW_TABLES() WHERE type = 'NODE' RETURN *;")
    nodes = [row[1] for row in resp]

    # Relationship tables
    resp = conn.execute("CALL SHOW_TABLES() WHERE type = 'REL' RETURN *;")
    rel_tables = [row[1] for row in resp]

    relationships: list[dict] = []
    for tbl_name in rel_tables:
        resp = conn.execute(f"CALL SHOW_CONNECTION('{tbl_name}') RETURN *;")
        for row in resp:
            relationships.append({"name": tbl_name, "from": row[0], "to": row[1]})

    schema = {"nodes": [], "edges": []}

    for node in nodes:
        node_schema = {"label": node, "properties": []}
        node_props = conn.execute(f"CALL TABLE_INFO('{node}') RETURN *;")
        for row in node_props:
            node_schema["properties"].append({"name": row[1], "type": row[2]})
        schema["nodes"].append(node_schema)

    for rel in relationships:
        edge = {
            "label": rel["name"],
            "from": rel["from"],
            "to": rel["to"],
            "properties": [],
        }
        rel_props = conn.execute(f"CALL TABLE_INFO('{rel['name']}') RETURN *;")
        for row in rel_props:
            edge["properties"].append({"name": row[1], "type": row[2]})
        schema["edges"].append(edge)

    return KuzuSchema(nodes=schema["nodes"], edges=schema["edges"])


class ManualGraphRAG(LLM):
    """
    Graph-RAG pipeline wrapped as an LLM backend.

    generate(prompt) treats `prompt` as the *user question* and returns a natural-language answer.

    Flags:
      - use_exemplars: enable/disable few-shot Text2Cypher
      - use_self_refine: enable/disable EXPLAIN + repair loop
      - use_postprocess: enable/disable rule-based Cypher post-processing
    """

    def __init__(
        self,
        llm: LLM,
        db_path: str,
        *,
        use_exemplars: bool = True,
        use_self_refine: bool = True,
        use_postprocess: bool = True,
        cache_text2cypher: bool = True,
        cache_maxsize: int = 256,
        cache_ttl_seconds: int = 0,
    ):
        self.llm = llm

        path = Path(db_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Kuzu database not found at {path}. "
                f"Did you run the build_db script to create nobel.kuzu?"
            )

        # IMPORTANT: open existing DB in read-only mode
        self.db = kuzu.Database(str(path), read_only=True)
        self.conn = kuzu.Connection(self.db)

        self.use_exemplars = use_exemplars
        self.use_self_refine = use_self_refine
        self.use_postprocess = use_postprocess

        # Pre-load schema once
        self.schema = _load_schema(self.conn)
        self._schema_json = json.dumps(
            {"nodes": self.schema.nodes, "edges": self.schema.edges}
        )

        # Text2Cypher cache (Task 2)
        if cache_text2cypher:
            self._text2cypher_cached = self._make_cached_text2cypher(
                maxsize=cache_maxsize, ttl_seconds=cache_ttl_seconds
            )
        else:
            self._text2cypher_cached = None

    # --------- LLM interface ---------

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """
        Treat `prompt` as the user question and run Graph-RAG to return an answer.
        """
        answer, _stats = self.query_with_stats(
            user_question=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return answer

    def stream(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Iterable[str]:
        # For now, just non-streaming fallback
        yield self.generate(prompt, max_tokens=max_tokens, temperature=temperature)

    # --------- Core RAG pipeline ---------

    def query(self, user_question: str) -> str:
        """Convenience wrapper: Graph-RAG answer only."""
        answer, _stats = self.query_with_stats(user_question)
        return answer

    def query_with_stats(
        self,
        user_question: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> tuple[str, dict[str, float]]:
        """
        Run the full Graph-RAG pipeline and return (answer, timing_stats).

        timing_stats keys:
          - total
          - text2cypher
          - refinement
          - db_query
          - answer
        """
        t0 = time.perf_counter()

        # 1. Text2Cypher (with optional caching)
        t1 = time.perf_counter()
        cypher = self._get_cypher_for_question(user_question)
        t2 = time.perf_counter()

        # 2. Self-refinement loop (EXPLAIN + repair)
        if self.use_self_refine:
            cypher = self._self_refine_cypher(user_question, cypher)
        t3 = time.perf_counter()

        # 3. Run DB query
        rows = self._run_cypher(cypher)
        t4 = time.perf_counter()

        # 4. Answer generation
        answer = self._answer_from_context(
            user_question=user_question,
            cypher=cypher,
            rows=rows,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        t5 = time.perf_counter()

        stats = {
            "total": t5 - t0,
            "text2cypher": t2 - t1,
            "refinement": t3 - t2,
            "db_query": t4 - t3,
            "answer": t5 - t4,
        }

        # Simple timing breakdown (Task 2)
        print("\n=== Timing breakdown (seconds) ===")
        for k, v in stats.items():
            print(f"{k:>12}: {v:.4f}")

        return answer, stats

    # --------- Text2Cypher with caching ---------

    def _make_cached_text2cypher(self, maxsize: int, ttl_seconds: int):
        @ttl_lru_cache(maxsize=maxsize, ttl_seconds=ttl_seconds)
        def _cached_call(key: str) -> str:
            payload = json.loads(key)
            return self._text2cypher_no_cache(
                question=payload["question"],
                schema_json=payload["schema_json"],
            )

        return _cached_call

    def _get_cypher_for_question(self, question: str) -> str:
        if self._text2cypher_cached is None:
            cypher = self._text2cypher_no_cache(
                question=question,
                schema_json=self._schema_json,
            )
        else:
            key = json.dumps(
                {
                    "question": question,
                    "schema_json": self._schema_json,
                },
                sort_keys=True,
            )
            cypher = self._text2cypher_cached(key)

        if self.use_postprocess:
            cypher = postprocess_cypher(cypher)

        print("\n=== Text2Cypher (after post-processing) ===")
        print(cypher)
        return cypher

    def _text2cypher_no_cache(self, question: str, schema_json: str) -> str:
        """Single-pass Text2Cypher using the underlying LLM."""
        exemplars_block = ""
        if self.use_exemplars:
            exs = select_exemplars(question, k=3)
            exemplars_block = format_exemplars_for_prompt(exs)

        prompt = f"""
You are an expert in writing Cypher queries for a Nobel laureate graph database.

<SCHEMA>
{schema_json}
</SCHEMA>

<QUESTION>
{question}
</QUESTION>

<GUIDELINES>
- When matching on Scholar names, ALWAYS match on the `knownName` property.
- For countries, cities, continents and institutions, you can match on the `name` property.
- Always label variables in MATCH, e.g. (s:Scholar)-[:WON]->(p:Prize).
- When comparing string properties:
  - Use the WHERE clause.
  - Lowercase the property values before comparison.
  - Use the CONTAINS operator for substring matching.
- NEVER use `.name` on :Prize or :Scholar.
  - For :Scholar use `knownName`.
  - For :Prize use `category` (and optionally `awardYear`).
- Do NOT use APOC.
- Return property values rather than whole nodes/relationships.
</GUIDELINES>

<EXEMPLARS>
{exemplars_block}
</EXEMPLARS>

Write a single Cypher query that answers the question. 
Return ONLY the Cypher query, no explanations, no backticks.
""".strip()

        cypher = self.llm.generate(prompt, max_tokens=512, temperature=0.0)
        return cypher.strip()

    # --------- Self-refinement with EXPLAIN (Task 1) ---------

    def _self_refine_cypher(
        self,
        question: str,
        cypher: str,
        max_attempts: int = 3,
    ) -> str:
        """
        Validate Cypher using EXPLAIN; if it fails, ask the LLM to repair.
        """
        attempt = 0
        last_error = None
        current = cypher

        while attempt < max_attempts:
            attempt += 1
            try:
                _ = self.conn.execute(f"EXPLAIN {current}")
                print(f"\nCypher validated successfully on attempt {attempt}.")
                return current
            except RuntimeError as e:
                last_error = str(e)
                print(f"\nEXPLAIN failed on attempt {attempt}: {last_error}")

                repair_prompt = f"""
You previously wrote the following Cypher query for the question:

QUESTION:
{question}

CURRENT QUERY:
{current}

The database returned the following error when running EXPLAIN:

ERROR:
{last_error}

Please return a REPAIRED Cypher query that fixes the error while still answering the question.
Follow the same schema and guidelines as before.
Return ONLY the fixed Cypher query, no explanations, no backticks.
""".strip()

                current = self.llm.generate(
                    repair_prompt,
                    max_tokens=512,
                    temperature=0.0,
                ).strip()

                if self.use_postprocess:
                    current = postprocess_cypher(current)

        print("\nWARNING: Cypher still invalid after refinement attempts; using last version.")
        return current

    # --------- DB + Answering ---------

    def _run_cypher(self, cypher: str) -> list[list[Any]]:
        try:
            result = self.conn.execute(cypher)
            rows = [list(row) for row in result]
            print("\n=== RAW DB RESULTS ===")
            print(rows)
            return rows
        except RuntimeError as e:
            print(f"\nError running query: {e}")
            return []

    def _answer_from_context(
        self,
        user_question: str,
        cypher: str,
        rows: list[list[Any]],
        *,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """
        Ask the LLM to answer based on the retrieved rows.
        """
        if not rows:
            return "Not enough context to answer the question from the graph."

        context_str = json.dumps(rows)

        prompt = f"""
You are helping to answer questions about Nobel laureates using a graph database.

The following Cypher query was executed:

{cypher}

It returned these rows (as a JSON-like array):

{context_str}

Using ONLY this context (do not make up facts), answer the user's question:

{user_question}

If the context is clearly insufficient, reply exactly with: "Not enough context".
When dealing with dates, mention the month in full.
""".strip()

        answer = self.llm.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return answer.strip()
