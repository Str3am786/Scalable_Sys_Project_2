from __future__ import annotations

# Standard library
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, override


import dspy
import kuzu
import hashlib

from ..cache.memo import ttl_lru_cache
from ..llm.base import LLM
from .prompts import (
    format_exemplars_for_prompt,
    postprocess_cypher,
    select_exemplars,
)



# =========================================================================
#  PART 1: DSPy Signatures
# =========================================================================

class Text2Cypher(dspy.Signature):
    """Translate a natural language question into a Cypher query for KuzuDB.

    IMPORTANT GUIDELINES:
    1. Do NOT filter by property (e.g. p.category = 'physics') unless the user EXPLICITLY asks for it.
    2. Use the schema strictly. Do not invent relationships like [:AFFILIATED_WITH]->(:Country) if they don't exist.
    3. If asking for 'Laureates from [Country]', use the [:BORN_IN] or [:IS_LOCATED_IN] paths as seen in the schema.
    """

    graph_schema = dspy.InputField(desc="The graph schema (nodes, edges, properties)")
    exemplars = dspy.InputField(desc="Relevant Q&A examples to guide the model")
    question = dspy.InputField(desc="The user's question")

    cypher = dspy.OutputField(desc="A valid Cypher query string. No markdown.")


class RepairCypher(dspy.Signature):
    """Fix a broken Cypher query based on the database error message."""

    graph_schema = dspy.InputField(desc="The graph schema")
    original_question = dspy.InputField()
    bad_cypher = dspy.InputField(desc="The query that failed")
    error_msg = dspy.InputField(desc="The error returned by KuzuDB")

    repaired_cypher = dspy.OutputField(desc="The fixed Cypher query")


class GenerateAnswer(dspy.Signature):
    """Answer the user question based on the database results."""
    question = dspy.InputField()
    context = dspy.InputField(desc="Structured data retrieved from the graph")
    answer = dspy.OutputField(desc="Natural language answer")


# =========================================================================
#  PART 2: DSPy Module
# =========================================================================

class RefinedCypherGenerator(dspy.Module):
    def __init__(self, schema_str, conn: kuzu.Connection, use_postprocess: bool):
        super().__init__()
        self.conn = conn
        self.schema_str = schema_str
        self.use_postprocess = use_postprocess

        self.generate = dspy.ChainOfThought(Text2Cypher)
        self.repair = dspy.ChainOfThought(RepairCypher)

    def validate_cypher(self, cypher_query) -> tuple[bool, str]:
        try:
            self.conn.execute(f"EXPLAIN {cypher_query}")
            return True, ""
        except Exception as e:
            return False, str(e)

    def forward(self, question, exemplars_str):
        # Step 1: Generation
        response = self.generate(
            graph_schema=self.schema_str,
            exemplars=exemplars_str,
            question=question
        )
        cypher = response.cypher

        # Step 2: Validation & Repair
        valid, error_msg = self.validate_cypher(cypher)

        max_retries = 3
        for i in range(max_retries):
            if valid:
                break
            print(f"  [Refining attempt {i + 1}] Error: {error_msg[:80]}...")

            response = self.repair(
                graph_schema=self.schema_str,
                original_question=question,
                bad_cypher=cypher,
                error_msg=error_msg
            )
            cypher = response.repaired_cypher
            valid, error_msg = self.validate_cypher(cypher)

        # Step 3: Post-process
        if self.use_postprocess:
            cypher = postprocess_cypher(cypher)

        return cypher


# =========================================================================
#  PART 3: GraphRAG Class (With Fallback)
# =========================================================================

class GraphRAG(LLM):
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
        self.base_llm = llm
        self.use_exemplars = use_exemplars

        # --- FIX: Auto-detect OpenAI provider for local models ---
        dspy_model = llm.model
        if not dspy_model.startswith("openai/"):
            dspy_model = "openai/" + dspy_model

        print(f"Configuring DSPy with model: {dspy_model} at {llm.client.base_url}")

        dspy_lm = dspy.LM(
            model=dspy_model,
            api_base=str(llm.client.base_url),
            api_key=llm.client.api_key,
        )
        dspy.configure(lm=dspy_lm)

        # Setup DB
        path = Path(db_path)
        if not path.exists():
            raise FileNotFoundError(f"DB not found at {path}")
        self.db = kuzu.Database(str(path), read_only=True)
        self.conn = kuzu.Connection(self.db)
        self.schema_str = self._get_schema_str()

        # Pipeline
        self.pipeline = RefinedCypherGenerator(
            schema_str=self.schema_str,
            conn=self.conn,
            use_postprocess=use_postprocess
        )
        self.answer_gen = dspy.Predict(GenerateAnswer)

        # Caching
        if cache_text2cypher:
            self._generate_cypher = self._make_cached_generator(
                maxsize=cache_maxsize, ttl=cache_ttl_seconds
            )
        else:
            self._generate_cypher = self._generate_cypher_no_cache

    def _get_schema_str(self):
        # Simplified schema dump
        try:
            nodes = self.conn.execute("CALL SHOW_TABLES() WHERE type = 'NODE' RETURN *;").get_as_df()['name'].tolist()
            rels = self.conn.execute("CALL SHOW_TABLES() WHERE type = 'REL' RETURN *;").get_as_df()['name'].tolist()
            return json.dumps({"nodes": nodes, "relationships": rels})
        except Exception:
            return "Schema unavailable"

    def _make_cached_generator(self, maxsize, ttl):
        @ttl_lru_cache(maxsize=maxsize, ttl_seconds=ttl)
        def _cached(complex_key: str):
            # Extract the payload from our custom key format
            # Format: "HASH_SIGNATURE|JSON_PAYLOAD"
            _, json_payload = complex_key.split("|", 1)
            data = json.loads(json_payload)
            return self._generate_cypher_no_cache(data['q'], data['ex'])

        def wrapper(q, ex):
            # [TASK 2] Keying by Question Hash and Schema Hash
            q_hash = hashlib.sha256(q.encode()).hexdigest()[:16]
            schema_hash = hashlib.sha256(self.schema_str.encode()).hexdigest()[:16]

            # We carry the payload for execution, but the prefix is the required hash
            # This ensures that if q or schema changes, the hash changes.
            payload = json.dumps({"q": q, "ex": ex})

            # The key effectively becomes the Identity of the request
            key = f"{q_hash}_{schema_hash}|{payload}"
            return _cached(key)

        return wrapper

    def _generate_cypher_no_cache(self, question, exemplars):
        return self.pipeline(question=question, exemplars_str=exemplars)

    def generate(self, prompt: str, **kwargs) -> str:
        answer, _ = self.query_with_stats(prompt)
        return answer

    def stream(self, prompt: str, **kwargs) -> Iterable[str]:
        yield self.generate(prompt)

    def query_with_stats(self, question: str) -> tuple[str, dict]:
        t0 = time.perf_counter()

        # 1. Exemplars
        exemplars_str = ""
        if self.use_exemplars:
            exs = select_exemplars(question, k=3)
            exemplars_str = format_exemplars_for_prompt(exs)

        t1 = time.perf_counter()

        # 2. Generate Cypher
        cypher = self._generate_cypher(question, exemplars_str)

        t2 = time.perf_counter()
        print(f"\n=== Generated Cypher ===\n{cypher}\n")

        # 3. Execute DB Query
        rows = []
        try:
            result = self.conn.execute(cypher)
            columns = result.get_column_names()
            while result.has_next():
                row = result.get_next()
                rows.append(dict(zip(columns, row)))
        except Exception as e:
            print(f"Execution error: {e}")

        t3 = time.perf_counter()

        # 4. Generate Answer or FALLBACK
        if not rows:
            print(">>> No data found in graph. Falling back to base LLM.")
            # Fallback: Just ask the LLM directly
            answer = self.base_llm.generate(question)
        else:
            # RAG: Answer using context
            context_str = json.dumps(rows[:20])
            ans_response = self.answer_gen(question=question, context=context_str)
            answer = ans_response.answer

        t4 = time.perf_counter()

        stats = {
            "total": t4 - t0,
            "text2cypher": t2 - t1,
            "db_exec": t3 - t2,
            "answer_gen": t4 - t3
        }

        print("\n=== Timing Stats ===")
        for k, v in stats.items():
            print(f"{k}: {v:.4f}s")

        return answer, stats


# =========================================================================
#  Manual Implementation
# =========================================================================

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
