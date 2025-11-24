from __future__ import annotations

import json
import time
import hashlib
from pathlib import Path
from typing import Any, Iterable

import dspy
import kuzu

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

    <SCHEMA_RULES>
    1. **Node Properties**:
       - `Scholar`: match names on property `knownName`.
       - `Country`, `City`, `Institution`: match names on property `name`.
       - `Prize`: match category on property `category`.
       - **Time**: Use `awardYear` for prize years.

    2. **Critical Data Types**:
       - `portion` (on :WON relationship) is a **STRING** (e.g., "1/3").
       - `awardYear` is an **INTEGER**.

    3. **Path Patterns**:
       - "Born in [City]" -> `(:Scholar)-[:BORN_IN]->(:City)`
       - "Won prize" -> `(:Scholar)-[:WON]->(:Prize)`
    
    4. **String Matching (CRITICAL)**:
       - **STOP!** Do NOT use inline property matching (e.g., `(:Institution {name: 'Harvard'})` is FORBIDDEN).
       - **ALWAYS** use `WHERE` clauses with `toLower(...)` and `CONTAINS`.
       - **Correct:** `MATCH (i:Institution) WHERE toLower(i.name) CONTAINS 'harvard'`
       - **Reason:** - Institutions: 'Harvard' must match 'Harvard Medical School'.
         - Categories: 'Medicine' must match 'Physiology or Medicine'.
         - Locations: 'Netherlands' must match 'the Netherlands'.

    5. **Formatting**:
       - Do NOT use `LIMIT` unless explicitly asked.
       - Return only the specific properties needed.
    
    6. **Historical Geography**:
       - Borders change over time. If a user asks about a country with historically shifting borders (e.g., Poland, Ukraine, Germany, Russia/USSR), you MUST check for BOTH the historical and modern country names using `OR`.
       - Example: `WHERE toLower(co.name) CONTAINS 'poland' OR toLower(co.name) CONTAINS 'ukraine'`

    </SCHEMA_RULES>
    """

    graph_schema = dspy.InputField(desc="The detailed schema definition of Nodes, Relationships, and Properties")
    exemplars = dspy.InputField(desc="Correct Q&A examples to learn from")
    question = dspy.InputField(desc="The user's question")

    cypher = dspy.OutputField(desc="A valid Cypher query string. No markdown.")


class RepairCypher(dspy.Signature):
    """Fix a broken Cypher query based on the database error message.
    
    Look at the <ERROR_MSG> and adjust the <BAD_CYPHER> to fix syntax or schema violations.
    Ensure you still adhere to the schema rules (e.g. correct property names).
    """

    graph_schema = dspy.InputField(desc="The graph schema")
    original_question = dspy.InputField()
    bad_cypher = dspy.InputField(desc="The query that failed")
    error_msg = dspy.InputField(desc="The error returned by KuzuDB")

    repaired_cypher = dspy.OutputField(desc="The fixed Cypher query. No markdown.")


class GenerateAnswer(dspy.Signature):
    """You are a precise Data Reporter for a Nobel Prize database. 
    Your job is to convert structured database rows into a natural language response.

    <CONTEXT_EXPLANATION>
    The provided <CONTEXT> consists of rows returned by a precise SQL/Cypher query that has ALREADY filtered the data based on the user's question.
    - If the context is `[{'name': 'Marie Curie'}]` and the question was "Who was born in Poland?", it means the database has ALREADY verified Marie Curie was born in Poland.
    - You do NOT need to see the field "Poland" in the context to trust this. The presence of the record IS the proof.
    </CONTEXT_EXPLANATION>

    <GUIDELINES>
    1. **Trust the Query**: Treat every record in the context as a valid answer. Do not filter them again yourself.
    2. **Completeness is Mandatory**: If the context contains 20 records, you MUST list all 20. Never summarize (e.g., do NOT say "and 5 others").
    3. **Missing Fields**: If the context lacks a specific column (like 'Death Date' or 'City'), state the information you DO have (Names, Years, Categories) and accept that the record is relevant.
    4. **Tone**: Be direct and objective, yet friendly.
    5. **Empty Context**: Only if the context is strictly `[]` (empty list), reply with "No information found in the database."
    </GUIDELINES>
    """
    question = dspy.InputField()
    context = dspy.InputField(desc="Structured data retrieved from the graph")
    answer = dspy.OutputField(desc="Natural language answer")


# =========================================================================
#  PART 2: DSPy Module (The Pipeline)
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
        # Generate
        response = self.generate(
            graph_schema=self.schema_str,
            exemplars=exemplars_str,
            question=question
        )
        cypher = response.cypher

        if self.use_postprocess:
            cypher = postprocess_cypher(cypher)

        # Validate & Repair Loop
        valid, error_msg = self.validate_cypher(cypher)
        max_retries = 3
        all_cyphers = [cypher]
        for i in range(max_retries):
            if valid:
                break
            # print(f"  [Refinement Attempt {i + 1}] Error: {error_msg[:100]}...")
            response = self.repair(
                graph_schema=self.schema_str,
                original_question=question,
                bad_cypher=cypher,
                error_msg=error_msg
            )
            cypher = response.repaired_cypher
            if self.use_postprocess:
                cypher = postprocess_cypher(cypher)
            all_cyphers.append(cypher)
            valid, error_msg = self.validate_cypher(cypher)

        return cypher, all_cyphers


# =========================================================================
#  Main GraphRAG Class
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

        # Configure DSPy
        dspy_model = llm.model
        if not dspy_model.startswith("openai/"):
            dspy_model = "openai/" + dspy_model

        print(f"Configuring DSPy with model: {dspy_model}")
        dspy_lm = dspy.LM(
            model=dspy_model,
            api_base=str(llm.client.base_url),
            api_key=llm.client.api_key,
            temperature=0.0
        )
        dspy.configure(lm=dspy_lm)
        dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

        # Setup DB
        path = Path(db_path)
        if not path.exists():
            raise FileNotFoundError(f"DB not found at {path}")
        
        self.db = kuzu.Database(str(path), read_only=True)
        self.conn = kuzu.Connection(self.db)
        self.schema_str = self._get_schema_str()

        # Initialize Pipeline
        self.pipeline = RefinedCypherGenerator(
            schema_str=self.schema_str,
            conn=self.conn,
            use_postprocess=use_postprocess
        )
        self.answer_gen = dspy.Predict(GenerateAnswer)

        # Setup Caching
        if cache_text2cypher:
            self._generate_cypher = self._make_cached_generator(
                maxsize=cache_maxsize, ttl=cache_ttl_seconds
            )
        else:
            self._generate_cypher = self._generate_cypher_no_cache

    def _get_schema_str(self) -> str:
        try:
            categories_df = self.conn.execute("MATCH (p:Prize) RETURN DISTINCT p.category").get_as_df()
            valid_categories = sorted(categories_df.iloc[:, 0].tolist())
            countries_df = self.conn.execute("MATCH (c:Country) RETURN DISTINCT c.name LIMIT 20").get_as_df()
            valid_countries_sample = sorted(countries_df.iloc[:, 0].tolist())
        except Exception as e:
            print(f"Schema generation warning: {e}")
            valid_categories = []
            valid_countries_sample = []

        schema_template = f"""
            <GRAPH_SCHEMA>
            <NODES>
                - (:Scholar) {{knownName: STRING, gender: STRING, birthDate: DATE, deathDate: DATE}}
                - (:Prize) {{category: STRING, awardYear: INTEGER}}
                   Categories: {json.dumps(valid_categories)}
                - (:Institution) {{name: STRING}}
                - (:City) {{name: STRING}}
                - (:Country) {{name: STRING}}
                   Examples: {json.dumps(valid_countries_sample)}...
            </NODES>
            <RELATIONSHIPS>
                - (:Scholar)-[:WON {{portion: STRING}}]->(:Prize)
                - (:Scholar)-[:BORN_IN]->(:City)
                - (:Scholar)-[:AFFILIATED_WITH]->(:Institution)
                - (:Institution)-[:IS_LOCATED_IN]->(:City)
                - (:City)-[:IS_CITY_IN]->(:Country)
            </RELATIONSHIPS>
            </GRAPH_SCHEMA>
        """
        return schema_template

    def _make_cached_generator(self, maxsize, ttl):
        @ttl_lru_cache(maxsize=maxsize, ttl_seconds=ttl)
        def _cached(complex_key: str):
            _, json_payload = complex_key.split("|", 1)
            data = json.loads(json_payload)
            return self._generate_cypher_no_cache(data['q'], data['ex'])

        def wrapper(q, ex):
            q_hash = hashlib.sha256(q.encode()).hexdigest()[:16]
            schema_hash = hashlib.sha256(self.schema_str.encode()).hexdigest()[:16]
            payload = json.dumps({"q": q, "ex": ex})
            key = f"{q_hash}_{schema_hash}|{payload}"
            return _cached(key)

        return wrapper

    def _generate_cypher_no_cache(self, question, exemplars):
        return self.pipeline(question=question, exemplars_str=exemplars)

    # --- LLM Interface Methods ---

    def generate(self, question: str):
        """
        Main entry point for Evaluation Pipeline.
        Returns: (answer, stats, cypher_query, all_tested_cyphers, results_context)
        """
        # Generate Cypher        
        exemplars_str = ""
        if self.use_exemplars:
            exs = select_exemplars(question, k=3)
            exemplars_str = format_exemplars_for_prompt(exs)

        cypher_query, all_tested_cyphers = self._generate_cypher(question, exemplars_str)
        
        # Execute Query
        results = []
        try:
            kuzu_result = self.conn.execute(cypher_query)
            # Convert Kuzu result to list of dicts
            columns = kuzu_result.get_column_names()
            while kuzu_result.has_next():
                row = kuzu_result.get_next()
                results.append(dict(zip(columns, row)))
        except Exception as e:
            print(f"Cypher Error: {e}")
            return "Error executing graph query.", {"error": str(e)}, cypher_query, [], []

        if not results:
            return "No information found in the database.", {"total": 0}, cypher_query, all_tested_cyphers, []
        
        # Generate Answer
        context_str = json.dumps(results, default=str, indent=2)
        prompt = f"Context:\n{context_str}\n\nQuestion: {question}"
        ans_response = self.answer_gen(question=question, context=context_str)
        answer = ans_response.answer

        # Return 5 values as expected by evaluate_pipeline.py
        return answer, {"total": 0}, cypher_query, all_tested_cyphers, results

    def stream(self, prompt: str, **kwargs) -> Iterable[str]:
        yield self.generate(prompt)[0]

    def query_with_stats(self, question: str) -> tuple[str, dict]:
        t0 = time.perf_counter()

        exemplars_str = ""
        if self.use_exemplars:
            exs = select_exemplars(question, k=3)
            exemplars_str = format_exemplars_for_prompt(exs)

        t1 = time.perf_counter()
        cypher, all_tested_cyphers = self._generate_cypher(question, exemplars_str)
        t2 = time.perf_counter()

        print(f"\n=== Final Cypher ===\n{cypher}")

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

        if not rows:
            print(">>> No data found. Fallback.")
            answer = self.base_llm.generate(question)
        else:
            context_str = json.dumps(rows, default=str)
            ans_response = self.answer_gen(question=question, context=context_str)
            answer = ans_response.answer

        t4 = time.perf_counter()
        stats = {
            "total": round(t4 - t0, 2),
            "text2cypher": round(t2 - t1, 2),
            "db_exec": round(t3 - t2, 2),
            "answer_gen": round(t4 - t3, 2)
        }
        return answer, stats
