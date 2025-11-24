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
       - `Scholar`: match names on property `knownName` (e.g., s.knownName).
       - `Country`, `City`, `Institution`: match names on property `name`.
       - `Prize`: match category on property `category` (e.g., 'physics').
       - **Time**: Use `awardYear` for prize years.

    2. **Path Patterns (Strictly Follow These)**:
       - "Born in [Country]" -> `(:Scholar)-[:BORN_IN]->(:City)-[:IS_CITY_IN]->(:Country)`
       - "Worked in / Affiliated with [Country]" -> `(:Scholar)-[:AFFILIATED_WITH]->(:Institution)-[:IS_LOCATED_IN]->(:City)-[:IS_CITY_IN]->(:Country)`
       - "Won prize" -> `(:Scholar)-[:WON]->(:Prize)` (Filter by `{awardYear: ...}` ONLY if a specific year is asked).

    3. **String Matching**:
       - ALWAYS use `toLower(...)` for name comparisons.
       - Use `CONTAINS` for loose name matching (e.g., "Einstein").
       - **CRITICAL**: Do NOT use inline dict filtering (e.g. `{category: 'physics'}`). 
         ALWAYS use a `WHERE` clause (e.g. `WHERE toLower(p.category) = 'physics'`).

    4. **Formatting**:
       - Return only the specific properties needed (e.g., `return s.knownName, p.awardYear`).
       - **Dates**: Always enclose dates in single quotes (e.g., s.deathDate >= '1935-01-01').
       - Do NOT return whole nodes/maps unless necessary.
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
    """Answer the user question based on the provided database context that is 100% factually accurate for the user's question.
    
    <GUIDELINES>
    1. The <CONTEXT> provides the ground truth data for the user's question. Trust it fully and use it to answer the question directly.
    2. If the context is an empty list `[]` or clearly irrelevant, ONLY THEN output: "Not enough context".
    3. Be concise and friendly.
    4. If the context includes a long list of names, institutions, countries, cities, etc... state ALL of them.
    5. When mentioning dates, format them clearly (e.g., "October" instead of "10").
    6. Use any additional knowledge that you may have to add valuable information to the answer, without distorting the ground truth given as context
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

        # Apply ChainOfThought so the model to plans the query before writing it
        self.generate = dspy.ChainOfThought(Text2Cypher)
        self.repair = dspy.ChainOfThought(RepairCypher)

    def validate_cypher(self, cypher_query) -> tuple[bool, str]:
        try:
            # Use EXPLAIN to check the syntax of the query
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

        # Post-process
        # We post-process immediately to strip markdown before validation
        if self.use_postprocess:
            cypher = postprocess_cypher(cypher)

        # Validate & Repair Loop
        valid, error_msg = self.validate_cypher(cypher)
        max_retries = 3
        all_cyphers = [cypher]
        for i in range(max_retries):
            if valid:
                break
            
            print(f"  [Refinement Attempt {i + 1}] Error: {error_msg[:100]}...")

            response = self.repair(
                graph_schema=self.schema_str,
                original_question=question,
                bad_cypher=cypher,
                error_msg=error_msg
            )
            cypher = response.repaired_cypher

            # Post-process again after repair to ensure clean string
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


        # Configure DSPy with the provided LLM
        dspy_model = llm.model
        if not dspy_model.startswith("openai/"):
            dspy_model = "openai/" + dspy_model

        print(f"Configuring DSPy with model: {dspy_model}")
        dspy_lm = dspy.LM(
            model=dspy_model,
            api_base=str(llm.client.base_url),
            api_key=llm.client.api_key,
            temperature=0.0 # Deterministic for code generation
        )
        
        dspy.configure(lm=dspy_lm)
        dspy.configure_cache(
            enable_disk_cache=False,
            enable_memory_cache=False,
        )

        # Setup DB
        path = Path(db_path)
        if not path.exists():
            raise FileNotFoundError(f"DB not found at {path}")
        
        # Read-only mode for safety
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
            print("CACHEEEE")
            self._generate_cypher = self._make_cached_generator(
                maxsize=cache_maxsize, ttl=cache_ttl_seconds
            )
        else:
            print("NO SULLA CARTA CACHEEEE")
            self._generate_cypher = self._generate_cypher_no_cache

    def _get_schema_str(self) -> str:
        """
        Dynamically builds a Rich Schema by querying the database for valid values.
        Combines STATIC rules with DYNAMIC data.
        """
        try:
            # Get all prize categories
            categories_df = self.conn.execute("MATCH (p:Prize) RETURN DISTINCT p.category").get_as_df()
            valid_categories = sorted(categories_df.iloc[:, 0].tolist())
            
            # Get 5 example countries
            countries_df = self.conn.execute("MATCH (c:Country) RETURN DISTINCT c.name LIMIT 5").get_as_df()
            valid_countries_sample = sorted(countries_df.iloc[:, 0].tolist())
            
        except Exception as e:
            print(f"Schema generation warning: {e}")
            valid_categories = ["(Error fetching categories)"]
            valid_countries_sample = []

        # Inject fetched values into this static template
        schema_template = f"""
            <GRAPH_SCHEMA>
            <NODES>
                - (:Scholar)
                    Properties: {{knownName: STRING, gender: STRING, birthDate: DATE, deathDate: DATE}}
                    Note: 'gender' is usually 'male' or 'female'.
                - (:Prize)
                    Properties: {{category: STRING, awardYear: INTEGER, sortOrder: INTEGER}}
                    Allowed Categories: {json.dumps(valid_categories)}
                - (:Institution)
                    Properties: {{name: STRING}}
                - (:City)
                    Properties: {{name: STRING}}
                - (:Country)
                    Properties: {{name: STRING}}
                    Examples: {json.dumps(valid_countries_sample)}...
            </NODES>

            <RELATIONSHIPS>
                - (:Scholar)-[:WON]->(:Prize)
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
            # Create a unique key based on Question + Schema
            q_hash = hashlib.sha256(q.encode()).hexdigest()[:16]
            schema_hash = hashlib.sha256(self.schema_str.encode()).hexdigest()[:16]
            payload = json.dumps({"q": q, "ex": ex})
            key = f"{q_hash}_{schema_hash}|{payload}"
            return _cached(key)

        return wrapper

    def _generate_cypher_no_cache(self, question, exemplars):
        return self.pipeline(question=question, exemplars_str=exemplars)

    # --- LLM Interface Methods ---

    def generate(self, prompt: str, **kwargs) -> str:
        answer, stats , cypher, all_tested_cyphers= self.query_with_stats(prompt)
        return answer, stats, cypher, all_tested_cyphers

    def stream(self, prompt: str, **kwargs) -> Iterable[str]:
        # Simple fallback to generate for now
        yield self.generate(prompt)

    # --- Core Logic ---

    def query_with_stats(self, question: str) -> tuple[str, dict]:
        t0 = time.perf_counter()

        # 1. Retrieve Exemplars
        exemplars_str = ""
        if self.use_exemplars:
            exs = select_exemplars(question, k=3)
            exemplars_str = format_exemplars_for_prompt(exs)

        t1 = time.perf_counter()

        # 2. Generate Cypher with ChainOfThought + Repair Loop + Postprocess
        cypher, all_tested_cyphers = self._generate_cypher(question, exemplars_str)

        t2 = time.perf_counter()

        # --- LOGGING FOR DEBUGGING ---
        print(f"\n=== Final Post-Processed Cypher ===\n{cypher}")
        # -----------------------------

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
        
        # --- LOGGING FOR DEBUGGING ---
        print(f"\n=== Raw DB Results ({len(rows)} rows) ===")
        print(rows)
        print("==========================================\n")
        # -----------------------------

        # 4. Generate Answer
        if not rows:
            print(">>> No data found in graph. Falling back to base LLM.")
            # Fallback: standard LLM generation without graph context
            answer = self.base_llm.generate(question)
        else:
            context_str = json.dumps(rows, default=str)
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
            print(f"{k:>12}: {v:.4f}s")

        return answer, stats, cypher , all_tested_cyphers