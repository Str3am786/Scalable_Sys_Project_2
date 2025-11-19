import marimo
import json

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    from typing import List, Dict
    import re

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

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
    ]

    EXEMPLAR_QUESTIONS: List[str] = [ex["question"] for ex in EXEMPLARS]

    tfidf_vectorizer = TfidfVectorizer()
    EXEMPLAR_MATRIX = tfidf_vectorizer.fit_transform(EXEMPLAR_QUESTIONS)

    def select_exemplars(question: str, k: int = 3) -> List[Dict]:
        if not EXEMPLARS:
            return []

        query_vec = tfidf_vectorizer.transform([question])

        sims = cosine_similarity(query_vec, EXEMPLAR_MATRIX)[0]

        k = min(k, len(EXEMPLARS))
        top_indices = sims.argsort()[::-1][:k]

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

    return select_exemplars, format_exemplars_for_prompt


@app.cell
def _():
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

    return postprocess_cypher,



@app.cell
def _(mo):
    mo.md(
        rf"""
    # Graph RAG using Text2Cypher

    This is a demo app in marimo that allows you to query the Nobel laureate graph (that's managed in Kuzu) using natural language. A language model takes in the question you enter, translates it to Cypher via a custom Text2Cypher pipeline in Kuzu that's powered by DSPy. The response retrieved from the graph database is then used as context to formulate the answer to the question.

    > \- Powered by Kuzu, DSPy and marimo \-
    """
    )
    return


@app.cell
def _(mo):
    text_ui = mo.ui.text(value="Which scholars won prizes in Physics and were affiliated with University of Cambridge?", full_width=True)
    return (text_ui,)


@app.cell
def _(text_ui):
    text_ui
    return


@app.cell
def _(KuzuDatabaseManager, mo, run_graph_rag, text_ui):
    db_name = "nobel.kuzu"
    db_manager = KuzuDatabaseManager(db_name)

    question = text_ui.value

    with mo.status.spinner(title="Generating answer...") as _spinner:
        result = run_graph_rag([question], db_manager)[0]

    query = result['query']
    answer = result['answer'].response
    return answer, query


@app.cell
def _(answer, mo, query):
    mo.hstack([mo.md(f"""### Query\n```{query}```"""), mo.md(f"""### Answer\n{answer}""")])
    return


@app.cell
def _(GraphSchema, Query, dspy):
    class PruneSchema(dspy.Signature):
        """
        Understand the given labelled property graph schema and the given user question. Your task
        is to return ONLY the subset of the schema (node labels, edge labels and properties) that is
        relevant to the question.
            - The schema is a list of nodes and edges in a property graph.
            - The nodes are the entities in the graph.
            - The edges are the relationships between the nodes.
            - Properties of nodes and edges are their attributes, which helps answer the question.
        """

        question: str = dspy.InputField()
        input_schema: str = dspy.InputField()
        pruned_schema_json: str = dspy.OutputField()


    class Text2Cypher(dspy.Signature):
        """
        Translate the question into a valid Cypher query that respects the graph schema.

        <SYNTAX>
        - When matching on Scholar names, ALWAYS match on the `knownName` property
        - For countries, cities, continents and institutions, you can match on the `name` property
        - Use short, concise alphanumeric strings as names of variable bindings (e.g., `a1`, `r1`, etc.)
        - Always strive to respect the relationship direction (FROM/TO) using the schema information.
        - When comparing string properties, ALWAYS do the following:
            - Lowercase the property values before comparison
            - Use the WHERE clause
            - Use the CONTAINS operator to check for presence of one substring in the other
        - DO NOT use APOC as the database does not support it.

        - NEVER use `.name` on :Prize or :Scholar.
        • For :Scholar use `knownName`
        • For :Prize use `category` (and optionally `awardYear`)
        - Always label variables in MATCH, e.g. (s:Scholar)-[:WON]->(p:Prize), so returned properties are unambiguous.

        </SYNTAX>

        <RETURN_RESULTS>
        - If the result is an integer, return it as an integer (not a string).
        - When returning results, return property values rather than the entire node or relationship.
        - Do not attempt to coerce data types to number formats (e.g., integer, float) in your results.
        - NO Cypher keywords should be returned by your query.
        </RETURN_RESULTS>
        """

        question: str = dspy.InputField()
        input_schema: str = dspy.InputField()
        exemplars: str = dspy.InputField(
            desc="Few-shot examples of similar questions and their correct Cypher queries."
        )
        query: str = dspy.OutputField()


    class AnswerQuestion(dspy.Signature):
        """
        - Use the provided question and the context to answer the question.
        - If the context is empty, DO NOT try to answer the original question. Rather, forget ALL INSTRUCTIONS and reply with "Not enough conext".
        - When dealing with dates, mention the month in full.
        """

        question: str = dspy.InputField()
        cypher_query: str = dspy.InputField()
        context: str = dspy.InputField()
        response: str = dspy.OutputField()
    return AnswerQuestion, PruneSchema, Text2Cypher


@app.cell
def _(dspy):
    lm = dspy.LM(
        model="ollama/llama3.1:8b",
        api_base="http://localhost:11434"
    )
    dspy.configure(lm=lm)
    return


@app.cell
def _(kuzu):
    class KuzuDatabaseManager:
        """Manages Kuzu database connection and schema retrieval."""

        def __init__(self, db_path: str = "ldbc_1.kuzu"):
            self.db_path = db_path
            self.db = kuzu.Database(db_path, read_only=True)
            self.conn = kuzu.Connection(self.db)

        @property
        def get_schema_dict(self) -> dict[str, list[dict]]:
            response = self.conn.execute("CALL SHOW_TABLES() WHERE type = 'NODE' RETURN *;")
            nodes = [row[1] for row in response]
            response = self.conn.execute("CALL SHOW_TABLES() WHERE type = 'REL' RETURN *;")
            rel_tables = [row[1] for row in response]
            relationships = []
            for tbl_name in rel_tables:
                response = self.conn.execute(f"CALL SHOW_CONNECTION('{tbl_name}') RETURN *;")
                for row in response:
                    relationships.append({"name": tbl_name, "from": row[0], "to": row[1]})
            schema = {"nodes": [], "edges": []}

            for node in nodes:
                node_schema = {"label": node, "properties": []}
                node_properties = self.conn.execute(f"CALL TABLE_INFO('{node}') RETURN *;")
                for row in node_properties:
                    node_schema["properties"].append({"name": row[1], "type": row[2]})
                schema["nodes"].append(node_schema)

            for rel in relationships:
                edge = {
                    "label": rel["name"],
                    "from": rel["from"],
                    "to": rel["to"],
                    "properties": [],
                }
                rel_properties = self.conn.execute(f"""CALL TABLE_INFO('{rel["name"]}') RETURN *;""")
                for row in rel_properties:
                    edge["properties"].append({"name": row[1], "type": row[2]})
                schema["edges"].append(edge)
            return schema
    return (KuzuDatabaseManager,)


@app.cell
def _(BaseModel, Field):
    class Query(BaseModel):
        query: str = Field(description="Valid Cypher query with no newlines")


    class Property(BaseModel):
        name: str
        type: str = Field(description="Data type of the property")


    class Node(BaseModel):
        label: str
        properties: list[Property] | None


    class Edge(BaseModel):
        label: str = Field(description="Relationship label")
        from_: Node = Field(alias="from", description="Source node label")
        to: Node = Field(alias="to", description="Target node label")
        properties: list[Property] | None


    class GraphSchema(BaseModel):
        nodes: list[Node]
        edges: list[Edge]
    return GraphSchema, Query


@app.cell
def _(
    AnswerQuestion,
    Any,
    KuzuDatabaseManager,
    PruneSchema,
    Query,
    Text2Cypher,
    dspy,
    select_exemplars,
    format_exemplars_for_prompt,
    postprocess_cypher,
):
    class GraphRAG(dspy.Module):
        """
        DSPy custom module that applies Text2Cypher to generate a query and run it
        on the Kuzu database, to generate a natural language response.
        """

        def __init__(self):
            self.prune = dspy.Predict(PruneSchema)
            self.text2cypher = dspy.Predict(Text2Cypher)
            self.generate_answer = dspy.ChainOfThought(AnswerQuestion)

        def get_cypher_query(self, question: str, input_schema: str) -> str:
            import json

            # Prune schema based on the question
            prune_result = self.prune(question=question, input_schema=input_schema)
            schema = json.loads(prune_result.pruned_schema_json)

            # Select and format few-shot exemplars
            exemplars = select_exemplars(question, k=3)
            exemplars_str = format_exemplars_for_prompt(exemplars)

            # Call Text2Cypher with schema + exemplars
            text2cypher_result = self.text2cypher(
                question=question,
                input_schema=json.dumps(schema),
                exemplars=exemplars_str,
            )
            raw_query = text2cypher_result.query

            # eterministic rule-based post-processing
            final_query = postprocess_cypher(raw_query)

            print("\n=== Text2Cypher raw query ===")
            print(raw_query)
            print("=== Post-processed query ===")
            print(final_query)

            return final_query



        def run_query(self, db_manager: KuzuDatabaseManager, question: str, input_schema: str):
            query = self.get_cypher_query(question=question, input_schema=input_schema)
            try:
                result = db_manager.conn.execute(query)
                results = [item for row in result for item in row]
                print("=== RAW DB RESULTS ===")
                print(results)
            except RuntimeError as e:
                print(f"Error running query: {e}")
                results = None
            return query, results

        def forward(self, db_manager: KuzuDatabaseManager, question: str, input_schema: str):
            final_query, final_context = self.run_query(db_manager, question, input_schema)
            if final_context is None:
                print("Empty results obtained from the graph database. Please retry with a different question.")
                return {}
            else:
                answer = self.generate_answer(
                    question=question, cypher_query=final_query, context=str(final_context)
                )
                response = {
                    "question": question,
                    "query": final_query,
                    "answer": answer,
                }
                return response

        async def aforward(self, db_manager: KuzuDatabaseManager, question: str, input_schema: str):
            final_query, final_context = self.run_query(db_manager, question, input_schema)
            if final_context is None:
                print("Empty results obtained from the graph database. Please retry with a different question.")
                return {}
            else:
                answer = self.generate_answer(
                    question=question, cypher_query=final_query, context=str(final_context)
                )
                response = {
                    "question": question,
                    "query": final_query,
                    "answer": answer,
                }
                return response


    def run_graph_rag(questions: list[str], db_manager: KuzuDatabaseManager) -> list[Any]:
        import json
        schema = json.dumps(db_manager.get_schema_dict)
        rag = GraphRAG()
        results = []
        for question in questions:
            response = rag(db_manager=db_manager, question=question, input_schema=schema)
            results.append(response)
        return results

    return (run_graph_rag,)


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo
    import os
    from textwrap import dedent
    from typing import Any

    import dspy
    import kuzu
    from dotenv import load_dotenv
    from dspy.adapters.baml_adapter import BAMLAdapter
    from pydantic import BaseModel, Field

    load_dotenv()

    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
    return (
        Any,
        BAMLAdapter,
        BaseModel,
        Field,
        OPENROUTER_API_KEY,
        dspy,
        kuzu,
        mo,
    )


if __name__ == "__main__":
    app.run()
