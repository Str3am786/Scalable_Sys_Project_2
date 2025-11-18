TEXT2CYPHER_PROMPT = """
You are a Cypher Query expert for KuzuDB.
Use the provided Schema to translate the user question into a Cypher query.

Schema:
{schema}

Question: {question}

Instructions:
1. ONLY output the Cypher query. No markdown, no explanation.
2. Use the relationship directions defined in the schema.
3. For string matching, use `CONTAINS` and `to_lower()` if unsure of casing.
4. Return specific properties, not full nodes (e.g., RETURN s.name, p.year).

Cypher Query:
"""

ANSWER_PROMPT = """
Context information is below.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {question}
Answer:
"""